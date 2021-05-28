/* Copyright (c) 2015-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>

#include "hip_event.hpp"

void ipcEventCallback(hipStream_t stream, hipError_t status, void* user_data)
{
  std::atomic<int> *signal = reinterpret_cast<std::atomic<int>*>(user_data);
  signal->store(0);
  return;
}

namespace hip {

bool Event::ready() {
  if (event_->status() != CL_COMPLETE) {
    event_->notifyCmdQueue();
  }

  return (event_->status() == CL_COMPLETE);
}

hipError_t Event::query() {
  amd::ScopedLock lock(lock_);

  // If event is not recorded, event_ is null, hence return hipSuccess
  if (event_ == nullptr) {
    return hipSuccess;
  }

  return ready() ? hipSuccess : hipErrorNotReady;
}

hipError_t Event::synchronize() {
  amd::ScopedLock lock(lock_);

  // If event is not recorded, event_ is null, hence return hipSuccess
  if (event_ == nullptr) {
    return hipSuccess;
  }

  event_->awaitCompletion();

  return hipSuccess;
}

hipError_t Event::elapsedTime(Event& eStop, float& ms) {
  amd::ScopedLock startLock(lock_);

  if (this == &eStop) {
    if (event_ == nullptr) {
      return hipErrorInvalidHandle;
    }

    if (flags & hipEventDisableTiming) {
      return hipErrorInvalidHandle;
    }

    if (!ready()) {
      return hipErrorNotReady;
    }

    ms = 0.f;
    return hipSuccess;
  }
  amd::ScopedLock stopLock(eStop.lock_);

  if (event_ == nullptr ||
      eStop.event_  == nullptr) {
    return hipErrorInvalidHandle;
  }

  if ((flags | eStop.flags) & hipEventDisableTiming) {
    return hipErrorInvalidHandle;
  }

  if (!ready() || !eStop.ready()) {
    return hipErrorNotReady;
  }

  if (event_ == eStop.event_ && recorded_ && eStop.recorded_) {
    // Events are the same, which indicates the stream is empty and likely
    // eventRecord is called on another stream. For such cases insert and measure a
    // marker.
    amd::Command* command = new amd::Marker(*event_->command().queue(), kMarkerDisableFlush);
    command->enqueue();
    command->awaitCompletion();
    ms = static_cast<float>(static_cast<int64_t>(command->event().profilingInfo().end_) - time())/1000000.f;
    command->release();
  } else {
    ms = static_cast<float>(eStop.time() - time())/1000000.f;
  }
  return hipSuccess;
}

int64_t Event::time() const {
  assert(event_ != nullptr);
  if (recorded_) {
    return static_cast<int64_t>(event_->profilingInfo().end_);
  } else {
    return static_cast<int64_t>(event_->profilingInfo().start_);
  }
}

hipError_t Event::streamWait(amd::HostQueue* hostQueue, uint flags) {
  if ((event_ == nullptr) || (event_->command().queue() == hostQueue)) {
    return hipSuccess;
  }

  amd::ScopedLock lock(lock_);
  bool retain = false;

  if (!event_->notifyCmdQueue()) {
    return hipErrorLaunchOutOfResources;
  }
  amd::Command::EventWaitList eventWaitList;
  eventWaitList.push_back(event_);

  amd::Command* command = new amd::Marker(*hostQueue, kMarkerDisableFlush, eventWaitList);
  if (command == NULL) {
    return hipErrorOutOfMemory;
  }
  command->enqueue();
  command->release();

  return hipSuccess;
}

void Event::addMarker(amd::HostQueue* queue, amd::Command* command, bool record) {
  // Keep the lock always at the beginning of this to avoid a race. SWDEV-277847
  amd::ScopedLock lock(lock_);

  if (command == nullptr) {
    bool recordExplicitGpuTs = !queue->properties().test(CL_QUEUE_PROFILING_ENABLE) &&
                               !(flags & hipEventDisableTiming);
    // Always submit a EventMarker. This would submit a NOP with a signal.
    command = new hip::EventMarker(*queue, !kMarkerDisableFlush, recordExplicitGpuTs);
    command->enqueue();
  }

  if (event_ == &command->event()) return;

  if (event_ != nullptr) {
    event_->release();
  }

  event_ = &command->event();
  recorded_ = record;
}

}

hipError_t ihipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  if (event == nullptr) {
    return hipErrorInvalidValue;
  }
#if !defined(_MSC_VER)
  unsigned supportedFlags = hipEventDefault | hipEventBlockingSync | hipEventDisableTiming |
                            hipEventReleaseToDevice | hipEventReleaseToSystem | hipEventInterprocess;
#else
  unsigned supportedFlags = hipEventDefault | hipEventBlockingSync | hipEventDisableTiming |
                            hipEventReleaseToDevice | hipEventReleaseToSystem;
#endif
  const unsigned releaseFlags = (hipEventReleaseToDevice | hipEventReleaseToSystem);

  const bool illegalFlags =
      (flags & ~supportedFlags) ||             // can't set any unsupported flags.
      (flags & releaseFlags) == releaseFlags;  // can't set both release flags

  if (!illegalFlags) {
    hip::Event* e = new hip::Event(flags);
    if (e == nullptr) {
      return hipErrorOutOfMemory;
    }
    *event = reinterpret_cast<hipEvent_t>(e);
  } else {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

hipError_t ihipEventQuery(hipEvent_t event) {
  if (event == nullptr) {
    return hipErrorInvalidHandle;
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  if ((e->flags & hipEventInterprocess) && (e->ipc_evt_.ipc_shmem_)) {
    int prev_read_idx = e->ipc_evt_.ipc_shmem_->read_index;
    int offset = (prev_read_idx % IPC_SIGNALS_PER_EVENT);
    if (e->ipc_evt_.ipc_shmem_->read_index < prev_read_idx+IPC_SIGNALS_PER_EVENT && e->ipc_evt_.ipc_shmem_->signal[offset] != 0) {
      return hipErrorNotReady;
    }
    return hipSuccess;
  } else {
    return e->query();
  }
}

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  HIP_INIT_API(hipEventCreateWithFlags, event, flags);
  HIP_RETURN(ihipEventCreateWithFlags(event, flags), *event);
}

hipError_t hipEventCreate(hipEvent_t* event) {
  HIP_INIT_API(hipEventCreate, event);
  HIP_RETURN(ihipEventCreateWithFlags(event, 0), *event);
}

hipError_t hipEventDestroy(hipEvent_t event) {
  HIP_INIT_API(hipEventDestroy, event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  if ((e->flags & hipEventInterprocess) && (e->ipc_evt_.ipc_shmem_)) {
    int owners = -- e->ipc_evt_.ipc_shmem_->owners;
    // Make sure event is synchronized
    hipEventSynchronize(event);
    if (!amd::Os::MemoryUnmapFile(e->ipc_evt_.ipc_shmem_,sizeof(hip::ihipIpcEventShmem_t))) {
      HIP_RETURN(hipErrorInvalidHandle);
    }
  }
  delete e;
  HIP_RETURN(hipSuccess);
}

hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  HIP_INIT_API(hipEventElapsedTime, ms, start, stop);

  if (start == nullptr || stop == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  if (ms == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::Event* eStart = reinterpret_cast<hip::Event*>(start);
  hip::Event* eStop  = reinterpret_cast<hip::Event*>(stop);

  if (eStart->deviceId() != eStop->deviceId()) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  HIP_RETURN(eStart->elapsedTime(*eStop, *ms), "Elapsed Time = ", *ms);
}

// ================================================================================================
bool createIpcEventShmemIfNeeded(hip::Event::ihipIpcEvent_t& ipc_evt) {
#if !defined(_MSC_VER)
  if (ipc_evt.ipc_shmem_) {
    // ipc_shmem_ already created, no need to create it again
    return true;
  }
  char name_template[] = "/tmp/eventXXXXXX";
  int temp_fd = mkstemp(name_template);

  ipc_evt.ipc_name_ = name_template;
  ipc_evt.ipc_name_.replace(0, 5, "/hip_");
  if (!amd::Os::MemoryMapFileTruncated(ipc_evt.ipc_name_.c_str(),
      const_cast<const void**> (reinterpret_cast<void**>(&(ipc_evt.ipc_shmem_))), sizeof(hip::ihipIpcEventShmem_t))) {
    return false;
  }
  ipc_evt.ipc_shmem_->owners = 1;
  ipc_evt.ipc_shmem_->read_index = -1;
  ipc_evt.ipc_shmem_->write_index = 0;
  for (uint32_t sig_idx = 0; sig_idx < IPC_SIGNALS_PER_EVENT; ++sig_idx) {
    ipc_evt.ipc_shmem_->signal[sig_idx] = 0;
  }

  close(temp_fd);
  return true;
#else
  return false;
#endif
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_INIT_API(hipEventRecord, event, stream);

  STREAM_CAPTURE(hipEventRecord, stream, event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  amd::HostQueue* queue = hip::getQueue(stream);

  if (g_devices[e->deviceId()]->devices()[0] != &queue->device()) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  bool isRecorded = e->isRecorded();
  if ((e->flags & hipEventInterprocess) && !isRecorded) {
    amd::Command* command = new amd::Marker(*queue, kMarkerDisableFlush);
    amd::Event& tEvent = command->event();
    createIpcEventShmemIfNeeded(e->ipc_evt_);
    int write_index = e->ipc_evt_.ipc_shmem_->write_index++;
    int offset = write_index % IPC_SIGNALS_PER_EVENT;
    while (e->ipc_evt_.ipc_shmem_->signal[offset] != 0) {
      amd::Os::sleep(1);
    }
    // Lock signal.
    e->ipc_evt_.ipc_shmem_->signal[offset] = 1;
    e->ipc_evt_.ipc_shmem_->owners_device_id = e->deviceId();

    std::atomic<int> *signal = &e->ipc_evt_.ipc_shmem_->signal[offset];
    StreamCallback* cbo = new StreamCallback(stream,
                    reinterpret_cast<hipStreamCallback_t> (ipcEventCallback), signal, command);
    if (!tEvent.setCallback(CL_COMPLETE, ihipStreamCallback,cbo)) {
      command->release();
      return hipErrorInvalidHandle;
    }
    command->enqueue();
    tEvent.notifyCmdQueue();
    // Update read index to indicate new signal.
    int expected = write_index - 1;
    while (!e->ipc_evt_.ipc_shmem_->read_index.compare_exchange_weak(expected, write_index)) {
      amd::Os::sleep(1);
    }
  } else {
    e->addMarker(queue, nullptr, true);
  }
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipEventSynchronize(hipEvent_t event) {
  HIP_INIT_API(hipEventSynchronize, event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  if ((e->flags & hipEventInterprocess) && (e->ipc_evt_.ipc_shmem_)) {
    int prev_read_idx = e->ipc_evt_.ipc_shmem_->read_index;
    if (prev_read_idx >= 0) {
      int offset = (prev_read_idx % IPC_SIGNALS_PER_EVENT);
      while ((e->ipc_evt_.ipc_shmem_->read_index < prev_read_idx + IPC_SIGNALS_PER_EVENT)
               && (e->ipc_evt_.ipc_shmem_->signal[offset] != 0)) {
        amd::Os::sleep(1);
      }
    }
    HIP_RETURN(hipSuccess);
  } else {
    HIP_RETURN(e->synchronize());
  }
}

hipError_t hipEventQuery(hipEvent_t event) {
  HIP_INIT_API(hipEventQuery, event);
  HIP_RETURN(ihipEventQuery(event));
}

hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
  HIP_INIT_API(hipIpcGetEventHandle, handle, event);
#if !defined(_MSC_VER)
  if (handle == nullptr || event == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  if (!(e->flags & hipEventInterprocess)) {
    HIP_RETURN(hipErrorInvalidConfiguration);
  }
  if (!createIpcEventShmemIfNeeded(e->ipc_evt_)) {
    HIP_RETURN(hipErrorInvalidConfiguration);
  }
  ihipIpcEventHandle_t* iHandle = reinterpret_cast<ihipIpcEventHandle_t*>(handle);
  memset(iHandle->shmem_name, 0, HIP_IPC_HANDLE_SIZE);
  e->ipc_evt_.ipc_name_.copy(iHandle->shmem_name, std::string::npos);
  HIP_RETURN(hipSuccess);
#else
  assert(0 && "Unimplemented");
  HIP_RETURN(hipErrorNotSupported);
#endif
}

hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
  HIP_INIT_API(NONE, event, handle);
#if !defined(_MSC_VER)
  hipError_t hip_err = hipSuccess;
  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip_err = ihipEventCreateWithFlags(event, hipEventDisableTiming | hipEventInterprocess);
  if (hip_err != hipSuccess) {
    HIP_RETURN(hip_err);
  }
  hip::Event* e = reinterpret_cast<hip::Event*>(*event);
  ihipIpcEventHandle_t* iHandle = reinterpret_cast<ihipIpcEventHandle_t*>(&handle);
  hip::Event::ihipIpcEvent_t& ipc_evt = e->ipc_evt_;
  ipc_evt.ipc_name_ = iHandle->shmem_name;
  if (!amd::Os::MemoryMapFileTruncated(ipc_evt.ipc_name_.c_str(),
                    (const void**) &(ipc_evt.ipc_shmem_), sizeof(hip::ihipIpcEventShmem_t))) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  ipc_evt.ipc_shmem_->owners += 1;
  e->setDeviceId(ipc_evt.ipc_shmem_->owners_device_id.load());

  HIP_RETURN(hipSuccess);
#else
  assert(0 && "Unimplemented");
  HIP_RETURN(hipErrorNotSupported);
#endif
}
