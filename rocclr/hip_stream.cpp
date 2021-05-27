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
#include "hip_internal.hpp"
#include "hip_event.hpp"
#include "thread/monitor.hpp"
#include "hip_prof_api.h"

extern api_callbacks_table_t callbacks_table;

static amd::Monitor streamSetLock{"Guards global stream set"};
static std::unordered_set<hip::Stream*> streamSet;
namespace hip {

// ================================================================================================
Stream::Stream(hip::Device* dev, Priority p, unsigned int f, bool null_stream,
               const std::vector<uint32_t>& cuMask, hipStreamCaptureStatus captureStatus)
    : queue_(nullptr),
      lock_("Stream Callback lock"),
      device_(dev),
      priority_(p),
      flags_(f),
      null_(null_stream),
      cuMask_(cuMask),
      captureStatus_(captureStatus) {}

// ================================================================================================
Stream::~Stream() {
  if (queue_ != nullptr) {
    amd::ScopedLock lock(streamSetLock);
    streamSet.erase(this);

    queue_->release();
    queue_ = nullptr;
  }
}

hipError_t Stream::EndCapture() {
  for (auto event : captureEvents_) {
    hip::Event* e = reinterpret_cast<hip::Event*>(event);
    e->EndCapture();
  }
  for (auto stream : parallelCaptureStreams_) {
    hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
    s->EndCapture();
  }
  captureStatus_ = hipStreamCaptureStatusNone;
  pCaptureGraph_ = nullptr;
  originStream_ = false;
  parentStream_ = nullptr;
  lastCapturedNodes_.clear();
  parallelCaptureStreams_.clear();
  captureEvents_.clear();

  return hipSuccess;
}
// ================================================================================================
bool Stream::Create() {
  // Enable queue profiling if a profiler is attached which sets the callback_table flag
  // or if we force it with env var. This would enable time stamp collection for every
  // command submitted to the stream(queue).
  bool isProfilerAttached = callbacks_table.is_enabled();
  cl_command_queue_properties properties = (isProfilerAttached || HIP_FORCE_QUEUE_PROFILING) ?
                                             CL_QUEUE_PROFILING_ENABLE : 0;
  amd::CommandQueue::Priority p;
  switch (priority_) {
    case Priority::High:
      p = amd::CommandQueue::Priority::High;
      break;
    case Priority::Low:
      p = amd::CommandQueue::Priority::Low;
      break;
    case Priority::Normal:
    default:
      p = amd::CommandQueue::Priority::Normal;
      break;
  }
  amd::HostQueue* queue = new amd::HostQueue(*device_->asContext(), *device_->devices()[0],
                                             properties, amd::CommandQueue::RealTimeDisabled,
                                             p, cuMask_);

  // Create a host queue
  bool result = (queue != nullptr) ? queue->create() : false;
  // Insert just created stream into the list of the blocking queues
  if (result) {
    amd::ScopedLock lock(streamSetLock);
    streamSet.insert(this);
    queue_ = queue;
    queue->vdev()->profilerAttach(isProfilerAttached);
  } else if (queue != nullptr) {
    queue->release();
  }

  return result;
}

// ================================================================================================
amd::HostQueue* Stream::asHostQueue(bool skip_alloc) {
  if (queue_ != nullptr) {
    return queue_;
  }
  // Access to the stream object is lock protected, because possible allocation
  amd::ScopedLock l(Lock());
  if (queue_ == nullptr) {
    // Create the host queue for the first time
    if (!skip_alloc) {
      Create();
    }
  }
  return queue_;
}

// ================================================================================================
void Stream::Finish() const {
  if (queue_ != nullptr) {
    queue_->finish();
  }
}

// ================================================================================================
int Stream::DeviceId() const {
  return device_->deviceId();
}

int Stream::DeviceId(const hipStream_t hStream) {
  hip::Stream* s = reinterpret_cast<hip::Stream*>(hStream);
  int deviceId = (s != nullptr)? s->DeviceId() : ihipGetDevice();
  assert(deviceId >= 0 && deviceId < static_cast<int>(g_devices.size()));
  return deviceId;
}

void Stream::syncNonBlockingStreams() {
  amd::ScopedLock lock(streamSetLock);
  for (auto& it : streamSet) {
    if (it->Flags() & hipStreamNonBlocking) {
      it->asHostQueue()->finish();
    }
  }
}

// ================================================================================================
bool isValid(hipStream_t stream) {
  // NULL stream is always valid
  if (stream == nullptr) {
    return true;
  }

  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  amd::ScopedLock lock(streamSetLock);
  if (streamSet.find(s) == streamSet.end()) {
    return false;
  }
  return true;
}

};// hip namespace

// ================================================================================================
void iHipWaitActiveStreams(amd::HostQueue* blocking_queue, bool wait_null_stream) {
  amd::Command::EventWaitList eventWaitList;
  {
    amd::ScopedLock lock(streamSetLock);

    for (const auto& stream : streamSet) {
      amd::HostQueue* active_queue = stream->asHostQueue();
      // If it's the current device
      if ((&active_queue->device() == &blocking_queue->device()) &&
          // Make sure it's a default stream
          ((stream->Flags() & hipStreamNonBlocking) == 0) &&
          // and it's not the current stream
          (active_queue != blocking_queue) &&
          // check for a wait on the null stream
          (stream->Null() == wait_null_stream)) {
        // Get the last valid command
        amd::Command* command = active_queue->getLastQueuedCommand(true);
        if (command != nullptr) {
          // Check the current active status
          if (command->status() != CL_COMPLETE) {
            command->notifyCmdQueue();
            eventWaitList.push_back(command);
          } else {
            command->release();
          }
        }
        // Nullstream, hence there is nothing else to wait
        if (wait_null_stream) {
          break;
        }
      }
    }
  }

  // Check if we have to wait anything
  if (eventWaitList.size() > 0) {
    amd::Command* command = new amd::Marker(*blocking_queue, kMarkerDisableFlush, eventWaitList);
    if (command != nullptr) {
      command->enqueue();
      command->release();
    }
  }

  // Release all active commands. It's safe after the marker was enqueued
  for (const auto& it : eventWaitList) {
    it->release();
  }
}

// ================================================================================================
void CL_CALLBACK ihipStreamCallback(cl_event event, cl_int command_exec_status, void* user_data) {
  hipError_t status = hipSuccess;
  StreamCallback* cbo = reinterpret_cast<StreamCallback*>(user_data);
  cbo->callBack_(cbo->stream_, status, cbo->userData_);
  cbo->command_->release();
  delete cbo;
}

// ================================================================================================
static hipError_t ihipStreamCreate(hipStream_t* stream,
                                  unsigned int flags, hip::Stream::Priority priority,
                                  const std::vector<uint32_t>& cuMask = {}) {
  if (flags != hipStreamDefault && flags != hipStreamNonBlocking) {
    return hipErrorInvalidValue;
  }
  hip::Stream* hStream = new hip::Stream(hip::getCurrentDevice(), priority, flags, false, cuMask);

  if (hStream == nullptr || !hStream->Create()) {
    delete hStream;
    return hipErrorOutOfMemory;
  }

  *stream = reinterpret_cast<hipStream_t>(hStream);

  return hipSuccess;
}

// ================================================================================================
hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags) {
  HIP_INIT_API(hipStreamCreateWithFlags, stream, flags);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipStreamCreate(stream, flags, hip::Stream::Priority::Normal), *stream);
}

// ================================================================================================
hipError_t hipStreamCreate(hipStream_t *stream) {
  HIP_INIT_API(hipStreamCreate, stream);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipStreamCreate(stream, hipStreamDefault, hip::Stream::Priority::Normal), *stream);
}

// ================================================================================================
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
  HIP_INIT_API(hipStreamCreateWithPriority, stream, flags, priority);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::Stream::Priority streamPriority;
  if (priority <= hip::Stream::Priority::High) {
    streamPriority = hip::Stream::Priority::High;
  } else if (priority >= hip::Stream::Priority::Low) {
    streamPriority = hip::Stream::Priority::Low;
  } else {
    streamPriority = hip::Stream::Priority::Normal;
  }

  HIP_RETURN(ihipStreamCreate(stream, flags, streamPriority), *stream);
}

// ================================================================================================
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  HIP_INIT_API(hipDeviceGetStreamPriorityRange, leastPriority, greatestPriority);

  if (leastPriority != nullptr) {
    *leastPriority = hip::Stream::Priority::Low;
  }
  if (greatestPriority != nullptr) {
    *greatestPriority = hip::Stream::Priority::High;
  }
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
  HIP_INIT_API(hipStreamGetFlags, stream, flags);

  if ((flags != nullptr) && (stream != nullptr)) {
    if (!hip::isValid(stream)) {
      return HIP_RETURN(hipErrorContextIsDestroyed);
    }
    *flags = reinterpret_cast<hip::Stream*>(stream)->Flags();
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamSynchronize(hipStream_t stream) {
  HIP_INIT_API(hipStreamSynchronize, stream);

  if (!hip::isValid(stream)) {
    return HIP_RETURN(hipErrorContextIsDestroyed);
  }

  // Wait for the current host queue
  hip::getQueue(stream)->finish();

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamDestroy(hipStream_t stream) {
  HIP_INIT_API(hipStreamDestroy, stream);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  if (!hip::isValid(stream)) {
    return HIP_RETURN(hipErrorContextIsDestroyed);
  }

  delete reinterpret_cast<hip::Stream*>(stream);

  HIP_RETURN(hipSuccess);
}

struct CallbackData {
    int previous_read_index;
    hip::ihipIpcEventShmem_t *shmem;
};

void WaitThenDecrementSignal(hipStream_t stream, hipError_t status, void* user_data){
    CallbackData *data = (CallbackData *)user_data;
    int offset = data->previous_read_index % IPC_SIGNALS_PER_EVENT;
    while (data->shmem->read_index < data->previous_read_index+IPC_SIGNALS_PER_EVENT
                    && data->shmem->signal[offset] != 0) {
    }
    delete data;
}

// ================================================================================================
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  HIP_INIT_API(hipStreamWaitEvent, stream, event, flags);

  EVENT_CAPTURE(hipStreamWaitEvent, event, stream, flags);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  if (!hip::isValid(stream)) {
    return HIP_RETURN(hipErrorContextIsDestroyed);
  }

  amd::HostQueue* queue = hip::getQueue(stream);

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  if (e->flags & hipEventInterprocess) {
    amd::Command* command = new amd::Marker(*queue, false);
    auto t{new CallbackData{e->ipc_evt_.ipc_shmem_->read_index, e->ipc_evt_.ipc_shmem_}};
    StreamCallback* cbo = new StreamCallback(stream,
                    reinterpret_cast<hipStreamCallback_t> (WaitThenDecrementSignal), t, command);
    if (!command->setCallback(CL_COMPLETE, ihipStreamCallback,cbo)) {
      command->release();
      return hipErrorInvalidHandle;
    }
    command->enqueue();
    command->awaitCompletion();
    HIP_RETURN(hipSuccess);
  } else {
    HIP_RETURN(e->streamWait(queue, flags));
  }
}

// ================================================================================================
hipError_t hipStreamQuery(hipStream_t stream) {
  HIP_INIT_API(hipStreamQuery, stream);

  if (!hip::isValid(stream)) {
    return HIP_RETURN(hipErrorContextIsDestroyed);
  }

  amd::HostQueue* hostQueue = hip::getQueue(stream);

  amd::Command* command = hostQueue->getLastQueuedCommand(true);
  if (command == nullptr) {
    // Nothing was submitted to the queue
    HIP_RETURN(hipSuccess);
  }

  amd::Event& event = command->event();
  if (command->type() != 0) {
    event.notifyCmdQueue();
  }
  hipError_t status = (command->status() == CL_COMPLETE) ? hipSuccess : hipErrorNotReady;
  command->release();
  HIP_RETURN(status);
}

// ================================================================================================
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags) {
  HIP_INIT_API(hipStreamAddCallback, stream, callback, userData, flags);
  //flags - Reserved for future use, must be 0
  if (callback == nullptr || flags != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!hip::isValid(stream)) {
    return HIP_RETURN(hipErrorContextIsDestroyed);
  }

  amd::HostQueue* hostQueue = hip::getQueue(stream);
  amd::Command* last_command = hostQueue->getLastQueuedCommand(true);
  amd::Command::EventWaitList eventWaitList;
  if (last_command != nullptr) {
    eventWaitList.push_back(last_command);
  }
  amd::Command* command = new amd::Marker(*hostQueue, !kMarkerDisableFlush, eventWaitList);
  if (command == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  StreamCallback* cbo = new StreamCallback(stream, callback, userData, command);

  if ((cbo == nullptr) || !command->setCallback(CL_COMPLETE, ihipStreamCallback, cbo)) {
    command->release();
    if (last_command != nullptr) {
      last_command->release();
    }
    return hipErrorInvalidHandle;
  }
  // Retain callback command for the blocking marker
  command->retain();
  command->enqueue();
  // @note: don't release the command here, because it will be released after HIP callback
  if (last_command != nullptr) {
    last_command->release();
  }
  // Add the new barrier to stall the stream, until the callback is done
  eventWaitList.clear();
  eventWaitList.push_back(command);
  amd::Command* block_command = new amd::Marker(*hostQueue, !kMarkerDisableFlush, eventWaitList);
  if (block_command == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  block_command->enqueue();
  block_command->release();
  // Release the callback marker
  command->release();

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t cuMaskSize,
                                        const uint32_t* cuMask) {
  HIP_INIT_API(hipExtStreamCreateWithCUMask, stream, cuMaskSize, cuMask);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }
  if (cuMaskSize == 0 || cuMask == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const std::vector<uint32_t> cuMaskv(cuMask, cuMask + cuMaskSize);

  HIP_RETURN(ihipStreamCreate(stream, hipStreamDefault, hip::Stream::Priority::Normal, cuMaskv), *stream);
}

// ================================================================================================
hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
  HIP_INIT_API(hipStreamGetPriority, stream, priority);
  if ((priority != nullptr) && (stream != nullptr)) {
    if (!hip::isValid(stream)) {
      return HIP_RETURN(hipErrorContextIsDestroyed);
    }
    *priority = static_cast<int>(reinterpret_cast<hip::Stream*>(stream)->GetPriority());
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);

}

// ================================================================================================
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t* cuMask) {
  HIP_INIT_API(hipExtStreamGetCUMask, stream, cuMaskSize, cuMask);

  if (cuMask == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  int deviceId = hip::getCurrentDevice()->deviceId();
  auto* deviceHandle = g_devices[deviceId]->devices()[0];
  const auto& info = deviceHandle->info();

  // find the minimum cuMaskSize required to present the CU mask bit-array in a patch of 32 bits
  // and return error if the cuMaskSize argument is less than cuMaskSizeRequired
  uint32_t cuMaskSizeRequired = info.maxComputeUnits_ / 32 +
    ((info.maxComputeUnits_ % 32) ? 1 : 0);

  if (cuMaskSize < cuMaskSizeRequired) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // make a default CU mask bit-array where all CUs are active
  // this default mask will be returned when there is no
  // custom or global CU mask defined
  std::vector<uint32_t> defaultCUMask;
  uint32_t temp = 0;
  uint32_t bit_index = 0;
  for (uint32_t i = 0; i < info.maxComputeUnits_; i++) {
    temp |= 1UL << bit_index;
    if (bit_index >= 32) {
      defaultCUMask.push_back(temp);
      temp = 0;
      bit_index = 0;
      temp |= 1UL << bit_index;
    }
    bit_index += 1;
  }
  if (bit_index != 0) {
    defaultCUMask.push_back(temp);
  }

  // if the stream is null then either return globalCUMask_ (if it is defined)
  // or return defaultCUMask
  if (stream == nullptr) {
    if (info.globalCUMask_.size() != 0) {
      std::copy(info.globalCUMask_.begin(), info.globalCUMask_.end(), cuMask);
    } else {
      std::copy(defaultCUMask.begin(), defaultCUMask.end(), cuMask);
    }
  } else {
  // if the stream is not null then get the stream's CU mask and return one of the below cases
  // case1 if globalCUMask_ is defined then return the AND of globalCUMask_ and stream's CU mask
  // case2 if globalCUMask_ is not defined then retuen AND of defaultCUMask and stream's CU mask
  // in both cases above if stream's CU mask is empty then either globalCUMask_ (for case1)
  // or defaultCUMask(for case2) will be returned
    std::vector<uint32_t> streamCUMask;
    streamCUMask = reinterpret_cast<hip::Stream*>(stream)->GetCUMask();
    std::vector<uint32_t> mask = {};
    if (info.globalCUMask_.size() != 0) {
      for (uint32_t i = 0; i < std::min(streamCUMask.size(), info.globalCUMask_.size()); i++) {
        mask.push_back(streamCUMask[i] & info.globalCUMask_[i]);
      }
    } else {
      for (uint32_t i = 0; i < std::min(streamCUMask.size(), defaultCUMask.size()); i++) {
        mask.push_back(streamCUMask[i] & defaultCUMask[i]);
      }
      // check to make sure after ANDing streamCUMask (custom-defined) with global CU mask,
      //we have non-zero mask, oterwise just return either globalCUMask_ or defaultCUMask
      bool zeroCUMask = true;
      for (auto m : mask) {
        if (m != 0) {
          zeroCUMask = false;
          break;
        }
      }
      if (zeroCUMask) {
        mask = (info.globalCUMask_.size() != 0) ? info.globalCUMask_ : defaultCUMask;
      }
      std::copy(mask.begin(), mask.end(), cuMask);
    }
  }
  HIP_RETURN(hipSuccess);
}
