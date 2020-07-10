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

static amd::Monitor streamSetLock{"Guards global stream set"};
static std::unordered_set<hip::Stream*> streamSet;

// Internal structure for stream callback handler
class StreamCallback {
   public:
    StreamCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                  amd::Command* command)
        : stream_(stream), callBack_(callback),
          userData_(userData), command_(command) {
        };
    hipStream_t stream_;
    hipStreamCallback_t callBack_;
    void* userData_;
    amd::Command* command_;
};

namespace hip {

// ================================================================================================
Stream::Stream(hip::Device* dev, Priority p,
    unsigned int f, bool null_stream, const std::vector<uint32_t>& cuMask)
  : queue_(nullptr), lock_("Stream Callback lock"), device_(dev),
    priority_(p), flags_(f), null_(null_stream), cuMask_(cuMask) {}

// ================================================================================================
bool Stream::Create() {
  cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
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
  amd::HostQueue* queue = new amd::HostQueue(*device_->asContext(), *device_->devices()[0], properties,
                             amd::CommandQueue::RealTimeDisabled, p, cuMask_);

  // Create a host queue
  bool result = (queue != nullptr) ? queue->create() : false;
  // Insert just created stream into the list of the blocking queues
  if (result) {
    amd::ScopedLock lock(streamSetLock);
    streamSet.insert(this);
    queue_ = queue;
  } else {
    queue_ = queue;
    Destroy();
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
void Stream::Destroy() {
  if (queue_ != nullptr) {
    amd::ScopedLock lock(streamSetLock);
    streamSet.erase(this);

    queue_->release();
    queue_ = nullptr;
  }
  delete this;
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

void Stream::syncNonBlockingStreams() {
  amd::ScopedLock lock(streamSetLock);
  for (auto& it : streamSet) {
    if (it->Flags() & hipStreamNonBlocking) {
      it->asHostQueue()->finish();
    }
  }
}

};

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
        if ((command != nullptr) &&
            // Check the current active status
            (command->status() != CL_COMPLETE)) {
          eventWaitList.push_back(command);
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
  hip::Stream* hStream = new hip::Stream(hip::getCurrentDevice(), priority, flags, false, cuMask);

  if (hStream == nullptr || !hStream->Create()) {
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
    *flags = reinterpret_cast<hip::Stream*>(stream)->Flags();
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamSynchronize(hipStream_t stream) {
  HIP_INIT_API(hipStreamSynchronize, stream);

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

  reinterpret_cast<hip::Stream*>(stream)->Destroy();

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  HIP_INIT_API(hipStreamWaitEvent, stream, event, flags);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  amd::HostQueue* queue = hip::getQueue(stream);

  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  HIP_RETURN(e->streamWait(queue, flags));
}

// ================================================================================================
hipError_t hipStreamQuery(hipStream_t stream) {
  HIP_INIT_API(hipStreamQuery, stream);

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

  amd::HostQueue* hostQueue = hip::getQueue(stream);
  amd::Command* command = hostQueue->getLastQueuedCommand(true);
  if (command == nullptr) {
    amd::Command::EventWaitList eventWaitList;
    command = new amd::Marker(*hostQueue, kMarkerDisableFlush, eventWaitList);
    command->enqueue();
  }
  amd::Event& event = command->event();
  StreamCallback* cbo = new StreamCallback(stream, callback, userData, command);

  if(!event.setCallback(CL_COMPLETE, ihipStreamCallback, reinterpret_cast<void*>(cbo))) {
    command->release();
    return hipErrorInvalidHandle;
  }

  event.notifyCmdQueue();

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
    *priority = static_cast<int>(reinterpret_cast<hip::Stream*>(stream)->GetPriority());
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);

}
