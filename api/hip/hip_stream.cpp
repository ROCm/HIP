/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
#include "hip_event.hpp"
#include "thread/monitor.hpp"

static amd::Monitor streamSetLock("Guards global stream set");
static std::unordered_set<amd::HostQueue*> streamSet;

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

void syncStreams() {
  amd::ScopedLock lock(streamSetLock);

  for (const auto& it : streamSet) {
    it->finish();
  }
}

};

void CL_CALLBACK ihipStreamCallback(cl_event event, cl_int command_exec_status, void* user_data) {

  hipError_t status = hipSuccess;
  StreamCallback* cbo = reinterpret_cast<StreamCallback*>(user_data);
  cbo->callBack_(cbo->stream_, status, cbo->userData_);
  cbo->command_->release();
  delete cbo;
}

static hipError_t ihipStreamCreate(hipStream_t *stream, unsigned int flags, amd::CommandQueue::Priority priority) {
  amd::Device* device = hip::getCurrentContext()->devices()[0];

  cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
  amd::HostQueue* queue = new amd::HostQueue(*hip::getCurrentContext(), *device, properties,
                                             amd::CommandQueue::RealTimeDisabled,
                                             priority);

  if (queue == nullptr || !queue->create()) {
    return hipErrorOutOfMemory;
  }

  if (!(flags & hipStreamNonBlocking)) {
    hip::syncStreams();

    {
      amd::ScopedLock lock(streamSetLock);
      streamSet.insert(queue);
    }
  }

  *stream = reinterpret_cast<hipStream_t>(as_cl(queue));

  return hipSuccess;
}

hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags) {
  HIP_INIT_API(stream, flags);

  HIP_RETURN(ihipStreamCreate(stream, flags, amd::CommandQueue::Priority::Normal));
}

hipError_t hipStreamCreate(hipStream_t *stream) {
  HIP_INIT_API(stream);

  HIP_RETURN(ihipStreamCreate(stream, hipStreamDefault, amd::CommandQueue::Priority::Normal));
}

hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
  HIP_INIT_API(stream, flags, priority);

  if (priority > static_cast<int>(amd::CommandQueue::Priority::High)) {
    priority = static_cast<int>(amd::CommandQueue::Priority::High);
  } else if (priority < static_cast<int>(amd::CommandQueue::Priority::Normal)) {
    priority = static_cast<int>(amd::CommandQueue::Priority::Normal);
  }

  return HIP_RETURN(ihipStreamCreate(stream, flags, static_cast<amd::CommandQueue::Priority>(priority)));
}

hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  HIP_INIT_API(leastPriority, greatestPriority);

  if (leastPriority != nullptr) {
    *leastPriority = static_cast<int>(amd::CommandQueue::Priority::Normal);
  }
  if (greatestPriority != nullptr) {
    // Only report one kind of priority for now.
    *greatestPriority = static_cast<int>(amd::CommandQueue::Priority::Normal);
  }
  return HIP_RETURN(hipSuccess);
}

hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags) {
  HIP_INIT_API(stream, flags);

  amd::HostQueue* hostQueue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  auto it = streamSet.find(hostQueue);

  if(flags != nullptr) {
    *flags = (it == streamSet.end()) ? hipStreamNonBlocking : hipStreamDefault;
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  HIP_INIT_API(stream);

  amd::HostQueue* hostQueue;

  if (stream == nullptr) {
    hip::syncStreams();

    hostQueue = hip::getNullStream();
  } else {
    hip::getNullStream()->finish();

    hostQueue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }

  if (hostQueue == nullptr) {
    HIP_RETURN(hipErrorUnknown);
  }

  hostQueue->finish();

  HIP_RETURN(hipSuccess);
}

hipError_t hipStreamDestroy(hipStream_t stream) {
  HIP_INIT_API(stream);

  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  amd::ScopedLock lock(streamSetLock);

  amd::HostQueue* hostQueue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();

  // Release last tracked command
  hostQueue->setLastQueuedCommand(nullptr);

  hostQueue->release();
  streamSet.erase(hostQueue);

  HIP_RETURN(hipSuccess);
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  HIP_INIT_API(stream, event, flags);

  if (stream == nullptr) {
    stream = reinterpret_cast<hipStream_t>(as_cl(hip::getNullStream()));
  }

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  return HIP_RETURN(e->streamWait(stream, flags));
}

hipError_t hipStreamQuery(hipStream_t stream) {
  HIP_INIT_API(stream);

  amd::HostQueue* hostQueue;
  if (stream == nullptr) {
    hostQueue = hip::getNullStream();
  } else {
    hostQueue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }

  amd::Command* command = hostQueue->getLastQueuedCommand(false);
  if (command == nullptr) {
    HIP_RETURN(hipSuccess);
  }

  amd::Event& event = command->event();

  if (command->type() != CL_COMMAND_MARKER) {
    event.notifyCmdQueue();
  }
  HIP_RETURN((command->status() == CL_COMPLETE) ? hipSuccess : hipErrorNotReady);
}

hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags) {
  HIP_INIT_API(stream, callback, userData, flags);

  amd::HostQueue* hostQueue = as_amd(reinterpret_cast<cl_command_queue>
                              (stream))->asHostQueue();
  amd::Command* command = hostQueue->getLastQueuedCommand(true);
  amd::Event& event = command->event();
  StreamCallback* cbo = new StreamCallback(stream, callback, userData, command);

  if(!event.setCallback(CL_COMPLETE, ihipStreamCallback, reinterpret_cast<void*>(cbo))) {
    command->release();
    return hipErrorInvalidResourceHandle;
  }

  event.notifyCmdQueue();

  HIP_RETURN(hipSuccess);
}


