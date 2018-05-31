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

namespace hip {

void syncStreams() {
  amd::ScopedLock lock(streamSetLock);

  for (const auto& it : streamSet) {
    it->finish();
  }
}

};

static hipError_t ihipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags) {
  amd::Device* device = hip::getCurrentContext()->devices()[0];

  amd::HostQueue* queue = new amd::HostQueue(*hip::getCurrentContext(), *device, 0,
                                             amd::CommandQueue::RealTimeDisabled,
                                             amd::CommandQueue::Priority::Normal);

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

  return ihipStreamCreateWithFlags(stream, flags);
}

hipError_t hipStreamCreate(hipStream_t *stream) {
  HIP_INIT_API(stream);

  return ihipStreamCreateWithFlags(stream, hipStreamDefault);
}

hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags) {
  HIP_INIT_API(stream, flags);

  amd::HostQueue* hostQueue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  auto it = streamSet.find(hostQueue);

  if(flags != nullptr) {
    *flags = (it == streamSet.end()) ? hipStreamNonBlocking : hipStreamDefault;
  } else {
    return hipErrorInvalidValue;
  }

  return hipSuccess;
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
    return hipErrorUnknown;
  }

  hostQueue->finish();

  return hipSuccess;
}

hipError_t hipStreamDestroy(hipStream_t stream) {
  HIP_INIT_API(stream);

  if (stream == nullptr) {
    return hipErrorInvalidResourceHandle;
  }

  amd::ScopedLock lock(streamSetLock);

  amd::HostQueue* hostQueue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  hostQueue->release();
  streamSet.erase(hostQueue);

  return hipSuccess;
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
  HIP_INIT_API(stream, event, flags);

  if (stream == nullptr || event == nullptr) {
    return hipErrorInvalidResourceHandle;
  }

  amd::HostQueue* hostQueue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  cl_event clEvent = as_cl(e->event_);

  amd::Command::EventWaitList eventWaitList;
  cl_int err = amd::clSetEventWaitList(eventWaitList, *hostQueue, 1, &clEvent);
  if (err != CL_SUCCESS) {
    return hipErrorUnknown;
  }

  amd::Command* command = new amd::Marker(*hostQueue, true, eventWaitList);
  if (command == NULL) {
    return hipErrorOutOfMemory;
  }
  command->enqueue();
  command->release();

  return hipSuccess;
}

hipError_t hipStreamQuery(hipStream_t stream) {
  HIP_INIT_API(stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags) {
  HIP_INIT_API(stream, callback, userData, flags);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}


