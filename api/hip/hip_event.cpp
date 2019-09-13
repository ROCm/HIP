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

#include "hip_event.hpp"

namespace hip {

bool Event::ready() {
  event_->notifyCmdQueue();

  return (event_->status() == CL_COMPLETE);
}

hipError_t Event::query() {
  amd::ScopedLock lock(lock_);

  if (event_ == nullptr) {
    return hipErrorInvalidResourceHandle;
  }

  return ready() ? hipSuccess : hipErrorNotReady;
}

hipError_t Event::synchronize() {
  amd::ScopedLock lock(lock_);

  if (event_ == nullptr) {
    return hipErrorInvalidResourceHandle;
  }

  event_->awaitCompletion();

  return hipSuccess;
}

hipError_t Event::elapsedTime(Event& eStop, float& ms) {
  amd::ScopedLock startLock(lock_);

  if (this == &eStop) {
    if (event_ == nullptr) {
      return hipErrorInvalidResourceHandle;
    }

    if (flags & hipEventDisableTiming) {
      return hipErrorInvalidResourceHandle;
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
    return hipErrorInvalidResourceHandle;
  }

  if ((flags | eStop.flags) & hipEventDisableTiming) {
    return hipErrorInvalidResourceHandle;
  }

  if (!ready() || !eStop.ready()) {
    return hipErrorNotReady;
  }

  if (event_ != eStop.event_) {
    ms = static_cast<float>(static_cast<int64_t>(eStop.event_->profilingInfo().end_ -
                            event_->profilingInfo().start_))/1000000.f;
  } else {
    ms = 0.f;
  }

  return hipSuccess;
}

hipError_t Event::streamWait(amd::HostQueue* hostQueue, uint flags) {
  if (stream_ == hostQueue) return hipSuccess;

  amd::ScopedLock lock(lock_);
  bool retain = false;

  if (event_ == nullptr) {
    event_ = stream_->getLastQueuedCommand(true);
    retain = true;
  }

  if (!event_->notifyCmdQueue()) {
    return hipErrorUnknown;
  }
  amd::Command::EventWaitList eventWaitList;
  eventWaitList.push_back(event_);

  amd::Command* command = new amd::Marker(*hostQueue, false, eventWaitList);
  if (command == NULL) {
    return hipErrorOutOfMemory;
  }
  command->enqueue();
  command->release();

  if (retain) {
    event_->release();
    event_ = nullptr;
  }

  return hipSuccess;
}

void Event::addMarker(amd::HostQueue* queue, amd::Command* command) {
  amd::ScopedLock lock(lock_);

  stream_ = queue;

  if (event_ != nullptr) {
    event_->release();
  }

  event_ = &command->event();
}

}

hipError_t ihipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  if (event == nullptr) {
    return hipErrorInvalidValue;
  }

  unsigned supportedFlags = hipEventDefault | hipEventBlockingSync | hipEventDisableTiming |
                            hipEventReleaseToDevice | hipEventReleaseToSystem;
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
    return hipErrorInvalidResourceHandle;
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  return e->query();
}

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  HIP_INIT_API(event, flags);

  HIP_RETURN(ihipEventCreateWithFlags(event, flags));
}  

hipError_t hipEventCreate(hipEvent_t* event) {
  HIP_INIT_API(event);

  HIP_RETURN(ihipEventCreateWithFlags(event, 0));
}

hipError_t hipEventDestroy(hipEvent_t event) {
  HIP_INIT_API(event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  delete reinterpret_cast<hip::Event*>(event);

  HIP_RETURN(hipSuccess);
}

hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  HIP_INIT_API(ms, start, stop);

  if (start == nullptr || stop == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  if (ms == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::Event* eStart = reinterpret_cast<hip::Event*>(start);
  hip::Event* eStop  = reinterpret_cast<hip::Event*>(stop);

  HIP_RETURN(eStart->elapsedTime(*eStop, *ms));
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_INIT_API(event, stream);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  amd::HostQueue* queue = hip::getQueue(stream);

  amd::Command* command = queue->getLastQueuedCommand(true);

  if (command == nullptr) {
    command = new amd::Marker(*queue, false);
    command->enqueue();
  }

  e->addMarker(queue, command);

  HIP_RETURN(hipSuccess);
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  HIP_INIT_API(event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  HIP_RETURN(e->synchronize());
}

hipError_t hipEventQuery(hipEvent_t event) {
  HIP_INIT_API(event);

  HIP_RETURN(ihipEventQuery(event));
}
