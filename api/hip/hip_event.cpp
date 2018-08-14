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

  if (e->event_ == nullptr) {
    return hipErrorInvalidResourceHandle;
  }

  e->event_->notifyCmdQueue();

  return (e->event_->status() == CL_COMPLETE) ? hipSuccess : hipErrorNotReady;
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

  hip::Event* eStart = reinterpret_cast<hip::Event*>(start);
  hip::Event* eStop  = reinterpret_cast<hip::Event*>(stop);

  if (eStart->event_ == nullptr ||
      eStop->event_  == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  if ((eStart->flags | eStop->flags) & hipEventDisableTiming) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  if (ihipEventQuery(start) == hipErrorNotReady ||
      ihipEventQuery(stop) == hipErrorNotReady) {
    HIP_RETURN(hipErrorNotReady);
  }

  if (ms == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *ms = static_cast<float>(eStop->event_->profilingInfo().submitted_ - eStart->event_->profilingInfo().submitted_)/1000000.f;

  HIP_RETURN(hipSuccess);
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_INIT_API(event, stream);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  if (stream == nullptr) {
    e->stream_ = hip::getNullStream();
  } else {
    e->stream_ = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }
  amd::Command* command = (e->flags & hipEventDisableTiming)? new amd::Marker(*e->stream_, true) :
    new hip::TimerMarker(*e->stream_);
  command->enqueue();

  if (e->event_ != nullptr) {
    e->event_->release();
  }

  e->event_ = &command->event();

  HIP_RETURN(hipSuccess);
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  HIP_INIT_API(event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  if (e->event_ == nullptr) {
    HIP_RETURN(hipErrorInvalidResourceHandle);
  }

  e->event_->awaitCompletion();

  HIP_RETURN(hipSuccess);
}

hipError_t hipEventQuery(hipEvent_t event) {
  HIP_INIT_API(event);

  HIP_RETURN(ihipEventQuery(event));
}
