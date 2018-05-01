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

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  HIP_INIT_API(event, flags);

  hip::Event* e = new hip::Event(flags);

  if (e == nullptr) {
    return hipErrorOutOfMemory;
  }

  *event = reinterpret_cast<hipEvent_t>(e);

  return hipSuccess;
}

hipError_t hipEventCreate(hipEvent_t* event) {
  HIP_INIT_API(event);

  hip::Event* e = new hip::Event(0);

  if (e == nullptr) {
    return hipErrorOutOfMemory;
  }

  *event = reinterpret_cast<hipEvent_t>(e);

  return hipSuccess;
}

hipError_t hipEventDestroy(hipEvent_t event) {
  HIP_INIT_API(event);

  delete reinterpret_cast<hip::Event*>(event);

  return hipSuccess;
}

hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  HIP_INIT_API(ms, start, stop);

  return hipErrorUnknown;
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_INIT_API(event, stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  HIP_INIT_API(event);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipEventQuery(hipEvent_t event) {
  HIP_INIT_API(event);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}
