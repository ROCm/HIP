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

#ifndef HIP_SRC_HIP_INTERNAL_H
#define HIP_SRC_HIP_INTERNAL_H

#include "cl_common.hpp"

#define HIP_INIT()\
  amd::Thread* thread = amd::Thread::current();              \
  if (!CL_CHECK_THREAD(thread)) {                            \
    return hipErrorOutOfMemory;                              \
  }


// This macro should be called at the beginning of every HIP API.
#define HIP_INIT_API(...) \
    HIP_INIT()

extern thread_local amd::Context* g_context;
extern std::vector<amd::Context*> g_devices;

hipError_t hipDeviceGetCount(int* count);

#endif // HIP_SRC_HIP_INTERNAL_H
