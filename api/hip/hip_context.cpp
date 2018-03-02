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
#include "platform/runtime.hpp"


amd::Context* g_context = nullptr;

hipError_t hipInit(unsigned int flags)
{
  HIP_INIT_API(flags);

  if (!amd::Runtime::initialized()) {
    amd::Runtime::init();
  }

  // FIXME: move the global VDI context to hipInit.
  g_context = new amd::Context(
      amd::Device::getDevices(CL_DEVICE_TYPE_GPU, false), amd::Context::Info());
  if (!g_context) return hipErrorOutOfMemory;

  if (g_context && CL_SUCCESS != g_context->create(nullptr)) {
    g_context->release();
    return hipErrorUnknown;
  }

  return hipSuccess;
}

hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags,  hipDevice_t device)
{
  HIP_INIT_API(ctx, flags, device);

  return hipSuccess;
}

