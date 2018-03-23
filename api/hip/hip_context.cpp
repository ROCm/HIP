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
#include "utils/versions.hpp"
#include <stack>

thread_local amd::Context* g_context = nullptr;
thread_local std::stack<amd::Context*> g_ctxtStack;

std::vector<amd::Context*> g_devices;

hipError_t hipInit(unsigned int flags)
{
  HIP_INIT_API(flags);

  if (!amd::Runtime::initialized()) {
    amd::Runtime::init();
  }

  const std::vector<amd::Device*>& devices = amd::Device::getDevices(CL_DEVICE_TYPE_GPU, false);

  for (unsigned int i=0; i<devices.size(); i++) {
    const std::vector<amd::Device*> device(1, devices[i]);
    amd::Context* context = new amd::Context(device, amd::Context::Info());
    if (!context) return hipErrorOutOfMemory;

    if (context && CL_SUCCESS != context->create(nullptr)) {
      context->release();
    } else {
      g_devices.push_back(context);
    }
  }

  return hipSuccess;
}

hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags,  hipDevice_t device)
{
  HIP_INIT_API(ctx, flags, device);

  if (static_cast<size_t>(device) >= g_devices.size()) {
    return hipErrorInvalidValue;
  }

  *ctx = reinterpret_cast<hipCtx_t>(g_devices[device]);

  // Increment ref count for device primary context
  g_devices[device]->retain();

  return hipSuccess;
}

hipError_t hipCtxSetCurrent(hipCtx_t ctx)
{
  HIP_INIT_API(ctx);

  if (ctx == nullptr) {
    if(!g_ctxtStack.empty()) {
      g_ctxtStack.pop();
    }
  } else {
    g_context = reinterpret_cast<amd::Context*>(as_amd(ctx));
    if(!g_ctxtStack.empty()) {
      g_ctxtStack.pop();
    }
    g_ctxtStack.push(g_context);
  }

  return hipSuccess;
}

hipError_t hipCtxGetCurrent(hipCtx_t* ctx)
{
  HIP_INIT_API(ctx);

  *ctx = reinterpret_cast<hipCtx_t>(g_context);

  return hipSuccess;
}

hipError_t hipRuntimeGetVersion(int *runtimeVersion)
{
  HIP_INIT_API(runtimeVersion);

  if (!runtimeVersion) {
    return hipErrorInvalidValue;
  }

  *runtimeVersion = AMD_PLATFORM_BUILD_NUMBER;

  return hipSuccess;
}

hipError_t hipCtxDestroy(hipCtx_t ctx)
{
  HIP_INIT_API(ctx);

  amd::Context* amdContext = reinterpret_cast<amd::Context*>(as_amd(ctx));
  if (amdContext == nullptr) {
    return hipErrorInvalidValue;
  }

  // Need to remove the ctx of calling thread if its the top one
  if (g_context == amdContext) {
    g_ctxtStack.pop();
  }

  // Remove context from global context list
  for (unsigned int i = 0; i < g_devices.size(); i++) {
    if (g_devices[i] == amdContext) {
      // Decrement ref count for device primary context
      amdContext->release();
    }
  }

  return hipSuccess;
}


hipError_t hipCtxPopCurrent(hipCtx_t* ctx)
{
  HIP_INIT_API(ctx);

  amd::Context* amdContext = reinterpret_cast<amd::Context*>(as_amd(ctx));
  if (amdContext == nullptr) {
    return hipErrorInvalidContext;
  }

  if (!g_ctxtStack.empty()) {
    amdContext = g_ctxtStack.top();
    g_ctxtStack.pop();
  } else {
    return hipErrorInvalidContext;
  }

  return hipSuccess;
}

hipError_t hipCtxPushCurrent(hipCtx_t ctx)
{
  HIP_INIT_API(ctx);

  amd::Context* amdContext = reinterpret_cast<amd::Context*>(as_amd(ctx));
  if (amdContext == nullptr) {
    return hipErrorInvalidContext;
  }

  g_context = amdContext;
  g_ctxtStack.push(g_context);

  return hipSuccess;
}