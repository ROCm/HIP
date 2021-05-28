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
#include "hip_platform.hpp"
#include "platform/runtime.hpp"
#include "utils/flags.hpp"
#include "utils/versions.hpp"

std::vector<hip::Device*> g_devices;

namespace hip {

thread_local Device* g_device = nullptr;
thread_local std::stack<Device*> g_ctxtStack;
thread_local hipError_t g_lastError = hipSuccess;
std::once_flag g_ihipInitialized;
Device* host_device = nullptr;

void init() {
  if (!amd::Runtime::initialized()) {
    amd::IS_HIP = true;
    GPU_NUM_MEM_DEPENDENCY = 0;
    AMD_DIRECT_DISPATCH = flagIsDefault(AMD_DIRECT_DISPATCH) ? false : AMD_DIRECT_DISPATCH;
    amd::Runtime::init();
  }

  const std::vector<amd::Device*>& devices = amd::Device::getDevices(CL_DEVICE_TYPE_GPU, false);

  for (unsigned int i=0; i<devices.size(); i++) {
    const std::vector<amd::Device*> device(1, devices[i]);
    amd::Context* context = new amd::Context(device, amd::Context::Info());
    if (!context) return;

    // Enable active wait on the device by default
    devices[i]->SetActiveWait(true);

    if (context && CL_SUCCESS != context->create(nullptr)) {
      context->release();
    } else {
      g_devices.push_back(new Device(context, i));
    }
  }

  amd::Context* hContext = new amd::Context(devices, amd::Context::Info());
  if (!hContext) return;

  if (CL_SUCCESS != hContext->create(nullptr)) {
    hContext->release();
  }
  host_device = new Device(hContext, -1);

  PlatformState::instance().init();
}

Device* getCurrentDevice() {
  return g_device;
}

void setCurrentDevice(unsigned int index) {
  assert(index<g_devices.size());
  g_device = g_devices[index];
}

amd::HostQueue* getQueue(hipStream_t stream) {
 if (stream == nullptr) {
    return getNullStream();
  } else {
    constexpr bool WaitNullStreamOnly = true;
    amd::HostQueue* queue = reinterpret_cast<hip::Stream*>(stream)->asHostQueue();
    if (!(reinterpret_cast<hip::Stream*>(stream)->Flags() & hipStreamNonBlocking)) {
      iHipWaitActiveStreams(queue, WaitNullStreamOnly);
    }
    return queue;
  }
}

// ================================================================================================
amd::HostQueue* getNullStream(amd::Context& ctx) {
  for (auto& it : g_devices) {
    if (it->asContext() == &ctx) {
      return it->NullStream();
    }
  }
  // If it's a pure SVM allocation with system memory access, then it shouldn't matter which device
  // runtime selects by default
  if (hip::host_device->asContext() == &ctx) {
    // Return current...
    return getNullStream();
  }
  return nullptr;
}

// ================================================================================================
amd::HostQueue* getNullStream() {
  Device* device = getCurrentDevice();
  return device ? device->NullStream() : nullptr;
}

};

using namespace hip;

hipError_t hipInit(unsigned int flags) {
  HIP_INIT_API(hipInit, flags);

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags,  hipDevice_t device) {
  HIP_INIT_API(hipCtxCreate, ctx, flags, device);

  if (static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *ctx = reinterpret_cast<hipCtx_t>(g_devices[device]);

  // Increment ref count for device primary context
  g_devices[device]->retain();
  g_ctxtStack.push(g_devices[device]);

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  HIP_INIT_API(hipCtxSetCurrent, ctx);

  if (ctx == nullptr) {
    if(!g_ctxtStack.empty()) {
      g_ctxtStack.pop();
    }
  } else {
    hip::g_device = reinterpret_cast<hip::Device*>(ctx);
    if(!g_ctxtStack.empty()) {
      g_ctxtStack.pop();
    }
    g_ctxtStack.push(hip::getCurrentDevice());
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
  HIP_INIT_API(hipCtxGetCurrent, ctx);

  *ctx = reinterpret_cast<hipCtx_t>(hip::getCurrentDevice());

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig) {
  HIP_INIT_API(hipCtxGetSharedMemConfig, pConfig);

  *pConfig = hipSharedMemBankSizeFourByte;

  HIP_RETURN(hipSuccess);
}

hipError_t hipRuntimeGetVersion(int *runtimeVersion) {
  HIP_INIT_API(hipRuntimeGetVersion, runtimeVersion);

  if (!runtimeVersion) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // HIP_VERSION = HIP_VERSION_MAJOR*100 + HIP_MINOR_VERSION
  *runtimeVersion = HIP_VERSION;

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxDestroy(hipCtx_t ctx) {
  HIP_INIT_API(hipCtxDestroy, ctx);

  hip::Device* dev = reinterpret_cast<hip::Device*>(ctx);
  if (dev == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Need to remove the ctx of calling thread if its the top one
  if (!g_ctxtStack.empty() && g_ctxtStack.top() == dev) {
    g_ctxtStack.pop();
  }

  // Remove context from global context list
  for (unsigned int i = 0; i < g_devices.size(); i++) {
    if (g_devices[i] == dev) {
      // Decrement ref count for device primary context
      dev->release();
    }
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
  HIP_INIT_API(hipCtxPopCurrent, ctx);

  hip::Device** dev = reinterpret_cast<hip::Device**>(ctx);
  if (!g_ctxtStack.empty()) {
    if (dev != nullptr) {
      *dev = g_ctxtStack.top();
    }
    g_ctxtStack.pop();
  } else {
    DevLogError("Context Stack empty \n");
    HIP_RETURN(hipErrorInvalidContext);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  HIP_INIT_API(hipCtxPushCurrent, ctx);

  hip::Device* dev = reinterpret_cast<hip::Device*>(ctx);
  if (dev == nullptr) {
    HIP_RETURN(hipErrorInvalidContext);
  }

  hip::g_device = dev;
  g_ctxtStack.push(hip::getCurrentDevice());

  HIP_RETURN(hipSuccess);
}

hipError_t hipDriverGetVersion(int* driverVersion) {
  HIP_INIT_API(hipDriverGetVersion, driverVersion);

  if (!driverVersion) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // HIP_VERSION = HIP_VERSION_MAJOR*100 + HIP_MINOR_VERSION
  *driverVersion = HIP_VERSION;

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxGetDevice(hipDevice_t* device) {
  HIP_INIT_API(hipCtxGetDevice, device);

  if (device != nullptr) {
    *device = hip::getCurrentDevice()->deviceId();
    HIP_RETURN(hipSuccess);
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipErrorInvalidContext);
}

hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion) {
  HIP_INIT_API(hipCtxGetApiVersion, apiVersion);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig) {
  HIP_INIT_API(hipCtxGetCacheConfig, cacheConfig);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
  HIP_INIT_API(hipCtxSetCacheConfig, cacheConfig);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
  HIP_INIT_API(hipCtxSetSharedMemConfig, config);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxSynchronize(void) {
  HIP_INIT_API(hipCtxSynchronize, 1);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxGetFlags(unsigned int* flags) {
  HIP_INIT_API(hipCtxGetFlags, flags);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active) {
  HIP_INIT_API(hipDevicePrimaryCtxGetState, dev, flags, active);

  if (static_cast<unsigned int>(dev) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (flags != nullptr) {
    *flags = 0;
  }

  if (active != nullptr) {
    *active = (g_devices[dev] == hip::getCurrentDevice())? 1 : 0;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
  HIP_INIT_API(hipDevicePrimaryCtxRelease, dev);

  if (static_cast<unsigned int>(dev) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
  HIP_INIT_API(hipDevicePrimaryCtxRetain, pctx, dev);

  if (static_cast<unsigned int>(dev) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  if (pctx == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pctx = reinterpret_cast<hipCtx_t>(g_devices[dev]);

  HIP_RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
  HIP_INIT_API(hipDevicePrimaryCtxReset, dev);

  HIP_RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
  HIP_INIT_API(hipDevicePrimaryCtxSetFlags, dev, flags);

  if (static_cast<unsigned int>(dev) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  } else {
    HIP_RETURN(hipErrorContextAlreadyInUse);
  }
}
