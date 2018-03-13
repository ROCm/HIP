/*
Copyright (c) 2018 - present Advanced Micro Devices, Inc. All rights reserved.

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

hipError_t hipDeviceGet(hipDevice_t *device, int deviceId)
{
  HIP_INIT_API(device, deviceId);

  if (device != nullptr) {
    *device = deviceId;
  } else {
    return hipErrorInvalidValue;
  }

  return hipSuccess;
};

hipError_t hipFuncSetCacheConfig (const void* func, hipFuncCache_t cacheConfig) {

  HIP_INIT_API(cacheConfig);

  // No way to set cache config yet.

  return hipSuccess;
}

hipError_t hipDeviceTotalMem (size_t *bytes, hipDevice_t device) {

  HIP_INIT_API(bytes, device);

  if (device < 0 || device > (cl_int)g_context->devices().size()) {
    return hipErrorInvalidDevice;
  }

  if (bytes == nullptr) {
    return hipErrorInvalidValue;
  }

  auto* deviceHandle = g_context->devices()[device];
  const auto& info = deviceHandle->info();

  *bytes = info.globalMemSize_;

  return hipSuccess;
}

hipError_t hipDeviceComputeCapability(int *major, int *minor, hipDevice_t device) {

  HIP_INIT_API(major, minor, device);

  if (device < 0 || device > (cl_int)g_context->devices().size()) {
    return hipErrorInvalidDevice;
  }

  if (major == nullptr || minor == nullptr) {
    return hipErrorInvalidValue;
  }

  auto* deviceHandle = g_context->devices()[device];
  const auto& info = deviceHandle->info();
  *major = info.gfxipVersion_ / 100;
  *minor = info.gfxipVersion_ % 100;

  return hipSuccess;
}

hipError_t hipDeviceGetName(char *name, int len, hipDevice_t device) {

  HIP_INIT_API((void*)name, len, device);

  if (device < 0 || device > (cl_int)g_context->devices().size()) {
    return hipErrorInvalidDevice;
  }

  if (name == nullptr) {
    return hipErrorInvalidValue;
  }

  auto* deviceHandle = g_context->devices()[device];
  const auto& info = deviceHandle->info();

  len = ((cl_uint)len < ::strlen(info.boardName_)) ? len : 128;
  ::strncpy(name, info.boardName_, len);

  return hipSuccess;
}
