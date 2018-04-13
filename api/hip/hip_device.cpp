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

hipError_t hipDeviceGet(hipDevice_t *device, int deviceId) {
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

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    return hipErrorInvalidDevice;
  }

  if (bytes == nullptr) {
    return hipErrorInvalidValue;
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();

  *bytes = info.globalMemSize_;

  return hipSuccess;
}

hipError_t hipDeviceComputeCapability(int *major, int *minor, hipDevice_t device) {

  HIP_INIT_API(major, minor, device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    return hipErrorInvalidDevice;
  }

  if (major == nullptr || minor == nullptr) {
    return hipErrorInvalidValue;
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();
  *major = info.gfxipVersion_ / 100;
  *minor = info.gfxipVersion_ % 100;

  return hipSuccess;
}

hipError_t hipDeviceGetCount(int* count) {
  HIP_INIT_API(count);

  return ihipDeviceGetCount(count);
}

hipError_t ihipDeviceGetCount(int* count) {
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }

  // Get all available devices
  *count = g_devices.size();

  return hipSuccess;
}

hipError_t hipDeviceGetName(char *name, int len, hipDevice_t device) {

  HIP_INIT_API((void*)name, len, device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    return hipErrorInvalidDevice;
  }

  if (name == nullptr) {
    return hipErrorInvalidValue;
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();

  len = ((cl_uint)len < ::strlen(info.boardName_)) ? len : 128;
  ::strncpy(name, info.boardName_, len);

  return hipSuccess;
}

hipError_t hipGetDeviceProperties ( hipDeviceProp_t* props, hipDevice_t device ) {
  HIP_INIT_API(props, device);

  if (props == nullptr) {
    return hipErrorInvalidValue;
  }

  if (unsigned(device) >= g_devices.size()) {
    return hipErrorInvalidDevice;
  }
  auto* deviceHandle = g_devices[device]->devices()[0];

  hipDeviceProp_t deviceProps = {0};

  const auto& info = deviceHandle->info();
  ::strncpy(deviceProps.name, info.boardName_, 128);
  deviceProps.totalGlobalMem = info.globalMemSize_;
  deviceProps.sharedMemPerBlock = info.localMemSizePerCU_;
  deviceProps.regsPerBlock = info.availableSGPRs_;
  deviceProps.warpSize = info.wavefrontWidth_;
  deviceProps.maxThreadsPerBlock = info.maxWorkGroupSize_;
  deviceProps.maxThreadsDim[0] = info.maxWorkItemSizes_[0];
  deviceProps.maxThreadsDim[1] = info.maxWorkItemSizes_[1];
  deviceProps.maxThreadsDim[2] = info.maxWorkItemSizes_[2];
  deviceProps.maxGridSize[0] = UINT32_MAX;
  deviceProps.maxGridSize[1] = UINT32_MAX;
  deviceProps.maxGridSize[2] = UINT32_MAX;
  deviceProps.clockRate = info.maxEngineClockFrequency_;
  deviceProps.memoryClockRate = info.maxMemoryClockFrequency_;
  deviceProps.memoryBusWidth = info.globalMemChannels_ * 32;
  deviceProps.totalConstMem = info.maxConstantBufferSize_;
  deviceProps.major = info.gfxipVersion_ / 100;
  deviceProps.minor = info.gfxipVersion_ % 100;
  deviceProps.multiProcessorCount = info.maxComputeUnits_;
  deviceProps.l2CacheSize = info.l2CacheSize_;
  deviceProps.maxThreadsPerMultiProcessor = info.simdPerCU_;
  deviceProps.computeMode = 0;
  deviceProps.clockInstructionRate = info.timeStampFrequency_;
  deviceProps.arch.hasGlobalInt32Atomics       = 1;
  deviceProps.arch.hasGlobalFloatAtomicExch    = 1;
  deviceProps.arch.hasSharedInt32Atomics       = 1;
  deviceProps.arch.hasSharedFloatAtomicExch    = 1;
  deviceProps.arch.hasFloatAtomicAdd           = 0;
  deviceProps.arch.hasGlobalInt64Atomics       = 1;
  deviceProps.arch.hasSharedInt64Atomics       = 1;
  deviceProps.arch.hasDoubles                  = 1;
  deviceProps.arch.hasWarpVote                 = 0;
  deviceProps.arch.hasWarpBallot               = 0;
  deviceProps.arch.hasWarpShuffle              = 0;
  deviceProps.arch.hasFunnelShift              = 0;
  deviceProps.arch.hasThreadFenceSystem        = 1;
  deviceProps.arch.hasSyncThreadsExt           = 0;
  deviceProps.arch.hasSurfaceFuncs             = 0;
  deviceProps.arch.has3dGrid                   = 1;
  deviceProps.arch.hasDynamicParallelism       = 0;
  deviceProps.concurrentKernels = 1;
  deviceProps.pciDomainID = info.deviceTopology_.pcie.function;
  deviceProps.pciBusID = info.deviceTopology_.pcie.bus;
  deviceProps.pciDeviceID = info.deviceTopology_.pcie.device;
  deviceProps.maxSharedMemoryPerMultiProcessor = info.localMemSizePerCU_;
  //deviceProps.isMultiGpuBoard = info.;
  deviceProps.canMapHostMemory = 1;
  deviceProps.gcnArch = info.gfxipVersion_;

  *props = deviceProps;
  return hipSuccess;
}

hipError_t hipHccGetAccelerator(int deviceId, hc::accelerator* acc) {
  HIP_INIT_API(deviceId, acc);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipHccGetAcceleratorView(hipStream_t stream, hc::accelerator_view** av) {
  HIP_INIT_API(stream, av);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}
