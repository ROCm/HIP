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

hipError_t hipGetDevice(int *deviceId) {

  HIP_INIT_API(deviceId);

  if (deviceId != NULL) {
    // this needs to return default device. For now return 0 always
    *deviceId = 0;
  } else {
    return hipErrorInvalidValue;
  }

  return hipSuccess;
}

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

hipError_t hipGetDeviceCount(int* count) {

  HIP_INIT_API(count);

  if (count == NULL) {
    return hipErrorInvalidValue;
  }

  // Get all available devices
  *count = g_context->devices().size();

  return hipSuccess;
}

hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device) {

  HIP_INIT_API(pi, attr, device);

  if (pi == nullptr) {
    return hipErrorInvalidValue;
  }

  //if (unsigned(device) >= g_context->devices().size()) {
  //  return hipErrorInvalidDevice;
  //}
  //auto* deviceHandle = g_context->devices()[device];

  //FIXME: should we cache the props, or just select from deviceHandle->info_?
  hipDeviceProp_t prop = {0};
  hipError_t err = hipGetDeviceProperties(&prop, device);
  if (err != hipSuccess) return err;

  switch (attr) {
  case hipDeviceAttributeMaxThreadsPerBlock:
    *pi = prop.maxThreadsPerBlock;
    break;
  case hipDeviceAttributeMaxBlockDimX:
    *pi = prop.maxThreadsDim[0];
    break;
  case hipDeviceAttributeMaxBlockDimY:
    *pi = prop.maxThreadsDim[1];
    break;
  case hipDeviceAttributeMaxBlockDimZ:
    *pi = prop.maxThreadsDim[2];
    break;
  case hipDeviceAttributeMaxGridDimX:
    *pi = prop.maxGridSize[0];
    break;
  case hipDeviceAttributeMaxGridDimY:
    *pi = prop.maxGridSize[1];
    break;
  case hipDeviceAttributeMaxGridDimZ:
    *pi = prop.maxGridSize[2];
    break;
  case hipDeviceAttributeMaxSharedMemoryPerBlock:
    *pi = prop.sharedMemPerBlock;
    break;
  case hipDeviceAttributeTotalConstantMemory:
    *pi = prop.totalConstMem;
    break;
  case hipDeviceAttributeWarpSize:
    *pi = prop.warpSize;
    break;
  case hipDeviceAttributeMaxRegistersPerBlock:
    *pi = prop.regsPerBlock;
    break;
  case hipDeviceAttributeClockRate:
    *pi = prop.clockRate;
    break;
  case hipDeviceAttributeMemoryClockRate:
    *pi = prop.memoryClockRate;
    break;
  case hipDeviceAttributeMemoryBusWidth:
    *pi = prop.memoryBusWidth;
    break;
  case hipDeviceAttributeMultiprocessorCount:
    *pi = prop.multiProcessorCount;
    break;
  case hipDeviceAttributeComputeMode:
    *pi = prop.computeMode;
    break;
  case hipDeviceAttributeL2CacheSize:
    *pi = prop.l2CacheSize;
    break;
  case hipDeviceAttributeMaxThreadsPerMultiProcessor:
    *pi = prop.maxThreadsPerMultiProcessor;
    break;
  case hipDeviceAttributeComputeCapabilityMajor:
    *pi = prop.major;
    break;
  case hipDeviceAttributeComputeCapabilityMinor:
    *pi = prop.minor;
    break;
  case hipDeviceAttributePciBusId:
    *pi = prop.pciBusID;
    break;
  case hipDeviceAttributeConcurrentKernels:
    *pi = prop.concurrentKernels;
    break;
  case hipDeviceAttributePciDeviceId:
    *pi = prop.pciDeviceID;
    break;
  case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
    *pi = prop.maxSharedMemoryPerMultiProcessor;
    break;
  case hipDeviceAttributeIsMultiGpuBoard:
    *pi = prop.isMultiGpuBoard;
    break;
  default:
    return hipErrorInvalidValue;
  }

  return hipSuccess;
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t* props, int device) {

  HIP_INIT_API(props, device);

  if (props == NULL) {
    return hipErrorInvalidValue;
  }

  if (unsigned(device) >= g_context->devices().size()) {
    return hipErrorInvalidDevice;
  }
  auto* deviceHandle = g_context->devices()[device];

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

hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {

  HIP_INIT_API(cacheConfig);

  // No way to set cache config yet.

  return hipSuccess;
}

hipError_t hipDeviceGetCacheConfig(hipFuncCache_t *cacheConfig) {
  HIP_INIT_API(cacheConfig);

  if(cacheConfig == NULL) {
    return hipErrorInvalidValue;
  }

  *cacheConfig = hipFuncCache_t();

  return hipSuccess;
}

hipError_t hipSetDeviceFlags(unsigned int flags) {

  HIP_INIT_API(flags);

  assert(0 && "Unimplemented");

  return hipSuccess;
};

hipError_t hipDeviceGetLimit (size_t *pValue, hipLimit_t limit) {

  HIP_INIT_API(pValue, limit);

  assert(0 && "Unimplemented");

  return hipSuccess;
}

hipError_t hipFuncSetCacheConfig (const void* func, hipFuncCache_t cacheConfig) {

  HIP_INIT_API(cacheConfig);

  assert(0 && "Not supported");

  return hipSuccess;
}

hipError_t hipDeviceSetSharedMemConfig (hipSharedMemConfig config) {

  HIP_INIT_API(config);

  assert(0 && "Not Supported");

  return hipSuccess;
}

hipError_t hipDeviceGetSharedMemConfig (hipSharedMemConfig *pConfig) {

  HIP_INIT_API(pConfig);

  assert(0 && "Not supported");

  return hipSuccess;
}


hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* properties) {

  HIP_INIT_API(device, properties);

  assert(0 && "Unimplemented");

  return hipSuccess;
}


hipError_t hipDeviceGetByPCIBusId (int*  device, const char* pciBusId) {

  HIP_INIT_API(device,pciBusId);

  assert(0 && "Unimplemented");

  return hipSuccess;
}


hipError_t hipDeviceTotalMem (size_t *bytes,hipDevice_t device) {

  HIP_INIT_API(bytes, device);

  assert(0 && "Unimplemented");

  return hipSuccess;
}

hipError_t hipDeviceComputeCapability(int *major, int *minor, hipDevice_t device) {

  HIP_INIT_API(major,minor, device);

  assert(0 && "Unimplemented");

  return hipSuccess;
}

hipError_t hipDeviceGetName(char *name,int len, hipDevice_t device) {

  HIP_INIT_API((void*)name,len, device);

  assert(0 && "Unimplemented");

  return hipSuccess;
}

hipError_t hipDeviceGetPCIBusId (char *pciBusId,int len, int device) {

  HIP_INIT_API((void*)pciBusId, len, device);

  assert(0 && "Unimplemented");

  return hipSuccess;
}

hipError_t hipDeviceSynchronize(void)
{
  // FIXME: should wait on all streams
  return hipSuccess;
}
