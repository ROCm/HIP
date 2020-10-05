/* Copyright (c) 2018-present Advanced Micro Devices, Inc.

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

namespace hip {

// ================================================================================================
amd::HostQueue* Device::NullStream(bool skip_alloc) {
  amd::HostQueue* null_queue = null_stream_.asHostQueue(skip_alloc);
  if (null_queue == nullptr) {
    return nullptr;
  }
  // Wait for all active streams before executing commands on the default
  iHipWaitActiveStreams(null_queue);
  return null_queue;
}

}

hipError_t hipDeviceGet(hipDevice_t *device, int deviceId) {
  HIP_INIT_API(hipDeviceGet, device, deviceId);

  if (device != nullptr) {
    *device = deviceId;
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
};

hipError_t hipDeviceTotalMem (size_t *bytes, hipDevice_t device) {

  HIP_INIT_API(hipDeviceTotalMem, bytes, device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (bytes == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();

  *bytes = info.globalMemSize_;

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceComputeCapability(int *major, int *minor, hipDevice_t device) {

  HIP_INIT_API(hipDeviceComputeCapability, major, minor, device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (major == nullptr || minor == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();
  *major = info.gfxipMajor_;
  *minor = info.gfxipMinor_;

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetCount(int* count) {
  HIP_INIT_API(hipDeviceGetCount, count);

  HIP_RETURN(ihipDeviceGetCount(count));
}

hipError_t ihipDeviceGetCount(int* count) {
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }

  // Get all available devices
  *count = g_devices.size();

  if (*count < 1) {
    return hipErrorNoDevice;
  }

  return hipSuccess;
}

hipError_t hipDeviceGetName(char *name, int len, hipDevice_t device) {

  HIP_INIT_API(hipDeviceGetName, (void*)name, len, device);

  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (name == nullptr || len <= 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto* deviceHandle = g_devices[device]->devices()[0];
  const auto& info = deviceHandle->info();
  const auto nameLen = ::strlen(info.boardName_);

  // Make sure that the size of `dest` is big enough to hold `src` including
  // trailing zero byte
  if (nameLen > (cl_uint)(len - 1)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  ::strncpy(name, info.boardName_, (nameLen + 1));

  HIP_RETURN(hipSuccess);
}

hipError_t hipGetDeviceProperties ( hipDeviceProp_t* props, hipDevice_t device ) {
  HIP_INIT_API(hipGetDeviceProperties, props, device);

  if (props == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (unsigned(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
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
  deviceProps.maxGridSize[0] = INT32_MAX;
  deviceProps.maxGridSize[1] = INT32_MAX;
  deviceProps.maxGridSize[2] = INT32_MAX;
  deviceProps.clockRate = info.maxEngineClockFrequency_ * 1000;
  deviceProps.memoryClockRate = info.maxMemoryClockFrequency_ * 1000;
  deviceProps.memoryBusWidth = info.globalMemChannels_;
  deviceProps.totalConstMem = info.maxConstantBufferSize_;
  deviceProps.major = info.gfxipMajor_;
  deviceProps.minor = info.gfxipMinor_;
  deviceProps.multiProcessorCount = info.maxComputeUnits_;
  deviceProps.l2CacheSize = info.l2CacheSize_;
  deviceProps.maxThreadsPerMultiProcessor = info.maxThreadsPerCU_;
  deviceProps.computeMode = 0;
  deviceProps.clockInstructionRate = info.timeStampFrequency_;
  deviceProps.arch.hasGlobalInt32Atomics       = 1;
  deviceProps.arch.hasGlobalFloatAtomicExch    = 1;
  deviceProps.arch.hasSharedInt32Atomics       = 1;
  deviceProps.arch.hasSharedFloatAtomicExch    = 1;
  deviceProps.arch.hasFloatAtomicAdd           = 1;
  deviceProps.arch.hasGlobalInt64Atomics       = 1;
  deviceProps.arch.hasSharedInt64Atomics       = 1;
  deviceProps.arch.hasDoubles                  = 1;
  deviceProps.arch.hasWarpVote                 = 1;
  deviceProps.arch.hasWarpBallot               = 1;
  deviceProps.arch.hasWarpShuffle              = 1;
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
  deviceProps.gcnArch = info.gfxipMajor_ * 100 + info.gfxipMinor_ * 10 + info.gfxipStepping_;
  sprintf(deviceProps.gcnArchName, "gfx%d%d%x", info.gfxipMajor_, info.gfxipMinor_, info.gfxipStepping_);
  deviceProps.cooperativeLaunch = info.cooperativeGroups_;
  deviceProps.cooperativeMultiDeviceLaunch = info.cooperativeMultiDeviceGroups_;

  deviceProps.cooperativeMultiDeviceUnmatchedFunc = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedGridDim = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedBlockDim = info.cooperativeMultiDeviceGroups_;
  deviceProps.cooperativeMultiDeviceUnmatchedSharedMem = info.cooperativeMultiDeviceGroups_;

  deviceProps.maxTexture1D = info.imageMaxBufferSize_;
  deviceProps.maxTexture2D[0] = info.image2DMaxWidth_;
  deviceProps.maxTexture2D[1] = info.image2DMaxHeight_;
  deviceProps.maxTexture3D[0] = info.image3DMaxWidth_;
  deviceProps.maxTexture3D[1] = info.image3DMaxHeight_;
  deviceProps.maxTexture3D[2] = info.image3DMaxDepth_;
  deviceProps.hdpMemFlushCntl = nullptr;
  deviceProps.hdpRegFlushCntl = nullptr;

  deviceProps.memPitch = info.maxMemAllocSize_;
  deviceProps.textureAlignment = info.imageBaseAddressAlignment_;
  deviceProps.texturePitchAlignment = info.imagePitchAlignment_;
  deviceProps.kernelExecTimeoutEnabled = 0;
  deviceProps.ECCEnabled = info.errorCorrectionSupport_? 1:0;
  deviceProps.isLargeBar = info.largeBar_ ? 1 : 0;
  deviceProps.asicRevision = info.asicRevision_;

  *props = deviceProps;
  HIP_RETURN(hipSuccess);
}

hipError_t hipHccGetAccelerator(int deviceId, hc::accelerator* acc) {
  HIP_INIT_API(hipHccGetAccelerator, deviceId, acc);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipHccGetAcceleratorView(hipStream_t stream, hc::accelerator_view** av) {
  HIP_INIT_API(hipHccGetAcceleratorView, stream, av);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}
