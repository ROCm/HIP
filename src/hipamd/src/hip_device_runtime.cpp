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

hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* properties) {

  HIP_INIT_API(hipChooseDevice, device, properties);

  if (device == nullptr || properties == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *device = 0;
  cl_uint maxMatchedCount = 0;
  int count = 0;
  ihipDeviceGetCount(&count);

  for (cl_int i = 0; i< count; ++i) {
    hipDeviceProp_t currentProp = {0};
    cl_uint validPropCount = 0;
    cl_uint matchedCount = 0;
    hipError_t err = hipGetDeviceProperties(&currentProp, i);
    if (properties->major != 0) {
      validPropCount++;
      if(currentProp.major >= properties->major) {
        matchedCount++;
      }
    }
    if (properties->minor != 0) {
      validPropCount++;
      if(currentProp.minor >= properties->minor) {
        matchedCount++;
      }
    }
    if(properties->totalGlobalMem != 0) {
        validPropCount++;
        if(currentProp.totalGlobalMem >= properties->totalGlobalMem) {
            matchedCount++;
        }
    }
    if(properties->sharedMemPerBlock != 0) {
        validPropCount++;
        if(currentProp.sharedMemPerBlock >= properties->sharedMemPerBlock) {
            matchedCount++;
        }
    }
    if(properties->maxThreadsPerBlock != 0) {
        validPropCount++;
        if(currentProp.maxThreadsPerBlock >= properties->maxThreadsPerBlock ) {
            matchedCount++;
        }
    }
    if(properties->totalConstMem != 0) {
        validPropCount++;
        if(currentProp.totalConstMem >= properties->totalConstMem ) {
            matchedCount++;
        }
    }
    if(properties->multiProcessorCount != 0) {
        validPropCount++;
        if(currentProp.multiProcessorCount >=
          properties->multiProcessorCount ) {
            matchedCount++;
        }
    }
    if(properties->maxThreadsPerMultiProcessor != 0) {
        validPropCount++;
        if(currentProp.maxThreadsPerMultiProcessor >=
          properties->maxThreadsPerMultiProcessor ) {
            matchedCount++;
        }
    }
    if(properties->memoryClockRate != 0) {
        validPropCount++;
        if(currentProp.memoryClockRate >= properties->memoryClockRate ) {
            matchedCount++;
        }
    }
    if(properties->memoryBusWidth != 0) {
        validPropCount++;
        if(currentProp.memoryBusWidth >= properties->memoryBusWidth ) {
            matchedCount++;
        }
    }
    if(properties->l2CacheSize != 0) {
        validPropCount++;
        if(currentProp.l2CacheSize >= properties->l2CacheSize ) {
            matchedCount++;
        }
    }
    if(properties->regsPerBlock != 0) {
        validPropCount++;
        if(currentProp.regsPerBlock >= properties->regsPerBlock ) {
            matchedCount++;
        }
    }
    if(properties->maxSharedMemoryPerMultiProcessor != 0) {
        validPropCount++;
        if(currentProp.maxSharedMemoryPerMultiProcessor >=
          properties->maxSharedMemoryPerMultiProcessor ) {
            matchedCount++;
        }
    }
    if(properties->warpSize != 0) {
        validPropCount++;
        if(currentProp.warpSize >= properties->warpSize ) {
            matchedCount++;
        }
    }
    if(validPropCount == matchedCount) {
      *device = matchedCount > maxMatchedCount ? i : *device;
      maxMatchedCount = std::max(matchedCount, maxMatchedCount);
    }
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device) {

  HIP_INIT_API(hipDeviceGetAttribute, pi, attr, device);

  if (pi == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  int count = 0;
  ihipDeviceGetCount(&count);
  if (device < 0 || device >= count) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  //FIXME: should we cache the props, or just select from deviceHandle->info_?
  hipDeviceProp_t prop = {0};
  hipError_t err = hipGetDeviceProperties(&prop, device);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }

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
  case hipDeviceAttributeCooperativeLaunch:
    *pi = prop.cooperativeLaunch;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceLaunch:
    *pi = prop.cooperativeMultiDeviceLaunch;
    break;
  case hipDeviceAttributeIntegrated:
    *pi = prop.integrated;
    break;
  case hipDeviceAttributeMaxTexture1DWidth:
    *pi = prop.maxTexture1D;
    break;
  case hipDeviceAttributeMaxTexture2DWidth:
    *pi = prop.maxTexture2D[0];
    break;
  case hipDeviceAttributeMaxTexture2DHeight:
    *pi = prop.maxTexture2D[1];
    break;
  case hipDeviceAttributeMaxTexture3DWidth:
    *pi = prop.maxTexture3D[0];
    break;
  case hipDeviceAttributeMaxTexture3DHeight:
    *pi = prop.maxTexture3D[1];
    break;
  case hipDeviceAttributeMaxTexture3DDepth:
    *pi = prop.maxTexture3D[2];
    break;
  case hipDeviceAttributeHdpMemFlushCntl:
    *reinterpret_cast<unsigned int**>(pi) = prop.hdpMemFlushCntl;
    break;
  case hipDeviceAttributeHdpRegFlushCntl:
    *reinterpret_cast<unsigned int**>(pi) = prop.hdpRegFlushCntl;
    break;
  case hipDeviceAttributeMaxPitch:
    *pi = prop.memPitch;
    break;
  case hipDeviceAttributeTextureAlignment:
    *pi = prop.textureAlignment;
    break;
  case hipDeviceAttributeTexturePitchAlignment:
    *pi = prop.texturePitchAlignment;
    break;
  case hipDeviceAttributeKernelExecTimeout:
    *pi = prop.kernelExecTimeoutEnabled;
    break;
  case hipDeviceAttributeCanMapHostMemory:
    *pi = prop.canMapHostMemory;
    break;
  case hipDeviceAttributeEccEnabled:
    *pi = prop.ECCEnabled;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc:
    *pi = prop.cooperativeMultiDeviceUnmatchedFunc;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim:
    *pi = prop.cooperativeMultiDeviceUnmatchedGridDim;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim:
    *pi = prop.cooperativeMultiDeviceUnmatchedBlockDim;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem:
    *pi = prop.cooperativeMultiDeviceUnmatchedSharedMem;
    break;
  case hipDeviceAttributeAsicRevision:
    *pi = prop.asicRevision;
    break;
  case hipDeviceAttributeManagedMemory:
    *pi = prop.managedMemory;
    break;
  case hipDeviceAttributeDirectManagedMemAccessFromHost:
    *pi = prop.directManagedMemAccessFromHost;
    break;
  case hipDeviceAttributeConcurrentManagedAccess:
    *pi = prop.concurrentManagedAccess;
    break;
  case hipDeviceAttributePageableMemoryAccess:
    *pi = prop.pageableMemoryAccess;
    break;
  case hipDeviceAttributePageableMemoryAccessUsesHostPageTables:
    *pi = prop.pageableMemoryAccessUsesHostPageTables;
    break;
  case hipDeviceAttributeCanUseStreamWaitValue:
    // hipStreamWaitValue64() and hipStreamWaitValue32() support
    *pi = g_devices[device]->devices()[0]->info().aqlBarrierValue_;
    break;
  default:
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetByPCIBusId(int* device, const char*pciBusIdstr) {

  HIP_INIT_API(hipDeviceGetByPCIBusId, device, pciBusIdstr);

  if (device == nullptr || pciBusIdstr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  int pciBusID = -1;
  int pciDeviceID = -1;
  int pciDomainID = -1;
  bool found = false;
  if (sscanf (pciBusIdstr, "%04x:%02x:%02x", &pciDomainID, &pciBusID, &pciDeviceID) == 0x3) {
    int count = 0;
    ihipDeviceGetCount(&count);
    for (cl_int i = 0; i < count; i++) {
      hipDevice_t dev;
      hipDeviceGet(&dev, i);
      hipDeviceProp_t prop;
      hipGetDeviceProperties(&prop, dev);

      if ((pciBusID == prop.pciBusID) && (pciDomainID == prop.pciDomainID)
                    && (pciDeviceID == prop.pciDeviceID)) {
        *device = i;
        found = true;
        break;
      }
    }
  }
  if (!found) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetCacheConfig ( hipFuncCache_t * cacheConfig ) {
  HIP_INIT_API(hipDeviceGetCacheConfig, cacheConfig);

  if(cacheConfig == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *cacheConfig = hipFuncCache_t();

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetLimit ( size_t* pValue, hipLimit_t limit ) {

  HIP_INIT_API(hipDeviceGetLimit, pValue, limit);

  if(pValue == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if(limit == hipLimitMallocHeapSize) {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, ihipGetDevice());

    *pValue = prop.totalGlobalMem;
    HIP_RETURN(hipSuccess);
  } else {
    HIP_RETURN(hipErrorUnsupportedLimit);
  }
}

hipError_t hipDeviceGetPCIBusId ( char* pciBusId, int  len, int  device ) {

  HIP_INIT_API(hipDeviceGetPCIBusId, (void*)pciBusId, len, device);

  int count;
  ihipDeviceGetCount(&count);
  if (device < 0 || device >= count) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (pciBusId == nullptr || len <= 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, device);

  snprintf (pciBusId, len, "%04x:%02x:%02x.0",
                    prop.pciDomainID,
                    prop.pciBusID,
                    prop.pciDeviceID);

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceGetSharedMemConfig ( hipSharedMemConfig * pConfig ) {
  HIP_INIT_API(hipDeviceGetSharedMemConfig, pConfig);
  if (pConfig == nullptr) {
    return HIP_RETURN(hipErrorInvalidValue);
  }
  *pConfig = hipSharedMemBankSizeFourByte;

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceReset ( void ) {
  HIP_INIT_API(hipDeviceReset);

  /* FIXME */

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceSetCacheConfig ( hipFuncCache_t cacheConfig ) {
  HIP_INIT_API(hipDeviceSetCacheConfig, cacheConfig);

  // No way to set cache config yet.

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceSetLimit ( hipLimit_t limit, size_t value ) {
  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipDeviceSetSharedMemConfig ( hipSharedMemConfig config ) {
  HIP_INIT_API(hipDeviceSetSharedMemConfig, config);

  // No way to set cache config yet.

  HIP_RETURN(hipSuccess);
}

hipError_t hipDeviceSynchronize ( void ) {
  HIP_INIT_API(hipDeviceSynchronize);

  amd::HostQueue* queue = hip::getNullStream();

  if (!queue) {
    HIP_RETURN(hipErrorOutOfMemory);
  }

  queue->finish();

  hip::Stream::syncNonBlockingStreams();

  HIP_RETURN(hipSuccess);
}

int ihipGetDevice() {
  hip::Device* device = hip::getCurrentDevice();
  if(device == nullptr){
    return -1;
  }
  return device->deviceId();
}

hipError_t hipGetDevice ( int* deviceId ) {
  HIP_INIT_API(hipGetDevice, deviceId);

  if (deviceId != nullptr) {
    int dev = ihipGetDevice();
    if (dev == -1) {
      HIP_RETURN(hipErrorNoDevice);
    }
    *deviceId = dev;
    HIP_RETURN(hipSuccess);
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }
}

hipError_t hipGetDeviceCount ( int* count ) {
  HIP_INIT_API(hipGetDeviceCount, count);

  HIP_RETURN(ihipDeviceGetCount(count));
}

hipError_t hipGetDeviceFlags ( unsigned int* flags ) {
  HIP_INIT_API(hipGetDeviceFlags, flags);
  if (flags == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *flags = hip::getCurrentDevice()->getFlags();
  HIP_RETURN(hipSuccess);
}

hipError_t hipSetDevice ( int  device ) {
  HIP_INIT_API(hipSetDevice, device);

  if (static_cast<unsigned int>(device) < g_devices.size()) {
    hip::setCurrentDevice(device);

    HIP_RETURN(hipSuccess);
  }
  HIP_RETURN(hipErrorInvalidDevice);
}

hipError_t hipSetDeviceFlags ( unsigned int  flags ) {
  HIP_INIT_API(hipSetDeviceFlags, flags);

  constexpr uint32_t supportedFlags =
      hipDeviceScheduleMask | hipDeviceMapHost | hipDeviceLmemResizeToMax;

  if (flags & ~supportedFlags) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  switch (flags & hipDeviceScheduleMask) {
    case hipDeviceScheduleAuto:
      // Current behavior is different from the spec, due to MT usage in runtime
      if (hip::host_device->devices().size() >= std::thread::hardware_concurrency()) {
        device->SetActiveWait(false);
        break;
      }
      // Fall through for active wait...
    case hipDeviceScheduleSpin:
    case hipDeviceScheduleYield:
      // The both options falls into yield, because MT usage in runtime
      device->SetActiveWait(true);
      break;
    case hipDeviceScheduleBlockingSync:
      device->SetActiveWait(false);
      break;
    default:
      break;
  }
  hip::getCurrentDevice()->setFlags(flags & hipDeviceScheduleMask);

  HIP_RETURN(hipSuccess);
}

hipError_t hipSetValidDevices ( int* device_arr, int  len ) {
  HIP_INIT_API(hipSetValidDevices, device_arr, len);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}
