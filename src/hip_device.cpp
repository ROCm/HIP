/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip_runtime.h"
#include "hcc_detail/hip_hcc.h"
#include "hcc_detail/trace_helper.h"

//-------------------------------------------------------------------------------------------------
//Devices
//-------------------------------------------------------------------------------------------------
// TODO - does this initialize HIP runtime?
hipError_t hipGetDevice(int *deviceId)
{
    HIP_INIT_API(deviceId);

    hipError_t e = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();

    if (ctx == nullptr) {
        e = hipErrorInvalidDevice; // TODO, check error code.
        *deviceId = -1;
    } else {
        *deviceId = ctx->getDevice()->_deviceId;
    }

    return ihipLogStatus(e);
}

// TODO - does this initialize HIP runtime?
hipError_t hipGetDeviceCount(int *count)
{
    HIP_INIT_API(count);

    *count = g_deviceCnt;

    if (*count > 0) {
        return ihipLogStatus(hipSuccess);
    } else {
        return ihipLogStatus(hipErrorNoDevice);
    }
}

hipError_t hipDeviceSetCacheConfig ( hipFuncCache cacheConfig )
{
    HIP_INIT_API(cacheConfig);

    // Nop, AMD does not support variable cache configs.

    return ihipLogStatus(hipSuccess);
}

hipError_t hipDeviceGetCacheConfig ( hipFuncCache *cacheConfig )
{
    HIP_INIT_API(cacheConfig);

    *cacheConfig = hipFuncCachePreferNone;

    return ihipLogStatus(hipSuccess);
}

hipError_t hipFuncSetCacheConfig ( hipFuncCache cacheConfig )
{
    HIP_INIT_API(cacheConfig);

    // Nop, AMD does not support variable cache configs.

    return ihipLogStatus(hipSuccess);
}

hipError_t hipDeviceSetSharedMemConfig ( hipSharedMemConfig config )
{
    HIP_INIT_API(config);

    // Nop, AMD does not support variable shared mem configs.

    return ihipLogStatus(hipSuccess);
}

hipError_t hipDeviceGetSharedMemConfig ( hipSharedMemConfig * pConfig )
{
    HIP_INIT_API(pConfig);

    *pConfig = hipSharedMemBankSizeFourByte;

    return ihipLogStatus(hipSuccess);
}

hipError_t hipSetDevice(int deviceId)
{
    HIP_INIT_API(deviceId);
    if ((deviceId < 0) || (deviceId >= g_deviceCnt)) {
        return ihipLogStatus(hipErrorInvalidDevice);
    } else {
        ihipSetTlsDefaultCtx(ihipGetPrimaryCtx(deviceId));
        return ihipLogStatus(hipSuccess);
    }
}

hipError_t hipDeviceSynchronize(void)
{
    HIP_INIT_API();
    return ihipLogStatus(ihipSynchronize());
}

hipError_t hipDeviceReset(void)
{
    HIP_INIT_API();

    auto *ctx = ihipGetTlsDefaultCtx();

    // TODO-HCC
    // This function currently does a user-level cleanup of known resources.
    // It could benefit from KFD support to perform a more "nuclear" clean that would include any associated kernel resources and page table entries.


    if (ctx) {
        // Release ctx resources (streams and memory):
        ctx->locked_reset();
    }

    return ihipLogStatus(hipSuccess);
}

hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device)
{
    HIP_INIT_API(pi, attr, device);

    hipError_t e = hipSuccess;

    auto * hipDevice = ihipGetDevice(device);
    hipDeviceProp_t *prop = &hipDevice->_props;
    if (hipDevice) {
        switch (attr) {
        case hipDeviceAttributeMaxThreadsPerBlock:
            *pi = prop->maxThreadsPerBlock; break;
        case hipDeviceAttributeMaxBlockDimX:
            *pi = prop->maxThreadsDim[0]; break;
        case hipDeviceAttributeMaxBlockDimY:
            *pi = prop->maxThreadsDim[1]; break;
        case hipDeviceAttributeMaxBlockDimZ:
            *pi = prop->maxThreadsDim[2]; break;
        case hipDeviceAttributeMaxGridDimX:
            *pi = prop->maxGridSize[0]; break;
        case hipDeviceAttributeMaxGridDimY:
            *pi = prop->maxGridSize[1]; break;
        case hipDeviceAttributeMaxGridDimZ:
            *pi = prop->maxGridSize[2]; break;
        case hipDeviceAttributeMaxSharedMemoryPerBlock:
            *pi = prop->sharedMemPerBlock; break;
        case hipDeviceAttributeTotalConstantMemory:
            *pi = prop->totalConstMem; break;
        case hipDeviceAttributeWarpSize:
            *pi = prop->warpSize; break;
        case hipDeviceAttributeMaxRegistersPerBlock:
            *pi = prop->regsPerBlock; break;
        case hipDeviceAttributeClockRate:
            *pi = prop->clockRate; break;
        case hipDeviceAttributeMemoryClockRate:
            *pi = prop->memoryClockRate; break;
        case hipDeviceAttributeMemoryBusWidth:
            *pi = prop->memoryBusWidth; break;
        case hipDeviceAttributeMultiprocessorCount:
            *pi = prop->multiProcessorCount; break;
        case hipDeviceAttributeComputeMode:
            *pi = prop->computeMode; break;
        case hipDeviceAttributeL2CacheSize:
            *pi = prop->l2CacheSize; break;
        case hipDeviceAttributeMaxThreadsPerMultiProcessor:
            *pi = prop->maxThreadsPerMultiProcessor; break;
        case hipDeviceAttributeComputeCapabilityMajor:
            *pi = prop->major; break;
        case hipDeviceAttributeComputeCapabilityMinor:
            *pi = prop->minor; break;
        case hipDeviceAttributePciBusId:
            *pi = prop->pciBusID; break;
        case hipDeviceAttributeConcurrentKernels:
            *pi = prop->concurrentKernels; break;
        case hipDeviceAttributePciDeviceId:
            *pi = prop->pciDeviceID; break;
        case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
            *pi = prop->maxSharedMemoryPerMultiProcessor; break;
        case hipDeviceAttributeIsMultiGpuBoard:
            *pi = prop->isMultiGpuBoard; break;
        default:
            e = hipErrorInvalidValue; break;
        }
    } else {
        e = hipErrorInvalidDevice;
    }
    return ihipLogStatus(e);
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t* props, int device)
{
    HIP_INIT_API(props, device);

    hipError_t e;

    auto * hipDevice = ihipGetDevice(device);
    if (hipDevice) {
        // copy saved props
        *props = hipDevice->_props;
        e = hipSuccess;
    } else {
        e = hipErrorInvalidDevice;
    }

    return ihipLogStatus(e);
}

hipError_t hipSetDeviceFlags( unsigned int flags)
{
    HIP_INIT_API(flags);

    hipError_t e;

    auto * ctx = ihipGetTlsDefaultCtx();

    // TODO : does this really OR in the flags or replaces previous flags:
    // TODO : Review error handling behavior for this function, it often returns ErrorSetOnActiveProcess
    if (ctx) {
       ctx->_ctxFlags = ctx->_ctxFlags | flags;
       e = hipSuccess;
    } else {
       e = hipErrorInvalidDevice;
    }

    return ihipLogStatus(e);
};

hipError_t hipDeviceComputeCapability(int *major, int *minor, hipDevice_t device)
{
    HIP_INIT_API(major,minor, device);
    hipError_t e = hipSuccess;
    int deviceId= device->_deviceId;
    e = hipDeviceGetAttribute(major, hipDeviceAttributeComputeCapabilityMajor, deviceId);
    e = hipDeviceGetAttribute(minor, hipDeviceAttributeComputeCapabilityMinor, deviceId);
    return ihipLogStatus(e);
}

hipError_t hipDeviceGetName(char *name,int len,hipDevice_t device)
{
    HIP_INIT_API(name,len, device);
    hipError_t e = hipSuccess;
    int nameLen = strlen(device->_props.name);
    if(nameLen <= len)
        memcpy(name,device->_props.name,nameLen);
    return ihipLogStatus(e);
}

hipError_t hipDeviceGetPCIBusId (int *pciBusId,int len,hipDevice_t device)
{
    HIP_INIT_API(pciBusId,len, device);
    hipError_t e = hipSuccess;
    int deviceId= device->_deviceId;
    e = hipDeviceGetAttribute(pciBusId, hipDeviceAttributePciBusId, deviceId);
    return ihipLogStatus(e);
}

hipError_t hipDeviceTotalMem (size_t *bytes,hipDevice_t device)
{
    HIP_INIT_API(bytes, device);
    hipError_t e = hipSuccess;
    *bytes= device->_props.totalGlobalMem;
    return ihipLogStatus(e);
}

hipError_t hipChooseDevice( int* device, const hipDeviceProp_t* prop )
{
    hipDeviceProp_t  tempProp;
    int deviceCount;
    int inPropCount=0;
    int matchedPropCount=0;
    hipError_t e = hipSuccess;
    hipGetDeviceCount( &deviceCount );
    *device = 0;
    for (int i=0; i< deviceCount; i++) {
        hipGetDeviceProperties( &tempProp, i );
        if(prop->major !=0) {
            inPropCount++;
            if(tempProp.major >= prop->major) {
                matchedPropCount++;
            }
            if(prop->minor !=0) {
                inPropCount++;
                 if(tempProp.minor >= prop->minor) {
                     matchedPropCount++;
                 }
            }
        }
        if(prop->totalGlobalMem != 0) {
            inPropCount++;
            if(tempProp.totalGlobalMem >= prop->totalGlobalMem) {
                matchedPropCount++;
            }
        }
        if(prop->sharedMemPerBlock != 0) {
            inPropCount++;
            if(tempProp.sharedMemPerBlock >= prop->sharedMemPerBlock) {
                matchedPropCount++;
            }
        }
        if(prop->maxThreadsPerBlock  != 0) {
            inPropCount++;
            if(tempProp.maxThreadsPerBlock >= prop->maxThreadsPerBlock ) {
                matchedPropCount++;
            }
        }
        if(prop->totalConstMem  != 0) {
            inPropCount++;
            if(tempProp.totalConstMem >= prop->totalConstMem ) {
                matchedPropCount++;
            }
        }
        if(prop->multiProcessorCount  != 0) {
            inPropCount++;
            if(tempProp.multiProcessorCount >= prop->multiProcessorCount ) {
                matchedPropCount++;
            }
        }
        if(prop->maxThreadsPerMultiProcessor  != 0) {
            inPropCount++;
            if(tempProp.maxThreadsPerMultiProcessor >= prop->maxThreadsPerMultiProcessor ) {
                matchedPropCount++;
            }
        }
        if(prop->memoryClockRate  != 0) {
            inPropCount++;
            if(tempProp.memoryClockRate >= prop->memoryClockRate ) {
                matchedPropCount++;
            }
        }
        if(inPropCount == matchedPropCount) {
            *device = i;
        }
#if 0
        else{
            e= hipErrorInvalidValue;
        }
#endif
    }
    return ihipLogStatus(e);
}

