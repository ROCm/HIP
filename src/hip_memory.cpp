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

#include <hc_am.hpp>
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"

__device__ char __hip_device_heap[__HIP_SIZE_OF_HEAP];
__device__ uint32_t __hip_device_page_flag[__HIP_NUM_PAGES];

// Internal HIP APIS:
namespace hip_internal {

hipError_t memcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                       hipStream_t stream) {
    hipError_t e = hipSuccess;

    // Return success if number of bytes to copy is 0
    if (sizeBytes == 0) return e;

    stream = ihipSyncAndResolveStream(stream);

    if ((dst == NULL) || (src == NULL)) {
        e = hipErrorInvalidValue;
    } else if (stream) {
        try {
            stream->locked_copyAsync(dst, src, sizeBytes, kind);
        } catch (ihipException& ex) {
            e = ex._code;
        }
    } else {
        e = hipErrorInvalidValue;
    }

    return e;
}

// return 0 on success or -1 on error:
int sharePtr(void* ptr, ihipCtx_t* ctx, bool shareWithAll, unsigned hipFlags) {
    int ret = 0;

    auto device = ctx->getWriteableDevice();

    if (shareWithAll) {
        // shareWithAll memory is not mapped to any device
        hc::am_memtracker_update(ptr, -1, hipFlags);
        hsa_status_t s = hsa_amd_agents_allow_access(g_deviceCnt + 1, g_allAgents, NULL, ptr);
        tprintf(DB_MEM, "    allow access to CPU + all %d GPUs (shareWithAll)\n", g_deviceCnt);
        if (s != HSA_STATUS_SUCCESS) {
            ret = -1;
        }
    } else {
#if USE_APP_PTR_FOR_CTX
        hc::am_memtracker_update(ptr, device->_deviceId, hipFlags, ctx);
#else
        hc::am_memtracker_update(ptr, device->_deviceId, hipFlags);
#endif
        int peerCnt = 0;
        {
            LockedAccessor_CtxCrit_t crit(ctx->criticalData());
            // the peerCnt always stores self so make sure the trace actually
            peerCnt = crit->peerCnt();
            tprintf(DB_MEM, "  allow access to %d other peer(s)\n", peerCnt - 1);
            if (peerCnt > 1) {
                // printf ("peer self access\n");

                // TODOD - remove me:
                for (auto iter = crit->_peers.begin(); iter != crit->_peers.end(); iter++) {
                    tprintf(DB_MEM, "    allow access to peer: %s%s\n", (*iter)->toString().c_str(),
                            (iter == crit->_peers.begin()) ? " (self)" : "");
                };

                hsa_status_t s =
                    hsa_amd_agents_allow_access(crit->peerCnt(), crit->peerAgents(), NULL, ptr);
                if (s != HSA_STATUS_SUCCESS) {
                    ret = -1;
                }
            }
        }
    }

    return ret;
}


// Allocate a new pointer with am_alloc and share with all valid peers.
// Returns null-ptr if a memory error occurs (either allocation or sharing)
void* allocAndSharePtr(const char* msg, size_t sizeBytes, ihipCtx_t* ctx, bool shareWithAll,
                       unsigned amFlags, unsigned hipFlags, size_t alignment) {
    void* ptr = nullptr;

    auto device = ctx->getWriteableDevice();

#if (__hcc_workweek__ >= 17332)
    if (alignment != 0) {
        ptr = hc::am_aligned_alloc(sizeBytes, device->_acc, amFlags, alignment);
    } else
#endif
    {
        ptr = hc::am_alloc(sizeBytes, device->_acc, amFlags);
    }
    tprintf(DB_MEM, " alloc %s ptr:%p-%p size:%zu on dev:%d\n", msg, ptr,
            static_cast<char*>(ptr) + sizeBytes, sizeBytes, device->_deviceId);

    if (HIP_INIT_ALLOC != -1) {
        // TODO , dont' call HIP API directly here:
        hipMemset(ptr, HIP_INIT_ALLOC, sizeBytes);
    }

    if (ptr != nullptr) {
        int r = sharePtr(ptr, ctx, shareWithAll, hipFlags);
        if (r != 0) {
            ptr = nullptr;
        }
    }

    return ptr;
}


}  // end namespace hip_internal

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Memory
//
//
//
// HIP uses several "app*" fields HC memory tracker to track state necessary for the HIP API.
//_appId : DeviceID.  For device mem, this is device where the memory is physically allocated.
//         For host or registered mem, this is the current device when the memory is allocated or
//         registered.  This device will have a GPUVM mapping for the host mem.
//
//_appAllocationFlags : These are flags provided by the user when allocation is performed. They are
//returned to user in hipHostGetFlags and other APIs.
// TODO - add more info here when available.
//
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
    HIP_INIT_API(hipPointerGetAttributes, attributes, ptr);

    hipError_t e = hipSuccess;
    if ((attributes == nullptr) || (ptr == nullptr)) {
        e = hipErrorInvalidValue;
    } else {
        hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
        if (status == AM_SUCCESS) {
            attributes->memoryType =
                amPointerInfo._isInDeviceMem ? hipMemoryTypeDevice : hipMemoryTypeHost;
            attributes->hostPointer = amPointerInfo._hostPointer;
            attributes->devicePointer = amPointerInfo._devicePointer;
            attributes->isManaged = 0;
            if (attributes->memoryType == hipMemoryTypeHost) {
                attributes->hostPointer = (void*)ptr;
            }
            if (attributes->memoryType == hipMemoryTypeDevice) {
                attributes->devicePointer = (void*)ptr;
            }
            attributes->allocationFlags = amPointerInfo._appAllocationFlags;
            attributes->device = amPointerInfo._appId;

            if (attributes->device < -1) {
                e = hipErrorInvalidDevice;
            }
        } else {
            attributes->memoryType = hipMemoryTypeDevice;
            attributes->hostPointer = 0;
            attributes->devicePointer = 0;
            attributes->device = -2;
            attributes->isManaged = 0;
            attributes->allocationFlags = 0;

            e = hipErrorUnknown;  // TODO - should be hipErrorInvalidValue ?
        }
    }
    return ihipLogStatus(e);
}


hipError_t hipHostGetDevicePointer(void** devicePointer, void* hostPointer, unsigned flags) {
    HIP_INIT_API(hipHostGetDevicePointer, devicePointer, hostPointer, flags);

    hipError_t e = hipSuccess;

    // Flags must be 0:
    if ((flags != 0) || (devicePointer == nullptr) || (hostPointer == nullptr)) {
        e = hipErrorInvalidValue;
    } else {
        hc::accelerator acc;
        *devicePointer = NULL;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, hostPointer);
        if (status == AM_SUCCESS) {
            *devicePointer =
                static_cast<char*>(amPointerInfo._devicePointer) +
                (static_cast<char*>(hostPointer) - static_cast<char*>(amPointerInfo._hostPointer));
            tprintf(DB_MEM, " host_ptr=%p returned device_pointer=%p\n", hostPointer,
                    *devicePointer);
        } else {
            e = hipErrorMemoryAllocation;
        }
    }
    return ihipLogStatus(e);
}


hipError_t hipMalloc(void** ptr, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMalloc, (TRACE_MEM), ptr, sizeBytes);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    // return NULL pointer when malloc size is 0
    if (sizeBytes == 0) {
        *ptr = NULL;
        hip_status = hipSuccess;

    } else if ((ctx == nullptr) || (ptr == nullptr)) {
        hip_status = hipErrorInvalidValue;

    } else {
        auto device = ctx->getWriteableDevice();
        *ptr = hip_internal::allocAndSharePtr("device_mem", sizeBytes, ctx, false /*shareWithAll*/,
                                              0 /*amFlags*/, 0 /*hipFlags*/, 0);

        if (sizeBytes && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        }
    }


    return ihipLogStatus(hip_status);
}


hipError_t hipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags) {
    HIP_INIT_SPECIAL_API(hipHostMalloc, (TRACE_MEM), ptr, sizeBytes, flags);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;

    if (HIP_SYNC_HOST_ALLOC) {
        hipDeviceSynchronize();
    }

    auto ctx = ihipGetTlsDefaultCtx();

    if (sizeBytes == 0) {
        hip_status = hipSuccess;
        // TODO - should size of 0 return err or be siliently ignored?
    } else if ((ctx == nullptr) || (ptr == nullptr)) {
        hip_status = hipErrorInvalidValue;
    } else {
        unsigned trueFlags = flags;
        if (flags == hipHostMallocDefault) {
            // HCC/ROCM provide a modern system with unified memory and should set both of these
            // flags by default:
            trueFlags = hipHostMallocMapped | hipHostMallocPortable;
        }


        const unsigned supportedFlags = hipHostMallocPortable | hipHostMallocMapped |
                                        hipHostMallocWriteCombined | hipHostMallocCoherent |
                                        hipHostMallocNonCoherent;


        const unsigned coherencyFlags = hipHostMallocCoherent | hipHostMallocNonCoherent;

        if ((flags & ~supportedFlags) || ((flags & coherencyFlags) == coherencyFlags)) {
            *ptr = nullptr;
            // can't specify unsupported flags, can't specify both Coherent + NonCoherent
            hip_status = hipErrorInvalidValue;
        } else {
            auto device = ctx->getWriteableDevice();

            unsigned amFlags = 0;
            if (flags & hipHostMallocCoherent) {
                amFlags = amHostCoherent;
            } else if (flags & hipHostMallocNonCoherent) {
                amFlags = amHostNonCoherent;
            } else {
                // depends on env variables:
                amFlags = HIP_HOST_COHERENT ? amHostCoherent : amHostNonCoherent;
            }


            *ptr = hip_internal::allocAndSharePtr(
                (amFlags & amHostCoherent) ? "finegrained_host" : "pinned_host", sizeBytes, ctx,
                (trueFlags & hipHostMallocPortable) /*shareWithAll*/, amFlags, flags, 0);

            if (sizeBytes && (*ptr == NULL)) {
                hip_status = hipErrorMemoryAllocation;
            }
        }
    }

    if (HIP_SYNC_HOST_ALLOC) {
        hipDeviceSynchronize();
    }
    return ihipLogStatus(hip_status);
}

// Deprecated function:
hipError_t hipMallocHost(void** ptr, size_t sizeBytes) { return hipHostMalloc(ptr, sizeBytes, 0); }


// Deprecated function:
hipError_t hipHostAlloc(void** ptr, size_t sizeBytes, unsigned int flags) {
    return hipHostMalloc(ptr, sizeBytes, flags);
};

// width in bytes
hipError_t ihipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height, size_t depth) {
    hipError_t hip_status = hipSuccess;
    if(ptr==NULL)
     {
	hip_status=hipErrorInvalidValue;
       	return hip_status;
     }
    // hardcoded 128 bytes
    *pitch = ((((int)width - 1) / 128) + 1) * 128;
    const size_t sizeBytes = (*pitch) * height;

    auto ctx = ihipGetTlsDefaultCtx();

    if (ctx) {
        hc::accelerator acc = ctx->getDevice()->_acc;
        hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

        size_t allocGranularity = 0;
        hsa_amd_memory_pool_t* allocRegion =
            static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_region());
        hsa_amd_memory_pool_get_info(*allocRegion, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                     &allocGranularity);

        hsa_ext_image_descriptor_t imageDescriptor;
        imageDescriptor.width = *pitch;
        imageDescriptor.height = height;
        imageDescriptor.depth = depth;
        imageDescriptor.array_size = 0;
        if (depth == 0)
            imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
        else
            imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_3D;
        imageDescriptor.format.channel_order = HSA_EXT_IMAGE_CHANNEL_ORDER_R;
        imageDescriptor.format.channel_type = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;

        hsa_access_permission_t permission = HSA_ACCESS_PERMISSION_RW;
        hsa_ext_image_data_info_t imageInfo;
        hsa_status_t status =
            hsa_ext_image_data_get_info(*agent, &imageDescriptor, permission, &imageInfo);
        size_t alignment = imageInfo.alignment <= allocGranularity ? 0 : imageInfo.alignment;

        const unsigned am_flags = 0;
        *ptr = hip_internal::allocAndSharePtr("device_pitch", sizeBytes, ctx,
                                              false /*shareWithAll*/, am_flags, 0, alignment);

        if (sizeBytes && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        }
    } else {
        hip_status = hipErrorMemoryAllocation;
    }

    return hip_status;
}

// width in bytes
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
    HIP_INIT_SPECIAL_API(hipMallocPitch, (TRACE_MEM), ptr, pitch, width, height);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;

    if (width == 0 || height == 0) return ihipLogStatus(hipErrorUnknown);

    hip_status = ihipMallocPitch(ptr, pitch, width, height, 0);
    return ihipLogStatus(hip_status);
}

hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
    HIP_INIT_API(hipMalloc3D, pitchedDevPtr, &extent);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;

    if (extent.width == 0 || extent.height == 0) return ihipLogStatus(hipErrorUnknown);
    if (!pitchedDevPtr) return ihipLogStatus(hipErrorInvalidValue);
    void* ptr;
    size_t pitch;

    hip_status =
        ihipMallocPitch(&pitchedDevPtr->ptr, &pitch, extent.width, extent.height, extent.depth);
    if (hip_status == hipSuccess) {
        pitchedDevPtr->pitch = pitch;
        pitchedDevPtr->xsize = extent.width;
        pitchedDevPtr->ysize = extent.height;
    }
    return ihipLogStatus(hip_status);
}

hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f) {
    hipChannelFormatDesc cd;
    cd.x = x;
    cd.y = y;
    cd.z = z;
    cd.w = w;
    cd.f = f;
    return cd;
}

extern void getChannelOrderAndType(const hipChannelFormatDesc& desc,
                                   enum hipTextureReadMode readMode,
                                   hsa_ext_image_channel_order_t* channelOrder,
                                   hsa_ext_image_channel_type_t* channelType);

hipError_t hipArrayCreate(hipArray** array, const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
    HIP_INIT_SPECIAL_API(hipArrayCreate, (TRACE_MEM), array, pAllocateArray);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;
    if (pAllocateArray->width > 0) {
        auto ctx = ihipGetTlsDefaultCtx();
        *array = (hipArray*)malloc(sizeof(hipArray));
        array[0]->drvDesc = *pAllocateArray;
        array[0]->width = pAllocateArray->width;
        array[0]->height = pAllocateArray->height;
        array[0]->isDrv = true;
        array[0]->textureType = hipTextureType2D;
        void** ptr = &array[0]->data;
        if (ctx) {
            const unsigned am_flags = 0;
            size_t size = pAllocateArray->width;
            if (pAllocateArray->height > 0) {
                size = size * pAllocateArray->height;
            }
            hsa_ext_image_channel_type_t channelType;
            size_t allocSize = 0;
            switch (pAllocateArray->format) {
                case HIP_AD_FORMAT_UNSIGNED_INT8:
                    allocSize = size * sizeof(uint8_t);
                    channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
                    break;
                case HIP_AD_FORMAT_UNSIGNED_INT16:
                    allocSize = size * sizeof(uint16_t);
                    channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
                    break;
                case HIP_AD_FORMAT_UNSIGNED_INT32:
                    allocSize = size * sizeof(uint32_t);
                    channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
                    break;
                case HIP_AD_FORMAT_SIGNED_INT8:
                    allocSize = size * sizeof(int8_t);
                    channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
                    break;
                case HIP_AD_FORMAT_SIGNED_INT16:
                    allocSize = size * sizeof(int16_t);
                    channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
                    break;
                case HIP_AD_FORMAT_SIGNED_INT32:
                    allocSize = size * sizeof(int32_t);
                    channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
                    break;
                case HIP_AD_FORMAT_HALF:
                    allocSize = size * sizeof(int16_t);
                    channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
                    break;
                case HIP_AD_FORMAT_FLOAT:
                    allocSize = size * sizeof(float);
                    channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT;
                    break;
                default:
                    hip_status = hipErrorUnknown;
                    break;
            }
            hc::accelerator acc = ctx->getDevice()->_acc;
            hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

            size_t allocGranularity = 0;
            hsa_amd_memory_pool_t* allocRegion =
                static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_region());
            hsa_amd_memory_pool_get_info(
                *allocRegion, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &allocGranularity);

            hsa_ext_image_descriptor_t imageDescriptor;

            imageDescriptor.width = pAllocateArray->width;
            imageDescriptor.height = pAllocateArray->height;
            imageDescriptor.depth = 0;
            imageDescriptor.array_size = 0;

            imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;

            hsa_ext_image_channel_order_t channelOrder;

            if (pAllocateArray->numChannels == 4) {
                channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
            } else if (pAllocateArray->numChannels == 2) {
                channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RG;
            } else if (pAllocateArray->numChannels == 1) {
                channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_R;
            }
            imageDescriptor.format.channel_order = channelOrder;
            imageDescriptor.format.channel_type = channelType;

            hsa_access_permission_t permission = HSA_ACCESS_PERMISSION_RW;
            hsa_ext_image_data_info_t imageInfo;
            hsa_status_t status =
                hsa_ext_image_data_get_info(*agent, &imageDescriptor, permission, &imageInfo);
            size_t alignment = imageInfo.alignment <= allocGranularity ? 0 : imageInfo.alignment;

            *ptr = hip_internal::allocAndSharePtr("device_array", allocSize, ctx,
                                                  false /*shareWithAll*/, am_flags, 0, alignment);
            if (size && (*ptr == NULL)) {
                hip_status = hipErrorMemoryAllocation;
            }
        } else {
            hip_status = hipErrorMemoryAllocation;
        }
    } else {
        hip_status = hipErrorInvalidValue;
    }

    return ihipLogStatus(hip_status);
}

hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc, size_t width,
                          size_t height, unsigned int flags) {
    HIP_INIT_SPECIAL_API(hipMallocArray, (TRACE_MEM), array, desc, width, height, flags);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;
    if (width > 0) {
        auto ctx = ihipGetTlsDefaultCtx();

        *array = (hipArray*)malloc(sizeof(hipArray));
        array[0]->type = flags;
        array[0]->width = width;
        array[0]->height = height;
        array[0]->depth = 1;
        array[0]->desc = *desc;
        array[0]->isDrv = false;
        array[0]->textureType = hipTextureType2D;
        void** ptr = &array[0]->data;

        if (ctx) {
            const unsigned am_flags = 0;
            size_t size = width;
            if (height > 0) {
                size = size * height;
            }

            const size_t allocSize = size * ((desc->x + desc->y + desc->z + desc->w) / 8);

            hc::accelerator acc = ctx->getDevice()->_acc;
            hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

            size_t allocGranularity = 0;
            hsa_amd_memory_pool_t* allocRegion =
                static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_region());
            hsa_amd_memory_pool_get_info(
                *allocRegion, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &allocGranularity);

            hsa_ext_image_descriptor_t imageDescriptor;

            imageDescriptor.width = width;
            imageDescriptor.height = height;
            imageDescriptor.depth = 0;
            imageDescriptor.array_size = 0;
            switch (flags) {
                case hipArrayLayered:
                case hipArrayCubemap:
                case hipArraySurfaceLoadStore:
                case hipArrayTextureGather:
                    assert(0);
                    break;
                case hipArrayDefault:
                default:
                    imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
                    break;
            }
            hsa_ext_image_channel_order_t channelOrder;
            hsa_ext_image_channel_type_t channelType;
            getChannelOrderAndType(*desc, hipReadModeElementType, &channelOrder, &channelType);
            imageDescriptor.format.channel_order = channelOrder;
            imageDescriptor.format.channel_type = channelType;

            hsa_access_permission_t permission = HSA_ACCESS_PERMISSION_RW;
            hsa_ext_image_data_info_t imageInfo;
            hsa_status_t status =
                hsa_ext_image_data_get_info(*agent, &imageDescriptor, permission, &imageInfo);
            size_t alignment = imageInfo.alignment <= allocGranularity ? 0 : imageInfo.alignment;

            *ptr = hip_internal::allocAndSharePtr("device_array", allocSize, ctx,
                                                  false /*shareWithAll*/, am_flags, 0, alignment);
            if (size && (*ptr == NULL)) {
                hip_status = hipErrorMemoryAllocation;
            }

        } else {
            hip_status = hipErrorMemoryAllocation;
        }
    } else {
        hip_status = hipErrorInvalidValue;
    }

    return ihipLogStatus(hip_status);
}

hipError_t hipArray3DCreate(hipArray** array, const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
    HIP_INIT_SPECIAL_API(hipArray3DCreate, (TRACE_MEM), array, pAllocateArray);
    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();

    *array = (hipArray*)malloc(sizeof(hipArray));
    array[0]->type = pAllocateArray->flags;
    array[0]->width = pAllocateArray->width;
    array[0]->height = pAllocateArray->height;
    array[0]->depth = pAllocateArray->depth;
    array[0]->drvDesc = *pAllocateArray;
    array[0]->isDrv = true;
    array[0]->textureType = hipTextureType3D;
    void** ptr = &array[0]->data;

    if (ctx) {
        const unsigned am_flags = 0;
        const size_t size = pAllocateArray->width * pAllocateArray->height * pAllocateArray->depth;

        size_t allocSize = 0;
        hsa_ext_image_channel_type_t channelType;
        switch (pAllocateArray->format) {
            case HIP_AD_FORMAT_UNSIGNED_INT8:
                allocSize = size * sizeof(uint8_t);
                channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8;
                break;
            case HIP_AD_FORMAT_UNSIGNED_INT16:
                allocSize = size * sizeof(uint16_t);
                channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16;
                break;
            case HIP_AD_FORMAT_UNSIGNED_INT32:
                allocSize = size * sizeof(uint32_t);
                channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32;
                break;
            case HIP_AD_FORMAT_SIGNED_INT8:
                allocSize = size * sizeof(int8_t);
                channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8;
                break;
            case HIP_AD_FORMAT_SIGNED_INT16:
                allocSize = size * sizeof(int16_t);
                channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16;
                break;
            case HIP_AD_FORMAT_SIGNED_INT32:
                allocSize = size * sizeof(int32_t);
                channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32;
                break;
            case HIP_AD_FORMAT_HALF:
                allocSize = size * sizeof(int16_t);
                channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT;
                break;
            case HIP_AD_FORMAT_FLOAT:
                allocSize = size * sizeof(float);
                channelType = HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT;
                break;
            default:
                hip_status = hipErrorUnknown;
                break;
        }

        hc::accelerator acc = ctx->getDevice()->_acc;
        hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

        size_t allocGranularity = 0;
        hsa_amd_memory_pool_t* allocRegion =
            static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_region());
        hsa_amd_memory_pool_get_info(*allocRegion, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                     &allocGranularity);

        hsa_ext_image_descriptor_t imageDescriptor;
        imageDescriptor.width = pAllocateArray->width;
        imageDescriptor.height = pAllocateArray->height;
        imageDescriptor.depth = 0;
        imageDescriptor.array_size = 0;
        switch (pAllocateArray->flags) {
            case hipArrayLayered:
                imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_2DA;
                imageDescriptor.array_size = pAllocateArray->depth;
                break;
            case hipArraySurfaceLoadStore:
            case hipArrayTextureGather:
            case hipArrayDefault:
                assert(0);
                break;
            case hipArrayCubemap:
            default:
                imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_3D;
                imageDescriptor.depth = pAllocateArray->depth;
                break;
        }
        hsa_ext_image_channel_order_t channelOrder;

        // getChannelOrderAndType(*desc, hipReadModeElementType, &channelOrder, &channelType);
        if (pAllocateArray->numChannels == 4) {
            channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
        } else if (pAllocateArray->numChannels == 2) {
            channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_RG;
        } else if (pAllocateArray->numChannels == 1) {
            channelOrder = HSA_EXT_IMAGE_CHANNEL_ORDER_R;
        }
        imageDescriptor.format.channel_order = channelOrder;
        imageDescriptor.format.channel_type = channelType;

        hsa_access_permission_t permission = HSA_ACCESS_PERMISSION_RW;
        hsa_ext_image_data_info_t imageInfo;
        hsa_status_t status =
            hsa_ext_image_data_get_info(*agent, &imageDescriptor, permission, &imageInfo);
        size_t alignment = imageInfo.alignment <= allocGranularity ? 0 : imageInfo.alignment;

        *ptr = hip_internal::allocAndSharePtr("device_array", allocSize, ctx, false, am_flags, 0,
                                              alignment);

        if (size && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        }

    } else {
        hip_status = hipErrorMemoryAllocation;
    }

    return ihipLogStatus(hip_status);
}

hipError_t hipMalloc3DArray(hipArray** array, const struct hipChannelFormatDesc* desc,
                            struct hipExtent extent, unsigned int flags) {



    HIP_INIT_API(hipMalloc3DArray, array, desc, &extent, flags);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;

    if(array==NULL )
    {
         hip_status=hipErrorInvalidValue;
        return ihipLogStatus(hip_status);
    }
    auto ctx = ihipGetTlsDefaultCtx();

    *array = (hipArray*)malloc(sizeof(hipArray));
    array[0]->type = flags;
    array[0]->width = extent.width;
    array[0]->height = extent.height;
    array[0]->depth = extent.depth;
    array[0]->desc = *desc;
    array[0]->isDrv = false;
    array[0]->textureType = hipTextureType3D;
    void** ptr = &array[0]->data;

    if (ctx) {
        const unsigned am_flags = 0;
        const size_t size = extent.width * extent.height * extent.depth;

        const size_t allocSize = size * ((desc->x + desc->y + desc->z + desc->w) / 8);

        hc::accelerator acc = ctx->getDevice()->_acc;
        hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

        size_t allocGranularity = 0;
        hsa_amd_memory_pool_t* allocRegion =
            static_cast<hsa_amd_memory_pool_t*>(acc.get_hsa_am_region());
        hsa_amd_memory_pool_get_info(*allocRegion, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                     &allocGranularity);

        hsa_ext_image_descriptor_t imageDescriptor;
        imageDescriptor.width = extent.width;
        imageDescriptor.height = extent.height;
        imageDescriptor.depth = extent.depth;
        imageDescriptor.array_size = 0;
        switch (flags) {
            case hipArrayLayered:
                imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_2DA;
                imageDescriptor.array_size = extent.depth;
                break;
            case hipArraySurfaceLoadStore:
            case hipArrayTextureGather:
            case hipArrayDefault:
                assert(0);
                break;
            case hipArrayCubemap:
            default:
                imageDescriptor.geometry = HSA_EXT_IMAGE_GEOMETRY_3D;
                imageDescriptor.depth = extent.depth;
                break;
        }
        hsa_ext_image_channel_order_t channelOrder;
        hsa_ext_image_channel_type_t channelType;
        getChannelOrderAndType(*desc, hipReadModeElementType, &channelOrder, &channelType);
        imageDescriptor.format.channel_order = channelOrder;
        imageDescriptor.format.channel_type = channelType;

        hsa_access_permission_t permission = HSA_ACCESS_PERMISSION_RW;
        hsa_ext_image_data_info_t imageInfo;
        hsa_status_t status =
            hsa_ext_image_data_get_info(*agent, &imageDescriptor, permission, &imageInfo);
        size_t alignment = imageInfo.alignment <= allocGranularity ? 0 : imageInfo.alignment;

        *ptr = hip_internal::allocAndSharePtr("device_array", allocSize, ctx, false, am_flags, 0,
                                              alignment);

        if (size && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        }

    } else {
        hip_status = hipErrorMemoryAllocation;
    }

    return ihipLogStatus(hip_status);
}

hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
    HIP_INIT_API(hipHostGetFlags, flagsPtr, hostPtr);

    hipError_t hip_status = hipSuccess;

    hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
    hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
    am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, hostPtr);
    if (status == AM_SUCCESS) {
        *flagsPtr = amPointerInfo._appAllocationFlags;
        if (*flagsPtr == 0) {
            hip_status = hipErrorInvalidValue;
        } else {
            hip_status = hipSuccess;
        }
        tprintf(DB_MEM, " %s: host ptr=%p\n", __func__, hostPtr);
    } else {
        hip_status = hipErrorInvalidValue;
    }
    return ihipLogStatus(hip_status);
}


// TODO - need to fix several issues here related to P2P access, host memory fallback.
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
    HIP_INIT_API(hipHostRegister, hostPtr, sizeBytes, flags);

    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if (hostPtr == NULL) {
        return ihipLogStatus(hipErrorInvalidValue);
    }

    hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
    hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
    am_status_t am_status = hc::am_memtracker_getinfo(&amPointerInfo, hostPtr);

    if (am_status == AM_SUCCESS) {
        hip_status = hipErrorHostMemoryAlreadyRegistered;
    } else {
        auto ctx = ihipGetTlsDefaultCtx();
        if (hostPtr == NULL) {
            return ihipLogStatus(hipErrorInvalidValue);
        }
        // TODO-test : multi-gpu access to registered host memory.
        if (ctx) {
            if (flags == hipHostRegisterDefault || flags == hipHostRegisterPortable ||
                flags == hipHostRegisterMapped) {
                auto device = ctx->getWriteableDevice();
                std::vector<hc::accelerator> vecAcc;
                for (int i = 0; i < g_deviceCnt; i++) {
                    vecAcc.push_back(ihipGetDevice(i)->_acc);
                }
                am_status = hc::am_memory_host_lock(device->_acc, hostPtr, sizeBytes, &vecAcc[0],
                                                    vecAcc.size());
                if ( am_status == AM_SUCCESS ) {
                     am_status = hc::am_memtracker_getinfo(&amPointerInfo, hostPtr);

                if ( am_status == AM_SUCCESS ) {
                    void *devPtr = amPointerInfo._devicePointer;
 #if USE_APP_PTR_FOR_CTX
                    hc::am_memtracker_update(hostPtr, device->_deviceId, flags, ctx);
                    hc::am_memtracker_update(devPtr, device->_deviceId, flags, ctx);
 #else
                    hc::am_memtracker_update(hostPtr, device->_deviceId, flags);
                    hc::am_memtracker_update(devPtr, device->_deviceId, flags);
 #endif
                    tprintf(DB_MEM, " %s registered ptr=%p and allowed access to %zu peers\n", __func__,
                                 hostPtr, vecAcc.size());
                };
                };
                if (am_status == AM_SUCCESS) {
                    hip_status = hipSuccess;
                } else {
                    hip_status = hipErrorMemoryAllocation;
                }
            } else {
                hip_status = hipErrorInvalidValue;
            }
        }
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipHostUnregister(void* hostPtr) {
    HIP_INIT_API(hipHostUnregister, hostPtr);
    auto ctx = ihipGetTlsDefaultCtx();
    hipError_t hip_status = hipSuccess;
    if (hostPtr == NULL) {
        hip_status = hipErrorInvalidValue;
    } else {
        auto device = ctx->getWriteableDevice();
        am_status_t am_status = hc::am_memory_host_unlock(device->_acc, hostPtr);
        tprintf(DB_MEM, " %s unregistered ptr=%p\n", __func__, hostPtr);
        if (am_status != AM_SUCCESS) {
            hip_status = hipErrorHostMemoryNotRegistered;
        }
    }
    return ihipLogStatus(hip_status);
}

namespace {
inline hipDeviceptr_t agent_address_for_symbol(const char* symbolName) {
    hipDeviceptr_t r = nullptr;

#if __hcc_workweek__ >= 17481
    size_t byte_cnt = 0u;
    ihipModuleGetGlobal(&r, &byte_cnt, 0, symbolName);
#else
    auto ctx = ihipGetTlsDefaultCtx();
    auto acc = ctx->getDevice()->_acc;
    r = acc.get_symbol_address(symbolName);
#endif

    return r;
}
}  // namespace

hipError_t hipMemcpyToSymbol(const void* symbolName, const void* src, size_t count, size_t offset,
                             hipMemcpyKind kind) {
    HIP_INIT_SPECIAL_API(hipMemcpyToSymbol, (TRACE_MCMD), symbolName, src, count, offset, kind);

    if (symbolName == nullptr) {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    auto ctx = ihipGetTlsDefaultCtx();

    hc::accelerator acc = ctx->getDevice()->_acc;

    hipDeviceptr_t dst = agent_address_for_symbol(static_cast<const char*>(symbolName));
    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbolName, dst);

    if (dst == nullptr) {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    if (kind == hipMemcpyHostToDevice || kind == hipMemcpyDefault ||
        kind == hipMemcpyDeviceToDevice || kind == hipMemcpyHostToHost) {
        stream->locked_copySync((char*)dst+offset, (void*)src, count, kind, false);
    } else {
        return ihipLogStatus(hipErrorInvalidValue);
    }

    return ihipLogStatus(hipSuccess);
}


hipError_t hipMemcpyFromSymbol(void* dst, const void* symbolName, size_t count, size_t offset,
                               hipMemcpyKind kind) {
    HIP_INIT_SPECIAL_API(hipMemcpyFromSymbol, (TRACE_MCMD), symbolName, dst, count, offset, kind);

    if (symbolName == nullptr) {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    auto ctx = ihipGetTlsDefaultCtx();

    hc::accelerator acc = ctx->getDevice()->_acc;

    hipDeviceptr_t src = agent_address_for_symbol(static_cast<const char*>(symbolName));
    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbolName, dst);

    if (dst == nullptr) {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    if (kind == hipMemcpyDefault || kind == hipMemcpyDeviceToHost ||
        kind == hipMemcpyDeviceToDevice || kind == hipMemcpyHostToHost) {
        stream->locked_copySync((void*)dst, (char*)src+offset, count, kind, false);
    } else {
        return ihipLogStatus(hipErrorInvalidValue);
    }

    return ihipLogStatus(hipSuccess);
}


hipError_t hipMemcpyToSymbolAsync(const void* symbolName, const void* src, size_t count,
                                  size_t offset, hipMemcpyKind kind, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpyToSymbolAsync, (TRACE_MCMD), symbolName, src, count, offset, kind, stream);

    if (symbolName == nullptr) {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    hipError_t e = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();

    hc::accelerator acc = ctx->getDevice()->_acc;

    hipDeviceptr_t dst = agent_address_for_symbol(static_cast<const char*>(symbolName));
    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbolName, dst);

    if (dst == nullptr) {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    if (stream) {
        try {
            hip_internal::memcpyAsync((char*)dst+offset, src, count, kind, stream);
        } catch (ihipException& ex) {
            e = ex._code;
        }
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}


hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbolName, size_t count, size_t offset,
                                    hipMemcpyKind kind, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpyFromSymbolAsync, (TRACE_MCMD), symbolName, dst, count, offset, kind, stream);

    if (symbolName == nullptr) {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    hipError_t e = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();

    hc::accelerator acc = ctx->getDevice()->_acc;

    hipDeviceptr_t src = agent_address_for_symbol(static_cast<const char*>(symbolName));
    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbolName, src);

    if (src == nullptr || dst == nullptr) {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    stream = ihipSyncAndResolveStream(stream);
    if (stream) {
        try {
            hip_internal::memcpyAsync(dst, (char*)src+offset, count, kind, stream);
        } catch (ihipException& ex) {
            e = ex._code;
        }
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}


hipError_t hipGetSymbolAddress(void** devPtr, const void* symbolName) {
    HIP_INIT_API(hipGetSymbolAddress, devPtr, symbolName);

    size_t size = 0;
    return ihipModuleGetGlobal(devPtr, &size, 0, static_cast<const char*>(symbolName));
}


hipError_t hipGetSymbolSize(size_t* size, const void* symbolName) {
    HIP_INIT_API(hipGetSymbolSize, size, symbolName);

    void* devPtr = nullptr;
    return ihipModuleGetGlobal(&devPtr, size, 0, static_cast<const char*>(symbolName));
}


//---
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
    HIP_INIT_SPECIAL_API(hipMemcpy, (TRACE_MCMD), dst, src, sizeBytes, kind);

    hipError_t e = hipSuccess;

    // Return success if number of bytes to copy is 0
    if (sizeBytes == 0) return ihipLogStatus(e);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    if(dst==NULL || src==NULL)
	{
	e=hipErrorInvalidValue;
	return ihipLogStatus(e);
	}
    try {
        stream->locked_copySync(dst, src, sizeBytes, kind);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}


hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMemcpyHtoD, (TRACE_MCMD), dst, src, sizeBytes);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {
        stream->locked_copySync((void*)dst, (void*)src, sizeBytes, hipMemcpyHostToDevice, false);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}


hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMemcpyDtoH, (TRACE_MCMD), dst, src, sizeBytes);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {
        stream->locked_copySync((void*)dst, (void*)src, sizeBytes, hipMemcpyDeviceToHost, false);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}


hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMemcpyDtoD, (TRACE_MCMD), dst, src, sizeBytes);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {
        stream->locked_copySync((void*)dst, (void*)src, sizeBytes, hipMemcpyDeviceToDevice, false);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyHtoH(void* dst, void* src, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMemcpyHtoH, (TRACE_MCMD), dst, src, sizeBytes);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {
        stream->locked_copySync((void*)dst, (void*)src, sizeBytes, hipMemcpyHostToHost, false);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpyAsync, (TRACE_MCMD), dst, src, sizeBytes, kind, stream);

    return ihipLogStatus(hip_internal::memcpyAsync(dst, src, sizeBytes, kind, stream));
}

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpyHtoDAsync, (TRACE_MCMD), dst, src, sizeBytes, stream);

    return ihipLogStatus(
        hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyHostToDevice, stream));
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpyDtoDAsync, (TRACE_MCMD), dst, src, sizeBytes, stream);

    return ihipLogStatus(
        hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, stream));
}

hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpyDtoHAsync, (TRACE_MCMD), dst, src, sizeBytes, stream);

    return ihipLogStatus(
        hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToHost, stream));
}

hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
    HIP_INIT_SPECIAL_API(hipMemcpy2DToArray, (TRACE_MCMD), dst, wOffset, hOffset, src, spitch, width, height, kind);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    size_t byteSize;
    if (dst) {
        switch (dst[0].desc.f) {
            case hipChannelFormatKindSigned:
                byteSize = sizeof(int);
                break;
            case hipChannelFormatKindUnsigned:
                byteSize = sizeof(unsigned int);
                break;
            case hipChannelFormatKindFloat:
                byteSize = sizeof(float);
                break;
            case hipChannelFormatKindNone:
                byteSize = sizeof(size_t);
                break;
            default:
                byteSize = 0;
                break;
        }
    } else {
        return ihipLogStatus(hipErrorUnknown);
    }

    if ((wOffset + width > (dst->width * byteSize)) || width > spitch) {
        return ihipLogStatus(hipErrorUnknown);
    }

    size_t src_w = spitch;
    size_t dst_w = (dst->width) * byteSize;

    try {
        for (int i = 0; i < height; ++i) {
            stream->locked_copySync((unsigned char*)dst->data + i * dst_w,
                                    (unsigned char*)src + i * src_w, width, kind);
        }
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind) {
    HIP_INIT_SPECIAL_API(hipMemcpyToArray, (TRACE_MCMD), dst, wOffset, hOffset, src, count, kind);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {
        stream->locked_copySync((char*)dst->data + wOffset, src, count, kind);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind) {
    HIP_INIT_SPECIAL_API(hipMemcpyFromArray, (TRACE_MCMD), dst, srcArray, wOffset, hOffset, count, kind);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {
        stream->locked_copySync((char*)dst, (char*)srcArray->data + wOffset, count, kind);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyHtoA(hipArray* dstArray, size_t dstOffset, const void* srcHost, size_t count) {
    HIP_INIT_SPECIAL_API(hipMemcpyHtoA, (TRACE_MCMD), dstArray, dstOffset, srcHost, count);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;
    try {
        stream->locked_copySync((char*)dstArray->data + dstOffset, srcHost, count,
                                hipMemcpyHostToDevice);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyAtoH(void* dst, hipArray* srcArray, size_t srcOffset, size_t count) {
    HIP_INIT_SPECIAL_API(hipMemcpyAtoH, (TRACE_MCMD), dst, srcArray, srcOffset, count);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {
        stream->locked_copySync((char*)dst, (char*)srcArray->data + srcOffset, count,
                                hipMemcpyDeviceToHost);
    } catch (ihipException& ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p) {
    HIP_INIT_SPECIAL_API(hipMemcpy3D, (TRACE_MCMD), p);
    hipError_t e = hipSuccess;
    if (p) {
        size_t byteSize;
        size_t depth;
        size_t height;
        size_t widthInBytes;
        size_t srcPitch;
        size_t dstPitch;
        void* srcPtr;
        void* dstPtr;
        size_t ySize;
        if (p->dstArray != nullptr) {
            if (p->dstArray->isDrv == false) {
                switch (p->dstArray->desc.f) {
                    case hipChannelFormatKindSigned:
                        byteSize = sizeof(int);
                        break;
                    case hipChannelFormatKindUnsigned:
                        byteSize = sizeof(unsigned int);
                        break;
                    case hipChannelFormatKindFloat:
                        byteSize = sizeof(float);
                        break;
                    case hipChannelFormatKindNone:
                        byteSize = sizeof(size_t);
                        break;
                    default:
                        byteSize = 0;
                        break;
                }
                depth = p->extent.depth;
                height = p->extent.height;
                widthInBytes = p->extent.width * byteSize;
                srcPitch = p->srcPtr.pitch;
                srcPtr = p->srcPtr.ptr;
                ySize = p->srcPtr.ysize;
                dstPitch = p->dstArray->width * byteSize;
                dstPtr = p->dstArray->data;
            } else {
                depth = p->Depth;
                height = p->Height;
                widthInBytes = p->WidthInBytes;
                dstPitch = p->dstArray->width * 4;
                srcPitch = p->srcPitch;
                srcPtr = (void*)p->srcHost;
                ySize = p->srcHeight;
                dstPtr = p->dstArray->data;
            }
        } else {
            // Non array destination
            depth = p->extent.depth;
            height = p->extent.height;
            widthInBytes = p->extent.width;
            srcPitch = p->srcPtr.pitch;
            srcPtr = p->srcPtr.ptr;
            dstPtr = p->dstPtr.ptr;
            ySize = p->srcPtr.ysize;
            dstPitch = p->dstPtr.pitch;
        }
        hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);
        hc::completion_future marker;
        try {
            if((widthInBytes == dstPitch) && (widthInBytes == srcPitch)) {
                stream->locked_copySync((void*)dstPtr, (void*)srcPtr, widthInBytes*height*depth, p->kind, false);
            } else {
                for (int i = 0; i < depth; i++) {
                    for (int j = 0; j < height; j++) {
                        // TODO: p->srcPos or p->dstPos are not 0.
                        unsigned char* src =
                             (unsigned char*)srcPtr + i * ySize * srcPitch + j * srcPitch;
                        unsigned char* dst =
                             (unsigned char*)dstPtr + i * height * dstPitch + j * dstPitch;
                        stream->locked_copySync(dst, src, widthInBytes, p->kind);
                     }
                }
           }
        } catch (ihipException ex) {
            e = ex._code;
        }
    } else {
        e = hipErrorInvalidValue;
    }
    return ihipLogStatus(e);
}

namespace {
template <uint32_t block_dim, typename RandomAccessIterator, typename N, typename T>
__global__ void hip_fill_n(RandomAccessIterator f, N n, T value) {
    const uint32_t grid_dim = gridDim.x * blockDim.x;

    size_t idx = blockIdx.x * block_dim + threadIdx.x;
    while (idx < n) {
        __builtin_memcpy(reinterpret_cast<void*>(&f[idx]), reinterpret_cast<const void*>(&value),
                         sizeof(T));
        idx += grid_dim;
    }
}

template <typename T, typename std::enable_if<std::is_integral<T>{}>::type* = nullptr>
inline const T& clamp_integer(const T& x, const T& lower, const T& upper) {
    assert(!(upper < lower));

    return std::min(upper, std::max(x, lower));
}

template <typename T>
__global__ void hip_copy2d_n(T* dst, const T* src, size_t width, size_t height, size_t destPitch, size_t srcPitch) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    size_t floorWidth = (width/sizeof(T));
    T *dstPtr = (T *)((uint8_t*) dst + idy * destPitch);
    T *srcPtr = (T *)((uint8_t*) src + idy * srcPitch);
    if((idx < floorWidth) && (idy < height)){
        dstPtr[idx] = srcPtr[idx];
    } else if((idx < width) && (idy < height)){
        size_t bytesToCopy = width - (floorWidth * sizeof(T));
        dstPtr += floorWidth;
        srcPtr += floorWidth;
        __builtin_memcpy(reinterpret_cast<uint8_t*>(dstPtr), reinterpret_cast<const uint8_t*>(srcPtr),bytesToCopy);
    }
}
}  // namespace

template <typename T>
void ihipMemsetKernel(hipStream_t stream, T* ptr, T val, size_t sizeBytes) {
    static constexpr uint32_t block_dim = 256;

    const uint32_t grid_dim = clamp_integer<size_t>(sizeBytes / block_dim, 1, UINT32_MAX);

    hipLaunchKernelGGL(hip_fill_n<block_dim>, dim3(grid_dim), dim3{block_dim}, 0u, stream, ptr,
                       sizeBytes, std::move(val));
}

template <typename T>
void ihipMemcpy2dKernel(hipStream_t stream, T* dst, const T* src, size_t width, size_t height, size_t destPitch, size_t srcPitch) {
    size_t threadsPerBlock_x = 64;
    size_t threadsPerBlock_y = 4;
    uint32_t grid_dim_x = clamp_integer<size_t>( (width+(threadsPerBlock_x*sizeof(T)-1)) / (threadsPerBlock_x*sizeof(T)), 1, UINT32_MAX);
    uint32_t grid_dim_y = clamp_integer<size_t>( (height+(threadsPerBlock_y-1)) / threadsPerBlock_y, 1, UINT32_MAX);
    hipLaunchKernelGGL(hip_copy2d_n, dim3(grid_dim_x,grid_dim_y), dim3(threadsPerBlock_x,threadsPerBlock_y), 0u, stream, dst, src,
                       width, height, destPitch, srcPitch);
}

typedef enum ihipMemsetDataType {
    ihipMemsetDataTypeChar   = 0,
    ihipMemsetDataTypeShort  = 1,
    ihipMemsetDataTypeInt    = 2
}ihipMemsetDataType;

hipError_t ihipMemset(void* dst, int  value, size_t sizeBytes, hipStream_t stream, enum ihipMemsetDataType copyDataType  )
{
    hipError_t e = hipSuccess;

    if (sizeBytes == 0) return e;

    if (stream && (dst != NULL)) {
        if(copyDataType == ihipMemsetDataTypeChar){
            if ((sizeBytes & 0x3) == 0) {
                // use a faster dword-per-workitem copy:
                try {
                    value = value & 0xff;
                    uint32_t value32 = (value << 24) | (value << 16) | (value << 8) | (value) ;
                    ihipMemsetKernel<uint32_t> (stream, static_cast<uint32_t*> (dst), value32, sizeBytes/sizeof(uint32_t));
                }
                catch (std::exception &ex) {
                    e = hipErrorInvalidValue;
                }
             } else {
                // use a slow byte-per-workitem copy:
                try {
                    ihipMemsetKernel<char> (stream, static_cast<char*> (dst), value, sizeBytes);
                }
                catch (std::exception &ex) {
                    e = hipErrorInvalidValue;
                }
            }
        } else {
           if(copyDataType == ihipMemsetDataTypeInt) { // 4 Bytes value
               try {
                   ihipMemsetKernel<uint32_t> (stream, static_cast<uint32_t*> (dst), value, sizeBytes);
               } catch (std::exception &ex) {
                   e = hipErrorInvalidValue;
               }
            } else if(copyDataType == ihipMemsetDataTypeShort) {
               try {
                   value = value & 0xffff;
                   ihipMemsetKernel<uint16_t> (stream, static_cast<uint16_t*> (dst), value, sizeBytes);
               } catch (std::exception &ex) {
                   e = hipErrorInvalidValue;
               }
            }
        }
        if (HIP_API_BLOCKING) {
            tprintf (DB_SYNC, "%s LAUNCH_BLOCKING wait for hipMemsetAsync.\n", ToString(stream).c_str());
            stream->locked_wait();
        }
    } else {
        e = hipErrorInvalidValue;
    }
    return e;
};

hipError_t getLockedPointer(void *hostPtr, size_t dataLen, void **devicePtrPtr)
{
        hc::accelerator acc;

#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, hostPtr);
        if (status == AM_SUCCESS) {
            *devicePtrPtr = static_cast<char*>(amPointerInfo._devicePointer) +
                (static_cast<char*>(hostPtr) - static_cast<char*>(amPointerInfo._hostPointer));
            return(hipSuccess);
        };
        return(hipErrorHostMemoryNotRegistered);
};

// TODO - review and optimize
hipError_t ihipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                        size_t height, hipMemcpyKind kind) {
    if (dst == nullptr || src == nullptr || width > dpitch || width > spitch) return hipErrorInvalidValue;

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);
    int isLocked = 0;
    void *pinnedPtr=NULL;
    void *actualSrc = (void*)src;
    void *actualDest = dst;
    if(kind == hipMemcpyHostToDevice ) {
        if(getLockedPointer((void*)src, spitch, &pinnedPtr) == hipSuccess ){
            isLocked = 1;
            actualSrc = pinnedPtr;
        }
    } else if(kind == hipMemcpyDeviceToHost) {
          if(getLockedPointer((void*)dst, dpitch, &pinnedPtr) == hipSuccess ){
            isLocked = 1;
            actualDest = pinnedPtr;
          }
    }
    hc::completion_future marker;

    hipError_t e = hipSuccess;
    if((width == dpitch) && (width == spitch)) {
        stream->locked_copySync((void*)dst, (void*)src, width*height, kind, false);
    } else {
        try {
            if(!isLocked) {
                for (int i = 0; i < height; ++i)
                    stream->locked_copySync((unsigned char*)dst + i * dpitch,
                                    (unsigned char*)src + i * spitch, width, kind);
            } else {
                stream->locked_copy2DSync(dst, src, width, height, spitch, dpitch, kind);
            }
        } catch (ihipException& ex) {
            e = ex._code;
        }
    }

    return e;
}

hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind) {
    HIP_INIT_SPECIAL_API(hipMemcpy2D, (TRACE_MCMD), dst, dpitch, src, spitch, width, height, kind);
    hipError_t e = hipSuccess;
    e = ihipMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
    return ihipLogStatus(e);
}

hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemcpy2DAsync, (TRACE_MCMD), dst, dpitch, src, spitch, width, height, kind, stream);
    if (dst == nullptr || src == nullptr || width > dpitch || width > spitch) return ihipLogStatus(hipErrorInvalidValue);
    hipError_t e = hipSuccess;
    int isLocked = 0;
    void *pinnedPtr=NULL;
    void *actualSrc = (void*)src;
    void *actualDest = dst;
    stream = ihipSyncAndResolveStream(stream);
    if(kind == hipMemcpyHostToDevice ) {
        if(getLockedPointer((void*)src, spitch, &pinnedPtr) == hipSuccess ){
            isLocked = 1;
            actualSrc = pinnedPtr;
        }
    } else if(kind == hipMemcpyDeviceToHost) {
          if(getLockedPointer((void*)dst, dpitch, &pinnedPtr) == hipSuccess ){
            isLocked = 1;
            actualDest = pinnedPtr;
          }
    }
    if((width == dpitch) && (width == spitch)) {
            hip_internal::memcpyAsync(dst, src, width*height, kind, stream);
    } else {
        try {
            if(!isLocked){
                for (int i = 0; i < height; ++i)
                    e = hip_internal::memcpyAsync((unsigned char*)dst + i * dpitch,
                                          (unsigned char*)src + i * spitch, width, kind, stream);
            } else{
                stream->locked_copy2DAsync(dst, src, width, height, spitch, dpitch, kind);
            }
        } catch (ihipException& ex) {
            e = ex._code;
        }
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
    HIP_INIT_SPECIAL_API(hipMemcpyParam2D, (TRACE_MCMD), pCopy);
    hipError_t e = hipSuccess;
    if (pCopy == nullptr) {
        e = hipErrorInvalidValue;
    }
    e = ihipMemcpy2D(pCopy->dstArray->data, pCopy->widthInBytes, pCopy->srcHost, pCopy->srcPitch,
                     pCopy->widthInBytes, pCopy->height, hipMemcpyDefault);
    return ihipLogStatus(e);
}

// TODO-sync: function is async unless target is pinned host memory - then these are fully sync.
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipMemsetAsync, (TRACE_MCMD), dst, value, sizeBytes, stream);

    hipError_t e = hipSuccess;

    stream = ihipSyncAndResolveStream(stream);

    e = ihipMemset(dst, value, sizeBytes, stream, ihipMemsetDataTypeChar);

    return ihipLogStatus(e);
};

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMemset, (TRACE_MCMD), dst, value, sizeBytes);

    hipError_t e = hipSuccess;

    hipStream_t stream = hipStreamNull;
    stream =  ihipSyncAndResolveStream(stream);
    if (stream) {
        e = ihipMemset(dst, value, sizeBytes, stream, ihipMemsetDataTypeChar);
        stream->locked_wait();
    } else {
        e = hipErrorInvalidValue;
    }
    return ihipLogStatus(e);
}

hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
    HIP_INIT_SPECIAL_API(hipMemset2D, (TRACE_MCMD), dst, pitch, value, width, height);

    hipError_t e = hipSuccess;

    hipStream_t stream = hipStreamNull;
    stream = ihipSyncAndResolveStream(stream);
    if (stream) {
        size_t sizeBytes = pitch * height;
        e = ihipMemset(dst, value, sizeBytes, stream, ihipMemsetDataTypeChar);
        stream->locked_wait();
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream )
{
    HIP_INIT_SPECIAL_API(hipMemset2DAsync, (TRACE_MCMD), dst, pitch, value, width, height, stream);

    hipError_t e = hipSuccess;

    stream =  ihipSyncAndResolveStream(stream);

    if (stream) {
        size_t sizeBytes = pitch * height;
        e = ihipMemset(dst, value, sizeBytes, stream, ihipMemsetDataTypeChar);
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
};

hipError_t hipMemsetD8(hipDeviceptr_t dst, unsigned char value, size_t sizeBytes) {
    HIP_INIT_SPECIAL_API(hipMemsetD8, (TRACE_MCMD), dst, value, sizeBytes);

    hipError_t e = hipSuccess;

    hipStream_t stream = hipStreamNull;
    stream = ihipSyncAndResolveStream(stream);
    if (stream) {
        e = ihipMemset(dst, value, sizeBytes, stream, ihipMemsetDataTypeChar);
        stream->locked_wait();
    } else {
        e = hipErrorInvalidValue;
    }
    return ihipLogStatus(e);
}

hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent )
{
    HIP_INIT_SPECIAL_API(hipMemset3D, (TRACE_MCMD), &pitchedDevPtr, value, &extent);
    hipError_t e = hipSuccess;

    hipStream_t stream = hipStreamNull;
    // TODO - call an ihip memset so HIP_TRACE is correct.
    stream =  ihipSyncAndResolveStream(stream);
    if (stream) {
        size_t sizeBytes = pitchedDevPtr.pitch * extent.height * extent.depth;
        e = ihipMemset(pitchedDevPtr.ptr, value, sizeBytes, stream, ihipMemsetDataTypeChar);
        stream->locked_wait();
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent ,hipStream_t stream )
{
    HIP_INIT_SPECIAL_API(hipMemset3DAsync, (TRACE_MCMD), &pitchedDevPtr, value, &extent);
    hipError_t e = hipSuccess;

    // TODO - call an ihip memset so HIP_TRACE is correct.
    stream =  ihipSyncAndResolveStream(stream);
    if (stream) {
        size_t sizeBytes = pitchedDevPtr.pitch * extent.height * extent.depth;
        e = ihipMemset(pitchedDevPtr.ptr, value, sizeBytes, stream, ihipMemsetDataTypeChar);
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemGetInfo(size_t* free, size_t* total) {
    HIP_INIT_API(hipMemGetInfo, free, total);

    hipError_t e = hipSuccess;

    ihipCtx_t* ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        auto device = ctx->getWriteableDevice();
        if (total) {
            *total = device->_props.totalGlobalMem;
        } else {
            e = hipErrorInvalidValue;
        }

        if (free) {
            // TODO - replace with kernel-level for reporting free memory:
            size_t deviceMemSize, hostMemSize, userMemSize;
            hc::am_memtracker_sizeinfo(device->_acc, &deviceMemSize, &hostMemSize, &userMemSize);

            *free = device->_props.totalGlobalMem - deviceMemSize;

            // Deduct the amount of memory from the free memory reported from the system
            if (HIP_HIDDEN_FREE_MEM) *free -= (size_t)HIP_HIDDEN_FREE_MEM * 1024 * 1024;
        } else {
            e = hipErrorInvalidValue;
        }

    } else {
        e = hipErrorInvalidDevice;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
    HIP_INIT_API(hipMemPtrGetInfo, ptr, size);

    hipError_t e = hipSuccess;

    if (ptr != nullptr && size != nullptr) {
        hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
        if (status == AM_SUCCESS) {
            *size = amPointerInfo._sizeBytes;
        } else {
            e = hipErrorInvalidValue;
        }
    } else {
        e = hipErrorInvalidValue;
    }
    return ihipLogStatus(e);
}


hipError_t hipFree(void* ptr) {
    HIP_INIT_SPECIAL_API(hipFree, (TRACE_MEM), ptr);

    hipError_t hipStatus = hipErrorInvalidDevicePointer;

    // Synchronize to ensure all work has finished.
    ihipGetTlsDefaultCtx()->locked_waitAllStreams();  // ignores non-blocking streams, this waits
                                                      // for all activity to finish.

    if (ptr) {
        hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
        if (status == AM_SUCCESS) {
            if (amPointerInfo._hostPointer == NULL) {
                hc::am_free(ptr);
                hipStatus = hipSuccess;
            }
        }
    } else {
        // free NULL pointer succeeds and is common technique to initialize runtime
        hipStatus = hipSuccess;
    }

    return ihipLogStatus(hipStatus);
}


hipError_t hipHostFree(void* ptr) {
    HIP_INIT_SPECIAL_API(hipHostFree, (TRACE_MEM), ptr);

    // Synchronize to ensure all work has finished.
    ihipGetTlsDefaultCtx()->locked_waitAllStreams();  // ignores non-blocking streams, this waits
                                                      // for all activity to finish.


    hipError_t hipStatus = hipErrorInvalidValue;
    if (ptr) {
        hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
        if (status == AM_SUCCESS) {
            if (amPointerInfo._hostPointer == ptr) {
                hc::am_free(ptr);
                hipStatus = hipSuccess;
            }
        }
    } else {
        // free NULL pointer succeeds and is common technique to initialize runtime
        hipStatus = hipSuccess;
    }

    return ihipLogStatus(hipStatus);
};


// Deprecated:
hipError_t hipFreeHost(void* ptr) { return hipHostFree(ptr); }

hipError_t hipFreeArray(hipArray* array) {
    HIP_INIT_SPECIAL_API(hipFreeArray, (TRACE_MEM), array);

    hipError_t hipStatus = hipErrorInvalidDevicePointer;

    // Synchronize to ensure all work has finished.
    ihipGetTlsDefaultCtx()->locked_waitAllStreams();  // ignores non-blocking streams, this waits
                                                      // for all activity to finish.

    if (array->data) {
        hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, array->data);
        if (status == AM_SUCCESS) {
            if (amPointerInfo._hostPointer == NULL) {
                hc::am_free(array->data);
                hipStatus = hipSuccess;
            }
        }
    }

    return ihipLogStatus(hipStatus);
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
    HIP_INIT_API(hipMemGetAddressRange, pbase, psize, dptr);
    hipError_t hipStatus = hipSuccess;
    hc::accelerator acc;
#if (__hcc_workweek__ >= 17332)
    hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
    am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, dptr);
    if (status == AM_SUCCESS) {
        *pbase = amPointerInfo._devicePointer;
        *psize = amPointerInfo._sizeBytes;
    } else
        hipStatus = hipErrorInvalidDevicePointer;
    return ihipLogStatus(hipStatus);
}


// TODO: IPC implementaiton:

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
    HIP_INIT_API(hipIpcGetMemHandle, handle, devPtr);
    hipError_t hipStatus = hipSuccess;
    // Get the size of allocated pointer
    size_t psize = 0u;
    hc::accelerator acc;
    if ((handle == NULL) || (devPtr == NULL)) {
        hipStatus = hipErrorInvalidResourceHandle;
    } else {
#if (__hcc_workweek__ >= 17332)
        hc::AmPointerInfo amPointerInfo(NULL, NULL, NULL, 0, acc, 0, 0);
#else
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
#endif
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, devPtr);
        if (status == AM_SUCCESS) {
            psize = (size_t)amPointerInfo._sizeBytes;
        } else {
            hipStatus = hipErrorInvalidResourceHandle;
        }
        ihipIpcMemHandle_t* iHandle = (ihipIpcMemHandle_t*)handle;
        // Save the size of the pointer to hipIpcMemHandle
        iHandle->psize = psize;

#if USE_IPC
        // Create HSA ipc memory
        hsa_status_t hsa_status =
            hsa_amd_ipc_memory_create(devPtr, psize, (hsa_amd_ipc_memory_t*)&(iHandle->ipc_handle));
        if (hsa_status != HSA_STATUS_SUCCESS) hipStatus = hipErrorMemoryAllocation;
#else
        hipStatus = hipErrorRuntimeOther;
#endif
    }
    return ihipLogStatus(hipStatus);
}

hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags) {
    HIP_INIT_API(hipIpcOpenMemHandle, devPtr, &handle, flags);
    hipError_t hipStatus = hipSuccess;
    if (devPtr == NULL) {
        hipStatus = hipErrorInvalidValue;
    } else {
#if USE_IPC
        // Get the current device agent.
        hc::accelerator acc;
        hsa_agent_t* agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());
        if (!agent) return hipErrorInvalidResourceHandle;

        ihipIpcMemHandle_t* iHandle = (ihipIpcMemHandle_t*)&handle;
        // Attach ipc memory
        auto ctx = ihipGetTlsDefaultCtx();
        {
            LockedAccessor_CtxCrit_t crit(ctx->criticalData());
            // the peerCnt always stores self so make sure the trace actually
            hsa_status_t hsa_status = hsa_amd_ipc_memory_attach(
                (hsa_amd_ipc_memory_t*)&(iHandle->ipc_handle), iHandle->psize, crit->peerCnt(),
                crit->peerAgents(), devPtr);
            if (hsa_status != HSA_STATUS_SUCCESS) hipStatus = hipErrorMapBufferObjectFailed;
        }
#else
        hipStatus = hipErrorRuntimeOther;
#endif
    }
    return ihipLogStatus(hipStatus);
}

hipError_t hipIpcCloseMemHandle(void* devPtr) {
    HIP_INIT_API(hipIpcCloseMemHandle, devPtr);
    hipError_t hipStatus = hipSuccess;
    if (devPtr == NULL) {
        hipStatus = hipErrorInvalidValue;
    } else {
#if USE_IPC
        hsa_status_t hsa_status = hsa_amd_ipc_memory_detach(devPtr);
        if (hsa_status != HSA_STATUS_SUCCESS) return hipErrorInvalidResourceHandle;
#else
        hipStatus = hipErrorRuntimeOther;
#endif
    }
    return ihipLogStatus(hipStatus);
}

// hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle){
//     return hipSuccess;
// }
