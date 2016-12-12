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
#include "hip_hcc.h"
#include "trace_helper.h"
#include "hip/hcc_detail/hip_texture.h"
#include <hc_am.hpp>

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Memory
//
//
hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, void* ptr)
{
    HIP_INIT_API(attributes, ptr);

    hipError_t e = hipSuccess;

    hc::accelerator acc;
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
    am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
    if (status == AM_SUCCESS) {

        attributes->memoryType    = amPointerInfo._isInDeviceMem ? hipMemoryTypeDevice: hipMemoryTypeHost;
        attributes->hostPointer   = amPointerInfo._hostPointer;
        attributes->devicePointer = amPointerInfo._devicePointer;
        attributes->isManaged     = 0;
        if(attributes->memoryType == hipMemoryTypeHost){
            attributes->hostPointer = ptr;
        }
        if(attributes->memoryType == hipMemoryTypeDevice){
            attributes->devicePointer = ptr;
        }
        attributes->allocationFlags = amPointerInfo._appAllocationFlags;
        attributes->device          = amPointerInfo._appId;

        if (attributes->device < 0) {
            e = hipErrorInvalidDevice;
        }


    } else {
        attributes->memoryType    = hipMemoryTypeDevice;
        attributes->hostPointer   = 0;
        attributes->devicePointer = 0;
        attributes->device        = -1;
        attributes->isManaged     = 0;
        attributes->allocationFlags = 0;

        e = hipErrorUnknown; // TODO - should be hipErrorInvalidValue ?
    }

    return ihipLogStatus(e);
}

hipError_t hipHostGetDevicePointer(void **devicePointer, void *hostPointer, unsigned flags)
{
    HIP_INIT_API(devicePointer, hostPointer, flags);

    hipError_t e = hipSuccess;

    *devicePointer = NULL;

    // Flags must be 0:
    if (flags != 0) {
        e = hipErrorInvalidValue;
    } else {
        hc::accelerator acc;
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, hostPointer);
        if (status == AM_SUCCESS) {
            *devicePointer = amPointerInfo._devicePointer;
        } else {
            e = hipErrorMemoryAllocation;
        }
    }
    return ihipLogStatus(e);
}

hipError_t hipMalloc(void** ptr, size_t sizeBytes)
{
    HIP_INIT_API(ptr, sizeBytes);

    hipError_t  hip_status = hipSuccess;
    // return NULL pointer when malloc size is 0
    if (sizeBytes == 0)
    {
        *ptr = NULL;
        return ihipLogStatus(hipSuccess);
    }

    auto ctx = ihipGetTlsDefaultCtx();

    if (ctx) {
        auto device = ctx->getWriteableDevice();
        const unsigned am_flags = 0;
        *ptr = hc::am_alloc(sizeBytes, device->_acc, am_flags);


        if (sizeBytes && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        } else {
            hc::am_memtracker_update(*ptr, device->_deviceId, 0);
            int peerCnt=0;
            {
                LockedAccessor_CtxCrit_t crit(ctx->criticalData());
                // the peerCnt always stores self so make sure the trace actually
                peerCnt = crit->peerCnt();
                tprintf(DB_MEM, " allocated device_mem ptr:%p size:%zu on dev:%d and allowed %d other peer(s) access\n",
                        *ptr, sizeBytes, device->_deviceId, peerCnt-1);
                if (peerCnt > 1) {

                    //printf ("peer self access\n");

                    // TODOD - remove me:
                    for (auto iter = crit->_peers.begin(); iter!=crit->_peers.end(); iter++) {
                        tprintf (DB_MEM, "   allow access to peer: %s%s\n", (*iter)->toString().c_str(), (iter == crit->_peers.begin()) ? " (self)":"");
                    };

                    hsa_status_t e = hsa_amd_agents_allow_access(crit->peerCnt(), crit->peerAgents(), NULL, *ptr);
                    if (e != HSA_STATUS_SUCCESS) {
                        hip_status = hipErrorMemoryAllocation;
                    }
                }
            }
        }
    } else {
        hip_status = hipErrorMemoryAllocation;
    }


    return ihipLogStatus(hip_status);
}


hipError_t hipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags)
{
    HIP_INIT_API(ptr, sizeBytes, flags);

    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();

    if (sizeBytes == 0) {
        hip_status = hipSuccess;
        // TODO - should size of 0 return err or be siliently ignored?
    } else if ((ctx==nullptr) || (ptr == nullptr)) {
        hip_status = hipErrorInvalidValue;
    } else {
        unsigned trueFlags = flags;
        if (flags == hipHostMallocDefault) {
            trueFlags = hipHostMallocMapped | hipHostMallocWriteCombined;
        }

        const unsigned supportedFlags = hipHostMallocPortable | hipHostMallocMapped | hipHostMallocWriteCombined;

        if (flags & ~supportedFlags) {
            hip_status = hipErrorInvalidValue;
        }
        else {
            auto device = ctx->getWriteableDevice();
            if(HIP_COHERENT_HOST_ALLOC){
                // Force to allocate finedgrained system memory
                *ptr = hc::am_alloc(sizeBytes, device->_acc, amHostPinned);
                if(sizeBytes < 1 && (*ptr == NULL)){
                    hip_status = hipErrorMemoryAllocation;
                } else {
                    hc::am_memtracker_update(*ptr, device->_deviceId, amHostCoherent);
                }
                tprintf(DB_MEM, " %s: finegrained system memory ptr=%p\n", __func__, *ptr);
             }
            else{
                // TODO - am_alloc requires writeable __acc, perhaps could be refactored?
                // TODO - hipHostMallocMapped is be ignored on ROCM - all memory is mapped to host address space as WC.
                *ptr = hc::am_alloc(sizeBytes, device->_acc, amHostPinned);
                if (*ptr == NULL) {
                    hip_status = hipErrorMemoryAllocation;
                } else {
                    hc::am_memtracker_update(*ptr, device->_deviceId, flags);
                    // TODO-hipHostMallocPortable should map the host memory into all contexts, regardless of peer status.
                    int peerCnt=0;
                    {
                        LockedAccessor_CtxCrit_t crit(ctx->criticalData());
                        peerCnt = crit->peerCnt();
                        if (peerCnt > 1) {
                            hsa_amd_agents_allow_access(crit->peerCnt(), crit->peerAgents(), NULL, *ptr);
                        }
                    }
                    tprintf(DB_MEM, "allocated pinned_host ptr:%p size:%zu on dev:%d and allow access to %d other peer(s)\n", *ptr, sizeBytes, device->_deviceId, peerCnt-1);
                }
            }
        }
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipMallocHost(void** ptr, size_t sizeBytes)
{
    return hipHostMalloc(ptr, sizeBytes, 0);
}

hipError_t hipHostAlloc(void** ptr, size_t sizeBytes, unsigned int flags)
{
    return hipHostMalloc(ptr, sizeBytes, flags);
};

// width in bytes
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height)
{
    HIP_INIT_API(ptr, pitch, width, height);

    hipError_t  hip_status = hipSuccess;

    if(width == 0 || height == 0)
        return ihipLogStatus(hipErrorUnknown);

    // hardcoded 128 bytes
    *pitch = ((((int)width-1)/128) + 1)*128;
    const size_t sizeBytes = (*pitch)*height;

    auto ctx = ihipGetTlsDefaultCtx();

    //err = hipMalloc(ptr, (*pitch)*height);
    if (ctx) {
        auto device = ctx->getWriteableDevice();

        const unsigned am_flags = 0;
        *ptr = hc::am_alloc(sizeBytes, device->_acc, am_flags);

        if (sizeBytes && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        } else {
            hc::am_memtracker_update(*ptr, device->_deviceId, 0);
            {
                LockedAccessor_CtxCrit_t crit(ctx->criticalData());
                if (crit->peerCnt() > 1) { // peerCnt includes self so only call allow_access if other peers involved:
                    hsa_status_t hsa_status = hsa_amd_agents_allow_access(crit->peerCnt(), crit->peerAgents(), NULL, *ptr);
                    if (hsa_status != HSA_STATUS_SUCCESS) {
                        hip_status = hipErrorMemoryAllocation;
                    }
                }
            }
        }
    } else {
        hip_status = hipErrorMemoryAllocation;
    }

    return ihipLogStatus(hip_status);
}

hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f)
{
    hipChannelFormatDesc cd;
    cd.x = x; cd.y = y; cd.z = z; cd.w = w;
    cd.f = f;
    return cd;
}

hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc,
        size_t width, size_t height, unsigned int flags)
{
    HIP_INIT_API(array, desc, width, height, flags);

    hipError_t  hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();

    *array = (hipArray*)malloc(sizeof(hipArray));
    array[0]->width = width;
    array[0]->height = height;

    array[0]->f = desc->f;

    void ** ptr = &array[0]->data;

    if (ctx) {
        auto device = ctx->getWriteableDevice();
        const unsigned am_flags = 0;
        const size_t size = width*height;

        switch(desc->f) {
            case hipChannelFormatKindSigned:
                *ptr = hc::am_alloc(size*sizeof(int), device->_acc, am_flags);
                break;
            case hipChannelFormatKindUnsigned:
                *ptr = hc::am_alloc(size*sizeof(unsigned int), device->_acc, am_flags);
                break;
            case hipChannelFormatKindFloat:
                *ptr = hc::am_alloc(size*sizeof(float), device->_acc, am_flags);
                break;
            case hipChannelFormatKindNone:
                *ptr = hc::am_alloc(size*sizeof(size_t), device->_acc, am_flags);
                break;
            default:
                hip_status = hipErrorUnknown;
                break;
        }
        if (size && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
        } else {
            hc::am_memtracker_update(*ptr, device->_deviceId, 0);
            {
                LockedAccessor_CtxCrit_t crit(ctx->criticalData());
                if (crit->peerCnt() > 1) { // peerCnt includes self so only call allow_access if other peers involved:
                    hsa_status_t hsa_status = hsa_amd_agents_allow_access(crit->peerCnt(), crit->peerAgents(), NULL, *ptr);
                    if (hsa_status != HSA_STATUS_SUCCESS) {
                        hip_status = hipErrorMemoryAllocation;
                    }
                }
            }
        }

    } else {
        hip_status = hipErrorMemoryAllocation;
    }

    return ihipLogStatus(hip_status);
}

hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr)
{
    HIP_INIT_API(flagsPtr, hostPtr);

    hipError_t hip_status = hipSuccess;

    hc::accelerator acc;
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
    am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, hostPtr);
    if(status == AM_SUCCESS){
        *flagsPtr = amPointerInfo._appAllocationFlags;
        if(*flagsPtr == 0){
            hip_status = hipErrorInvalidValue;
        }
        else{
            hip_status = hipSuccess;
        }
        tprintf(DB_MEM, " %s: host ptr=%p\n", __func__, hostPtr);
    }else{
        hip_status = hipErrorInvalidValue;
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipHostRegister(void *hostPtr, size_t sizeBytes, unsigned int flags)
{
    HIP_INIT_API(hostPtr, sizeBytes, flags);

    hipError_t hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();
    if(hostPtr == NULL){
        return ihipLogStatus(hipErrorInvalidValue);
    }

    hc::accelerator acc;
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
    am_status_t am_status = hc::am_memtracker_getinfo(&amPointerInfo, hostPtr);

    if(am_status == AM_SUCCESS){
        hip_status = hipErrorHostMemoryAlreadyRegistered;
    } else {
        auto ctx = ihipGetTlsDefaultCtx();
        if(hostPtr == NULL){
            return ihipLogStatus(hipErrorInvalidValue);
        }
        if (ctx) {
            auto device = ctx->getWriteableDevice();
            if(flags == hipHostRegisterDefault || flags == hipHostRegisterPortable || flags == hipHostRegisterMapped){
                std::vector<hc::accelerator>vecAcc;
                for(int i=0;i<g_deviceCnt;i++){
                    vecAcc.push_back(ihipGetDevice(i)->_acc);
                }
                am_status = hc::am_memory_host_lock(device->_acc, hostPtr, sizeBytes, &vecAcc[0], vecAcc.size());

                tprintf(DB_MEM, " %s registered ptr=%p\n", __func__, hostPtr);
                if(am_status == AM_SUCCESS){
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

hipError_t hipHostUnregister(void *hostPtr)
{
    HIP_INIT_API(hostPtr);
    auto ctx = ihipGetTlsDefaultCtx();
    hipError_t hip_status = hipSuccess;
    if(hostPtr == NULL){
        hip_status = hipErrorInvalidValue;
    }else{
        auto device = ctx->getWriteableDevice();
        am_status_t am_status = hc::am_memory_host_unlock(device->_acc, hostPtr);
        tprintf(DB_MEM, " %s unregistered ptr=%p\n", __func__, hostPtr);
        if(am_status != AM_SUCCESS){
            hip_status = hipErrorHostMemoryNotRegistered;
        }
    }
    return ihipLogStatus(hip_status);
}

hipError_t hipMemcpyToSymbol(const char* symbolName, const void *src, size_t count, size_t offset, hipMemcpyKind kind)
{
    HIP_INIT_API(symbolName, src, count, offset, kind);

    if(symbolName == nullptr)
    {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    auto ctx = ihipGetTlsDefaultCtx();

    hc::accelerator acc = ctx->getDevice()->_acc;

    void *ptr = acc.get_symbol_address(symbolName);
    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbolName, ptr);

    if(ptr == nullptr)
    {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    stream->locked_copySync(ptr, src, count + offset, kind);

    return ihipLogStatus(hipSuccess);
}

hipError_t hipMemcpyToSymbolAsync(const char* symbolName, const void *src, size_t count, size_t offset, hipMemcpyKind kind, hipStream_t stream)
{
    HIP_INIT_API(symbolName, src, count, offset, kind, stream);

    if(symbolName == nullptr)
    {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    hipError_t e = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();

    hc::accelerator acc = ctx->getDevice()->_acc;

    void *ptr = acc.get_symbol_address(symbolName);
    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbolName, ptr);

    if(ptr == nullptr)
    {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    if (stream) {
        try {
            stream->locked_copyAsync(ptr, src, count + offset, kind);
        }
        catch (ihipException ex) {
            e = ex._code;
        }
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);


}

//---
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
{
    HIP_INIT_API(dst, src, sizeBytes, kind);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {

        stream->locked_copySync(dst, src, sizeBytes, kind);
    }
    catch (ihipException ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes)
{
    HIP_INIT_API(dst, src, sizeBytes);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {

        stream->locked_copySync((void*)dst, (void*)src, sizeBytes, hipMemcpyHostToDevice, false);
    }
    catch (ihipException ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes)
{
    HIP_INIT_API(dst, src, sizeBytes);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {

        stream->locked_copySync((void*)dst, (void*)src, sizeBytes, hipMemcpyDeviceToHost, false);
    }
    catch (ihipException ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes)
{
    HIP_INIT_API(dst, src, sizeBytes);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {

        stream->locked_copySync((void*)dst, (void*)src, sizeBytes, hipMemcpyDeviceToDevice, false);
    }
    catch (ihipException ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyHtoH(void* dst, void* src, size_t sizeBytes)
{
    HIP_INIT_API(dst, src, sizeBytes);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {

        stream->locked_copySync((void*)dst, (void*)src, sizeBytes, hipMemcpyHostToHost, false);
    }
    catch (ihipException ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}



// Internal copy sync:
namespace hip_internal {

hipError_t memcpyAsync (void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)
{
    hipError_t e = hipSuccess;

    stream = ihipSyncAndResolveStream(stream);


    if ((dst == NULL) || (src == NULL)) {
        e= hipErrorInvalidValue;
    } else if (stream) {
        try {
            stream->locked_copyAsync(dst, src, sizeBytes, kind);
        }
        catch (ihipException ex) {
            e = ex._code;
        }
    } else {
        e = hipErrorInvalidValue;
    }

    return e;
}
} // end namespace hip_internal


hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)
{
    HIP_INIT_API(dst, src, sizeBytes, kind, stream);

    return ihipLogStatus(hip_internal::memcpyAsync(dst, src, sizeBytes, kind, stream));

}


hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes, hipStream_t stream)
{
    HIP_INIT_API(dst, src, sizeBytes, stream);

    return ihipLogStatus(hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyHostToDevice, stream));
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)
{
    HIP_INIT_API(dst, src, sizeBytes, stream);

    return ihipLogStatus(hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, stream));
}

hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)
{
    HIP_INIT_API(dst, src, sizeBytes, stream);

    return ihipLogStatus(hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToHost, stream));
}

// TODO - review and optimize
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
        size_t width, size_t height, hipMemcpyKind kind) {

    HIP_INIT_API(dst, dpitch, src, spitch, width, height, kind);

    if(width > dpitch || width > spitch)
        return ihipLogStatus(hipErrorUnknown);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {
        for(int i = 0; i < height; ++i) {
            stream->locked_copySync((unsigned char*)dst + i*dpitch, (unsigned char*)src + i*spitch, width, kind);
        }
    }
    catch (ihipException ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
        size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {

    HIP_INIT_API(dst, wOffset, hOffset, src, spitch, width, height, kind);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    size_t byteSize;
    if(dst) {
        switch(dst[0].f) {
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

    if((wOffset + width > (dst->width * byteSize)) || width > spitch) {
        return ihipLogStatus(hipErrorUnknown);
    }

    size_t src_w = spitch;
    size_t dst_w = (dst->width)*byteSize;

    try {
        for(int i = 0; i < height; ++i) {
            stream->locked_copySync((unsigned char*)dst->data + i*dst_w, (unsigned char*)src + i*src_w, width, kind);
        }
    }
    catch (ihipException ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset,
        const void* src, size_t count, hipMemcpyKind kind) {

    HIP_INIT_API(dst, wOffset, hOffset, src, count, kind);

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    hc::completion_future marker;

    hipError_t e = hipSuccess;

    try {
        stream->locked_copySync((char *)dst->data + wOffset, src, count, kind);
    }
    catch (ihipException ex) {
        e = ex._code;
    }

    return ihipLogStatus(e);
}

// TODO - make member function of stream?
template <typename T>
hc::completion_future
ihipMemsetKernel(hipStream_t stream,
    LockedAccessor_StreamCrit_t &crit,
    T * ptr, T val, size_t sizeBytes)
{
    int wg = std::min((unsigned)8, stream->getDevice()->_computeUnits);
    const int threads_per_wg = 256;

    int threads = wg * threads_per_wg;
    if (threads > sizeBytes) {
        threads = ((sizeBytes + threads_per_wg - 1) / threads_per_wg) * threads_per_wg;
    }


    hc::extent<1> ext(threads);
    auto ext_tile = ext.tile(threads_per_wg);

    hc::completion_future cf =
    hc::parallel_for_each(
            crit->_av,
            ext_tile,
            [=] (hc::tiled_index<1> idx)
            __attribute__((hc))
    {
        int offset = amp_get_global_id(0);
        // TODO-HCC - change to hc_get_local_size()
        int stride = amp_get_local_size(0) * hc_get_num_groups(0) ;

        for (int i=offset; i<sizeBytes; i+=stride) {
            ptr[i] = val;
        }
    });

    return cf;
}

// TODO-sync: function is async unless target is pinned host memory - then these are fully sync.
hipError_t hipMemsetAsync(void* dst, int  value, size_t sizeBytes, hipStream_t stream )
{
    HIP_INIT_API(dst, value, sizeBytes, stream);

    hipError_t e = hipSuccess;

    stream =  ihipSyncAndResolveStream(stream);

    if (stream) {
        auto crit = stream->lockopen_preKernelCommand();

        hc::completion_future cf ;

        if ((sizeBytes & 0x3) == 0) {
            // use a faster dword-per-workitem copy:
            try {
                value = value & 0xff;
                unsigned value32 = (value << 24) | (value << 16) | (value << 8) | (value) ;
                cf = ihipMemsetKernel<unsigned> (stream, crit, static_cast<unsigned*> (dst), value32, sizeBytes/sizeof(unsigned));
            }
            catch (std::exception &ex) {
                e = hipErrorInvalidValue;
            }
        } else {
            // use a slow byte-per-workitem copy:
            try {
                cf = ihipMemsetKernel<char> (stream, crit, static_cast<char*> (dst), value, sizeBytes);
            }
            catch (std::exception &ex) {
                e = hipErrorInvalidValue;
            }
        }

        stream->lockclose_postKernelCommand("hipMemsetAsync", &crit->_av);


        if (HIP_API_BLOCKING) {
            tprintf (DB_SYNC, "%s LAUNCH_BLOCKING wait for hipMemsetAsync.\n", ToString(stream).c_str());
            cf.wait();
            //tprintf (DB_SYNC, "'%s' LAUNCH_BLOCKING memset completed [stream:%p].\n", __func__, (void*)stream);
        }
    } else {
        e = hipErrorInvalidValue;
    }


    return ihipLogStatus(e);
};

hipError_t hipMemset(void* dst, int  value, size_t sizeBytes )
{
    hipStream_t stream = hipStreamNull;
    // TODO - call an ihip memset so HIP_TRACE is correct.
    HIP_INIT_API(dst, value, sizeBytes, stream);

    hipError_t e = hipSuccess;

    stream =  ihipSyncAndResolveStream(stream);

    if (stream) {
        auto crit = stream->lockopen_preKernelCommand();

        hc::completion_future cf ;

        if ((sizeBytes & 0x3) == 0) {
            // use a faster dword-per-workitem copy:
            try {
                value = value & 0xff;
                unsigned value32 = (value << 24) | (value << 16) | (value << 8) | (value) ;
                cf = ihipMemsetKernel<unsigned> (stream, crit, static_cast<unsigned*> (dst), value32, sizeBytes/sizeof(unsigned));
            }
            catch (std::exception &ex) {
                e = hipErrorInvalidValue;
            }
        } else {
            // use a slow byte-per-workitem copy:
            try {
                cf = ihipMemsetKernel<char> (stream, crit, static_cast<char*> (dst), value, sizeBytes);
            }
            catch (std::exception &ex) {
                e = hipErrorInvalidValue;
            }
        }
        // TODO - is hipMemset supposed to be async?
        cf.wait();

        stream->lockclose_postKernelCommand("hipMemset", &crit->_av);


        if (HIP_LAUNCH_BLOCKING) {
            tprintf (DB_SYNC, "'%s' LAUNCH_BLOCKING wait for memset [stream:%p].\n", __func__, (void*)stream);
            cf.wait();
            tprintf (DB_SYNC, "'%s' LAUNCH_BLOCKING memset completed [stream:%p].\n", __func__, (void*)stream);
        }
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemGetInfo  (size_t *free, size_t *total)
{
    HIP_INIT_API(free, total);

    hipError_t e = hipSuccess;

    ihipCtx_t * ctx = ihipGetTlsDefaultCtx();
    if (ctx) {
        auto device = ctx->getWriteableDevice();
        if (total) {
            *total = device->_props.totalGlobalMem;
        }
        else {
             e = hipErrorInvalidValue;
        }

        if (free) {
            // TODO - replace with kernel-level for reporting free memory:
            size_t deviceMemSize, hostMemSize, userMemSize;
            hc::am_memtracker_sizeinfo(device->_acc, &deviceMemSize, &hostMemSize, &userMemSize);

            *free =  device->_props.totalGlobalMem - deviceMemSize;
        }
        else {
             e = hipErrorInvalidValue;
        }

    } else {
        e = hipErrorInvalidDevice;
    }

    return ihipLogStatus(e);
}

hipError_t hipFree(void* ptr)
{
    HIP_INIT_API(ptr);

    hipError_t hipStatus = hipErrorInvalidDevicePointer;

    // Synchronize to ensure all work has finished.
    ihipGetTlsDefaultCtx()->locked_waitAllStreams(); // ignores non-blocking streams, this waits for all activity to finish.

    if (ptr) {
        hc::accelerator acc;
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
        if(status == AM_SUCCESS){
            if(amPointerInfo._hostPointer == NULL){
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

hipError_t hipHostFree(void* ptr)
{
    HIP_INIT_API(ptr);

    // Synchronize to ensure all work has finished.
    ihipGetTlsDefaultCtx()->locked_waitAllStreams(); // ignores non-blocking streams, this waits for all activity to finish.


    hipError_t hipStatus = hipErrorInvalidValue;
    if (ptr) {
        hc::accelerator acc;
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
        if(status == AM_SUCCESS){
            if(amPointerInfo._hostPointer == ptr){
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

hipError_t hipFreeHost(void* ptr)
{
    return hipHostFree(ptr);
}

hipError_t hipFreeArray(hipArray* array)
{
    HIP_INIT_API(array);

    hipError_t hipStatus = hipErrorInvalidDevicePointer;

    // Synchronize to ensure all work has finished.
    ihipGetTlsDefaultCtx()->locked_waitAllStreams(); // ignores non-blocking streams, this waits for all activity to finish.

    if(array->data) {
        hc::accelerator acc;
        hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
        am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, array->data);
        if(status == AM_SUCCESS){
            if(amPointerInfo._hostPointer == NULL){
                hc::am_free(array->data);
                hipStatus = hipSuccess;
            }
        }
    }

    return ihipLogStatus(hipStatus);
}

hipError_t hipMemGetAddressRange ( hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr )
{
    HIP_INIT_API ( pbase , psize , dptr );
    hipError_t hipStatus = hipSuccess;
    hc::accelerator acc;
    hc::AmPointerInfo amPointerInfo( NULL , NULL , 0 , acc , 0 , 0 );
    am_status_t status = hc::am_memtracker_getinfo( &amPointerInfo , dptr );
    if (status == AM_SUCCESS) {
        *pbase = amPointerInfo._devicePointer;
        *psize = amPointerInfo._sizeBytes;
    }
    else
        hipStatus = hipErrorInvalidDevicePointer;
    return hipStatus;
}


//TODO: IPC implementaiton:

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr){
    HIP_INIT_API ( handle, devPtr);
    hipError_t hipStatus = hipSuccess;
    // Get the size of allocated pointer
    size_t psize;
    hc::accelerator acc;
    hc::AmPointerInfo amPointerInfo( NULL , NULL , 0 , acc , 0 , 0 );
    am_status_t status = hc::am_memtracker_getinfo( &amPointerInfo , devPtr );
    if (status == AM_SUCCESS) {
        psize = (size_t)amPointerInfo._sizeBytes;
    }
    else
        hipStatus = hipErrorInvalidResourceHandle;

    // Save the size of the pointer to hipIpcMemHandle
    (*handle)->psize = psize;

    // Create HSA ipc memory
    hsa_status_t hsa_status =
        hsa_amd_ipc_memory_create(devPtr, psize, &(*handle)->ipc_handle);
    if(hsa_status!= HSA_STATUS_SUCCESS)
        hipStatus = hipErrorMemoryAllocation;

    return hipStatus;
}

hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags){
// HIP_INIT_API ( devPtr, handle.handle , flags);
    hipError_t hipStatus = hipSuccess;

    // Get the current device agent.
    hc::accelerator acc;
    hsa_agent_t *agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());
    if(!agent)
        return hipErrorInvalidResourceHandle;

    //Attach ipc memory
    hsa_status_t hsa_status =
        hsa_amd_ipc_memory_attach(&handle->ipc_handle, handle->psize, 1, agent, devPtr);
    if(hsa_status != HSA_STATUS_SUCCESS)
        hipStatus = hipErrorMapBufferObjectFailed;

    return hipStatus;
}

hipError_t hipIpcCloseMemHandle(void *devPtr){
    HIP_INIT_API ( devPtr );
    hipError_t hipStatus = hipSuccess;

    hsa_status_t hsa_status =
        hsa_amd_ipc_memory_detach(devPtr);
    if(hsa_status != HSA_STATUS_SUCCESS)
        return hipErrorInvalidResourceHandle;
    return hipStatus;
}

// hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle){
//     return hipSuccess;
// }
