/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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



// Internal HIP APIS:
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

// return 0 on success or -1 on error:
int sharePtr(void *ptr, ihipCtx_t *ctx, unsigned hipFlags)
{
    int ret = 0;

    auto device = ctx->getWriteableDevice();

    hc::am_memtracker_update(ptr, device->_deviceId, hipFlags);
    int peerCnt=0;
    {
        LockedAccessor_CtxCrit_t crit(ctx->criticalData());
        // the peerCnt always stores self so make sure the trace actually
        peerCnt = crit->peerCnt();
        tprintf(DB_MEM, "  allow access to %d other peer(s)\n", peerCnt-1);
        if (peerCnt > 1) {

            //printf ("peer self access\n");

            // TODOD - remove me:
            for (auto iter = crit->_peers.begin(); iter!=crit->_peers.end(); iter++) {
                tprintf (DB_MEM, "    allow access to peer: %s%s\n", (*iter)->toString().c_str(), (iter == crit->_peers.begin()) ? " (self)":"");
            };

            hsa_status_t s = hsa_amd_agents_allow_access(crit->peerCnt(), crit->peerAgents(), NULL, ptr);
            if (s != HSA_STATUS_SUCCESS) {
                ret = -1;
            }
        }
    }

    return ret;
}




// Allocate a new pointer with am_alloc and share with all valid peers.
// Returns null-ptr if a memory error occurs (either allocation or sharing)
void * allocAndSharePtr(const char *msg, size_t sizeBytes, ihipCtx_t *ctx, unsigned amFlags, unsigned hipFlags)
{

    void *ptr = nullptr;

    auto device = ctx->getWriteableDevice();

    ptr = hc::am_alloc(sizeBytes, device->_acc, amFlags);
    tprintf(DB_MEM, " alloc %s ptr:%p size:%zu on dev:%d\n",
            msg, ptr, sizeBytes, device->_deviceId);

    if (ptr != nullptr) {
        int r = sharePtr(ptr, ctx, hipFlags);
        if (r != 0) {
            ptr = nullptr;
        }
    }

    return ptr;
}


} // end namespace hip_internal

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Memory
//
//
//
//HIP uses several "app*" fields HC memory tracker to track state necessary for the HIP API.
//_appId : DeviceID.  For device mem, this is device where the memory is physically allocated.
//         For host or registered mem, this is the current device when the memory is allocated or registered.  This device will have a GPUVM mapping for the host mem.
//
//_appAllocationFlags : These are flags provided by the user when allocation is performed. They are returned to user in hipHostGetFlags and other APIs.
// TODO - add more info here when available.
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
    HIP_SET_DEVICE();
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
        *ptr = hip_internal::allocAndSharePtr("device_mem", sizeBytes, ctx,  0/*amFlags*/, 0/*hipFlags*/);

    } else {
        hip_status = hipErrorMemoryAllocation;
    }


    return ihipLogStatus(hip_status);
}


hipError_t hipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags)
{
    HIP_INIT_CMD_API(ptr, sizeBytes, flags);
    HIP_SET_DEVICE();
    hipError_t hip_status = hipSuccess;

    if (HIP_SYNC_HOST_ALLOC) {
        hipDeviceSynchronize();
    }

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
            unsigned amFlags = HIP_COHERENT_HOST_ALLOC ? amHostCoherent : amHostPinned;

            *ptr = hip_internal::allocAndSharePtr(HIP_COHERENT_HOST_ALLOC ? "finegrained_host":"pinned_host", 
                                                  sizeBytes, ctx, amFlags, flags);
            if(sizeBytes  && (*ptr == NULL)){
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
hipError_t hipMallocHost(void** ptr, size_t sizeBytes)
{
    return hipHostMalloc(ptr, sizeBytes, 0);
}


// Deprecated function:
hipError_t hipHostAlloc(void** ptr, size_t sizeBytes, unsigned int flags)
{
    return hipHostMalloc(ptr, sizeBytes, flags);
};


// width in bytes
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height)
{
    HIP_INIT_CMD_API(ptr, pitch, width, height);
    HIP_SET_DEVICE();
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
        *ptr = hip_internal::allocAndSharePtr("device_pitch", sizeBytes, ctx, am_flags, 0);

        if (sizeBytes && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
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
    HIP_INIT_CMD_API(array, desc, width, height, flags);
    HIP_SET_DEVICE();
    hipError_t  hip_status = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();

    *array = (hipArray*)malloc(sizeof(hipArray));
    array[0]->width = width;
    array[0]->height = height;

    array[0]->f = desc->f;

    void ** ptr = &array[0]->data;

    if (ctx) {
        const unsigned am_flags = 0;
        const size_t size = width*height;

        size_t allocSize = 0;
        switch(desc->f) {
            case hipChannelFormatKindSigned:
                allocSize = size * sizeof(int);
                break;
            case hipChannelFormatKindUnsigned:
                allocSize = size * sizeof(unsigned int);
                break;
            case hipChannelFormatKindFloat:
                allocSize = size * sizeof(float);
                break;
            case hipChannelFormatKindNone:
                allocSize = size * sizeof(size_t);
                break;
            default:
                hip_status = hipErrorUnknown;
                break;
        }
        *ptr = hip_internal::allocAndSharePtr("device_array", allocSize, ctx, am_flags, 0);
        if (size && (*ptr == NULL)) {
            hip_status = hipErrorMemoryAllocation;
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


// TODO - need to fix several issues here related to P2P access, host memory fallback.
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
        if (hostPtr == NULL) {
            return ihipLogStatus(hipErrorInvalidValue);
        }
        //TODO-test : multi-gpu access to registered host memory.
        if (ctx) {
            if(flags == hipHostRegisterDefault || flags == hipHostRegisterPortable || flags == hipHostRegisterMapped){
                auto device = ctx->getWriteableDevice();
                std::vector<hc::accelerator>vecAcc;
                for(int i=0;i<g_deviceCnt;i++){
                    vecAcc.push_back(ihipGetDevice(i)->_acc);
                }
                am_status = hc::am_memory_host_lock(device->_acc, hostPtr, sizeBytes, &vecAcc[0], vecAcc.size());
                hc::am_memtracker_update(hostPtr, device->_deviceId, flags);

                tprintf(DB_MEM, " %s registered ptr=%p and allowed access to %zu peers\n", __func__, hostPtr, vecAcc.size());
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
    HIP_INIT_CMD_API(symbolName, src, count, offset, kind);

    if(symbolName == nullptr)
    {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    auto ctx = ihipGetTlsDefaultCtx();

    hc::accelerator acc = ctx->getDevice()->_acc;

    void *dst = acc.get_symbol_address(symbolName);
    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbolName, dst);

    if(dst == nullptr)
    {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    if(kind == hipMemcpyHostToDevice || kind == hipMemcpyDeviceToHost || kind == hipMemcpyDeviceToDevice || kind == hipMemcpyHostToHost)
    {
      stream->lockedSymbolCopySync(acc, dst, (void*)src, count, offset, kind);
    //  acc.memcpy_symbol(dst, (void*)src, count+offset);
    } else {
      return ihipLogStatus(hipErrorInvalidValue);
    }

    return ihipLogStatus(hipSuccess);
}


hipError_t hipMemcpyFromSymbol(void* dst, const char* symbolName, size_t count, size_t offset, hipMemcpyKind kind)
{
    HIP_INIT_CMD_API(symbolName, dst, count, offset, kind);

    if(symbolName == nullptr)
    {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    auto ctx = ihipGetTlsDefaultCtx();

    hc::accelerator acc = ctx->getDevice()->_acc;

    void *src = acc.get_symbol_address(symbolName);
    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbolName, dst);

    if(dst == nullptr)
    {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    hipStream_t stream = ihipSyncAndResolveStream(hipStreamNull);

    if(kind == hipMemcpyHostToDevice || kind == hipMemcpyDeviceToHost || kind == hipMemcpyDeviceToDevice || kind == hipMemcpyHostToHost)
    {
      stream->lockedSymbolCopySync(acc, dst, (void*)src, count,  offset, kind);
    }
    else {
      return ihipLogStatus(hipErrorInvalidValue);
    }

    return ihipLogStatus(hipSuccess);
}


hipError_t hipMemcpyToSymbolAsync(const char* symbolName, const void *src, size_t count, size_t offset, hipMemcpyKind kind, hipStream_t stream)
{
    HIP_INIT_CMD_API(symbolName, src, count, offset, kind, stream);

    if(symbolName == nullptr)
    {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    hipError_t e = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();

    hc::accelerator acc = ctx->getDevice()->_acc;

    void *dst = acc.get_symbol_address(symbolName);
    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbolName, dst);

    if(dst == nullptr)
    {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    if (stream) {
        try {
          stream->lockedSymbolCopyAsync(acc, dst, (void*)src, count + offset, kind);
        }
        catch (ihipException ex) {
            e = ex._code;
        }
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}


hipError_t hipMemcpyFromSymbolAsync(void* dst, const char* symbolName, size_t count, size_t offset, hipMemcpyKind kind, hipStream_t stream)
{
    HIP_INIT_CMD_API(symbolName, dst, count, offset, kind, stream);

    if(symbolName == nullptr)
    {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    hipError_t e = hipSuccess;

    auto ctx = ihipGetTlsDefaultCtx();

    hc::accelerator acc = ctx->getDevice()->_acc;

    void *src = acc.get_symbol_address(symbolName);
    tprintf(DB_MEM, " symbol '%s' resolved to address:%p\n", symbolName, src);

    if(src == nullptr || dst == nullptr)
    {
        return ihipLogStatus(hipErrorInvalidSymbol);
    }

    if (stream) {
        try {
          stream->lockedSymbolCopyAsync(acc, dst, src, count + offset, kind);
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
    HIP_INIT_CMD_API(dst, src, sizeBytes, kind);

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
    HIP_INIT_CMD_API(dst, src, sizeBytes);

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
    HIP_INIT_CMD_API(dst, src, sizeBytes);

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
    HIP_INIT_CMD_API(dst, src, sizeBytes);

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
    HIP_INIT_CMD_API(dst, src, sizeBytes);

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





hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)
{
    HIP_INIT_CMD_API(dst, src, sizeBytes, kind, stream);

    return ihipLogStatus(hip_internal::memcpyAsync(dst, src, sizeBytes, kind, stream));

}


hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes, hipStream_t stream)
{
    HIP_INIT_CMD_API(dst, src, sizeBytes, stream);

    return ihipLogStatus(hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyHostToDevice, stream));
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)
{
    HIP_INIT_CMD_API(dst, src, sizeBytes, stream);

    return ihipLogStatus(hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, stream));
}

hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)
{
    HIP_INIT_CMD_API(dst, src, sizeBytes, stream);

    return ihipLogStatus(hip_internal::memcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToHost, stream));
}

// TODO - review and optimize
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
        size_t width, size_t height, hipMemcpyKind kind) {

    HIP_INIT_CMD_API(dst, dpitch, src, spitch, width, height, kind);

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

    HIP_INIT_CMD_API(dst, wOffset, hOffset, src, spitch, width, height, kind);

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

    HIP_INIT_CMD_API(dst, wOffset, hOffset, src, count, kind);

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
void
ihipMemsetKernel(hipStream_t stream,
    LockedAccessor_StreamCrit_t &crit,
    T * ptr, T val, size_t sizeBytes,
    hc::completion_future *cf)
{
    int wg = std::min((unsigned)8, stream->getDevice()->_computeUnits);
    const int threads_per_wg = 256;

    int threads = wg * threads_per_wg;
    if (threads > sizeBytes) {
        threads = ((sizeBytes + threads_per_wg - 1) / threads_per_wg) * threads_per_wg;
    }


    hc::extent<1> ext(threads);
    auto ext_tile = ext.tile(threads_per_wg);

    *cf =
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

}

// TODO-sync: function is async unless target is pinned host memory - then these are fully sync.
hipError_t hipMemsetAsync(void* dst, int  value, size_t sizeBytes, hipStream_t stream )
{
    HIP_INIT_CMD_API(dst, value, sizeBytes, stream);

    hipError_t e = hipSuccess;

    stream =  ihipSyncAndResolveStream(stream);

    if (stream) {
        auto crit = stream->lockopen_preKernelCommand();

        stream->ensureHaveQueue(crit);

        hc::completion_future cf ;

        if ((sizeBytes & 0x3) == 0) {
            // use a faster dword-per-workitem copy:
            try {
                value = value & 0xff;
                uint32_t value32 = (value << 24) | (value << 16) | (value << 8) | (value) ;
                ihipMemsetKernel<uint32_t> (stream, crit, static_cast<uint32_t*> (dst), value32, sizeBytes/sizeof(uint32_t), &cf);
            }
            catch (std::exception &ex) {
                e = hipErrorInvalidValue;
            }
        } else {
            // use a slow byte-per-workitem copy:
            try {
                ihipMemsetKernel<char> (stream, crit, static_cast<char*> (dst), value, sizeBytes, &cf);
            }
            catch (std::exception &ex) {
                e = hipErrorInvalidValue;
            }
        }

        stream->lockclose_postKernelCommand("hipMemsetAsync", &crit->_av);


        if (HIP_API_BLOCKING) {
            tprintf (DB_SYNC, "%s LAUNCH_BLOCKING wait for hipMemsetAsync.\n", ToString(stream).c_str());
            cf.wait();
        }
    } else {
        e = hipErrorInvalidValue;
    }


    return ihipLogStatus(e);
};

hipError_t hipMemset(void* dst, int  value, size_t sizeBytes )
{
    HIP_INIT_CMD_API(dst, value, sizeBytes);

    hipError_t e = hipSuccess;

    hipStream_t stream = hipStreamNull;
    // TODO - call an ihip memset so HIP_TRACE is correct.
    stream =  ihipSyncAndResolveStream(stream);

    if (stream) {
        auto crit = stream->lockopen_preKernelCommand();

        stream->ensureHaveQueue(crit);
        hc::completion_future cf ;

        if ((sizeBytes & 0x3) == 0) {
            // use a faster dword-per-workitem copy:
            try {
                value = value & 0xff;
                uint32_t value32 = (value << 24) | (value << 16) | (value << 8) | (value) ;
                ihipMemsetKernel<uint32_t> (stream, crit, static_cast<uint32_t*> (dst), value32, sizeBytes/sizeof(uint32_t), &cf);
            }
            catch (std::exception &ex) {
                e = hipErrorInvalidValue;
            }
        } else {
            // use a slow byte-per-workitem copy:
            try {
                ihipMemsetKernel<char> (stream, crit, static_cast<char*> (dst), value, sizeBytes, &cf);
            }
            catch (std::exception &ex) {
                e = hipErrorInvalidValue;
            }
        }
        // TODO - is hipMemset supposed to be async?
        cf.wait();

        stream->lockclose_postKernelCommand("hipMemset", &crit->_av);


        if (HIP_LAUNCH_BLOCKING) {
            tprintf (DB_SYNC, "'%s' LAUNCH_BLOCKING wait for memset in %s.\n", __func__, ToString(stream).c_str());
            cf.wait();
            tprintf (DB_SYNC, "'%s' LAUNCH_BLOCKING memset completed in %s.\n", __func__, ToString(stream).c_str());
        }
    } else {
        e = hipErrorInvalidValue;
    }

    return ihipLogStatus(e);
}

hipError_t hipMemsetD8(hipDeviceptr_t dst, unsigned char  value, size_t sizeBytes )
{
    HIP_INIT_CMD_API(dst, value, sizeBytes);

    hipError_t e = hipSuccess;

    hipStream_t stream = hipStreamNull;
    // TODO - call an ihip memset so HIP_TRACE is correct.
    stream =  ihipSyncAndResolveStream(stream);

    if (stream) {
        auto crit = stream->lockopen_preKernelCommand();

        stream->ensureHaveQueue(crit);
        hc::completion_future cf ;

        if ((sizeBytes & 0x3) == 0) {
            // use a faster dword-per-workitem copy:
            try {
                uint32_t value32 = (value << 24) | (value << 16) | (value << 8) | (value) ;
                ihipMemsetKernel<uint32_t> (stream, crit, static_cast<uint32_t*> (dst), value32, sizeBytes/sizeof(uint32_t), &cf);
            }
            catch (std::exception &ex) {
                e = hipErrorInvalidValue;
            }
        } else {
            // use a slow byte-per-workitem copy:
            try {
                ihipMemsetKernel<char> (stream, crit, static_cast<char*> (dst), value, sizeBytes, &cf);
            }
            catch (std::exception &ex) {
                e = hipErrorInvalidValue;
            }
        }
        cf.wait();

        stream->lockclose_postKernelCommand("hipMemsetD8", &crit->_av);


        if (HIP_LAUNCH_BLOCKING) {
            tprintf (DB_SYNC, "'%s' LAUNCH_BLOCKING wait for memset in %s.\n", __func__, ToString(stream).c_str());
            cf.wait();
            tprintf (DB_SYNC, "'%s' LAUNCH_BLOCKING memset completed in %s.\n", __func__, ToString(stream).c_str());
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

hipError_t hipMemPtrGetInfo(void *ptr, size_t *size)
{
  HIP_INIT_API(ptr, size);

  hipError_t e = hipSuccess;

  if(ptr != nullptr && size != nullptr){
    hc::accelerator acc;
    hc::AmPointerInfo amPointerInfo(NULL, NULL, 0, acc, 0, 0);
    am_status_t status = hc::am_memtracker_getinfo(&amPointerInfo, ptr);
    if(status == AM_SUCCESS){
      *size = amPointerInfo._sizeBytes;
    }else{
      e = hipErrorInvalidValue;
    }
  }else{
    e = hipErrorInvalidValue;
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


// Deprecated:
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
    return ihipLogStatus(hipStatus);
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
    ihipIpcMemHandle_t* iHandle = (ihipIpcMemHandle_t*) handle;
    // Save the size of the pointer to hipIpcMemHandle
    iHandle->psize = psize;

#if USE_IPC
    // Create HSA ipc memory
    hsa_status_t hsa_status =
        hsa_amd_ipc_memory_create(devPtr, psize, (hsa_amd_ipc_memory_t*) &(iHandle->ipc_handle));
    if(hsa_status!= HSA_STATUS_SUCCESS)
        hipStatus = hipErrorMemoryAllocation;
#else
    hipStatus = hipErrorRuntimeOther;
#endif

    return ihipLogStatus(hipStatus);
}

hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags){
    HIP_INIT_API ( devPtr, &handle , flags);
    hipError_t hipStatus = hipSuccess;

#if USE_IPC
    // Get the current device agent.
    hc::accelerator acc;
    hsa_agent_t *agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());
    if(!agent)
        return hipErrorInvalidResourceHandle;

    ihipIpcMemHandle_t* iHandle = (ihipIpcMemHandle_t*) &handle;
    //Attach ipc memory
    hsa_status_t hsa_status =
        hsa_amd_ipc_memory_attach((hsa_amd_ipc_memory_t*)&(iHandle->ipc_handle), iHandle->psize, 1, agent, devPtr);
    if(hsa_status != HSA_STATUS_SUCCESS)
        hipStatus = hipErrorMapBufferObjectFailed;
#else
    hipStatus = hipErrorRuntimeOther;
#endif
    return ihipLogStatus(hipStatus);
}

hipError_t hipIpcCloseMemHandle(void *devPtr){
    HIP_INIT_API ( devPtr );
    hipError_t hipStatus = hipSuccess;

#if USE_IPC
    hsa_status_t hsa_status =
        hsa_amd_ipc_memory_detach(devPtr);
    if(hsa_status != HSA_STATUS_SUCCESS)
        return hipErrorInvalidResourceHandle;
#else
    hipStatus = hipErrorRuntimeOther;
#endif
    return ihipLogStatus(hipStatus);
}

// hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle){
//     return hipSuccess;
// }
