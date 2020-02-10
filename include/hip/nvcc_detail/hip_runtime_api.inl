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
#include "hip/hip_runtime_api.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
#ifdef NVCC_INLINE_DISABLED
#define __dparm(x)
#else
#define __dparm(x) = x
#endif
#else
#define __dparm(x)
#endif

HIP_NVCC_INLINE hipError_t hipCUDAErrorTohipError(cudaError_t cuError) {
    switch (cuError) {
        case cudaSuccess:
            return hipSuccess;
        case cudaErrorProfilerDisabled:
            return hipErrorProfilerDisabled;
        case cudaErrorProfilerNotInitialized:
            return hipErrorProfilerNotInitialized;
        case cudaErrorProfilerAlreadyStarted:
            return hipErrorProfilerAlreadyStarted;
        case cudaErrorProfilerAlreadyStopped:
            return hipErrorProfilerAlreadyStopped;
        case cudaErrorInsufficientDriver:
            return hipErrorInsufficientDriver;
        case cudaErrorUnsupportedLimit:
            return hipErrorUnsupportedLimit;
        case cudaErrorPeerAccessUnsupported:
            return hipErrorPeerAccessUnsupported;
        case cudaErrorInvalidGraphicsContext:
            return hipErrorInvalidGraphicsContext;
        case cudaErrorSharedObjectSymbolNotFound:
            return hipErrorSharedObjectSymbolNotFound;
        case cudaErrorSharedObjectInitFailed:
            return hipErrorSharedObjectInitFailed;
        case cudaErrorOperatingSystem:
            return hipErrorOperatingSystem;
        case cudaErrorSetOnActiveProcess:
            return hipErrorSetOnActiveProcess;
        case cudaErrorIllegalAddress:
            return hipErrorIllegalAddress;
        case cudaErrorInvalidSymbol:
            return hipErrorInvalidSymbol;
        case cudaErrorMissingConfiguration:
            return hipErrorMissingConfiguration;
        case cudaErrorMemoryAllocation:
            return hipErrorOutOfMemory;
        case cudaErrorInitializationError:
            return hipErrorNotInitialized;
        case cudaErrorLaunchFailure:
            return hipErrorLaunchFailure;
        case cudaErrorPriorLaunchFailure:
            return hipErrorPriorLaunchFailure;
        case cudaErrorLaunchOutOfResources:
            return hipErrorLaunchOutOfResources;
        case cudaErrorInvalidDeviceFunction:
            return hipErrorInvalidDeviceFunction;
        case cudaErrorInvalidConfiguration:
            return hipErrorInvalidConfiguration;
        case cudaErrorInvalidDevice:
            return hipErrorInvalidDevice;
        case cudaErrorInvalidValue:
            return hipErrorInvalidValue;
        case cudaErrorInvalidDevicePointer:
            return hipErrorInvalidDevicePointer;
        case cudaErrorInvalidMemcpyDirection:
            return hipErrorInvalidMemcpyDirection;
        case cudaErrorInvalidResourceHandle:
            return hipErrorInvalidHandle;
        case cudaErrorNotReady:
            return hipErrorNotReady;
        case cudaErrorNoDevice:
            return hipErrorNoDevice;
        case cudaErrorPeerAccessAlreadyEnabled:
            return hipErrorPeerAccessAlreadyEnabled;
        case cudaErrorPeerAccessNotEnabled:
            return hipErrorPeerAccessNotEnabled;
        case cudaErrorHostMemoryAlreadyRegistered:
            return hipErrorHostMemoryAlreadyRegistered;
        case cudaErrorHostMemoryNotRegistered:
            return hipErrorHostMemoryNotRegistered;
        case cudaErrorMapBufferObjectFailed:
            return hipErrorMapFailed;
        case cudaErrorAssert:
            return hipErrorAssert;
        case cudaErrorNotSupported:
            return hipErrorNotSupported;
        case cudaErrorCudartUnloading:
            return hipErrorDeinitialized;
        case cudaErrorInvalidKernelImage:
            return hipErrorInvalidImage;
        case cudaErrorUnmapBufferObjectFailed:
            return hipErrorUnmapFailed;
        case cudaErrorNoKernelImageForDevice:
            return hipErrorNoBinaryForGpu;
        case cudaErrorECCUncorrectable:
            return hipErrorECCNotCorrectable;
        case cudaErrorDeviceAlreadyInUse:
            return hipErrorContextAlreadyInUse;
        case cudaErrorInvalidPtx:
            return hipErrorInvalidKernelFile;
        case cudaErrorLaunchTimeout:
            return hipErrorLaunchTimeOut;
#if CUDA_VERSION >= 10010
        case cudaErrorInvalidSource:
            return hipErrorInvalidSource;
        case cudaErrorFileNotFound:
            return hipErrorFileNotFound;
        case cudaErrorSymbolNotFound:
            return hipErrorNotFound;
        case cudaErrorArrayIsMapped:
            return hipErrorArrayIsMapped;
        case cudaErrorNotMappedAsPointer:
            return hipErrorNotMappedAsPointer;
        case cudaErrorNotMappedAsArray:
            return hipErrorNotMappedAsArray;
        case cudaErrorNotMapped:
            return hipErrorNotMapped;
        case cudaErrorAlreadyAcquired:
            return hipErrorAlreadyAcquired;
        case cudaErrorAlreadyMapped:
            return hipErrorAlreadyMapped;
#endif
#if CUDA_VERSION >= 10020
        case cudaErrorDeviceUninitialized:
            return hipErrorInvalidContext;
#endif
        case cudaErrorUnknown:
        default:
            return hipErrorUnknown;  // Note - translated error.
    }
}

HIP_NVCC_INLINE hipError_t hipCUResultTohipError(CUresult cuError) {  // TODO Populate further
    switch (cuError) {
        case CUDA_SUCCESS:
            return hipSuccess;
        case CUDA_ERROR_OUT_OF_MEMORY:
            return hipErrorOutOfMemory;
        case CUDA_ERROR_INVALID_VALUE:
            return hipErrorInvalidValue;
        case CUDA_ERROR_INVALID_DEVICE:
            return hipErrorInvalidDevice;
        case CUDA_ERROR_DEINITIALIZED:
            return hipErrorDeinitialized;
        case CUDA_ERROR_NO_DEVICE:
            return hipErrorNoDevice;
        case CUDA_ERROR_INVALID_CONTEXT:
            return hipErrorInvalidContext;
        case CUDA_ERROR_NOT_INITIALIZED:
            return hipErrorNotInitialized;
        case CUDA_ERROR_INVALID_HANDLE:
            return hipErrorInvalidHandle;
        case CUDA_ERROR_MAP_FAILED:
            return hipErrorMapFailed;
        case CUDA_ERROR_PROFILER_DISABLED:
            return hipErrorProfilerDisabled;
        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
            return hipErrorProfilerNotInitialized;
        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
            return hipErrorProfilerAlreadyStarted;
        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
            return hipErrorProfilerAlreadyStopped;
        case CUDA_ERROR_INVALID_IMAGE:
            return hipErrorInvalidImage;
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
            return hipErrorContextAlreadyCurrent;
        case CUDA_ERROR_UNMAP_FAILED:
            return hipErrorUnmapFailed;
        case CUDA_ERROR_ARRAY_IS_MAPPED:
            return hipErrorArrayIsMapped;
        case CUDA_ERROR_ALREADY_MAPPED:
            return hipErrorAlreadyMapped;
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            return hipErrorNoBinaryForGpu;
        case CUDA_ERROR_ALREADY_ACQUIRED:
            return hipErrorAlreadyAcquired;
        case CUDA_ERROR_NOT_MAPPED:
            return hipErrorNotMapped;
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
            return hipErrorNotMappedAsArray;
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
            return hipErrorNotMappedAsPointer;
        case CUDA_ERROR_ECC_UNCORRECTABLE:
            return hipErrorECCNotCorrectable;
        case CUDA_ERROR_UNSUPPORTED_LIMIT:
            return hipErrorUnsupportedLimit;
        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
            return hipErrorContextAlreadyInUse;
        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
            return hipErrorPeerAccessUnsupported;
        case CUDA_ERROR_INVALID_PTX:
            return hipErrorInvalidKernelFile;
        case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
            return hipErrorInvalidGraphicsContext;
        case CUDA_ERROR_INVALID_SOURCE:
            return hipErrorInvalidSource;
        case CUDA_ERROR_FILE_NOT_FOUND:
            return hipErrorFileNotFound;
        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return hipErrorSharedObjectSymbolNotFound;
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return hipErrorSharedObjectInitFailed;
        case CUDA_ERROR_OPERATING_SYSTEM:
            return hipErrorOperatingSystem;
        case CUDA_ERROR_NOT_FOUND:
            return hipErrorNotFound;
        case CUDA_ERROR_NOT_READY:
            return hipErrorNotReady;
        case CUDA_ERROR_ILLEGAL_ADDRESS:
            return hipErrorIllegalAddress;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return hipErrorLaunchOutOfResources;
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            return hipErrorLaunchTimeOut;
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            return hipErrorPeerAccessAlreadyEnabled;
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            return hipErrorPeerAccessNotEnabled;
        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            return hipErrorSetOnActiveProcess;
        case CUDA_ERROR_ASSERT:
            return hipErrorAssert;
        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            return hipErrorHostMemoryAlreadyRegistered;
        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            return hipErrorHostMemoryNotRegistered;
        case CUDA_ERROR_LAUNCH_FAILED:
            return hipErrorLaunchFailure;
        case CUDA_ERROR_NOT_SUPPORTED:
            return hipErrorNotSupported;
        case CUDA_ERROR_UNKNOWN:
        default:
            return hipErrorUnknown;  // Note - translated error.
    }
}

// TODO   match the error enum names of hip and cuda
HIP_NVCC_INLINE cudaError_t hipErrorToCudaError(hipError_t hError) {
    switch (hError) {
        case hipSuccess:
            return cudaSuccess;
        case hipErrorOutOfMemory:
            return cudaErrorMemoryAllocation;
        case hipErrorProfilerDisabled:
            return cudaErrorProfilerDisabled;
        case hipErrorProfilerNotInitialized:
            return cudaErrorProfilerNotInitialized;
        case hipErrorProfilerAlreadyStarted:
            return cudaErrorProfilerAlreadyStarted;
        case hipErrorProfilerAlreadyStopped:
            return cudaErrorProfilerAlreadyStopped;
        case hipErrorInvalidConfiguration:
            return cudaErrorInvalidConfiguration;
        case hipErrorLaunchOutOfResources:
            return cudaErrorLaunchOutOfResources;
        case hipErrorInvalidValue:
            return cudaErrorInvalidValue;
        case hipErrorInvalidHandle:
            return cudaErrorInvalidResourceHandle;
        case hipErrorInvalidDevice:
            return cudaErrorInvalidDevice;
        case hipErrorInvalidMemcpyDirection:
            return cudaErrorInvalidMemcpyDirection;
        case hipErrorInvalidDevicePointer:
            return cudaErrorInvalidDevicePointer;
        case hipErrorNotInitialized:
            return cudaErrorInitializationError;
        case hipErrorNoDevice:
            return cudaErrorNoDevice;
        case hipErrorNotReady:
            return cudaErrorNotReady;
        case hipErrorPeerAccessNotEnabled:
            return cudaErrorPeerAccessNotEnabled;
        case hipErrorPeerAccessAlreadyEnabled:
            return cudaErrorPeerAccessAlreadyEnabled;
        case hipErrorHostMemoryAlreadyRegistered:
            return cudaErrorHostMemoryAlreadyRegistered;
        case hipErrorHostMemoryNotRegistered:
            return cudaErrorHostMemoryNotRegistered;
        case hipErrorDeinitialized:
            return cudaErrorCudartUnloading;
        case hipErrorInvalidSymbol:
            return cudaErrorInvalidSymbol;
        case hipErrorInsufficientDriver:
            return cudaErrorInsufficientDriver;
        case hipErrorMissingConfiguration:
            return cudaErrorMissingConfiguration;
        case hipErrorPriorLaunchFailure:
            return cudaErrorPriorLaunchFailure;
        case hipErrorInvalidDeviceFunction:
            return cudaErrorInvalidDeviceFunction;
        case hipErrorInvalidImage:
            return cudaErrorInvalidKernelImage;
        case hipErrorInvalidContext:
#if CUDA_VERSION >= 10020
            return cudaErrorDeviceUninitialized;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorMapFailed:
            return cudaErrorMapBufferObjectFailed;
        case hipErrorUnmapFailed:
            return cudaErrorUnmapBufferObjectFailed;
        case hipErrorArrayIsMapped:
#if CUDA_VERSION >= 10010
            return cudaErrorArrayIsMapped;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorAlreadyMapped:
#if CUDA_VERSION >= 10010
            return cudaErrorAlreadyMapped;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorNoBinaryForGpu:
            return cudaErrorNoKernelImageForDevice;
        case hipErrorAlreadyAcquired:
#if CUDA_VERSION >= 10010
            return cudaErrorAlreadyAcquired;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorNotMapped:
#if CUDA_VERSION >= 10010
            return cudaErrorNotMapped;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorNotMappedAsArray:
#if CUDA_VERSION >= 10010
            return cudaErrorNotMappedAsArray;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorNotMappedAsPointer:
#if CUDA_VERSION >= 10010
            return cudaErrorNotMappedAsPointer;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorECCNotCorrectable:
            return cudaErrorECCUncorrectable;
        case hipErrorUnsupportedLimit:
            return cudaErrorUnsupportedLimit;
        case hipErrorContextAlreadyInUse:
            return cudaErrorDeviceAlreadyInUse;
        case hipErrorPeerAccessUnsupported:
            return cudaErrorPeerAccessUnsupported;
        case hipErrorInvalidKernelFile:
            return cudaErrorInvalidPtx;
        case hipErrorInvalidGraphicsContext:
            return cudaErrorInvalidGraphicsContext;
        case hipErrorInvalidSource:
#if CUDA_VERSION >= 10010
            return cudaErrorInvalidSource;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorFileNotFound:
#if CUDA_VERSION >= 10010
            return cudaErrorFileNotFound;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorSharedObjectSymbolNotFound:
            return cudaErrorSharedObjectSymbolNotFound;
        case hipErrorSharedObjectInitFailed:
            return cudaErrorSharedObjectInitFailed;
        case hipErrorOperatingSystem:
            return cudaErrorOperatingSystem;
        case hipErrorNotFound:
#if CUDA_VERSION >= 10010
            return cudaErrorSymbolNotFound;
#else
            return cudaErrorUnknown;
#endif
        case hipErrorIllegalAddress:
            return cudaErrorIllegalAddress;
        case hipErrorLaunchTimeOut:
            return cudaErrorLaunchTimeout;
        case hipErrorSetOnActiveProcess:
            return cudaErrorSetOnActiveProcess;
        case hipErrorLaunchFailure:
            return cudaErrorLaunchFailure;
        case hipErrorNotSupported:
            return cudaErrorNotSupported;
        // HSA: does not exist in CUDA
        case hipErrorRuntimeMemory:
        // HSA: does not exist in CUDA
        case hipErrorRuntimeOther:
        case hipErrorUnknown:
        case hipErrorTbd:
        default:
            return cudaErrorUnknown;  // Note - translated error.
    }
}

HIP_NVCC_INLINE enum cudaMemcpyKind hipMemcpyKindToCudaMemcpyKind(hipMemcpyKind kind) {
    switch (kind) {
        case hipMemcpyHostToHost:
            return cudaMemcpyHostToHost;
        case hipMemcpyHostToDevice:
            return cudaMemcpyHostToDevice;
        case hipMemcpyDeviceToHost:
            return cudaMemcpyDeviceToHost;
        case hipMemcpyDeviceToDevice:
            return cudaMemcpyDeviceToDevice;
        default:
            return cudaMemcpyDefault;
    }
}

HIP_NVCC_INLINE enum cudaTextureAddressMode hipTextureAddressModeToCudaTextureAddressMode(
    hipTextureAddressMode kind) {
    switch (kind) {
        case hipAddressModeWrap:
            return cudaAddressModeWrap;
        case hipAddressModeClamp:
            return cudaAddressModeClamp;
        case hipAddressModeMirror:
            return cudaAddressModeMirror;
        case hipAddressModeBorder:
            return cudaAddressModeBorder;
        default:
            return cudaAddressModeWrap;
    }
}

HIP_NVCC_INLINE enum cudaTextureFilterMode hipTextureFilterModeToCudaTextureFilterMode(
    hipTextureFilterMode kind) {
    switch (kind) {
        case hipFilterModePoint:
            return cudaFilterModePoint;
        case hipFilterModeLinear:
            return cudaFilterModeLinear;
        default:
            return cudaFilterModePoint;
    }
}

HIP_NVCC_INLINE enum cudaTextureReadMode hipTextureReadModeToCudaTextureReadMode(
    hipTextureReadMode kind) {
    switch (kind) {
        case hipReadModeElementType:
            return cudaReadModeElementType;
        case hipReadModeNormalizedFloat:
            return cudaReadModeNormalizedFloat;
        default:
            return cudaReadModeElementType;
    }
}

HIP_NVCC_INLINE enum cudaChannelFormatKind hipChannelFormatKindToCudaChannelFormatKind(
    hipChannelFormatKind kind) {
    switch (kind) {
        case hipChannelFormatKindSigned:
            return cudaChannelFormatKindSigned;
        case hipChannelFormatKindUnsigned:
            return cudaChannelFormatKindUnsigned;
        case hipChannelFormatKindFloat:
            return cudaChannelFormatKindFloat;
        case hipChannelFormatKindNone:
            return cudaChannelFormatKindNone;
        default:
            return cudaChannelFormatKindNone;
    }
}

/**
 * Stream CallBack struct
 */
#define HIPRT_CB CUDART_CB
typedef void(HIPRT_CB* hipStreamCallback_t)(hipStream_t stream, hipError_t status, void* userData);
HIP_NVCC_INLINE hipError_t hipInit(unsigned int flags) {
    return hipCUResultTohipError(cuInit(flags));
}

HIP_NVCC_INLINE hipError_t hipDeviceReset() { return hipCUDAErrorTohipError(cudaDeviceReset()); }

HIP_NVCC_INLINE hipError_t hipGetLastError() { return hipCUDAErrorTohipError(cudaGetLastError()); }

HIP_NVCC_INLINE hipError_t hipPeekAtLastError() {
    return hipCUDAErrorTohipError(cudaPeekAtLastError());
}

HIP_NVCC_INLINE hipError_t hipMalloc(void** ptr, size_t size) {
    return hipCUDAErrorTohipError(cudaMalloc(ptr, size));
}

HIP_NVCC_INLINE hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
    return hipCUDAErrorTohipError(cudaMallocPitch(ptr, pitch, width, height));
}

HIP_NVCC_INLINE hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch,
                                            size_t widthInBytes, size_t height,
                                            unsigned int elementSizeBytes) {
    return hipCUResultTohipError(
        cuMemAllocPitch(dptr, pitch, widthInBytes, height, elementSizeBytes));
}

HIP_NVCC_INLINE hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
    return hipCUDAErrorTohipError(cudaMalloc3D(pitchedDevPtr, extent));
}

HIP_NVCC_INLINE hipError_t hipFree(void* ptr) { return hipCUDAErrorTohipError(cudaFree(ptr)); }

HIP_NVCC_INLINE hipError_t hipMallocHost(void** ptr, size_t size) {
    return hipCUDAErrorTohipError(cudaMallocHost(ptr, size));
}

HIP_NVCC_INLINE hipError_t hipMemAllocHost(void** ptr, size_t size) {
    return hipCUResultTohipError(cuMemAllocHost(ptr, size));
}

HIP_NVCC_INLINE hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaHostAlloc(ptr, size, flags));
}

HIP_NVCC_INLINE hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaHostAlloc(ptr, size, flags));
}

HIP_NVCC_INLINE hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaMallocManaged(ptr, size, flags));
}

HIP_NVCC_INLINE hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc,
                                          size_t width, size_t height,
                                          unsigned int flags __dparm(hipArrayDefault)) {
    return hipCUDAErrorTohipError(cudaMallocArray(array, desc, width, height, flags));
}

HIP_NVCC_INLINE hipError_t hipMalloc3DArray(hipArray** array, const hipChannelFormatDesc* desc,
                                            hipExtent extent, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaMalloc3DArray(array, desc, extent, flags));
}

HIP_NVCC_INLINE hipError_t hipFreeArray(hipArray* array) {
    return hipCUDAErrorTohipError(cudaFreeArray(array));
}

HIP_NVCC_INLINE hipError_t hipHostGetDevicePointer(void** devPtr, void* hostPtr,
                                                   unsigned int flags) {
    return hipCUDAErrorTohipError(cudaHostGetDevicePointer(devPtr, hostPtr, flags));
}

HIP_NVCC_INLINE hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
    return hipCUDAErrorTohipError(cudaHostGetFlags(flagsPtr, hostPtr));
}

HIP_NVCC_INLINE hipError_t hipHostRegister(void* ptr, size_t size, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaHostRegister(ptr, size, flags));
}

HIP_NVCC_INLINE hipError_t hipHostUnregister(void* ptr) {
    return hipCUDAErrorTohipError(cudaHostUnregister(ptr));
}

HIP_NVCC_INLINE hipError_t hipFreeHost(void* ptr) {
    return hipCUDAErrorTohipError(cudaFreeHost(ptr));
}

HIP_NVCC_INLINE hipError_t hipHostFree(void* ptr) {
    return hipCUDAErrorTohipError(cudaFreeHost(ptr));
}

HIP_NVCC_INLINE hipError_t hipSetDevice(int device) {
    return hipCUDAErrorTohipError(cudaSetDevice(device));
}

HIP_NVCC_INLINE hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop) {
    struct cudaDeviceProp cdprop;
    memset(&cdprop, 0x0, sizeof(struct cudaDeviceProp));
    cdprop.major = prop->major;
    cdprop.minor = prop->minor;
    cdprop.totalGlobalMem = prop->totalGlobalMem;
    cdprop.sharedMemPerBlock = prop->sharedMemPerBlock;
    cdprop.regsPerBlock = prop->regsPerBlock;
    cdprop.warpSize = prop->warpSize;
    cdprop.maxThreadsPerBlock = prop->maxThreadsPerBlock;
    cdprop.clockRate = prop->clockRate;
    cdprop.totalConstMem = prop->totalConstMem;
    cdprop.multiProcessorCount = prop->multiProcessorCount;
    cdprop.l2CacheSize = prop->l2CacheSize;
    cdprop.maxThreadsPerMultiProcessor = prop->maxThreadsPerMultiProcessor;
    cdprop.computeMode = prop->computeMode;
    cdprop.canMapHostMemory = prop->canMapHostMemory;
    cdprop.memoryClockRate = prop->memoryClockRate;
    cdprop.memoryBusWidth = prop->memoryBusWidth;
    return hipCUDAErrorTohipError(cudaChooseDevice(device, &cdprop));
}

HIP_NVCC_INLINE hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t size) {
    return hipCUResultTohipError(cuMemcpyHtoD(dst, src, size));
}

HIP_NVCC_INLINE hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t size) {
    return hipCUResultTohipError(cuMemcpyDtoH(dst, src, size));
}

HIP_NVCC_INLINE hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size) {
    return hipCUResultTohipError(cuMemcpyDtoD(dst, src, size));
}

HIP_NVCC_INLINE hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t size,
                                              hipStream_t stream) {
    return hipCUResultTohipError(cuMemcpyHtoDAsync(dst, src, size, stream));
}

HIP_NVCC_INLINE hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t size,
                                              hipStream_t stream) {
    return hipCUResultTohipError(cuMemcpyDtoHAsync(dst, src, size, stream));
}

HIP_NVCC_INLINE hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size,
                                              hipStream_t stream) {
    return hipCUResultTohipError(cuMemcpyDtoDAsync(dst, src, size, stream));
}

HIP_NVCC_INLINE hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                                     hipMemcpyKind copyKind) {
    return hipCUDAErrorTohipError(
        cudaMemcpy(dst, src, sizeBytes, hipMemcpyKindToCudaMemcpyKind(copyKind)));
}

HIP_NVCC_INLINE hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes,
                                               hipMemcpyKind copyKind, hipStream_t stream) {
    cudaError_t error =
        cudaMemcpyAsync(dst, src, sizeBytes, hipMemcpyKindToCudaMemcpyKind(copyKind), stream);

    if (error != cudaSuccess) return hipCUDAErrorTohipError(error);

    return hipCUDAErrorTohipError(cudaStreamSynchronize(stream));
}

HIP_NVCC_INLINE hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                                          hipMemcpyKind copyKind, hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(
        cudaMemcpyAsync(dst, src, sizeBytes, hipMemcpyKindToCudaMemcpyKind(copyKind), stream));
}

HIP_NVCC_INLINE hipError_t
hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes, size_t offset __dparm(0),
                  hipMemcpyKind copyType __dparm(hipMemcpyHostToDevice)) {
    return hipCUDAErrorTohipError(cudaMemcpyToSymbol(symbol, src, sizeBytes, offset,
                                                     hipMemcpyKindToCudaMemcpyKind(copyType)));
}

HIP_NVCC_INLINE hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src,
                                                  size_t sizeBytes, size_t offset,
                                                  hipMemcpyKind copyType,
                                                  hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemcpyToSymbolAsync(
        symbol, src, sizeBytes, offset, hipMemcpyKindToCudaMemcpyKind(copyType), stream));
}

HIP_NVCC_INLINE hipError_t hipMemcpyFromSymbol(void* dst, const void* symbolName, size_t sizeBytes,
                                               size_t offset __dparm(0),
                                               hipMemcpyKind kind __dparm(hipMemcpyDeviceToHost)) {
    return hipCUDAErrorTohipError(cudaMemcpyFromSymbol(dst, symbolName, sizeBytes, offset,
                                                       hipMemcpyKindToCudaMemcpyKind(kind)));
}

HIP_NVCC_INLINE hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbolName,
                                                    size_t sizeBytes, size_t offset,
                                                    hipMemcpyKind kind,
                                                    hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemcpyFromSymbolAsync(
        dst, symbolName, sizeBytes, offset, hipMemcpyKindToCudaMemcpyKind(kind), stream));
}

HIP_NVCC_INLINE hipError_t hipGetSymbolAddress(void** devPtr, const void* symbolName) {
    return hipCUDAErrorTohipError(cudaGetSymbolAddress(devPtr, symbolName));
}

HIP_NVCC_INLINE hipError_t hipGetSymbolSize(size_t* size, const void* symbolName) {
    return hipCUDAErrorTohipError(cudaGetSymbolSize(size, symbolName));
}

HIP_NVCC_INLINE hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                                       size_t width, size_t height, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(
        cudaMemcpy2D(dst, dpitch, src, spitch, width, height, hipMemcpyKindToCudaMemcpyKind(kind)));
}

HIP_NVCC_INLINE hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
    return hipCUResultTohipError(cuMemcpy2D(pCopy));
}

HIP_NVCC_INLINE hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy,
                                                 hipStream_t stream __dparm(0)) {
    return hipCUResultTohipError(cuMemcpy2DAsync(pCopy, stream));
}

HIP_NVCC_INLINE hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p) {
    return hipCUDAErrorTohipError(cudaMemcpy3D(p));
}

HIP_NVCC_INLINE hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaMemcpy3DAsync(p, stream));
}

HIP_NVCC_INLINE hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
                                            size_t spitch, size_t width, size_t height,
                                            hipMemcpyKind kind, hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                                    hipMemcpyKindToCudaMemcpyKind(kind), stream));
}

HIP_NVCC_INLINE hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset,
                                              const void* src, size_t spitch, size_t width,
                                              size_t height, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width,
                                                      height, hipMemcpyKindToCudaMemcpyKind(kind)));
}

HIP_NVCC_INLINE hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset,
                                            const void* src, size_t count, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(
        cudaMemcpyToArray(dst, wOffset, hOffset, src, count, hipMemcpyKindToCudaMemcpyKind(kind)));
}

HIP_NVCC_INLINE hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset,
                                              size_t hOffset, size_t count, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaMemcpyFromArray(dst, srcArray, wOffset, hOffset, count,
                                                      hipMemcpyKindToCudaMemcpyKind(kind)));
}

HIP_NVCC_INLINE hipError_t hipMemcpyAtoH(void* dst, hipArray* srcArray, size_t srcOffset,
                                         size_t count) {
    return hipCUResultTohipError(cuMemcpyAtoH(dst, (CUarray)srcArray, srcOffset, count));
}

HIP_NVCC_INLINE hipError_t hipMemcpyHtoA(hipArray* dstArray, size_t dstOffset, const void* srcHost,
                                         size_t count) {
    return hipCUResultTohipError(cuMemcpyHtoA((CUarray)dstArray, dstOffset, srcHost, count));
}

HIP_NVCC_INLINE hipError_t hipDeviceSynchronize() {
    return hipCUDAErrorTohipError(cudaDeviceSynchronize());
}

HIP_NVCC_INLINE hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* pCacheConfig) {
    return hipCUDAErrorTohipError(cudaDeviceGetCacheConfig(pCacheConfig));
}

HIP_NVCC_INLINE hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
    return hipCUDAErrorTohipError(cudaDeviceSetCacheConfig(cacheConfig));
}

HIP_NVCC_INLINE const char* hipGetErrorString(hipError_t error) {
    return cudaGetErrorString(hipErrorToCudaError(error));
}

HIP_NVCC_INLINE const char* hipGetErrorName(hipError_t error) {
    return cudaGetErrorName(hipErrorToCudaError(error));
}

HIP_NVCC_INLINE hipError_t hipGetDeviceCount(int* count) {
    return hipCUDAErrorTohipError(cudaGetDeviceCount(count));
}

HIP_NVCC_INLINE hipError_t hipGetDevice(int* device) {
    return hipCUDAErrorTohipError(cudaGetDevice(device));
}

HIP_NVCC_INLINE hipError_t hipIpcCloseMemHandle(void* devPtr) {
    return hipCUDAErrorTohipError(cudaIpcCloseMemHandle(devPtr));
}

HIP_NVCC_INLINE hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaIpcGetEventHandle(handle, event));
}

HIP_NVCC_INLINE hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
    return hipCUDAErrorTohipError(cudaIpcGetMemHandle(handle, devPtr));
}

HIP_NVCC_INLINE hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
    return hipCUDAErrorTohipError(cudaIpcOpenEventHandle(event, handle));
}

HIP_NVCC_INLINE hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle,
                                               unsigned int flags) {
    return hipCUDAErrorTohipError(cudaIpcOpenMemHandle(devPtr, handle, flags));
}

HIP_NVCC_INLINE hipError_t hipMemset(void* devPtr, int value, size_t count) {
    return hipCUDAErrorTohipError(cudaMemset(devPtr, value, count));
}

HIP_NVCC_INLINE hipError_t hipMemsetD32(hipDeviceptr_t devPtr, int value, size_t count) {
    return hipCUResultTohipError(cuMemsetD32(devPtr, value, count));
}

HIP_NVCC_INLINE hipError_t hipMemsetAsync(void* devPtr, int value, size_t count,
                                          hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemsetAsync(devPtr, value, count, stream));
}

HIP_NVCC_INLINE hipError_t hipMemsetD32Async(hipDeviceptr_t devPtr, int value, size_t count,
                                             hipStream_t stream __dparm(0)) {
    return hipCUResultTohipError(cuMemsetD32Async(devPtr, value, count, stream));
}

HIP_NVCC_INLINE hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t sizeBytes) {
    return hipCUResultTohipError(cuMemsetD8(dest, value, sizeBytes));
}

HIP_NVCC_INLINE hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value,
                                            size_t sizeBytes, hipStream_t stream __dparm(0)) {
    return hipCUResultTohipError(cuMemsetD8Async(dest, value, sizeBytes, stream));
}

HIP_NVCC_INLINE hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value,
                                        size_t sizeBytes) {
    return hipCUResultTohipError(cuMemsetD16(dest, value, sizeBytes));
}

HIP_NVCC_INLINE hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value,
                                             size_t sizeBytes, hipStream_t stream __dparm(0)) {
    return hipCUResultTohipError(cuMemsetD16Async(dest, value, sizeBytes, stream));
}

HIP_NVCC_INLINE hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width,
                                       size_t height) {
    return hipCUDAErrorTohipError(cudaMemset2D(dst, pitch, value, width, height));
}

HIP_NVCC_INLINE hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width,
                                            size_t height, hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemset2DAsync(dst, pitch, value, width, height, stream));
}

HIP_NVCC_INLINE hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
    return hipCUDAErrorTohipError(cudaMemset3D(pitchedDevPtr, value, extent));
}

HIP_NVCC_INLINE hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value,
                                            hipExtent extent, hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemset3DAsync(pitchedDevPtr, value, extent, stream));
}

HIP_NVCC_INLINE hipError_t hipGetDeviceProperties(hipDeviceProp_t* p_prop, int device) {
    struct cudaDeviceProp cdprop;
    cudaError_t cerror;
    cerror = cudaGetDeviceProperties(&cdprop, device);

    strncpy(p_prop->name, cdprop.name, 256);
    p_prop->totalGlobalMem = cdprop.totalGlobalMem;
    p_prop->sharedMemPerBlock = cdprop.sharedMemPerBlock;
    p_prop->regsPerBlock = cdprop.regsPerBlock;
    p_prop->warpSize = cdprop.warpSize;
    p_prop->maxThreadsPerBlock = cdprop.maxThreadsPerBlock;
    for (int i = 0; i < 3; i++) {
        p_prop->maxThreadsDim[i] = cdprop.maxThreadsDim[i];
        p_prop->maxGridSize[i] = cdprop.maxGridSize[i];
    }
    p_prop->clockRate = cdprop.clockRate;
    p_prop->memoryClockRate = cdprop.memoryClockRate;
    p_prop->memoryBusWidth = cdprop.memoryBusWidth;
    p_prop->totalConstMem = cdprop.totalConstMem;
    p_prop->major = cdprop.major;
    p_prop->minor = cdprop.minor;
    p_prop->multiProcessorCount = cdprop.multiProcessorCount;
    p_prop->l2CacheSize = cdprop.l2CacheSize;
    p_prop->maxThreadsPerMultiProcessor = cdprop.maxThreadsPerMultiProcessor;
    p_prop->computeMode = cdprop.computeMode;
    p_prop->clockInstructionRate = cdprop.clockRate;  // Same as clock-rate:

    int ccVers = p_prop->major * 100 + p_prop->minor * 10;
    p_prop->arch.hasGlobalInt32Atomics = (ccVers >= 110);
    p_prop->arch.hasGlobalFloatAtomicExch = (ccVers >= 110);
    p_prop->arch.hasSharedInt32Atomics = (ccVers >= 120);
    p_prop->arch.hasSharedFloatAtomicExch = (ccVers >= 120);
    p_prop->arch.hasFloatAtomicAdd = (ccVers >= 200);
    p_prop->arch.hasGlobalInt64Atomics = (ccVers >= 120);
    p_prop->arch.hasSharedInt64Atomics = (ccVers >= 110);
    p_prop->arch.hasDoubles = (ccVers >= 130);
    p_prop->arch.hasWarpVote = (ccVers >= 120);
    p_prop->arch.hasWarpBallot = (ccVers >= 200);
    p_prop->arch.hasWarpShuffle = (ccVers >= 300);
    p_prop->arch.hasFunnelShift = (ccVers >= 350);
    p_prop->arch.hasThreadFenceSystem = (ccVers >= 200);
    p_prop->arch.hasSyncThreadsExt = (ccVers >= 200);
    p_prop->arch.hasSurfaceFuncs = (ccVers >= 200);
    p_prop->arch.has3dGrid = (ccVers >= 200);
    p_prop->arch.hasDynamicParallelism = (ccVers >= 350);

    p_prop->concurrentKernels = cdprop.concurrentKernels;
    p_prop->pciDomainID = cdprop.pciDomainID;
    p_prop->pciBusID = cdprop.pciBusID;
    p_prop->pciDeviceID = cdprop.pciDeviceID;
    p_prop->maxSharedMemoryPerMultiProcessor = cdprop.sharedMemPerMultiprocessor;
    p_prop->isMultiGpuBoard = cdprop.isMultiGpuBoard;
    p_prop->canMapHostMemory = cdprop.canMapHostMemory;
    p_prop->gcnArch = 0;  // Not a GCN arch
    p_prop->integrated = cdprop.integrated;
    p_prop->cooperativeLaunch = cdprop.cooperativeLaunch;
    p_prop->cooperativeMultiDeviceLaunch = cdprop.cooperativeMultiDeviceLaunch;

    p_prop->maxTexture1D = cdprop.maxTexture1D;
    p_prop->maxTexture2D[0] = cdprop.maxTexture2D[0];
    p_prop->maxTexture2D[1] = cdprop.maxTexture2D[1];
    p_prop->maxTexture3D[0] = cdprop.maxTexture3D[0];
    p_prop->maxTexture3D[1] = cdprop.maxTexture3D[1];
    p_prop->maxTexture3D[2] = cdprop.maxTexture3D[2];

    p_prop->memPitch = cdprop.memPitch;
    p_prop->textureAlignment = cdprop.textureAlignment;
    p_prop->texturePitchAlignment = cdprop.texturePitchAlignment;
    p_prop->kernelExecTimeoutEnabled = cdprop.kernelExecTimeoutEnabled;
    p_prop->ECCEnabled = cdprop.ECCEnabled;
    p_prop->tccDriver = cdprop.tccDriver;

    return hipCUDAErrorTohipError(cerror);
}

HIP_NVCC_INLINE hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device) {
    enum cudaDeviceAttr cdattr;
    cudaError_t cerror;

    switch (attr) {
        case hipDeviceAttributeMaxThreadsPerBlock:
            cdattr = cudaDevAttrMaxThreadsPerBlock;
            break;
        case hipDeviceAttributeMaxBlockDimX:
            cdattr = cudaDevAttrMaxBlockDimX;
            break;
        case hipDeviceAttributeMaxBlockDimY:
            cdattr = cudaDevAttrMaxBlockDimY;
            break;
        case hipDeviceAttributeMaxBlockDimZ:
            cdattr = cudaDevAttrMaxBlockDimZ;
            break;
        case hipDeviceAttributeMaxGridDimX:
            cdattr = cudaDevAttrMaxGridDimX;
            break;
        case hipDeviceAttributeMaxGridDimY:
            cdattr = cudaDevAttrMaxGridDimY;
            break;
        case hipDeviceAttributeMaxGridDimZ:
            cdattr = cudaDevAttrMaxGridDimZ;
            break;
        case hipDeviceAttributeMaxSharedMemoryPerBlock:
            cdattr = cudaDevAttrMaxSharedMemoryPerBlock;
            break;
        case hipDeviceAttributeTotalConstantMemory:
            cdattr = cudaDevAttrTotalConstantMemory;
            break;
        case hipDeviceAttributeWarpSize:
            cdattr = cudaDevAttrWarpSize;
            break;
        case hipDeviceAttributeMaxRegistersPerBlock:
            cdattr = cudaDevAttrMaxRegistersPerBlock;
            break;
        case hipDeviceAttributeClockRate:
            cdattr = cudaDevAttrClockRate;
            break;
        case hipDeviceAttributeMemoryClockRate:
            cdattr = cudaDevAttrMemoryClockRate;
            break;
        case hipDeviceAttributeMemoryBusWidth:
            cdattr = cudaDevAttrGlobalMemoryBusWidth;
            break;
        case hipDeviceAttributeMultiprocessorCount:
            cdattr = cudaDevAttrMultiProcessorCount;
            break;
        case hipDeviceAttributeComputeMode:
            cdattr = cudaDevAttrComputeMode;
            break;
        case hipDeviceAttributeL2CacheSize:
            cdattr = cudaDevAttrL2CacheSize;
            break;
        case hipDeviceAttributeMaxThreadsPerMultiProcessor:
            cdattr = cudaDevAttrMaxThreadsPerMultiProcessor;
            break;
        case hipDeviceAttributeComputeCapabilityMajor:
            cdattr = cudaDevAttrComputeCapabilityMajor;
            break;
        case hipDeviceAttributeComputeCapabilityMinor:
            cdattr = cudaDevAttrComputeCapabilityMinor;
            break;
        case hipDeviceAttributeConcurrentKernels:
            cdattr = cudaDevAttrConcurrentKernels;
            break;
        case hipDeviceAttributePciBusId:
            cdattr = cudaDevAttrPciBusId;
            break;
        case hipDeviceAttributePciDeviceId:
            cdattr = cudaDevAttrPciDeviceId;
            break;
        case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
            cdattr = cudaDevAttrMaxSharedMemoryPerMultiprocessor;
            break;
        case hipDeviceAttributeIsMultiGpuBoard:
            cdattr = cudaDevAttrIsMultiGpuBoard;
            break;
        case hipDeviceAttributeIntegrated:
            cdattr = cudaDevAttrIntegrated;
            break;
        case hipDeviceAttributeMaxTexture1DWidth:
            cdattr = cudaDevAttrMaxTexture1DWidth;
            break;
        case hipDeviceAttributeMaxTexture2DWidth:
            cdattr = cudaDevAttrMaxTexture2DWidth;
            break;
        case hipDeviceAttributeMaxTexture2DHeight:
            cdattr = cudaDevAttrMaxTexture2DHeight;
            break;
        case hipDeviceAttributeMaxTexture3DWidth:
            cdattr = cudaDevAttrMaxTexture3DWidth;
            break;
        case hipDeviceAttributeMaxTexture3DHeight:
            cdattr = cudaDevAttrMaxTexture3DHeight;
            break;
        case hipDeviceAttributeMaxTexture3DDepth:
            cdattr = cudaDevAttrMaxTexture3DDepth;
            break;
        case hipDeviceAttributeMaxPitch:
            cdattr = cudaDevAttrMaxPitch;
            break;
        case hipDeviceAttributeTextureAlignment:
            cdattr = cudaDevAttrTextureAlignment;
            break;
        case hipDeviceAttributeTexturePitchAlignment:
            cdattr = cudaDevAttrTexturePitchAlignment;
            break;
        case hipDeviceAttributeKernelExecTimeout:
            cdattr = cudaDevAttrKernelExecTimeout;
            break;
        case hipDeviceAttributeCanMapHostMemory:
            cdattr = cudaDevAttrCanMapHostMemory;
            break;
        case hipDeviceAttributeEccEnabled:
            cdattr = cudaDevAttrEccEnabled;
            break;
        default:
            return hipCUDAErrorTohipError(cudaErrorInvalidValue);
    }

    cerror = cudaDeviceGetAttribute(pi, cdattr, device);

    return hipCUDAErrorTohipError(cerror);
}

HIP_NVCC_INLINE hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                                        const void* func,
                                                                        int blockSize,
                                                                        size_t dynamicSMemSize) {
    cudaError_t cerror;
    cerror =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
    return hipCUDAErrorTohipError(cerror);
}

HIP_NVCC_INLINE hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes,
                                                   const void* ptr) {
    struct cudaPointerAttributes cPA;
    hipError_t err = hipCUDAErrorTohipError(cudaPointerGetAttributes(&cPA, ptr));
    if (err == hipSuccess) {
        switch (cPA.memoryType) {
            case cudaMemoryTypeDevice:
                attributes->memoryType = hipMemoryTypeDevice;
                break;
            case cudaMemoryTypeHost:
                attributes->memoryType = hipMemoryTypeHost;
                break;
            default:
                return hipErrorUnknown;
        }
        attributes->device = cPA.device;
        attributes->devicePointer = cPA.devicePointer;
        attributes->hostPointer = cPA.hostPointer;
        attributes->isManaged = 0;
        attributes->allocationFlags = 0;
    }
    return err;
}

HIP_NVCC_INLINE hipError_t hipMemGetInfo(size_t* free, size_t* total) {
    return hipCUDAErrorTohipError(cudaMemGetInfo(free, total));
}

HIP_NVCC_INLINE hipError_t hipEventCreate(hipEvent_t* event) {
    return hipCUDAErrorTohipError(cudaEventCreate(event));
}

HIP_NVCC_INLINE hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream __dparm(NULL)) {
    return hipCUDAErrorTohipError(cudaEventRecord(event, stream));
}

HIP_NVCC_INLINE hipError_t hipEventSynchronize(hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaEventSynchronize(event));
}

HIP_NVCC_INLINE hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    return hipCUDAErrorTohipError(cudaEventElapsedTime(ms, start, stop));
}

HIP_NVCC_INLINE hipError_t hipEventDestroy(hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaEventDestroy(event));
}

HIP_NVCC_INLINE hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaStreamCreateWithFlags(stream, flags));
}

HIP_NVCC_INLINE hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags,
                                                       int priority) {
    return hipCUDAErrorTohipError(cudaStreamCreateWithPriority(stream, flags, priority));
}

HIP_NVCC_INLINE hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority,
                                                           int* greatestPriority) {
    return hipCUDAErrorTohipError(
        cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority));
}

HIP_NVCC_INLINE hipError_t hipStreamCreate(hipStream_t* stream) {
    return hipCUDAErrorTohipError(cudaStreamCreate(stream));
}

HIP_NVCC_INLINE hipError_t hipStreamSynchronize(hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaStreamSynchronize(stream));
}

HIP_NVCC_INLINE hipError_t hipStreamDestroy(hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaStreamDestroy(stream));
}

HIP_NVCC_INLINE hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
    return hipCUDAErrorTohipError(cudaStreamGetFlags(stream, flags));
}

HIP_NVCC_INLINE hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
    return hipCUDAErrorTohipError(cudaStreamGetPriority(stream, priority));
}

HIP_NVCC_INLINE hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                                              unsigned int flags) {
    return hipCUDAErrorTohipError(cudaStreamWaitEvent(stream, event, flags));
}

HIP_NVCC_INLINE hipError_t hipStreamQuery(hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaStreamQuery(stream));
}

HIP_NVCC_INLINE hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback,
                                                void* userData, unsigned int flags) {
    return hipCUDAErrorTohipError(
        cudaStreamAddCallback(stream, (cudaStreamCallback_t)callback, userData, flags));
}

HIP_NVCC_INLINE hipError_t hipDriverGetVersion(int* driverVersion) {
    cudaError_t err = cudaDriverGetVersion(driverVersion);

    // Override driver version to match version reported on HCC side.
    *driverVersion = 4;

    return hipCUDAErrorTohipError(err);
}

HIP_NVCC_INLINE hipError_t hipRuntimeGetVersion(int* runtimeVersion) {
    return hipCUDAErrorTohipError(cudaRuntimeGetVersion(runtimeVersion));
}

HIP_NVCC_INLINE hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) {
    return hipCUDAErrorTohipError(cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice));
}

HIP_NVCC_INLINE hipError_t hipDeviceDisablePeerAccess(int peerDevice) {
    return hipCUDAErrorTohipError(cudaDeviceDisablePeerAccess(peerDevice));
}

HIP_NVCC_INLINE hipError_t hipDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaDeviceEnablePeerAccess(peerDevice, flags));
}

HIP_NVCC_INLINE hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
    return hipCUResultTohipError(cuCtxDisablePeerAccess(peerCtx));
}

HIP_NVCC_INLINE hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
    return hipCUResultTohipError(cuCtxEnablePeerAccess(peerCtx, flags));
}

HIP_NVCC_INLINE hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags,
                                                       int* active) {
    return hipCUResultTohipError(cuDevicePrimaryCtxGetState(dev, flags, active));
}

HIP_NVCC_INLINE hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
    return hipCUResultTohipError(cuDevicePrimaryCtxRelease(dev));
}

HIP_NVCC_INLINE hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
    return hipCUResultTohipError(cuDevicePrimaryCtxRetain(pctx, dev));
}

HIP_NVCC_INLINE hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
    return hipCUResultTohipError(cuDevicePrimaryCtxReset(dev));
}

HIP_NVCC_INLINE hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
    return hipCUResultTohipError(cuDevicePrimaryCtxSetFlags(dev, flags));
}

HIP_NVCC_INLINE hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize,
                                                 hipDeviceptr_t dptr) {
    return hipCUResultTohipError(cuMemGetAddressRange(pbase, psize, dptr));
}

HIP_NVCC_INLINE hipError_t hipMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice,
                                         size_t count) {
    return hipCUDAErrorTohipError(cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count));
}

HIP_NVCC_INLINE hipError_t hipMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                                              int srcDevice, size_t count,
                                              hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(
        cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream));
}

// Profile APIs:
HIP_NVCC_INLINE hipError_t hipProfilerStart() {
    return hipCUDAErrorTohipError(cudaProfilerStart());
}

HIP_NVCC_INLINE hipError_t hipProfilerStop() { return hipCUDAErrorTohipError(cudaProfilerStop()); }

HIP_NVCC_INLINE hipError_t hipSetDeviceFlags(unsigned int flags) {
    return hipCUDAErrorTohipError(cudaSetDeviceFlags(flags));
}

HIP_NVCC_INLINE hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaEventCreateWithFlags(event, flags));
}

HIP_NVCC_INLINE hipError_t hipEventQuery(hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaEventQuery(event));
}

HIP_NVCC_INLINE hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
    return hipCUResultTohipError(cuCtxCreate(ctx, flags, device));
}

HIP_NVCC_INLINE hipError_t hipCtxDestroy(hipCtx_t ctx) {
    return hipCUResultTohipError(cuCtxDestroy(ctx));
}

HIP_NVCC_INLINE hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
    return hipCUResultTohipError(cuCtxPopCurrent(ctx));
}

HIP_NVCC_INLINE hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
    return hipCUResultTohipError(cuCtxPushCurrent(ctx));
}

HIP_NVCC_INLINE hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
    return hipCUResultTohipError(cuCtxSetCurrent(ctx));
}

HIP_NVCC_INLINE hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
    return hipCUResultTohipError(cuCtxGetCurrent(ctx));
}

HIP_NVCC_INLINE hipError_t hipCtxGetDevice(hipDevice_t* device) {
    return hipCUResultTohipError(cuCtxGetDevice(device));
}

HIP_NVCC_INLINE hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion) {
    return hipCUResultTohipError(cuCtxGetApiVersion(ctx, (unsigned int*)apiVersion));
}

HIP_NVCC_INLINE hipError_t hipCtxGetCacheConfig(hipFuncCache* cacheConfig) {
    return hipCUResultTohipError(cuCtxGetCacheConfig(cacheConfig));
}

HIP_NVCC_INLINE hipError_t hipCtxSetCacheConfig(hipFuncCache cacheConfig) {
    return hipCUResultTohipError(cuCtxSetCacheConfig(cacheConfig));
}

HIP_NVCC_INLINE hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
    return hipCUResultTohipError(cuCtxSetSharedMemConfig((CUsharedconfig)config));
}

HIP_NVCC_INLINE hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig) {
    return hipCUResultTohipError(cuCtxGetSharedMemConfig((CUsharedconfig*)pConfig));
}

HIP_NVCC_INLINE hipError_t hipCtxSynchronize(void) {
    return hipCUResultTohipError(cuCtxSynchronize());
}

HIP_NVCC_INLINE hipError_t hipCtxGetFlags(unsigned int* flags) {
    return hipCUResultTohipError(cuCtxGetFlags(flags));
}

HIP_NVCC_INLINE hipError_t hipCtxDetach(hipCtx_t ctx) {
    return hipCUResultTohipError(cuCtxDetach(ctx));
}

HIP_NVCC_INLINE hipError_t hipDeviceGet(hipDevice_t* device, int ordinal) {
    return hipCUResultTohipError(cuDeviceGet(device, ordinal));
}

HIP_NVCC_INLINE hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device) {
    return hipCUResultTohipError(cuDeviceComputeCapability(major, minor, device));
}

HIP_NVCC_INLINE hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device) {
    return hipCUResultTohipError(cuDeviceGetName(name, len, device));
}

HIP_NVCC_INLINE hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, hipDevice_t device) {
    return hipCUDAErrorTohipError(cudaDeviceGetPCIBusId(pciBusId, len, device));
}

HIP_NVCC_INLINE hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId) {
    return hipCUDAErrorTohipError(cudaDeviceGetByPCIBusId(device, pciBusId));
}

HIP_NVCC_INLINE hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* config) {
    return hipCUDAErrorTohipError(cudaDeviceGetSharedMemConfig(config));
}

HIP_NVCC_INLINE hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) {
    return hipCUDAErrorTohipError(cudaDeviceSetSharedMemConfig(config));
}

HIP_NVCC_INLINE hipError_t hipDeviceGetLimit(size_t* pValue, hipLimit_t limit) {
    return hipCUDAErrorTohipError(cudaDeviceGetLimit(pValue, limit));
}

HIP_NVCC_INLINE hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device) {
    return hipCUResultTohipError(cuDeviceTotalMem(bytes, device));
}

HIP_NVCC_INLINE hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
    return hipCUResultTohipError(cuModuleLoad(module, fname));
}

HIP_NVCC_INLINE hipError_t hipModuleUnload(hipModule_t hmod) {
    return hipCUResultTohipError(cuModuleUnload(hmod));
}

HIP_NVCC_INLINE hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                                const char* kname) {
    return hipCUResultTohipError(cuModuleGetFunction(function, module, kname));
}

HIP_NVCC_INLINE hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func) {
    return hipCUDAErrorTohipError(cudaFuncGetAttributes(attr, func));
}

HIP_NVCC_INLINE hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib,
                                               hipFunction_t hfunc) {
    return hipCUResultTohipError(cuFuncGetAttribute(value, attrib, hfunc));
}

HIP_NVCC_INLINE hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                                              const char* name) {
    return hipCUResultTohipError(cuModuleGetGlobal(dptr, bytes, hmod, name));
}

HIP_NVCC_INLINE hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
    return hipCUResultTohipError(cuModuleLoadData(module, image));
}

HIP_NVCC_INLINE hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                               unsigned int numOptions, hipJitOption* options,
                                               void** optionValues) {
    return hipCUResultTohipError(
        cuModuleLoadDataEx(module, image, numOptions, options, optionValues));
}

HIP_NVCC_INLINE hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks,
                                           dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                           hipStream_t stream) {
    return hipCUDAErrorTohipError(
        cudaLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream));
}

HIP_NVCC_INLINE hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                                 unsigned int gridDimY, unsigned int gridDimZ,
                                                 unsigned int blockDimX, unsigned int blockDimY,
                                                 unsigned int blockDimZ,
                                                 unsigned int sharedMemBytes, hipStream_t stream,
                                                 void** kernelParams, void** extra) {
    return hipCUResultTohipError(cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                                blockDimY, blockDimZ, sharedMemBytes, stream,
                                                kernelParams, extra));
}

HIP_NVCC_INLINE hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t cacheConfig) {
    return hipCUDAErrorTohipError(cudaFuncSetCacheConfig(func, cacheConfig));
}

HIP_NVCC_INLINE hipError_t hipBindTexture(size_t* offset, struct textureReference* tex,
                                          const void* devPtr, const hipChannelFormatDesc* desc,
                                          size_t size __dparm(UINT_MAX)) {
    return hipCUDAErrorTohipError(cudaBindTexture(offset, tex, devPtr, desc, size));
}

HIP_NVCC_INLINE hipError_t hipBindTexture2D(size_t* offset, struct textureReference* tex,
                                            const void* devPtr, const hipChannelFormatDesc* desc,
                                            size_t width, size_t height, size_t pitch) {
    return hipCUDAErrorTohipError(
        cudaBindTexture2D(offset, tex, devPtr, desc, width, height, pitch));
}

HIP_NVCC_INLINE hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w,
                                                          hipChannelFormatKind f) {
    return cudaCreateChannelDesc(x, y, z, w, hipChannelFormatKindToCudaChannelFormatKind(f));
}

HIP_NVCC_INLINE hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject,
                                                  const hipResourceDesc* pResDesc,
                                                  const hipTextureDesc* pTexDesc,
                                                  const hipResourceViewDesc* pResViewDesc) {
    return hipCUDAErrorTohipError(
        cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc));
}

HIP_NVCC_INLINE hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
    return hipCUDAErrorTohipError(cudaDestroyTextureObject(textureObject));
}

HIP_NVCC_INLINE hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject,
                                                  const hipResourceDesc* pResDesc) {
    return hipCUDAErrorTohipError(cudaCreateSurfaceObject(pSurfObject, pResDesc));
}

HIP_NVCC_INLINE hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject) {
    return hipCUDAErrorTohipError(cudaDestroySurfaceObject(surfaceObject));
}

HIP_NVCC_INLINE hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                                           hipTextureObject_t textureObject) {
    return hipCUDAErrorTohipError(cudaGetTextureObjectResourceDesc(pResDesc, textureObject));
}

HIP_NVCC_INLINE hipError_t hipGetTextureAlignmentOffset(size_t* offset,
                                                        const struct textureReference* texref) {
    return hipCUDAErrorTohipError(cudaGetTextureAlignmentOffset(offset, texref));
}

HIP_NVCC_INLINE hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array) {
    return hipCUDAErrorTohipError(cudaGetChannelDesc(desc, array));
}


#ifdef __cplusplus
}
#endif
