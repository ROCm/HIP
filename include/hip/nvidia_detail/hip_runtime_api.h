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

#ifndef HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_RUNTIME_API_H
#define HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_RUNTIME_API_H

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
#define __dparm(x) = x
#else
#define __dparm(x)
#endif

// Add Deprecated Support for CUDA Mapped HIP APIs
#if defined(__DOXYGEN_ONLY__) || defined(HIP_ENABLE_DEPRECATED)
#define __HIP_DEPRECATED
#elif defined(_MSC_VER)
#define __HIP_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define __HIP_DEPRECATED __attribute__((deprecated))
#else
#define __HIP_DEPRECATED
#endif


// TODO -move to include/hip_runtime_api.h as a common implementation.
/**
 * Memory copy types
 *
 */
typedef enum hipMemcpyKind {
    hipMemcpyHostToHost,
    hipMemcpyHostToDevice,
    hipMemcpyDeviceToHost,
    hipMemcpyDeviceToDevice,
    hipMemcpyDefault
} hipMemcpyKind;

typedef enum hipMemoryAdvise {
    hipMemAdviseSetReadMostly,
    hipMemAdviseUnsetReadMostly,
    hipMemAdviseSetPreferredLocation,
    hipMemAdviseUnsetPreferredLocation,
    hipMemAdviseSetAccessedBy,
    hipMemAdviseUnsetAccessedBy
} hipMemoryAdvise;

typedef enum hipMemRangeAttribute {
    hipMemRangeAttributeReadMostly,
    hipMemRangeAttributePreferredLocation,
    hipMemRangeAttributeAccessedBy,
    hipMemRangeAttributeLastPrefetchLocation
} hipMemRangeAttribute;

// hipDataType
#define hipDataType cudaDataType
#define HIP_R_16F CUDA_R_16F
#define HIP_R_32F CUDA_R_32F
#define HIP_R_64F CUDA_R_64F
#define HIP_C_16F CUDA_C_16F
#define HIP_C_32F CUDA_C_32F
#define HIP_C_64F CUDA_C_64F

// hipLibraryPropertyType
#define hipLibraryPropertyType libraryPropertyType
#define HIP_LIBRARY_MAJOR_VERSION MAJOR_VERSION
#define HIP_LIBRARY_MINOR_VERSION MINOR_VERSION
#define HIP_LIBRARY_PATCH_LEVEL PATCH_LEVEL

#define HIP_ARRAY_DESCRIPTOR CUDA_ARRAY_DESCRIPTOR
#define HIP_ARRAY3D_DESCRIPTOR CUDA_ARRAY3D_DESCRIPTOR

//hipArray_Format
#define HIP_AD_FORMAT_UNSIGNED_INT8   CU_AD_FORMAT_UNSIGNED_INT8
#define HIP_AD_FORMAT_UNSIGNED_INT16  CU_AD_FORMAT_UNSIGNED_INT16
#define HIP_AD_FORMAT_UNSIGNED_INT32  CU_AD_FORMAT_UNSIGNED_INT32
#define HIP_AD_FORMAT_SIGNED_INT8     CU_AD_FORMAT_SIGNED_INT8
#define HIP_AD_FORMAT_SIGNED_INT16    CU_AD_FORMAT_SIGNED_INT16
#define HIP_AD_FORMAT_SIGNED_INT32    CU_AD_FORMAT_SIGNED_INT32
#define HIP_AD_FORMAT_HALF            CU_AD_FORMAT_HALF
#define HIP_AD_FORMAT_FLOAT           CU_AD_FORMAT_FLOAT

// hipArray_Format
#define hipArray_Format CUarray_format

inline static CUarray_format hipArray_FormatToCUarray_format(
    hipArray_Format format) {
    switch (format) {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            return CU_AD_FORMAT_UNSIGNED_INT8;
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            return CU_AD_FORMAT_UNSIGNED_INT16;
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            return CU_AD_FORMAT_UNSIGNED_INT32;
        case HIP_AD_FORMAT_SIGNED_INT8:
            return CU_AD_FORMAT_SIGNED_INT8;
        case HIP_AD_FORMAT_SIGNED_INT16:
            return CU_AD_FORMAT_SIGNED_INT16;
        case HIP_AD_FORMAT_SIGNED_INT32:
            return CU_AD_FORMAT_SIGNED_INT32;
        case HIP_AD_FORMAT_HALF:
            return CU_AD_FORMAT_HALF;
        case HIP_AD_FORMAT_FLOAT:
            return CU_AD_FORMAT_FLOAT;
        default:
            return CU_AD_FORMAT_UNSIGNED_INT8;
    }
}

#define HIP_TR_ADDRESS_MODE_WRAP   CU_TR_ADDRESS_MODE_WRAP
#define HIP_TR_ADDRESS_MODE_CLAMP  CU_TR_ADDRESS_MODE_CLAMP
#define HIP_TR_ADDRESS_MODE_MIRROR CU_TR_ADDRESS_MODE_MIRROR
#define HIP_TR_ADDRESS_MODE_BORDER CU_TR_ADDRESS_MODE_BORDER

// hipAddress_mode
#define hipAddress_mode CUaddress_mode

inline static CUaddress_mode hipAddress_modeToCUaddress_mode(
    hipAddress_mode mode) {
    switch (mode) {
        case HIP_TR_ADDRESS_MODE_WRAP:
            return CU_TR_ADDRESS_MODE_WRAP;
        case HIP_TR_ADDRESS_MODE_CLAMP:
            return CU_TR_ADDRESS_MODE_CLAMP;
        case HIP_TR_ADDRESS_MODE_MIRROR:
            return CU_TR_ADDRESS_MODE_MIRROR;
        case HIP_TR_ADDRESS_MODE_BORDER:
            return CU_TR_ADDRESS_MODE_BORDER;
        default:
            return CU_TR_ADDRESS_MODE_WRAP;
    }
}

#define HIP_TR_FILTER_MODE_POINT   CU_TR_FILTER_MODE_POINT
#define HIP_TR_FILTER_MODE_LINEAR  CU_TR_FILTER_MODE_LINEAR

// hipFilter_mode
#define hipFilter_mode CUfilter_mode

inline static CUfilter_mode hipFilter_mode_enumToCUfilter_mode(
    hipFilter_mode mode) {
    switch (mode) {
        case HIP_TR_FILTER_MODE_POINT:
            return CU_TR_FILTER_MODE_POINT;
        case HIP_TR_FILTER_MODE_LINEAR:
            return CU_TR_FILTER_MODE_LINEAR;
        default:
            return CU_TR_FILTER_MODE_POINT;
    }
}

//hipResourcetype
#define HIP_RESOURCE_TYPE_ARRAY            CU_RESOURCE_TYPE_ARRAY
#define HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY  CU_RESOURCE_TYPE_MIPMAPPED_ARRAY
#define HIP_RESOURCE_TYPE_LINEAR           CU_RESOURCE_TYPE_LINEAR
#define HIP_RESOURCE_TYPE_PITCH2D          CU_RESOURCE_TYPE_PITCH2D

// hipResourcetype
#define hipResourcetype CUresourcetype

inline static CUresourcetype hipResourcetype_enumToCUresourcetype(
    hipResourcetype resType) {
    switch (resType) {
        case HIP_RESOURCE_TYPE_ARRAY:
            return CU_RESOURCE_TYPE_ARRAY;
        case HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY:
            return CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        case HIP_RESOURCE_TYPE_LINEAR:
            return CU_RESOURCE_TYPE_LINEAR;
        case HIP_RESOURCE_TYPE_PITCH2D:
            return CU_RESOURCE_TYPE_PITCH2D;
        default:
            return CU_RESOURCE_TYPE_ARRAY;
    }
}

#define hipTexRef CUtexref
#define hiparray CUarray

// hipTextureAddressMode
typedef enum cudaTextureAddressMode hipTextureAddressMode;
#define hipAddressModeWrap cudaAddressModeWrap
#define hipAddressModeClamp cudaAddressModeClamp
#define hipAddressModeMirror cudaAddressModeMirror
#define hipAddressModeBorder cudaAddressModeBorder

// hipTextureFilterMode
typedef enum cudaTextureFilterMode hipTextureFilterMode;
#define hipFilterModePoint cudaFilterModePoint
#define hipFilterModeLinear cudaFilterModeLinear

// hipTextureReadMode
typedef enum cudaTextureReadMode hipTextureReadMode;
#define hipReadModeElementType cudaReadModeElementType
#define hipReadModeNormalizedFloat cudaReadModeNormalizedFloat

// hipChannelFormatKind
typedef enum cudaChannelFormatKind hipChannelFormatKind;
#define hipChannelFormatKindSigned      cudaChannelFormatKindSigned
#define hipChannelFormatKindUnsigned    cudaChannelFormatKindUnsigned
#define hipChannelFormatKindFloat       cudaChannelFormatKindFloat
#define hipChannelFormatKindNone        cudaChannelFormatKindNone

#define hipSurfaceBoundaryMode cudaSurfaceBoundaryMode
#define hipBoundaryModeZero cudaBoundaryModeZero
#define hipBoundaryModeTrap cudaBoundaryModeTrap
#define hipBoundaryModeClamp cudaBoundaryModeClamp

// hipFuncCache
#define hipFuncCachePreferNone cudaFuncCachePreferNone
#define hipFuncCachePreferShared cudaFuncCachePreferShared
#define hipFuncCachePreferL1 cudaFuncCachePreferL1
#define hipFuncCachePreferEqual cudaFuncCachePreferEqual

// hipResourceType
#define hipResourceType cudaResourceType
#define hipResourceTypeArray cudaResourceTypeArray
#define hipResourceTypeMipmappedArray cudaResourceTypeMipmappedArray
#define hipResourceTypeLinear cudaResourceTypeLinear
#define hipResourceTypePitch2D cudaResourceTypePitch2D
//
// hipErrorNoDevice.


//! Flags that can be used with hipEventCreateWithFlags:
#define hipEventDefault cudaEventDefault
#define hipEventBlockingSync cudaEventBlockingSync
#define hipEventDisableTiming cudaEventDisableTiming
#define hipEventInterprocess cudaEventInterprocess
#define hipEventReleaseToDevice 0 /* no-op on CUDA platform */
#define hipEventReleaseToSystem 0 /* no-op on CUDA platform */


#define hipHostMallocDefault cudaHostAllocDefault
#define hipHostMallocPortable cudaHostAllocPortable
#define hipHostMallocMapped cudaHostAllocMapped
#define hipHostMallocWriteCombined cudaHostAllocWriteCombined
#define hipHostMallocCoherent 0x0
#define hipHostMallocNonCoherent 0x0

#define hipMemAttachGlobal cudaMemAttachGlobal
#define hipMemAttachHost cudaMemAttachHost
#define hipMemAttachSingle cudaMemAttachSingle

#define hipHostRegisterDefault cudaHostRegisterDefault
#define hipHostRegisterPortable cudaHostRegisterPortable
#define hipHostRegisterMapped cudaHostRegisterMapped
#define hipHostRegisterIoMemory cudaHostRegisterIoMemory

#define HIP_LAUNCH_PARAM_BUFFER_POINTER CU_LAUNCH_PARAM_BUFFER_POINTER
#define HIP_LAUNCH_PARAM_BUFFER_SIZE CU_LAUNCH_PARAM_BUFFER_SIZE
#define HIP_LAUNCH_PARAM_END CU_LAUNCH_PARAM_END
#define hipLimitMallocHeapSize cudaLimitMallocHeapSize
#define hipIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess

#define hipOccupancyDefault cudaOccupancyDefault

#define hipCooperativeLaunchMultiDeviceNoPreSync    \
        cudaCooperativeLaunchMultiDeviceNoPreSync
#define hipCooperativeLaunchMultiDeviceNoPostSync   \
        cudaCooperativeLaunchMultiDeviceNoPostSync


// enum CUjit_option redefines
#define hipJitOptionMaxRegisters CU_JIT_MAX_REGISTERS
#define hipJitOptionThreadsPerBlock CU_JIT_THREADS_PER_BLOCK
#define hipJitOptionWallTime CU_JIT_WALL_TIME
#define hipJitOptionInfoLogBuffer CU_JIT_INFO_LOG_BUFFER
#define hipJitOptionInfoLogBufferSizeBytes CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
#define hipJitOptionErrorLogBuffer CU_JIT_ERROR_LOG_BUFFER
#define hipJitOptionErrorLogBufferSizeBytes CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
#define hipJitOptionOptimizationLevel CU_JIT_OPTIMIZATION_LEVEL
#define hipJitOptionTargetFromContext CU_JIT_TARGET_FROM_CUCONTEXT
#define hipJitOptionTarget CU_JIT_TARGET
#define hipJitOptionFallbackStrategy CU_JIT_FALLBACK_STRATEGY
#define hipJitOptionGenerateDebugInfo CU_JIT_GENERATE_DEBUG_INFO
#define hipJitOptionLogVerbose CU_JIT_LOG_VERBOSE
#define hipJitOptionGenerateLineInfo CU_JIT_GENERATE_LINE_INFO
#define hipJitOptionCacheMode CU_JIT_CACHE_MODE
#define hipJitOptionSm3xOpt CU_JIT_NEW_SM3X_OPT
#define hipJitOptionFastCompile CU_JIT_FAST_COMPILE
#define hipJitOptionNumOptions CU_JIT_NUM_OPTIONS

typedef cudaEvent_t hipEvent_t;
typedef cudaStream_t hipStream_t;
typedef cudaIpcEventHandle_t hipIpcEventHandle_t;
typedef cudaIpcMemHandle_t hipIpcMemHandle_t;
typedef enum cudaLimit hipLimit_t;
typedef enum cudaFuncAttribute hipFuncAttribute;
typedef enum cudaFuncCache hipFuncCache_t;
typedef CUcontext hipCtx_t;
typedef enum cudaSharedMemConfig hipSharedMemConfig;
typedef CUfunc_cache hipFuncCache;
typedef CUjit_option hipJitOption;
typedef CUdevice hipDevice_t;
typedef enum cudaDeviceP2PAttr hipDeviceP2PAttr;
#define hipDevP2PAttrPerformanceRank cudaDevP2PAttrPerformanceRank
#define hipDevP2PAttrAccessSupported cudaDevP2PAttrAccessSupported
#define hipDevP2PAttrNativeAtomicSupported cudaDevP2PAttrNativeAtomicSupported
#define hipDevP2PAttrHipArrayAccessSupported cudaDevP2PAttrCudaArrayAccessSupported
#define hipFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize
#define hipFuncAttributePreferredSharedMemoryCarveout cudaFuncAttributePreferredSharedMemoryCarveout

typedef CUmodule hipModule_t;
typedef CUfunction hipFunction_t;
typedef CUdeviceptr hipDeviceptr_t;
typedef struct cudaArray hipArray;
typedef struct cudaArray* hipArray_t;
typedef struct cudaArray* hipArray_const_t;
typedef struct cudaFuncAttributes hipFuncAttributes;
typedef struct cudaLaunchParams hipLaunchParams;
#define hipFunction_attribute CUfunction_attribute
#define hip_Memcpy2D CUDA_MEMCPY2D
#define HIP_MEMCPY3D CUDA_MEMCPY3D
#define hipMemcpy3DParms cudaMemcpy3DParms
#define hipArrayDefault cudaArrayDefault
#define hipArrayLayered cudaArrayLayered
#define hipArraySurfaceLoadStore cudaArraySurfaceLoadStore
#define hipArrayCubemap cudaArrayCubemap
#define hipArrayTextureGather cudaArrayTextureGather

typedef cudaTextureObject_t hipTextureObject_t;
typedef cudaSurfaceObject_t hipSurfaceObject_t;
#define hipTextureType1D cudaTextureType1D
#define hipTextureType1DLayered cudaTextureType1DLayered
#define hipTextureType2D cudaTextureType2D
#define hipTextureType2DLayered cudaTextureType2DLayered
#define hipTextureType3D cudaTextureType3D

#define hipDeviceScheduleAuto cudaDeviceScheduleAuto
#define hipDeviceScheduleSpin cudaDeviceScheduleSpin
#define hipDeviceScheduleYield cudaDeviceScheduleYield
#define hipDeviceScheduleBlockingSync cudaDeviceScheduleBlockingSync
#define hipDeviceScheduleMask cudaDeviceScheduleMask
#define hipDeviceMapHost cudaDeviceMapHost
#define hipDeviceLmemResizeToMax cudaDeviceLmemResizeToMax

#define hipCpuDeviceId cudaCpuDeviceId
#define hipInvalidDeviceId cudaInvalidDeviceId
typedef struct cudaExtent hipExtent;
typedef struct cudaPitchedPtr hipPitchedPtr;
#define make_hipExtent make_cudaExtent
#define make_hipPos make_cudaPos
#define make_hipPitchedPtr make_cudaPitchedPtr
// Flags that can be used with hipStreamCreateWithFlags
#define hipStreamDefault cudaStreamDefault
#define hipStreamNonBlocking cudaStreamNonBlocking

typedef struct cudaChannelFormatDesc hipChannelFormatDesc;
typedef struct cudaResourceDesc hipResourceDesc;
typedef struct cudaTextureDesc hipTextureDesc;
typedef struct cudaResourceViewDesc hipResourceViewDesc;
// adding code for hipmemSharedConfig
#define hipSharedMemBankSizeDefault cudaSharedMemBankSizeDefault
#define hipSharedMemBankSizeFourByte cudaSharedMemBankSizeFourByte
#define hipSharedMemBankSizeEightByte cudaSharedMemBankSizeEightByte

//Function Attributes
#define HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_NUM_REGS CU_FUNC_ATTRIBUTE_NUM_REGS
#define HIP_FUNC_ATTRIBUTE_PTX_VERSION CU_FUNC_ATTRIBUTE_PTX_VERSION
#define HIP_FUNC_ATTRIBUTE_BINARY_VERSION CU_FUNC_ATTRIBUTE_BINARY_VERSION
#define HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA CU_FUNC_ATTRIBUTE_CACHE_MODE_CA
#define HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
#define HIP_FUNC_ATTRIBUTE_MAX CU_FUNC_ATTRIBUTE_MAX

#if CUDA_VERSION >= 9000
#define __shfl(...)      __shfl_sync(0xffffffff, __VA_ARGS__)
#define __shfl_up(...)   __shfl_up_sync(0xffffffff, __VA_ARGS__)
#define __shfl_down(...) __shfl_down_sync(0xffffffff, __VA_ARGS__)
#define __shfl_xor(...)  __shfl_xor_sync(0xffffffff, __VA_ARGS__)
#endif // CUDA_VERSION >= 9000

inline static hipError_t hipCUDAErrorTohipError(cudaError_t cuError) {
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
        case cudaErrorCooperativeLaunchTooLarge:
            return hipErrorCooperativeLaunchTooLarge;
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

inline static hipError_t hipCUResultTohipError(CUresult cuError) {
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
        case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
            return hipErrorCooperativeLaunchTooLarge;
        case CUDA_ERROR_NOT_SUPPORTED:
            return hipErrorNotSupported;
        case CUDA_ERROR_UNKNOWN:
        default:
            return hipErrorUnknown;  // Note - translated error.
    }
}

inline static cudaError_t hipErrorToCudaError(hipError_t hError) {
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
        case hipErrorCooperativeLaunchTooLarge:
            return cudaErrorCooperativeLaunchTooLarge;
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

inline static enum cudaMemcpyKind hipMemcpyKindToCudaMemcpyKind(hipMemcpyKind kind) {
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

inline static enum cudaTextureAddressMode hipTextureAddressModeToCudaTextureAddressMode(
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

inline static enum cudaMemRangeAttribute hipMemRangeAttributeTocudaMemRangeAttribute(
   hipMemRangeAttribute kind) {
   switch (kind) {
       case hipMemRangeAttributeReadMostly:
           return cudaMemRangeAttributeReadMostly;
       case hipMemRangeAttributePreferredLocation:
           return cudaMemRangeAttributePreferredLocation;
       case hipMemRangeAttributeAccessedBy:
           return cudaMemRangeAttributeAccessedBy;
       case hipMemRangeAttributeLastPrefetchLocation:
           return cudaMemRangeAttributeLastPrefetchLocation;
       default:
           return cudaMemRangeAttributeReadMostly;
   }
}

inline static enum cudaMemoryAdvise hipMemoryAdviseTocudaMemoryAdvise(
    hipMemoryAdvise kind) {
   switch (kind) {
       case hipMemAdviseSetReadMostly:
           return cudaMemAdviseSetReadMostly;
       case hipMemAdviseUnsetReadMostly :
           return cudaMemAdviseUnsetReadMostly ;
       case hipMemAdviseSetPreferredLocation:
           return cudaMemAdviseSetPreferredLocation;
       case hipMemAdviseUnsetPreferredLocation:
           return cudaMemAdviseUnsetPreferredLocation;
       case hipMemAdviseSetAccessedBy:
           return cudaMemAdviseSetAccessedBy;
       case hipMemAdviseUnsetAccessedBy:
           return cudaMemAdviseUnsetAccessedBy;
       default:
           return cudaMemAdviseSetReadMostly;
   }
}

inline static enum cudaTextureFilterMode hipTextureFilterModeToCudaTextureFilterMode(
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

inline static enum cudaTextureReadMode hipTextureReadModeToCudaTextureReadMode(hipTextureReadMode kind) {
    switch (kind) {
        case hipReadModeElementType:
            return cudaReadModeElementType;
        case hipReadModeNormalizedFloat:
            return cudaReadModeNormalizedFloat;
        default:
            return cudaReadModeElementType;
    }
}

inline static enum cudaChannelFormatKind hipChannelFormatKindToCudaChannelFormatKind(
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
inline static hipError_t hipInit(unsigned int flags) {
    return hipCUResultTohipError(cuInit(flags));
}

inline static hipError_t hipDeviceReset() { return hipCUDAErrorTohipError(cudaDeviceReset()); }

inline static hipError_t hipGetLastError() { return hipCUDAErrorTohipError(cudaGetLastError()); }

inline static hipError_t hipPeekAtLastError() {
    return hipCUDAErrorTohipError(cudaPeekAtLastError());
}

inline static hipError_t hipMalloc(void** ptr, size_t size) {
    return hipCUDAErrorTohipError(cudaMalloc(ptr, size));
}

inline static hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
    return hipCUDAErrorTohipError(cudaMallocPitch(ptr, pitch, width, height));
}

inline static hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr,size_t* pitch,size_t widthInBytes,size_t height,unsigned int elementSizeBytes){
    return hipCUResultTohipError(cuMemAllocPitch(dptr,pitch,widthInBytes,height,elementSizeBytes));
}

inline static hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
    return hipCUDAErrorTohipError(cudaMalloc3D(pitchedDevPtr, extent));
}

inline static hipError_t hipFree(void* ptr) { return hipCUDAErrorTohipError(cudaFree(ptr)); }

inline static hipError_t hipMallocHost(void** ptr, size_t size)
    __attribute__((deprecated("use hipHostMalloc instead")));
inline static hipError_t hipMallocHost(void** ptr, size_t size) {
    return hipCUDAErrorTohipError(cudaMallocHost(ptr, size));
}

inline static hipError_t hipMemAllocHost(void** ptr, size_t size)
    __attribute__((deprecated("use hipHostMalloc instead")));
inline static hipError_t hipMemAllocHost(void** ptr, size_t size) {
    return hipCUResultTohipError(cuMemAllocHost(ptr, size));
}

inline static hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags)
    __attribute__((deprecated("use hipHostMalloc instead")));
inline static hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaHostAlloc(ptr, size, flags));
}

inline static hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaHostAlloc(ptr, size, flags));
}

inline static hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                                      int device) {
    return hipCUDAErrorTohipError(cudaMemAdvise(dev_ptr, count,
        hipMemoryAdviseTocudaMemoryAdvise(advice), device));
}

inline static hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device,
                                             hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemPrefetchAsync(dev_ptr, count, device, stream));
}

inline static hipError_t hipMemRangeGetAttribute(void* data, size_t data_size,
                                                 hipMemRangeAttribute attribute,
                                                 const void* dev_ptr, size_t count) {
    return hipCUDAErrorTohipError(cudaMemRangeGetAttribute(data, data_size,
        hipMemRangeAttributeTocudaMemRangeAttribute(attribute), dev_ptr, count));
}

inline static hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes,
                                                  hipMemRangeAttribute* attributes,
                                                  size_t num_attributes, const void* dev_ptr,
                                                  size_t count) {
    auto attrs = hipMemRangeAttributeTocudaMemRangeAttribute(*attributes);
    return hipCUDAErrorTohipError(cudaMemRangeGetAttributes(data, data_sizes, &attrs,
        num_attributes, dev_ptr, count));
}

inline static hipError_t hipStreamAttachMemAsync(hipStream_t stream, hipDeviceptr_t* dev_ptr,
                                                 size_t length __dparm(0),
                                                 unsigned int flags __dparm(hipMemAttachSingle)) {
    return hipCUDAErrorTohipError(cudaStreamAttachMemAsync(stream, dev_ptr, length, flags));
}

inline static hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaMallocManaged(ptr, size, flags));
}

inline static hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc,
                                        size_t width, size_t height,
                                        unsigned int flags __dparm(hipArrayDefault)) {
    return hipCUDAErrorTohipError(cudaMallocArray(array, desc, width, height, flags));
}

inline static hipError_t hipMalloc3DArray(hipArray** array, const hipChannelFormatDesc* desc,
                             hipExtent extent, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaMalloc3DArray(array, desc, extent, flags));
}

inline static hipError_t hipFreeArray(hipArray* array) {
    return hipCUDAErrorTohipError(cudaFreeArray(array));
}

inline static hipError_t hipHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaHostGetDevicePointer(devPtr, hostPtr, flags));
}

inline static hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
    return hipCUDAErrorTohipError(cudaHostGetFlags(flagsPtr, hostPtr));
}

inline static hipError_t hipHostRegister(void* ptr, size_t size, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaHostRegister(ptr, size, flags));
}

inline static hipError_t hipHostUnregister(void* ptr) {
    return hipCUDAErrorTohipError(cudaHostUnregister(ptr));
}

inline static hipError_t hipFreeHost(void* ptr)
    __attribute__((deprecated("use hipHostFree instead")));
inline static hipError_t hipFreeHost(void* ptr) {
    return hipCUDAErrorTohipError(cudaFreeHost(ptr));
}

inline static hipError_t hipHostFree(void* ptr) {
    return hipCUDAErrorTohipError(cudaFreeHost(ptr));
}

inline static hipError_t hipSetDevice(int device) {
    return hipCUDAErrorTohipError(cudaSetDevice(device));
}

inline static hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop) {
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

inline static hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t size) {
    return hipCUResultTohipError(cuMemcpyHtoD(dst, src, size));
}

inline static hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t size) {
    return hipCUResultTohipError(cuMemcpyDtoH(dst, src, size));
}

inline static hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size) {
    return hipCUResultTohipError(cuMemcpyDtoD(dst, src, size));
}

inline static hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t size,
                                            hipStream_t stream) {
    return hipCUResultTohipError(cuMemcpyHtoDAsync(dst, src, size, stream));
}

inline static hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t size,
                                            hipStream_t stream) {
    return hipCUResultTohipError(cuMemcpyDtoHAsync(dst, src, size, stream));
}

inline static hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size,
                                            hipStream_t stream) {
    return hipCUResultTohipError(cuMemcpyDtoDAsync(dst, src, size, stream));
}

inline static hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                                   hipMemcpyKind copyKind) {
    return hipCUDAErrorTohipError(
        cudaMemcpy(dst, src, sizeBytes, hipMemcpyKindToCudaMemcpyKind(copyKind)));
}


inline static hipError_t hipMemcpyWithStream(void* dst, const void* src,
				      size_t sizeBytes, hipMemcpyKind copyKind,
				      hipStream_t stream) {
	cudaError_t error = cudaMemcpyAsync(dst, src, sizeBytes, 
										hipMemcpyKindToCudaMemcpyKind(copyKind),
										stream);
	
	if (error != cudaSuccess) return hipCUDAErrorTohipError(error);
	
	return hipCUDAErrorTohipError(cudaStreamSynchronize(stream));
}

inline static hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                                        hipMemcpyKind copyKind, hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(
        cudaMemcpyAsync(dst, src, sizeBytes, hipMemcpyKindToCudaMemcpyKind(copyKind), stream));
}

inline static hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes,
                                           size_t offset __dparm(0),
                                           hipMemcpyKind copyType __dparm(hipMemcpyHostToDevice)) {
    return hipCUDAErrorTohipError(cudaMemcpyToSymbol(symbol, src, sizeBytes, offset,
                                                     hipMemcpyKindToCudaMemcpyKind(copyType)));
}

inline static hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src,
                                                size_t sizeBytes, size_t offset,
                                                hipMemcpyKind copyType,
                                                hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemcpyToSymbolAsync(
        symbol, src, sizeBytes, offset, hipMemcpyKindToCudaMemcpyKind(copyType), stream));
}

inline static hipError_t hipMemcpyFromSymbol(void* dst, const void* symbolName, size_t sizeBytes,
                                             size_t offset __dparm(0),
                                             hipMemcpyKind kind __dparm(hipMemcpyDeviceToHost)) {
    return hipCUDAErrorTohipError(cudaMemcpyFromSymbol(dst, symbolName, sizeBytes, offset,
                                                       hipMemcpyKindToCudaMemcpyKind(kind)));
}

inline static hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbolName,
                                                  size_t sizeBytes, size_t offset,
                                                  hipMemcpyKind kind,
                                                  hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemcpyFromSymbolAsync(
        dst, symbolName, sizeBytes, offset, hipMemcpyKindToCudaMemcpyKind(kind), stream));
}

inline static hipError_t hipGetSymbolAddress(void** devPtr, const void* symbolName) {
    return hipCUDAErrorTohipError(cudaGetSymbolAddress(devPtr, symbolName));
}

inline static hipError_t hipGetSymbolSize(size_t* size, const void* symbolName) {
    return hipCUDAErrorTohipError(cudaGetSymbolSize(size, symbolName));
}

inline static hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                                     size_t width, size_t height, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(
        cudaMemcpy2D(dst, dpitch, src, spitch, width, height, hipMemcpyKindToCudaMemcpyKind(kind)));
}

inline static hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
  return hipCUResultTohipError(cuMemcpy2D(pCopy));
}

inline static hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t stream __dparm(0)) {
  return hipCUResultTohipError(cuMemcpy2DAsync(pCopy, stream));
}

inline static hipError_t hipMemcpy3D(const struct hipMemcpy3DParms *p) {
    return hipCUDAErrorTohipError(cudaMemcpy3D(p));
}

inline static hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms *p, hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaMemcpy3DAsync(p, stream));
}

inline static hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy) {
    return hipCUResultTohipError(cuMemcpy3D(pCopy));
}

inline static hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream) {
    return hipCUResultTohipError(cuMemcpy3DAsync(pCopy, stream));
}

inline static hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                                          size_t width, size_t height, hipMemcpyKind kind,
                                          hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                                    hipMemcpyKindToCudaMemcpyKind(kind), stream));
}

inline static hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray* src,
                                              size_t wOffset, size_t hOffset, size_t width,
                                              size_t height, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width,
                                                        height,
                                                        hipMemcpyKindToCudaMemcpyKind(kind)));
}

inline static hipError_t hipMemcpy2DFromArrayAsync(void* dst, size_t dpitch, hipArray* src,
                                                   size_t wOffset, size_t hOffset, size_t width,
                                                   size_t height, hipMemcpyKind kind,
                                                   hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset,
                                                             width, height,
                                                             hipMemcpyKindToCudaMemcpyKind(kind),
                                                             stream));
}

inline static hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset,
                                            const void* src, size_t spitch, size_t width,
                                            size_t height, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width,
                                                      height, hipMemcpyKindToCudaMemcpyKind(kind)));
}

inline static hipError_t hipMemcpy2DToArrayAsync(hipArray* dst, size_t wOffset, size_t hOffset,
                                                 const void* src, size_t spitch, size_t width,
                                                 size_t height, hipMemcpyKind kind,
                                                 hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch,
                                                           width, height,
                                                           hipMemcpyKindToCudaMemcpyKind(kind),
                                                           stream));
}

__HIP_DEPRECATED inline static hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset,
                                                           size_t hOffset, const void* src,
                                                           size_t count, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(
        cudaMemcpyToArray(dst, wOffset, hOffset, src, count, hipMemcpyKindToCudaMemcpyKind(kind)));
}

__HIP_DEPRECATED inline static hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray,
                                                             size_t wOffset, size_t hOffset,
                                                             size_t count, hipMemcpyKind kind) {
    return hipCUDAErrorTohipError(cudaMemcpyFromArray(dst, srcArray, wOffset, hOffset, count,
                                                      hipMemcpyKindToCudaMemcpyKind(kind)));
}

inline static hipError_t hipMemcpyAtoH(void* dst, hipArray* srcArray, size_t srcOffset,
                                       size_t count) {
    return hipCUResultTohipError(cuMemcpyAtoH(dst, (CUarray)srcArray, srcOffset, count));
}

inline static hipError_t hipMemcpyHtoA(hipArray* dstArray, size_t dstOffset, const void* srcHost,
                                       size_t count) {
    return hipCUResultTohipError(cuMemcpyHtoA((CUarray)dstArray, dstOffset, srcHost, count));
}

inline static hipError_t hipDeviceSynchronize() {
    return hipCUDAErrorTohipError(cudaDeviceSynchronize());
}

inline static hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* pCacheConfig) {
    return hipCUDAErrorTohipError(cudaDeviceGetCacheConfig(pCacheConfig));
}

inline static hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value) {
    return hipCUDAErrorTohipError(cudaFuncSetAttribute(func, attr, value));
}

inline static hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
    return hipCUDAErrorTohipError(cudaDeviceSetCacheConfig(cacheConfig));
}

inline static hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config) {
    return hipCUDAErrorTohipError(cudaFuncSetSharedMemConfig(func, config));
}

inline static const char* hipGetErrorString(hipError_t error) {
    return cudaGetErrorString(hipErrorToCudaError(error));
}

inline static const char* hipGetErrorName(hipError_t error) {
    return cudaGetErrorName(hipErrorToCudaError(error));
}

inline static hipError_t hipGetDeviceCount(int* count) {
    return hipCUDAErrorTohipError(cudaGetDeviceCount(count));
}

inline static hipError_t hipGetDevice(int* device) {
    return hipCUDAErrorTohipError(cudaGetDevice(device));
}

inline static hipError_t hipIpcCloseMemHandle(void* devPtr) {
    return hipCUDAErrorTohipError(cudaIpcCloseMemHandle(devPtr));
}

inline static hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaIpcGetEventHandle(handle, event));
}

inline static hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
    return hipCUDAErrorTohipError(cudaIpcGetMemHandle(handle, devPtr));
}

inline static hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
    return hipCUDAErrorTohipError(cudaIpcOpenEventHandle(event, handle));
}

inline static hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle,
                                             unsigned int flags) {
    return hipCUDAErrorTohipError(cudaIpcOpenMemHandle(devPtr, handle, flags));
}

inline static hipError_t hipMemset(void* devPtr, int value, size_t count) {
    return hipCUDAErrorTohipError(cudaMemset(devPtr, value, count));
}

inline static hipError_t hipMemsetD32(hipDeviceptr_t devPtr, int value, size_t count) {
    return hipCUResultTohipError(cuMemsetD32(devPtr, value, count));
}

inline static hipError_t hipMemsetAsync(void* devPtr, int value, size_t count,
                                        hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemsetAsync(devPtr, value, count, stream));
}

inline static hipError_t hipMemsetD32Async(hipDeviceptr_t devPtr, int value, size_t count,
                                           hipStream_t stream __dparm(0)) {
    return hipCUResultTohipError(cuMemsetD32Async(devPtr, value, count, stream));
}

inline static hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t sizeBytes) {
    return hipCUResultTohipError(cuMemsetD8(dest, value, sizeBytes));
}

inline static hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t sizeBytes,
                                          hipStream_t stream __dparm(0)) {
    return hipCUResultTohipError(cuMemsetD8Async(dest, value, sizeBytes, stream));
}

inline static hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t sizeBytes) {
    return hipCUResultTohipError(cuMemsetD16(dest, value, sizeBytes));
}

inline static hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t sizeBytes,
                                           hipStream_t stream __dparm(0)) {
    return hipCUResultTohipError(cuMemsetD16Async(dest, value, sizeBytes, stream));
}

inline static hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
    return hipCUDAErrorTohipError(cudaMemset2D(dst, pitch, value, width, height));
}

inline static hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(cudaMemset2DAsync(dst, pitch, value, width, height, stream));
}

inline static hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent ){
    return hipCUDAErrorTohipError(cudaMemset3D(pitchedDevPtr, value, extent));
}

inline static hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent, hipStream_t stream __dparm(0) ){
    return hipCUDAErrorTohipError(cudaMemset3DAsync(pitchedDevPtr, value, extent, stream));
}

inline static hipError_t hipGetDeviceProperties(hipDeviceProp_t* p_prop, int device) {
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
    p_prop->clockInstructionRate = cdprop.clockRate; // Same as clock-rate:

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
    p_prop->gcnArch = 0; // Not a GCN arch
    p_prop->integrated = cdprop.integrated;
    p_prop->cooperativeLaunch = cdprop.cooperativeLaunch;
    p_prop->cooperativeMultiDeviceLaunch = cdprop.cooperativeMultiDeviceLaunch;
    p_prop->cooperativeMultiDeviceUnmatchedFunc = 0;
    p_prop->cooperativeMultiDeviceUnmatchedGridDim = 0;
    p_prop->cooperativeMultiDeviceUnmatchedBlockDim = 0;
    p_prop->cooperativeMultiDeviceUnmatchedSharedMem = 0;

    p_prop->maxTexture1D    = cdprop.maxTexture1D;
    p_prop->maxTexture2D[0] = cdprop.maxTexture2D[0];
    p_prop->maxTexture2D[1] = cdprop.maxTexture2D[1];
    p_prop->maxTexture3D[0] = cdprop.maxTexture3D[0];
    p_prop->maxTexture3D[1] = cdprop.maxTexture3D[1];
    p_prop->maxTexture3D[2] = cdprop.maxTexture3D[2];

    p_prop->memPitch                 = cdprop.memPitch;
    p_prop->textureAlignment         = cdprop.textureAlignment;
    p_prop->texturePitchAlignment    = cdprop.texturePitchAlignment;
    p_prop->kernelExecTimeoutEnabled = cdprop.kernelExecTimeoutEnabled;
    p_prop->ECCEnabled               = cdprop.ECCEnabled;
    p_prop->tccDriver                = cdprop.tccDriver;

    return hipCUDAErrorTohipError(cerror);
}

inline static hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device) {
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
        case hipDeviceAttributeCooperativeLaunch:
            cdattr = cudaDevAttrCooperativeLaunch;
            break;
        case hipDeviceAttributeCooperativeMultiDeviceLaunch:
            cdattr = cudaDevAttrCooperativeMultiDeviceLaunch;
            break;
        default:
            return hipCUDAErrorTohipError(cudaErrorInvalidValue);
    }

    cerror = cudaDeviceGetAttribute(pi, cdattr, device);

    return hipCUDAErrorTohipError(cerror);
}

inline static hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                                      const void* func,
                                                                      int blockSize,
                                                                      size_t dynamicSMemSize) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func,
                                                              blockSize, dynamicSMemSize));
}

inline static hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                                      const void* func,
                                                                      int blockSize,
                                                                      size_t dynamicSMemSize,
                                                                      unsigned int flags) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func,
                                                      blockSize, dynamicSMemSize, flags));
}

inline static hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, 
                                                                 hipFunction_t f,
                                                                 int  blockSize,
                                                                 size_t dynamicSMemSize ){
    return hipCUResultTohipError(cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f,
                                                                   blockSize, dynamicSMemSize));
}

inline static hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                                          hipFunction_t f,
                                                                          int  blockSize,
                                                                          size_t dynamicSMemSize,
                                                                          unsigned int  flags ) {
    return hipCUResultTohipError(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks,f,
                                                                blockSize, dynamicSMemSize, flags));
}

//TODO - Match CUoccupancyB2DSize
inline static hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit){
    return hipCUResultTohipError(cuOccupancyMaxPotentialBlockSize(gridSize, blockSize, f, NULL,
                                 dynSharedMemPerBlk, blockSizeLimit));
}

//TODO - Match CUoccupancyB2DSize
inline static hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit, unsigned int  flags){
    return hipCUResultTohipError(cuOccupancyMaxPotentialBlockSizeWithFlags(gridSize, blockSize, f, NULL,
                                 dynSharedMemPerBlk, blockSizeLimit, flags));
}

inline static hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
    struct cudaPointerAttributes cPA;
    hipError_t err = hipCUDAErrorTohipError(cudaPointerGetAttributes(&cPA, ptr));
    if (err == hipSuccess) {
#if (CUDART_VERSION >= 11000)
        auto memType = cPA.type;
#else
        unsigned memType = cPA.memoryType; // No auto because cuda 10.2 doesnt force c++11
#endif
        switch (memType) {
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

inline static hipError_t hipMemGetInfo(size_t* free, size_t* total) {
    return hipCUDAErrorTohipError(cudaMemGetInfo(free, total));
}

inline static hipError_t hipEventCreate(hipEvent_t* event) {
    return hipCUDAErrorTohipError(cudaEventCreate(event));
}

inline static hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream __dparm(NULL)) {
    return hipCUDAErrorTohipError(cudaEventRecord(event, stream));
}

inline static hipError_t hipEventSynchronize(hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaEventSynchronize(event));
}

inline static hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    return hipCUDAErrorTohipError(cudaEventElapsedTime(ms, start, stop));
}

inline static hipError_t hipEventDestroy(hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaEventDestroy(event));
}

inline static hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaStreamCreateWithFlags(stream, flags));
}

inline static hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
    return hipCUDAErrorTohipError(cudaStreamCreateWithPriority(stream, flags, priority));
}

inline static hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    return hipCUDAErrorTohipError(cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority));
}

inline static hipError_t hipStreamCreate(hipStream_t* stream) {
    return hipCUDAErrorTohipError(cudaStreamCreate(stream));
}

inline static hipError_t hipStreamSynchronize(hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaStreamSynchronize(stream));
}

inline static hipError_t hipStreamDestroy(hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaStreamDestroy(stream));
}

inline static hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags) {
    return hipCUDAErrorTohipError(cudaStreamGetFlags(stream, flags));
}

inline static hipError_t hipStreamGetPriority(hipStream_t stream, int *priority) {
    return hipCUDAErrorTohipError(cudaStreamGetPriority(stream, priority));
}

inline static hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                                            unsigned int flags) {
    return hipCUDAErrorTohipError(cudaStreamWaitEvent(stream, event, flags));
}

inline static hipError_t hipStreamQuery(hipStream_t stream) {
    return hipCUDAErrorTohipError(cudaStreamQuery(stream));
}

inline static hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback,
                                              void* userData, unsigned int flags) {
    return hipCUDAErrorTohipError(
        cudaStreamAddCallback(stream, (cudaStreamCallback_t)callback, userData, flags));
}

inline static hipError_t hipDriverGetVersion(int* driverVersion) {
    cudaError_t err = cudaDriverGetVersion(driverVersion);

    // Override driver version to match version reported on HCC side.
    *driverVersion = 4;

    return hipCUDAErrorTohipError(err);
}

inline static hipError_t hipRuntimeGetVersion(int* runtimeVersion) {
    return hipCUDAErrorTohipError(cudaRuntimeGetVersion(runtimeVersion));
}

inline static hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) {
    return hipCUDAErrorTohipError(cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice));
}

inline static hipError_t hipDeviceDisablePeerAccess(int peerDevice) {
    return hipCUDAErrorTohipError(cudaDeviceDisablePeerAccess(peerDevice));
}

inline static hipError_t hipDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaDeviceEnablePeerAccess(peerDevice, flags));
}

inline static hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
    return hipCUResultTohipError(cuCtxDisablePeerAccess(peerCtx));
}

inline static hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
    return hipCUResultTohipError(cuCtxEnablePeerAccess(peerCtx, flags));
}

inline static hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags,
                                                     int* active) {
    return hipCUResultTohipError(cuDevicePrimaryCtxGetState(dev, flags, active));
}

inline static hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
    return hipCUResultTohipError(cuDevicePrimaryCtxRelease(dev));
}

inline static hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
    return hipCUResultTohipError(cuDevicePrimaryCtxRetain(pctx, dev));
}

inline static hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
    return hipCUResultTohipError(cuDevicePrimaryCtxReset(dev));
}

inline static hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
    return hipCUResultTohipError(cuDevicePrimaryCtxSetFlags(dev, flags));
}

inline static hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize,
                                               hipDeviceptr_t dptr) {
    return hipCUResultTohipError(cuMemGetAddressRange(pbase, psize, dptr));
}

inline static hipError_t hipMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice,
                                       size_t count) {
    return hipCUDAErrorTohipError(cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count));
}

inline static hipError_t hipMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                                            int srcDevice, size_t count,
                                            hipStream_t stream __dparm(0)) {
    return hipCUDAErrorTohipError(
        cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream));
}

// Profile APIs:
inline static hipError_t hipProfilerStart() { return hipCUDAErrorTohipError(cudaProfilerStart()); }

inline static hipError_t hipProfilerStop() { return hipCUDAErrorTohipError(cudaProfilerStop()); }

inline static hipError_t hipGetDeviceFlags(unsigned int* flags) {
    return hipCUDAErrorTohipError(cudaGetDeviceFlags(flags));
}

inline static hipError_t hipSetDeviceFlags(unsigned int flags) {
    return hipCUDAErrorTohipError(cudaSetDeviceFlags(flags));
}

inline static hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags) {
    return hipCUDAErrorTohipError(cudaEventCreateWithFlags(event, flags));
}

inline static hipError_t hipEventQuery(hipEvent_t event) {
    return hipCUDAErrorTohipError(cudaEventQuery(event));
}

inline static hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
    return hipCUResultTohipError(cuCtxCreate(ctx, flags, device));
}

inline static hipError_t hipCtxDestroy(hipCtx_t ctx) {
    return hipCUResultTohipError(cuCtxDestroy(ctx));
}

inline static hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
    return hipCUResultTohipError(cuCtxPopCurrent(ctx));
}

inline static hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
    return hipCUResultTohipError(cuCtxPushCurrent(ctx));
}

inline static hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
    return hipCUResultTohipError(cuCtxSetCurrent(ctx));
}

inline static hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
    return hipCUResultTohipError(cuCtxGetCurrent(ctx));
}

inline static hipError_t hipCtxGetDevice(hipDevice_t* device) {
    return hipCUResultTohipError(cuCtxGetDevice(device));
}

inline static hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion) {
    return hipCUResultTohipError(cuCtxGetApiVersion(ctx, (unsigned int*)apiVersion));
}

inline static hipError_t hipCtxGetCacheConfig(hipFuncCache* cacheConfig) {
    return hipCUResultTohipError(cuCtxGetCacheConfig(cacheConfig));
}

inline static hipError_t hipCtxSetCacheConfig(hipFuncCache cacheConfig) {
    return hipCUResultTohipError(cuCtxSetCacheConfig(cacheConfig));
}

inline static hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
    return hipCUResultTohipError(cuCtxSetSharedMemConfig((CUsharedconfig)config));
}

inline static hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig) {
    return hipCUResultTohipError(cuCtxGetSharedMemConfig((CUsharedconfig*)pConfig));
}

inline static hipError_t hipCtxSynchronize(void) {
    return hipCUResultTohipError(cuCtxSynchronize());
}

inline static hipError_t hipCtxGetFlags(unsigned int* flags) {
    return hipCUResultTohipError(cuCtxGetFlags(flags));
}

inline static hipError_t hipCtxDetach(hipCtx_t ctx) {
    return hipCUResultTohipError(cuCtxDetach(ctx));
}

inline static hipError_t hipDeviceGet(hipDevice_t* device, int ordinal) {
    return hipCUResultTohipError(cuDeviceGet(device, ordinal));
}

inline static hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device) {
    return hipCUResultTohipError(cuDeviceComputeCapability(major, minor, device));
}

inline static hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device) {
    return hipCUResultTohipError(cuDeviceGetName(name, len, device));
}

inline static hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr,
                                                  int srcDevice, int dstDevice) {
    return hipCUDAErrorTohipError(cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice));
}

inline static hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, hipDevice_t device) {
    return hipCUDAErrorTohipError(cudaDeviceGetPCIBusId(pciBusId, len, device));
}

inline static hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId) {
    return hipCUDAErrorTohipError(cudaDeviceGetByPCIBusId(device, pciBusId));
}

inline static hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* config) {
    return hipCUDAErrorTohipError(cudaDeviceGetSharedMemConfig(config));
}

inline static hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config) {
    return hipCUDAErrorTohipError(cudaDeviceSetSharedMemConfig(config));
}

inline static hipError_t hipDeviceGetLimit(size_t* pValue, hipLimit_t limit) {
    return hipCUDAErrorTohipError(cudaDeviceGetLimit(pValue, limit));
}

inline static hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device) {
    return hipCUResultTohipError(cuDeviceTotalMem(bytes, device));
}

inline static hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
    return hipCUResultTohipError(cuModuleLoad(module, fname));
}

inline static hipError_t hipModuleUnload(hipModule_t hmod) {
    return hipCUResultTohipError(cuModuleUnload(hmod));
}

inline static hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                              const char* kname) {
    return hipCUResultTohipError(cuModuleGetFunction(function, module, kname));
}

inline static hipError_t hipModuleGetTexRef(hipTexRef* pTexRef, hipModule_t hmod, const char* name){
    hipCUResultTohipError(cuModuleGetTexRef(pTexRef, hmod, name));
}

inline static hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func) {
    return hipCUDAErrorTohipError(cudaFuncGetAttributes(attr, func));
}

inline static hipError_t hipFuncGetAttribute (int* value, hipFunction_attribute attrib, hipFunction_t hfunc) {
    return hipCUResultTohipError(cuFuncGetAttribute(value, attrib, hfunc));
}

inline static hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                                            const char* name) {
    return hipCUResultTohipError(cuModuleGetGlobal(dptr, bytes, hmod, name));
}

inline static hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
    return hipCUResultTohipError(cuModuleLoadData(module, image));
}

inline static hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                             unsigned int numOptions, hipJitOption* options,
                                             void** optionValues) {
    return hipCUResultTohipError(
        cuModuleLoadDataEx(module, image, numOptions, options, optionValues));
}

inline static hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks,
					 dim3 dimBlocks, void** args, size_t sharedMemBytes,
					 hipStream_t stream)
{
   return hipCUDAErrorTohipError(cudaLaunchKernel(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream));
}

inline static hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                               unsigned int gridDimY, unsigned int gridDimZ,
                                               unsigned int blockDimX, unsigned int blockDimY,
                                               unsigned int blockDimZ, unsigned int sharedMemBytes,
                                               hipStream_t stream, void** kernelParams,
                                               void** extra) {
    return hipCUResultTohipError(cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                                blockDimY, blockDimZ, sharedMemBytes, stream,
                                                kernelParams, extra));
}

inline static hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t cacheConfig) {
    return hipCUDAErrorTohipError(cudaFuncSetCacheConfig(func, cacheConfig));
}

__HIP_DEPRECATED inline static hipError_t hipBindTexture(size_t* offset,
                                                         struct textureReference* tex,
                                                         const void* devPtr,
                                                         const hipChannelFormatDesc* desc,
                                                         size_t size __dparm(UINT_MAX)) {
    return hipCUDAErrorTohipError(cudaBindTexture(offset, tex, devPtr, desc, size));
}

__HIP_DEPRECATED inline static hipError_t hipBindTexture2D(
    size_t* offset, struct textureReference* tex, const void* devPtr,
    const hipChannelFormatDesc* desc, size_t width, size_t height, size_t pitch) {
    return hipCUDAErrorTohipError(cudaBindTexture2D(offset, tex, devPtr, desc, width, height, pitch));
}

inline static hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w,
                                                        hipChannelFormatKind f) {
    return cudaCreateChannelDesc(x, y, z, w, hipChannelFormatKindToCudaChannelFormatKind(f));
}

inline static hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject,
                                                const hipResourceDesc* pResDesc,
                                                const hipTextureDesc* pTexDesc,
                                                const hipResourceViewDesc* pResViewDesc) {
    return hipCUDAErrorTohipError(
        cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc));
}

inline static hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
    return hipCUDAErrorTohipError(cudaDestroyTextureObject(textureObject));
}

inline static hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject,
                                                const hipResourceDesc* pResDesc) {
    return hipCUDAErrorTohipError(cudaCreateSurfaceObject(pSurfObject, pResDesc));
}

inline static hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject) {
    return hipCUDAErrorTohipError(cudaDestroySurfaceObject(surfaceObject));
}

inline static hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject) {
    return hipCUDAErrorTohipError(cudaGetTextureObjectResourceDesc( pResDesc, textureObject));
}

__HIP_DEPRECATED inline static hipError_t hipGetTextureAlignmentOffset(
    size_t* offset, const struct textureReference* texref) {
    return hipCUDAErrorTohipError(cudaGetTextureAlignmentOffset(offset,texref));
}

inline static hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array)
{
    return hipCUDAErrorTohipError(cudaGetChannelDesc(desc,array));
}

inline static hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDim,
                                      void** kernelParams, unsigned int sharedMemBytes,
                                      hipStream_t stream) {
    return hipCUDAErrorTohipError(
            cudaLaunchCooperativeKernel(f, gridDim, blockDim, kernelParams, sharedMemBytes, stream));
}

inline static hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                 int  numDevices, unsigned int  flags) {
    return hipCUDAErrorTohipError(cudaLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags));
}

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__

template<class T>
inline static hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                                      T func,
                                                                      int blockSize,
                                                                      size_t dynamicSMemSize) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func,
                                                            blockSize, dynamicSMemSize));
}

template <class T>
inline static hipError_t hipOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func,
                                                           size_t dynamicSMemSize = 0,
                                                           int blockSizeLimit = 0) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func,
                                                           dynamicSMemSize, blockSizeLimit));
}

template <class T>
inline static hipError_t hipOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, T func,
                                                           size_t dynamicSMemSize = 0,
                                                           int blockSizeLimit = 0, unsigned int  flags = 0) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func,
                                                           dynamicSMemSize, blockSizeLimit, flags));
}

template <class T>
inline static hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags( int* numBlocks, T func,
                                              int  blockSize, size_t dynamicSMemSize,unsigned int flags) {
    return hipCUDAErrorTohipError(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func,
                                                                 blockSize, dynamicSMemSize, flags));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
inline static hipError_t hipBindTexture(size_t* offset, const struct texture<T, dim, readMode>& tex,
                                        const void* devPtr, size_t size = UINT_MAX) {
    return hipCUDAErrorTohipError(cudaBindTexture(offset, tex, devPtr, size));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
inline static hipError_t hipBindTexture(size_t* offset, struct texture<T, dim, readMode>& tex,
                                        const void* devPtr, const hipChannelFormatDesc& desc,
                                        size_t size = UINT_MAX) {
    return hipCUDAErrorTohipError(cudaBindTexture(offset, tex, devPtr, desc, size));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
__HIP_DEPRECATED inline static hipError_t hipUnbindTexture(struct texture<T, dim, readMode>* tex) {
    return hipCUDAErrorTohipError(cudaUnbindTexture(tex));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
__HIP_DEPRECATED inline static hipError_t hipUnbindTexture(struct texture<T, dim, readMode>& tex) {
    return hipCUDAErrorTohipError(cudaUnbindTexture(tex));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
__HIP_DEPRECATED inline static hipError_t hipBindTextureToArray(
    struct texture<T, dim, readMode>& tex, hipArray_const_t array,
    const hipChannelFormatDesc& desc) {
    return hipCUDAErrorTohipError(cudaBindTextureToArray(tex, array, desc));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
__HIP_DEPRECATED inline static hipError_t hipBindTextureToArray(
    struct texture<T, dim, readMode>* tex, hipArray_const_t array,
    const hipChannelFormatDesc* desc) {
    return hipCUDAErrorTohipError(cudaBindTextureToArray(tex, array, desc));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
__HIP_DEPRECATED inline static hipError_t hipBindTextureToArray(
    struct texture<T, dim, readMode>& tex, hipArray_const_t array) {
    return hipCUDAErrorTohipError(cudaBindTextureToArray(tex, array));
}

template <class T>
inline static hipChannelFormatDesc hipCreateChannelDesc() {
    return cudaCreateChannelDesc<T>();
}

template <class T>
inline static hipError_t hipLaunchCooperativeKernel(T f, dim3 gridDim, dim3 blockDim,
                                             void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream) {
    return hipCUDAErrorTohipError(
            cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(f), gridDim, blockDim, kernelParams, sharedMemBytes, stream));
}

inline static hipError_t hipTexRefSetAddressMode(hipTexRef hTexRef, int dim, hipAddress_mode am){
    return hipCUResultTohipError(cuTexRefSetAddressMode(hTexRef,dim,am));
}

inline static hipError_t hipTexRefSetFilterMode(hipTexRef hTexRef, hipFilter_mode fm){
    return hipCUResultTohipError(cuTexRefSetFilterMode(hTexRef,fm));
}

inline static hipError_t hipTexRefSetAddress(size_t *ByteOffset, hipTexRef hTexRef, hipDeviceptr_t dptr, size_t bytes){
   return hipCUResultTohipError(cuTexRefSetAddress(ByteOffset,hTexRef,dptr,bytes));
}

inline static hipError_t hipTexRefSetAddress2D(hipTexRef hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, hipDeviceptr_t dptr, size_t Pitch){
   return hipCUResultTohipError(cuTexRefSetAddress2D(hTexRef,desc,dptr,Pitch));
}

inline static hipError_t hipTexRefSetFormat(hipTexRef hTexRef, hipArray_Format fmt, int NumPackedComponents){
   return hipCUResultTohipError(cuTexRefSetFormat(hTexRef,fmt,NumPackedComponents));
}

inline static hipError_t hipTexRefSetFlags(hipTexRef hTexRef, unsigned int Flags){
   return hipCUResultTohipError(cuTexRefSetFlags(hTexRef,Flags));
}

inline static hipError_t hipTexRefSetArray(hipTexRef hTexRef, hiparray hArray, unsigned int Flags){
   return hipCUResultTohipError(cuTexRefSetArray(hTexRef,hArray,Flags));
}

inline static hipError_t hipArrayCreate(hiparray* pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray){
   return hipCUResultTohipError(cuArrayCreate(pHandle, pAllocateArray));
}

inline static hipError_t hipArrayDestroy(hiparray hArray){
   return hipCUResultTohipError(cuArrayDestroy(hArray));
}

inline static hipError_t hipArray3DCreate(hiparray* pHandle,
                                          const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray){
   return hipCUResultTohipError(cuArray3DCreate(pHandle, pAllocateArray));
}

#endif  //__CUDACC__

#endif  // HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_RUNTIME_API_H
