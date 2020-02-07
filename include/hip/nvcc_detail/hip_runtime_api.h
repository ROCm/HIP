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

#ifndef HIP_INCLUDE_HIP_NVCC_DETAIL_HIP_RUNTIME_API_H
#define HIP_INCLUDE_HIP_NVCC_DETAIL_HIP_RUNTIME_API_H

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
#define __dparm(x) = x
#else
#define __dparm(x)
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
#define hipChannelFormatKindSigned cudaChannelFormatKindSigned
#define hipChannelFormatKindUnsigned cudaChannelFormatKindUnsigned
#define hipChannelFormatKindFloat cudaChannelFormatKindFloat
#define hipChannelFormatKindNone cudaChannelFormatKindNone

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

#define hipHostRegisterDefault cudaHostRegisterDefault
#define hipHostRegisterPortable cudaHostRegisterPortable
#define hipHostRegisterMapped cudaHostRegisterMapped
#define hipHostRegisterIoMemory cudaHostRegisterIoMemory

#define HIP_LAUNCH_PARAM_BUFFER_POINTER CU_LAUNCH_PARAM_BUFFER_POINTER
#define HIP_LAUNCH_PARAM_BUFFER_SIZE CU_LAUNCH_PARAM_BUFFER_SIZE
#define HIP_LAUNCH_PARAM_END CU_LAUNCH_PARAM_END
#define hipLimitMallocHeapSize cudaLimitMallocHeapSize
#define hipIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess

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
typedef enum cudaFuncCache hipFuncCache_t;
typedef CUcontext hipCtx_t;
typedef enum cudaSharedMemConfig hipSharedMemConfig;
typedef CUfunc_cache hipFuncCache;
typedef CUjit_option hipJitOption;
typedef CUdevice hipDevice_t;
typedef CUmodule hipModule_t;
typedef CUfunction hipFunction_t;
typedef CUdeviceptr hipDeviceptr_t;
typedef struct cudaArray hipArray;
typedef struct cudaArray* hipArray_t;
typedef struct cudaArray* hipArray_const_t;
typedef struct cudaFuncAttributes hipFuncAttributes;
#define hipFunction_attribute CUfunction_attribute
#define hip_Memcpy2D CUDA_MEMCPY2D
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
#define hipDeviceMapHost cudaDeviceMapHost

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

// Function Attributes
#define HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_NUM_REGS CU_FUNC_ATTRIBUTE_NUM_REGS
#define HIP_FUNC_ATTRIBUTE_PTX_VERSION CU_FUNC_ATTRIBUTE_PTX_VERSION
#define HIP_FUNC_ATTRIBUTE_BINARY_VERSION CU_FUNC_ATTRIBUTE_BINARY_VERSION
#define HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA CU_FUNC_ATTRIBUTE_CACHE_MODE_CA
#define HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES                                           \
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT                                        \
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
#define HIP_FUNC_ATTRIBUTE_MAX CU_FUNC_ATTRIBUTE_MAX

#ifdef NVCC_INLINE_DISABLED

#define HIP_NVCC_INLINE

hipError_t hipCUDAErrorTohipError(cudaError_t cuError);

hipError_t hipCUResultTohipError(CUresult cuError);

// TODO   match the error enum names of hip and cuda
cudaError_t hipErrorToCudaError(hipError_t hError);

enum cudaMemcpyKind hipMemcpyKindToCudaMemcpyKind(hipMemcpyKind kind);

enum cudaTextureAddressMode hipTextureAddressModeToCudaTextureAddressMode(
    hipTextureAddressMode kind);

enum cudaTextureFilterMode hipTextureFilterModeToCudaTextureFilterMode(hipTextureFilterMode kind);

enum cudaTextureReadMode hipTextureReadModeToCudaTextureReadMode(hipTextureReadMode kind);

enum cudaChannelFormatKind hipChannelFormatKindToCudaChannelFormatKind(hipChannelFormatKind kind);

/**
 * Stream CallBack struct
 */
#define HIPRT_CB CUDART_CB
typedef void(HIPRT_CB* hipStreamCallback_t)(hipStream_t stream, hipError_t status, void* userData);
hipError_t hipInit(unsigned int flags);

hipError_t hipDeviceReset();

hipError_t hipGetLastError();

hipError_t hipPeekAtLastError();

hipError_t hipMalloc(void** ptr, size_t size);

hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);

hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height,
                            unsigned int elementSizeBytes);

hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent);

hipError_t hipFree(void* ptr);

hipError_t hipMallocHost(void** ptr, size_t size)
    __attribute__((deprecated("use hipHostMalloc instead")));

hipError_t hipMemAllocHost(void** ptr, size_t size)
    __attribute__((deprecated("use hipHostMalloc instead")));

hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags)
    __attribute__((deprecated("use hipHostMalloc instead")));

hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);

hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags);

hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc, size_t width,
                          size_t height, unsigned int flags __dparm(hipArrayDefault));

hipError_t hipMalloc3DArray(hipArray** array, const hipChannelFormatDesc* desc, hipExtent extent,
                            unsigned int flags);

hipError_t hipFreeArray(hipArray* array);

hipError_t hipHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags);

hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr);

hipError_t hipHostRegister(void* ptr, size_t size, unsigned int flags);

hipError_t hipHostUnregister(void* ptr);

hipError_t hipFreeHost(void* ptr) __attribute__((deprecated("use hipHostFree instead")));

hipError_t hipHostFree(void* ptr);

hipError_t hipSetDevice(int device);

hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop);

hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t size);

hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t size);

hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size);

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t size, hipStream_t stream);

hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t size, hipStream_t stream);

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size,
                              hipStream_t stream);

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind copyKind);

hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind copyKind,
                               hipStream_t stream);

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind copyKind,
                          hipStream_t stream);

hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes, size_t offset,
                             hipMemcpyKind copyType);

hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t sizeBytes,
                                  size_t offset, hipMemcpyKind copyType, hipStream_t stream);

hipError_t hipMemcpyFromSymbol(void* dst, const void* symbolName, size_t sizeBytes, size_t offset,
                               hipMemcpyKind kind);

hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbolName, size_t sizeBytes,
                                    size_t offset, hipMemcpyKind kind, hipStream_t stream);

hipError_t hipGetSymbolAddress(void** devPtr, const void* symbolName);

hipError_t hipGetSymbolSize(size_t* size, const void* symbolName);

hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind);

hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy);

hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t stream);

hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p);

hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream);

hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream);

hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind);

hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind);

hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind);

hipError_t hipMemcpyAtoH(void* dst, hipArray* srcArray, size_t srcOffset, size_t count);

hipError_t hipMemcpyHtoA(hipArray* dstArray, size_t dstOffset, const void* srcHost, size_t count);

hipError_t hipDeviceSynchronize();

hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* pCacheConfig);

hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig);

const char* hipGetErrorString(hipError_t error);

const char* hipGetErrorName(hipError_t error);

hipError_t hipGetDeviceCount(int* count);

hipError_t hipGetDevice(int* device);

hipError_t hipIpcCloseMemHandle(void* devPtr);

hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event);

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);

hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle);

hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags);

hipError_t hipMemset(void* devPtr, int value, size_t count);

hipError_t hipMemsetD32(hipDeviceptr_t devPtr, int value, size_t count);

hipError_t hipMemsetAsync(void* devPtr, int value, size_t count, hipStream_t stream);

hipError_t hipMemsetD32Async(hipDeviceptr_t devPtr, int value, size_t count, hipStream_t stream);

hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t sizeBytes);

hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t sizeBytes,
                            hipStream_t stream);

hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t sizeBytes);

hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t sizeBytes,
                             hipStream_t stream);

hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height);

hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height,
                            hipStream_t stream);

hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent);

hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                            hipStream_t stream);

hipError_t hipGetDeviceProperties(hipDeviceProp_t* p_prop, int device);

hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device);

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func,
                                                        int blockSize, size_t dynamicSMemSize);

hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr);

hipError_t hipMemGetInfo(size_t* free, size_t* total);

hipError_t hipEventCreate(hipEvent_t* event);

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);

hipError_t hipEventSynchronize(hipEvent_t event);

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);

hipError_t hipEventDestroy(hipEvent_t event);

hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);

hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority);

hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);

hipError_t hipStreamCreate(hipStream_t* stream);

hipError_t hipStreamSynchronize(hipStream_t stream);

hipError_t hipStreamDestroy(hipStream_t stream);

hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags);

hipError_t hipStreamGetPriority(hipStream_t stream, int* priority);

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);

hipError_t hipStreamQuery(hipStream_t stream);

hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags);

hipError_t hipDriverGetVersion(int* driverVersion);

hipError_t hipRuntimeGetVersion(int* runtimeVersion);

hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);

hipError_t hipDeviceDisablePeerAccess(int peerDevice);

hipError_t hipDeviceEnablePeerAccess(int peerDevice, unsigned int flags);

hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx);

hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags);

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active);

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);

hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);

hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr);

hipError_t hipMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count);

hipError_t hipMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice,
                              size_t count, hipStream_t stream);

// Profile APIs:
hipError_t hipProfilerStart();

hipError_t hipProfilerStop();

hipError_t hipSetDeviceFlags(unsigned int flags);

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags);

hipError_t hipEventQuery(hipEvent_t event);

hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);

hipError_t hipCtxDestroy(hipCtx_t ctx);

hipError_t hipCtxPopCurrent(hipCtx_t* ctx);

hipError_t hipCtxPushCurrent(hipCtx_t ctx);

hipError_t hipCtxSetCurrent(hipCtx_t ctx);

hipError_t hipCtxGetCurrent(hipCtx_t* ctx);

hipError_t hipCtxGetDevice(hipDevice_t* device);

hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion);

hipError_t hipCtxGetCacheConfig(hipFuncCache* cacheConfig);

hipError_t hipCtxSetCacheConfig(hipFuncCache cacheConfig);

hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config);

hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig);

hipError_t hipCtxSynchronize(void);

hipError_t hipCtxGetFlags(unsigned int* flags);

hipError_t hipCtxDetach(hipCtx_t ctx);

hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);

hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device);

hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);

hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, hipDevice_t device);

hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId);

hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* config);

hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config);

hipError_t hipDeviceGetLimit(size_t* pValue, hipLimit_t limit);

hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device);

hipError_t hipModuleLoad(hipModule_t* module, const char* fname);

hipError_t hipModuleUnload(hipModule_t hmod);

hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname);

hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func);

hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc);

hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                              const char* name);

hipError_t hipModuleLoadData(hipModule_t* module, const void* image);

hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions,
                               hipJitOption* options, void** optionValues);

hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                           void** args, size_t sharedMemBytes, hipStream_t stream);

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX,
                                 unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra);

hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t cacheConfig);

hipError_t hipBindTexture(size_t* offset, struct textureReference* tex, const void* devPtr,
                          const hipChannelFormatDesc* desc, size_t size);

hipError_t hipBindTexture2D(size_t* offset, struct textureReference* tex, const void* devPtr,
                            const hipChannelFormatDesc* desc, size_t width, size_t height,
                            size_t pitch);

hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f);

hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const hipResourceViewDesc* pResViewDesc);

hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject);

hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject, const hipResourceDesc* pResDesc);

hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject);

hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject);

hipError_t hipGetTextureAlignmentOffset(size_t* offset, const struct textureReference* texref);

hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array);

#endif

#ifdef __cplusplus
}
#endif

#ifdef NVCC_INLINE_DISABLED

#define HIP_NVCC_INLINE

#ifdef __CUDACC__

template <class T>
HIP_NVCC_INLINE hipError_t hipOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize,
                                                             T func, size_t dynamicSMemSize,
                                                             int blockSizeLimit) {
    cudaError_t cerror;
    cerror = cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, dynamicSMemSize,
                                                blockSizeLimit);
    return hipCUDAErrorTohipError(cerror);
}

template <class T, int dim, enum cudaTextureReadMode readMode>
HIP_NVCC_INLINE hipError_t hipBindTexture(size_t* offset,
                                          const struct texture<T, dim, readMode>& tex,
                                          const void* devPtr, size_t size) {
    return hipCUDAErrorTohipError(cudaBindTexture(offset, tex, devPtr, size));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
HIP_NVCC_INLINE hipError_t hipBindTexture(size_t* offset, struct texture<T, dim, readMode>& tex,
                                          const void* devPtr, const hipChannelFormatDesc& desc,
                                          size_t size) {
    return hipCUDAErrorTohipError(cudaBindTexture(offset, tex, devPtr, desc, size));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
HIP_NVCC_INLINE hipError_t hipUnbindTexture(struct texture<T, dim, readMode>* tex) {
    return hipCUDAErrorTohipError(cudaUnbindTexture(tex));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
HIP_NVCC_INLINE hipError_t hipUnbindTexture(struct texture<T, dim, readMode>& tex) {
    return hipCUDAErrorTohipError(cudaUnbindTexture(tex));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
HIP_NVCC_INLINE hipError_t hipBindTextureToArray(struct texture<T, dim, readMode>& tex,
                                                 hipArray_const_t array,
                                                 const hipChannelFormatDesc& desc) {
    return hipCUDAErrorTohipError(cudaBindTextureToArray(tex, array, desc));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
HIP_NVCC_INLINE hipError_t hipBindTextureToArray(struct texture<T, dim, readMode>* tex,
                                                 hipArray_const_t array,
                                                 const hipChannelFormatDesc* desc) {
    return hipCUDAErrorTohipError(cudaBindTextureToArray(tex, array, desc));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
HIP_NVCC_INLINE hipError_t hipBindTextureToArray(struct texture<T, dim, readMode>& tex,
                                                 hipArray_const_t array) {
    return hipCUDAErrorTohipError(cudaBindTextureToArray(tex, array));
}

template <class T>
HIP_NVCC_INLINE hipChannelFormatDesc hipCreateChannelDesc() {
    return cudaCreateChannelDesc<T>();
}

#endif  //__CUDACC__

#else

#define HIP_NVCC_INLINE static inline

#include "hip/nvcc_detail/hip_runtime_api.inl"

#endif


#endif  // HIP_INCLUDE_HIP_NVCC_DETAIL_HIP_RUNTIME_API_H
