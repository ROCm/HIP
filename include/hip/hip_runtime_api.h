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

/**
 * @file hip_runtime_api.h
 *
 * @brief Defines the API signatures for HIP runtime.
 * This file can be compiled with a standard compiler.
 */

#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_API_H
#define HIP_INCLUDE_HIP_HIP_RUNTIME_API_H


#include <string.h>  // for getDeviceProp
#include <hip/hip_version.h>
#include <hip/hip_common.h>

enum {
    HIP_SUCCESS = 0,
    HIP_ERROR_INVALID_VALUE,
    HIP_ERROR_NOT_INITIALIZED,
    HIP_ERROR_LAUNCH_OUT_OF_RESOURCES
};

typedef struct {
    // 32-bit Atomics
    unsigned hasGlobalInt32Atomics : 1;     ///< 32-bit integer atomics for global memory.
    unsigned hasGlobalFloatAtomicExch : 1;  ///< 32-bit float atomic exch for global memory.
    unsigned hasSharedInt32Atomics : 1;     ///< 32-bit integer atomics for shared memory.
    unsigned hasSharedFloatAtomicExch : 1;  ///< 32-bit float atomic exch for shared memory.
    unsigned hasFloatAtomicAdd : 1;  ///< 32-bit float atomic add in global and shared memory.

    // 64-bit Atomics
    unsigned hasGlobalInt64Atomics : 1;  ///< 64-bit integer atomics for global memory.
    unsigned hasSharedInt64Atomics : 1;  ///< 64-bit integer atomics for shared memory.

    // Doubles
    unsigned hasDoubles : 1;  ///< Double-precision floating point.

    // Warp cross-lane operations
    unsigned hasWarpVote : 1;     ///< Warp vote instructions (__any, __all).
    unsigned hasWarpBallot : 1;   ///< Warp ballot instructions (__ballot).
    unsigned hasWarpShuffle : 1;  ///< Warp shuffle operations. (__shfl_*).
    unsigned hasFunnelShift : 1;  ///< Funnel two words into one with shift&mask caps.

    // Sync
    unsigned hasThreadFenceSystem : 1;  ///< __threadfence_system.
    unsigned hasSyncThreadsExt : 1;     ///< __syncthreads_count, syncthreads_and, syncthreads_or.

    // Misc
    unsigned hasSurfaceFuncs : 1;        ///< Surface functions.
    unsigned has3dGrid : 1;              ///< Grid and group dims are 3D (rather than 2D).
    unsigned hasDynamicParallelism : 1;  ///< Dynamic parallelism.
} hipDeviceArch_t;


//---
// Common headers for both NVCC and HCC paths:

/**
 * hipDeviceProp
 *
 */
typedef struct hipDeviceProp_t {
    char name[256];            ///< Device name.
    size_t totalGlobalMem;     ///< Size of global memory region (in bytes).
    size_t sharedMemPerBlock;  ///< Size of shared memory region (in bytes).
    int regsPerBlock;          ///< Registers per block.
    int warpSize;              ///< Warp size.
    int maxThreadsPerBlock;    ///< Max work items per work group or workgroup max size.
    int maxThreadsDim[3];      ///< Max number of threads in each dimension (XYZ) of a block.
    int maxGridSize[3];        ///< Max grid dimensions (XYZ).
    int clockRate;             ///< Max clock frequency of the multiProcessors in khz.
    int memoryClockRate;       ///< Max global memory clock frequency in khz.
    int memoryBusWidth;        ///< Global memory bus width in bits.
    size_t totalConstMem;      ///< Size of shared memory region (in bytes).
    int major;  ///< Major compute capability.  On HCC, this is an approximation and features may
                ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                ///< feature caps.
    int minor;  ///< Minor compute capability.  On HCC, this is an approximation and features may
                ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                ///< feature caps.
    int multiProcessorCount;          ///< Number of multi-processors (compute units).
    int l2CacheSize;                  ///< L2 cache size.
    int maxThreadsPerMultiProcessor;  ///< Maximum resident threads per multi-processor.
    int computeMode;                  ///< Compute mode.
    int clockInstructionRate;  ///< Frequency in khz of the timer used by the device-side "clock*"
                               ///< instructions.  New for HIP.
    hipDeviceArch_t arch;      ///< Architectural feature flags.  New for HIP.
    int concurrentKernels;     ///< Device can possibly execute multiple kernels concurrently.
    int pciDomainID;           ///< PCI Domain ID
    int pciBusID;              ///< PCI Bus ID.
    int pciDeviceID;           ///< PCI Device ID.
    size_t maxSharedMemoryPerMultiProcessor;  ///< Maximum Shared Memory Per Multiprocessor.
    int isMultiGpuBoard;                      ///< 1 if device is on a multi-GPU board, 0 if not.
    int canMapHostMemory;                     ///< Check whether HIP can map host memory
    int gcnArch;                              ///< DEPRECATED: use gcnArchName instead
    char gcnArchName[256];                    ///< AMD GCN Arch Name.
    int integrated;            ///< APU vs dGPU
    int cooperativeLaunch;            ///< HIP device supports cooperative launch
    int cooperativeMultiDeviceLaunch; ///< HIP device supports cooperative launch on multiple devices
    int maxTexture1DLinear;    ///< Maximum size for 1D textures bound to linear memory
    int maxTexture1D;          ///< Maximum number of elements in 1D images
    int maxTexture2D[2];       ///< Maximum dimensions (width, height) of 2D images, in image elements
    int maxTexture3D[3];       ///< Maximum dimensions (width, height, depth) of 3D images, in image elements
    unsigned int* hdpMemFlushCntl;      ///< Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
    unsigned int* hdpRegFlushCntl;      ///< Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
    size_t memPitch;                 ///<Maximum pitch in bytes allowed by memory copies
    size_t textureAlignment;         ///<Alignment requirement for textures
    size_t texturePitchAlignment;    ///<Pitch alignment requirement for texture references bound to pitched memory
    int kernelExecTimeoutEnabled;    ///<Run time limit for kernels executed on the device
    int ECCEnabled;                  ///<Device has ECC support enabled
    int tccDriver;                   ///< 1:If device is Tesla device using TCC driver, else 0
    int cooperativeMultiDeviceUnmatchedFunc;        ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched functions
    int cooperativeMultiDeviceUnmatchedGridDim;     ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched grid dimensions
    int cooperativeMultiDeviceUnmatchedBlockDim;    ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched block dimensions
    int cooperativeMultiDeviceUnmatchedSharedMem;   ///< HIP device supports cooperative launch on multiple
                                                    ///devices with unmatched shared memories
    int isLargeBar;                  ///< 1: if it is a large PCI bar device, else 0
    int asicRevision;                ///< Revision of the GPU in this device
    int managedMemory;               ///< Device supports allocating managed memory on this system
    int directManagedMemAccessFromHost; ///< Host can directly access managed memory on the device without migration
    int concurrentManagedAccess;     ///< Device can coherently access managed memory concurrently with the CPU
    int pageableMemoryAccess;        ///< Device supports coherently accessing pageable memory
                                     ///< without calling hipHostRegister on it
    int pageableMemoryAccessUsesHostPageTables; ///< Device accesses pageable memory via the host's page tables
} hipDeviceProp_t;


/**
 * Memory type (for pointer attributes)
 */
typedef enum hipMemoryType {
    hipMemoryTypeHost,    ///< Memory is physically located on host
    hipMemoryTypeDevice,  ///< Memory is physically located on device. (see deviceId for specific
                          ///< device)
    hipMemoryTypeArray,  ///< Array memory, physically located on device. (see deviceId for specific
                         ///< device)
    hipMemoryTypeUnified  ///< Not used currently
}hipMemoryType;


/**
 * Pointer attributes
 */
typedef struct hipPointerAttribute_t {
    enum hipMemoryType memoryType;
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
    unsigned allocationFlags; /* flags specified when memory was allocated*/
    /* peers? */
} hipPointerAttribute_t;


// hack to get these to show up in Doxygen:
/**
 *     @defgroup GlobalDefs Global enum and defines
 *     @{
 *
 */

// Ignoring error-code return values from hip APIs is discouraged. On C++17,
// we can make that yield a warning
#if __cplusplus >= 201703L
#define __HIP_NODISCARD [[nodiscard]]
#else
#define __HIP_NODISCARD
#endif

/*
 * @brief hipError_t
 * @enum
 * @ingroup Enumerations
 */
// Developer note - when updating these, update the hipErrorName and hipErrorString functions in
// NVCC and HCC paths Also update the hipCUDAErrorTohipError function in NVCC path.

typedef enum __HIP_NODISCARD hipError_t {
    hipSuccess = 0,  ///< Successful completion.
    hipErrorInvalidValue = 1,  ///< One or more of the parameters passed to the API call is NULL
                               ///< or not in an acceptable range.
    hipErrorOutOfMemory = 2,
    // Deprecated
    hipErrorMemoryAllocation = 2,  ///< Memory allocation error.
    hipErrorNotInitialized = 3,
    // Deprecated
    hipErrorInitializationError = 3,
    hipErrorDeinitialized = 4,
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    hipErrorInvalidConfiguration = 9,
    hipErrorInvalidPitchValue = 12,
    hipErrorInvalidSymbol = 13,
    hipErrorInvalidDevicePointer = 17,  ///< Invalid Device Pointer
    hipErrorInvalidMemcpyDirection = 21,  ///< Invalid memory copy direction
    hipErrorInsufficientDriver = 35,
    hipErrorMissingConfiguration = 52,
    hipErrorPriorLaunchFailure = 53,
    hipErrorInvalidDeviceFunction = 98,
    hipErrorNoDevice = 100,  ///< Call to hipGetDeviceCount returned 0 devices
    hipErrorInvalidDevice = 101,  ///< DeviceID must be in range 0...#compute-devices.
    hipErrorInvalidImage = 200,
    hipErrorInvalidContext = 201,  ///< Produced when input context is invalid.
    hipErrorContextAlreadyCurrent = 202,
    hipErrorMapFailed = 205,
    // Deprecated
    hipErrorMapBufferObjectFailed = 205,  ///< Produced when the IPC memory attach failed from ROCr.
    hipErrorUnmapFailed = 206,
    hipErrorArrayIsMapped = 207,
    hipErrorAlreadyMapped = 208,
    hipErrorNoBinaryForGpu = 209,
    hipErrorAlreadyAcquired = 210,
    hipErrorNotMapped = 211,
    hipErrorNotMappedAsArray = 212,
    hipErrorNotMappedAsPointer = 213,
    hipErrorECCNotCorrectable = 214,
    hipErrorUnsupportedLimit = 215,
    hipErrorContextAlreadyInUse = 216,
    hipErrorPeerAccessUnsupported = 217,
    hipErrorInvalidKernelFile = 218,  ///< In CUDA DRV, it is CUDA_ERROR_INVALID_PTX
    hipErrorInvalidGraphicsContext = 219,
    hipErrorInvalidSource = 300,
    hipErrorFileNotFound = 301,
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed = 303,
    hipErrorOperatingSystem = 304,
    hipErrorInvalidHandle = 400,
    // Deprecated
    hipErrorInvalidResourceHandle = 400,  ///< Resource handle (hipEvent_t or hipStream_t) invalid.
    hipErrorNotFound = 500,
    hipErrorNotReady = 600,  ///< Indicates that asynchronous operations enqueued earlier are not
                             ///< ready.  This is not actually an error, but is used to distinguish
                             ///< from hipSuccess (which indicates completion).  APIs that return
                             ///< this error include hipEventQuery and hipStreamQuery.
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701,  ///< Out of resources error.
    hipErrorLaunchTimeOut = 702,
    hipErrorPeerAccessAlreadyEnabled =
        704,  ///< Peer access was already enabled from the current device.
    hipErrorPeerAccessNotEnabled =
        705,  ///< Peer access was never enabled from the current device.
    hipErrorSetOnActiveProcess = 708,
    hipErrorAssert = 710,  ///< Produced when the kernel calls assert.
    hipErrorHostMemoryAlreadyRegistered =
        712,  ///< Produced when trying to lock a page-locked memory.
    hipErrorHostMemoryNotRegistered =
        713,  ///< Produced when trying to unlock a non-page-locked memory.
    hipErrorLaunchFailure =
        719,  ///< An exception occurred on the device while executing a kernel.
    hipErrorCooperativeLaunchTooLarge =
        720,  ///< This error indicates that the number of blocks launched per grid for a kernel
              ///< that was launched via cooperative launch APIs exceeds the maximum number of
              ///< allowed blocks for the current device
    hipErrorNotSupported = 801,  ///< Produced when the hip API is not supported/implemented
    hipErrorUnknown = 999,  //< Unknown error.
    // HSA Runtime Error Codes start here.
    hipErrorRuntimeMemory = 1052,  ///< HSA runtime memory call returned error.  Typically not seen
                                   ///< in production systems.
    hipErrorRuntimeOther = 1053,  ///< HSA runtime call other than memory returned error.  Typically
                                  ///< not seen in production systems.
    hipErrorTbd  ///< Marker that more error codes are needed.
} hipError_t;

#undef __HIP_NODISCARD

/*
 * @brief hipDeviceAttribute_t
 * @enum
 * @ingroup Enumerations
 */
typedef enum hipDeviceAttribute_t {
    hipDeviceAttributeMaxThreadsPerBlock,       ///< Maximum number of threads per block.
    hipDeviceAttributeMaxBlockDimX,             ///< Maximum x-dimension of a block.
    hipDeviceAttributeMaxBlockDimY,             ///< Maximum y-dimension of a block.
    hipDeviceAttributeMaxBlockDimZ,             ///< Maximum z-dimension of a block.
    hipDeviceAttributeMaxGridDimX,              ///< Maximum x-dimension of a grid.
    hipDeviceAttributeMaxGridDimY,              ///< Maximum y-dimension of a grid.
    hipDeviceAttributeMaxGridDimZ,              ///< Maximum z-dimension of a grid.
    hipDeviceAttributeMaxSharedMemoryPerBlock,  ///< Maximum shared memory available per block in
                                                ///< bytes.
    hipDeviceAttributeTotalConstantMemory,      ///< Constant memory size in bytes.
    hipDeviceAttributeWarpSize,                 ///< Warp size in threads.
    hipDeviceAttributeMaxRegistersPerBlock,  ///< Maximum number of 32-bit registers available to a
                                             ///< thread block. This number is shared by all thread
                                             ///< blocks simultaneously resident on a
                                             ///< multiprocessor.
    hipDeviceAttributeClockRate,             ///< Peak clock frequency in kilohertz.
    hipDeviceAttributeMemoryClockRate,       ///< Peak memory clock frequency in kilohertz.
    hipDeviceAttributeMemoryBusWidth,        ///< Global memory bus width in bits.
    hipDeviceAttributeMultiprocessorCount,   ///< Number of multiprocessors on the device.
    hipDeviceAttributeComputeMode,           ///< Compute mode that device is currently in.
    hipDeviceAttributeL2CacheSize,  ///< Size of L2 cache in bytes. 0 if the device doesn't have L2
                                    ///< cache.
    hipDeviceAttributeMaxThreadsPerMultiProcessor,  ///< Maximum resident threads per
                                                    ///< multiprocessor.
    hipDeviceAttributeComputeCapabilityMajor,       ///< Major compute capability version number.
    hipDeviceAttributeComputeCapabilityMinor,       ///< Minor compute capability version number.
    hipDeviceAttributeConcurrentKernels,  ///< Device can possibly execute multiple kernels
                                          ///< concurrently.
    hipDeviceAttributePciBusId,           ///< PCI Bus ID.
    hipDeviceAttributePciDeviceId,        ///< PCI Device ID.
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,  ///< Maximum Shared Memory Per
                                                         ///< Multiprocessor.
    hipDeviceAttributeIsMultiGpuBoard,                   ///< Multiple GPU devices.
    hipDeviceAttributeIntegrated,                        ///< iGPU
    hipDeviceAttributeCooperativeLaunch,                 ///< Support cooperative launch
    hipDeviceAttributeCooperativeMultiDeviceLaunch,      ///< Support cooperative launch on multiple devices
    hipDeviceAttributeMaxTexture1DWidth,    ///< Maximum number of elements in 1D images
    hipDeviceAttributeMaxTexture2DWidth,    ///< Maximum dimension width of 2D images in image elements
    hipDeviceAttributeMaxTexture2DHeight,   ///< Maximum dimension height of 2D images in image elements
    hipDeviceAttributeMaxTexture3DWidth,    ///< Maximum dimension width of 3D images in image elements
    hipDeviceAttributeMaxTexture3DHeight,   ///< Maximum dimensions height of 3D images in image elements
    hipDeviceAttributeMaxTexture3DDepth,    ///< Maximum dimensions depth of 3D images in image elements

    hipDeviceAttributeHdpMemFlushCntl,      ///< Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeHdpRegFlushCntl,      ///< Address of the HDP_REG_COHERENCY_FLUSH_CNTL register

    hipDeviceAttributeMaxPitch,             ///< Maximum pitch in bytes allowed by memory copies
    hipDeviceAttributeTextureAlignment,     ///<Alignment requirement for textures
    hipDeviceAttributeTexturePitchAlignment, ///<Pitch alignment requirement for 2D texture references bound to pitched memory;
    hipDeviceAttributeKernelExecTimeout,    ///<Run time limit for kernels executed on the device
    hipDeviceAttributeCanMapHostMemory,     ///<Device can map host memory into device address space
    hipDeviceAttributeEccEnabled,           ///<Device has ECC support enabled

    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,        ///< Supports cooperative launch on multiple
                                                                  ///devices with unmatched functions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,     ///< Supports cooperative launch on multiple
                                                                  ///devices with unmatched grid dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,    ///< Supports cooperative launch on multiple
                                                                  ///devices with unmatched block dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,   ///< Supports cooperative launch on multiple
                                                                  ///devices with unmatched shared memories
    hipDeviceAttributeAsicRevision,         ///< Revision of the GPU in this device
    hipDeviceAttributeManagedMemory,        ///< Device supports allocating managed memory on this system
    hipDeviceAttributeDirectManagedMemAccessFromHost, ///< Host can directly access managed memory on
                                                      /// the device without migration
    hipDeviceAttributeConcurrentManagedAccess,  ///< Device can coherently access managed memory
                                                /// concurrently with the CPU
    hipDeviceAttributePageableMemoryAccess,     ///< Device supports coherently accessing pageable memory
                                                /// without calling hipHostRegister on it
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables, ///< Device accesses pageable memory via
                                                              /// the host's page tables
    hipDeviceAttributeCanUseStreamWaitValue ///< '1' if Device supports hipStreamWaitValue32() and
                                            ///< hipStreamWaitValue64() , '0' otherwise.

} hipDeviceAttribute_t;

enum hipComputeMode {
    hipComputeModeDefault = 0,
    hipComputeModeExclusive = 1,
    hipComputeModeProhibited = 2,
    hipComputeModeExclusiveProcess = 3
};

/**
 *     @}
 */

#if (defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)) && !(defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__))
#include "hip/amd_detail/hip_runtime_api.h"
#elif !(defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)) && (defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__))
#include "hip/nvidia_detail/hip_runtime_api.h"
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif


/**
 * @brief: C++ wrapper for hipMalloc
 *
 * Perform automatic type conversion to eliminate need for excessive typecasting (ie void**)
 *
 * __HIP_DISABLE_CPP_FUNCTIONS__ macro can be defined to suppress these
 * wrappers. It is useful for applications which need to obtain decltypes of
 * HIP runtime APIs.
 *
 * @see hipMalloc
 */
#if defined(__cplusplus) && !defined(__HIP_DISABLE_CPP_FUNCTIONS__)
template <class T>
static inline hipError_t hipMalloc(T** devPtr, size_t size) {
    return hipMalloc((void**)devPtr, size);
}

// Provide an override to automatically typecast the pointer type from void**, and also provide a
// default for the flags.
template <class T>
static inline hipError_t hipHostMalloc(T** ptr, size_t size,
                                       unsigned int flags = hipHostMallocDefault) {
    return hipHostMalloc((void**)ptr, size, flags);
}

template <class T>
static inline hipError_t hipMallocManaged(T** devPtr, size_t size,
                                       unsigned int flags = hipMemAttachGlobal) {
    return hipMallocManaged((void**)devPtr, size, flags);
}
#endif

#endif
