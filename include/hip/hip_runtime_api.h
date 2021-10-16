/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
    hipErrorIllegalState = 401, ///< Resource required is not in a valid state to perform operation.
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
    hipErrorContextIsDestroyed = 709,
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
    hipErrorStreamCaptureUnsupported = 900,  ///< The operation is not permitted when the stream
                                             ///< is capturing.
    hipErrorStreamCaptureInvalidated = 901,  ///< The current capture sequence on the stream
                                             ///< has been invalidated due to a previous error.
    hipErrorStreamCaptureMerge = 902,  ///< The operation would have resulted in a merge of
                                       ///< two independent capture sequences.
    hipErrorStreamCaptureUnmatched = 903,  ///< The capture was not initiated in this stream.
    hipErrorStreamCaptureUnjoined = 904,  ///< The capture sequence contains a fork that was not
                                          ///< joined to the primary stream.
    hipErrorStreamCaptureIsolation = 905,  ///< A dependency would have been created which crosses
                                           ///< the capture sequence boundary. Only implicit
                                           ///< in-stream ordering dependencies  are allowed
                                           ///< to cross the boundary
    hipErrorStreamCaptureImplicit = 906,  ///< The operation would have resulted in a disallowed
                                          ///< implicit dependency on a current capture sequence
                                          ///< from hipStreamLegacy.
    hipErrorCapturedEvent = 907,  ///< The operation is not permitted on an event which was last
                                  ///< recorded in a capturing stream.
    hipErrorStreamCaptureWrongThread = 908,  ///< A stream capture sequence not initiated with
                                             ///< the hipStreamCaptureModeRelaxed argument to
                                             ///< hipStreamBeginCapture was passed to
                                             ///< hipStreamEndCapture in a different thread.
    hipErrorGraphExecUpdateFailure = 910,  ///< This error indicates that the graph update
                                           ///< not performed because it included changes which
                                           ///< violated constraintsspecific to instantiated graph
                                           ///< update.
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
    hipDeviceAttributeCudaCompatibleBegin = 0,

    hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin, ///< Whether ECC support is enabled.
    hipDeviceAttributeAccessPolicyMaxWindowSize,        ///< Cuda only. The maximum size of the window policy in bytes.
    hipDeviceAttributeAsyncEngineCount,                 ///< Cuda only. Asynchronous engines number.
    hipDeviceAttributeCanMapHostMemory,                 ///< Whether host memory can be mapped into device address space
    hipDeviceAttributeCanUseHostPointerForRegisteredMem,///< Cuda only. Device can access host registered memory
                                                        ///< at the same virtual address as the CPU
    hipDeviceAttributeClockRate,                        ///< Peak clock frequency in kilohertz.
    hipDeviceAttributeComputeMode,                      ///< Compute mode that device is currently in.
    hipDeviceAttributeComputePreemptionSupported,       ///< Cuda only. Device supports Compute Preemption.
    hipDeviceAttributeConcurrentKernels,                ///< Device can possibly execute multiple kernels concurrently.
    hipDeviceAttributeConcurrentManagedAccess,          ///< Device can coherently access managed memory concurrently with the CPU
    hipDeviceAttributeCooperativeLaunch,                ///< Support cooperative launch
    hipDeviceAttributeCooperativeMultiDeviceLaunch,     ///< Support cooperative launch on multiple devices
    hipDeviceAttributeDeviceOverlap,                    ///< Cuda only. Device can concurrently copy memory and execute a kernel.
                                                        ///< Deprecated. Use instead asyncEngineCount.
    hipDeviceAttributeDirectManagedMemAccessFromHost,   ///< Host can directly access managed memory on
                                                        ///< the device without migration
    hipDeviceAttributeGlobalL1CacheSupported,           ///< Cuda only. Device supports caching globals in L1
    hipDeviceAttributeHostNativeAtomicSupported,        ///< Cuda only. Link between the device and the host supports native atomic operations
    hipDeviceAttributeIntegrated,                       ///< Device is integrated GPU
    hipDeviceAttributeIsMultiGpuBoard,                  ///< Multiple GPU devices.
    hipDeviceAttributeKernelExecTimeout,                ///< Run time limit for kernels executed on the device
    hipDeviceAttributeL2CacheSize,                      ///< Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
    hipDeviceAttributeLocalL1CacheSupported,            ///< caching locals in L1 is supported
    hipDeviceAttributeLuid,                             ///< Cuda only. 8-byte locally unique identifier in 8 bytes. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeLuidDeviceNodeMask,               ///< Cuda only. Luid device node mask. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeComputeCapabilityMajor,           ///< Major compute capability version number.
    hipDeviceAttributeManagedMemory,                    ///< Device supports allocating managed memory on this system
    hipDeviceAttributeMaxBlocksPerMultiProcessor,       ///< Cuda only. Max block size per multiprocessor
    hipDeviceAttributeMaxBlockDimX,                     ///< Max block size in width.
    hipDeviceAttributeMaxBlockDimY,                     ///< Max block size in height.
    hipDeviceAttributeMaxBlockDimZ,                     ///< Max block size in depth.
    hipDeviceAttributeMaxGridDimX,                      ///< Max grid size  in width.
    hipDeviceAttributeMaxGridDimY,                      ///< Max grid size  in height.
    hipDeviceAttributeMaxGridDimZ,                      ///< Max grid size  in depth.
    hipDeviceAttributeMaxSurface1D,                     ///< Maximum size of 1D surface.
    hipDeviceAttributeMaxSurface1DLayered,              ///< Cuda only. Maximum dimensions of 1D layered surface.
    hipDeviceAttributeMaxSurface2D,                     ///< Maximum dimension (width, height) of 2D surface.
    hipDeviceAttributeMaxSurface2DLayered,              ///< Cuda only. Maximum dimensions of 2D layered surface.
    hipDeviceAttributeMaxSurface3D,                     ///< Maximum dimension (width, height, depth) of 3D surface.
    hipDeviceAttributeMaxSurfaceCubemap,                ///< Cuda only. Maximum dimensions of Cubemap surface.
    hipDeviceAttributeMaxSurfaceCubemapLayered,         ///< Cuda only. Maximum dimension of Cubemap layered surface.
    hipDeviceAttributeMaxTexture1DWidth,                ///< Maximum size of 1D texture.
    hipDeviceAttributeMaxTexture1DLayered,              ///< Cuda only. Maximum dimensions of 1D layered texture.
    hipDeviceAttributeMaxTexture1DLinear,               ///< Maximum number of elements allocatable in a 1D linear texture.
                                                        ///< Use cudaDeviceGetTexture1DLinearMaxWidth() instead on Cuda.
    hipDeviceAttributeMaxTexture1DMipmap,               ///< Cuda only. Maximum size of 1D mipmapped texture.
    hipDeviceAttributeMaxTexture2DWidth,                ///< Maximum dimension width of 2D texture.
    hipDeviceAttributeMaxTexture2DHeight,               ///< Maximum dimension hight of 2D texture.
    hipDeviceAttributeMaxTexture2DGather,               ///< Cuda only. Maximum dimensions of 2D texture if gather operations  performed.
    hipDeviceAttributeMaxTexture2DLayered,              ///< Cuda only. Maximum dimensions of 2D layered texture.
    hipDeviceAttributeMaxTexture2DLinear,               ///< Cuda only. Maximum dimensions (width, height, pitch) of 2D textures bound to pitched memory.
    hipDeviceAttributeMaxTexture2DMipmap,               ///< Cuda only. Maximum dimensions of 2D mipmapped texture.
    hipDeviceAttributeMaxTexture3DWidth,                ///< Maximum dimension width of 3D texture.
    hipDeviceAttributeMaxTexture3DHeight,               ///< Maximum dimension height of 3D texture.
    hipDeviceAttributeMaxTexture3DDepth,                ///< Maximum dimension depth of 3D texture.
    hipDeviceAttributeMaxTexture3DAlt,                  ///< Cuda only. Maximum dimensions of alternate 3D texture.
    hipDeviceAttributeMaxTextureCubemap,                ///< Cuda only. Maximum dimensions of Cubemap texture
    hipDeviceAttributeMaxTextureCubemapLayered,         ///< Cuda only. Maximum dimensions of Cubemap layered texture.
    hipDeviceAttributeMaxThreadsDim,                    ///< Maximum dimension of a block
    hipDeviceAttributeMaxThreadsPerBlock,               ///< Maximum number of threads per block.
    hipDeviceAttributeMaxThreadsPerMultiProcessor,      ///< Maximum resident threads per multiprocessor.
    hipDeviceAttributeMaxPitch,                         ///< Maximum pitch in bytes allowed by memory copies
    hipDeviceAttributeMemoryBusWidth,                   ///< Global memory bus width in bits.
    hipDeviceAttributeMemoryClockRate,                  ///< Peak memory clock frequency in kilohertz.
    hipDeviceAttributeComputeCapabilityMinor,           ///< Minor compute capability version number.
    hipDeviceAttributeMultiGpuBoardGroupID,             ///< Cuda only. Unique ID of device group on the same multi-GPU board
    hipDeviceAttributeMultiprocessorCount,              ///< Number of multiprocessors on the device.
    hipDeviceAttributeName,                             ///< Device name.
    hipDeviceAttributePageableMemoryAccess,             ///< Device supports coherently accessing pageable memory
                                                        ///< without calling hipHostRegister on it
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables, ///< Device accesses pageable memory via the host's page tables
    hipDeviceAttributePciBusId,                         ///< PCI Bus ID.
    hipDeviceAttributePciDeviceId,                      ///< PCI Device ID.
    hipDeviceAttributePciDomainID,                      ///< PCI Domain ID.
    hipDeviceAttributePersistingL2CacheMaxSize,         ///< Cuda11 only. Maximum l2 persisting lines capacity in bytes
    hipDeviceAttributeMaxRegistersPerBlock,             ///< 32-bit registers available to a thread block. This number is shared
                                                        ///< by all thread blocks simultaneously resident on a multiprocessor.
    hipDeviceAttributeMaxRegistersPerMultiprocessor,    ///< 32-bit registers available per block.
    hipDeviceAttributeReservedSharedMemPerBlock,        ///< Cuda11 only. Shared memory reserved by CUDA driver per block.
    hipDeviceAttributeMaxSharedMemoryPerBlock,          ///< Maximum shared memory available per block in bytes.
    hipDeviceAttributeSharedMemPerBlockOptin,           ///< Cuda only. Maximum shared memory per block usable by special opt in.
    hipDeviceAttributeSharedMemPerMultiprocessor,       ///< Cuda only. Shared memory available per multiprocessor.
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio, ///< Cuda only. Performance ratio of single precision to double precision.
    hipDeviceAttributeStreamPrioritiesSupported,        ///< Cuda only. Whether to support stream priorities.
    hipDeviceAttributeSurfaceAlignment,                 ///< Cuda only. Alignment requirement for surfaces
    hipDeviceAttributeTccDriver,                        ///< Cuda only. Whether device is a Tesla device using TCC driver
    hipDeviceAttributeTextureAlignment,                 ///< Alignment requirement for textures
    hipDeviceAttributeTexturePitchAlignment,            ///< Pitch alignment requirement for 2D texture references bound to pitched memory;
    hipDeviceAttributeTotalConstantMemory,              ///< Constant memory size in bytes.
    hipDeviceAttributeTotalGlobalMem,                   ///< Global memory available on devicice.
    hipDeviceAttributeUnifiedAddressing,                ///< Cuda only. An unified address space shared with the host.
    hipDeviceAttributeUuid,                             ///< Cuda only. Unique ID in 16 byte.
    hipDeviceAttributeWarpSize,                         ///< Warp size in threads.

    hipDeviceAttributeCudaCompatibleEnd = 9999,
    hipDeviceAttributeAmdSpecificBegin = 10000,

    hipDeviceAttributeClockInstructionRate = hipDeviceAttributeAmdSpecificBegin,  ///< Frequency in khz of the timer used by the device-side "clock*"
    hipDeviceAttributeArch,                                     ///< Device architecture
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,         ///< Maximum Shared Memory PerMultiprocessor.
    hipDeviceAttributeGcnArch,                                  ///< Device gcn architecture
    hipDeviceAttributeGcnArchName,                              ///< Device gcnArch name in 256 bytes
    hipDeviceAttributeHdpMemFlushCntl,                          ///< Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeHdpRegFlushCntl,                          ///< Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,      ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched functions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,   ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched grid dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,  ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched block dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem, ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched shared memories
    hipDeviceAttributeIsLargeBar,                               ///< Whether it is LargeBar
    hipDeviceAttributeAsicRevision,                             ///< Revision of the GPU in this device
    hipDeviceAttributeCanUseStreamWaitValue,                    ///< '1' if Device supports hipStreamWaitValue32() and
                                                                ///< hipStreamWaitValue64() , '0' otherwise.

    hipDeviceAttributeAmdSpecificEnd = 19999,
    hipDeviceAttributeVendorSpecificBegin = 20000,
    // Extended attributes for vendors
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

#include <stdint.h>
#include <stddef.h>
#ifndef GENERIC_GRID_LAUNCH
#define GENERIC_GRID_LAUNCH 1
#endif
#include <hip/amd_detail/host_defines.h>
#include <hip/driver_types.h>
#include <hip/texture_types.h>
#include <hip/surface_types.h>
#if defined(_MSC_VER)
#define DEPRECATED(msg) __declspec(deprecated(msg))
#else // !defined(_MSC_VER)
#define DEPRECATED(msg) __attribute__ ((deprecated(msg)))
#endif // !defined(_MSC_VER)
#define DEPRECATED_MSG "This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_deprecated_api_list.md"
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#define HIP_LAUNCH_PARAM_END ((void*)0x03)
#ifdef __cplusplus
  #define __dparm(x) \
          = x
#else
  #define __dparm(x)
#endif
#ifdef __GNUC__
#pragma GCC visibility push (default)
#endif
#ifdef __cplusplus
namespace hip_impl {
hipError_t hip_init();
}  // namespace hip_impl
#endif
// Structure definitions:
#ifdef __cplusplus
extern "C" {
#endif
//---
// API-visible structures
typedef struct ihipCtx_t* hipCtx_t;
// Note many APIs also use integer deviceIds as an alternative to the device pointer:
typedef int hipDevice_t;
typedef enum hipDeviceP2PAttr {
  hipDevP2PAttrPerformanceRank = 0,
  hipDevP2PAttrAccessSupported,
  hipDevP2PAttrNativeAtomicSupported,
  hipDevP2PAttrHipArrayAccessSupported
} hipDeviceP2PAttr;
typedef struct ihipStream_t* hipStream_t;
#define hipIpcMemLazyEnablePeerAccess 0
#define HIP_IPC_HANDLE_SIZE 64
typedef struct hipIpcMemHandle_st {
    char reserved[HIP_IPC_HANDLE_SIZE];
} hipIpcMemHandle_t;
typedef struct hipIpcEventHandle_st {
    char reserved[HIP_IPC_HANDLE_SIZE];
} hipIpcEventHandle_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;
typedef struct hipFuncAttributes {
    int binaryVersion;
    int cacheModeCA;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxDynamicSharedSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int preferredShmemCarveout;
    int ptxVersion;
    size_t sharedSizeBytes;
} hipFuncAttributes;
typedef struct ihipEvent_t* hipEvent_t;
enum hipLimit_t {
    hipLimitPrintfFifoSize = 0x01,
    hipLimitMallocHeapSize = 0x02,
};
/**
 * @addtogroup GlobalDefs More
 * @{
 */
//! Flags that can be used with hipStreamCreateWithFlags
#define hipStreamDefault                                                                           \
    0x00  ///< Default stream creation flags. These are used with hipStreamCreate().
#define hipStreamNonBlocking 0x01  ///< Stream does not implicitly synchronize with null stream
//! Flags that can be used with hipEventCreateWithFlags:
#define hipEventDefault 0x0  ///< Default flags
#define hipEventBlockingSync                                                                       \
    0x1  ///< Waiting will yield CPU.  Power-friendly and usage-friendly but may increase latency.
#define hipEventDisableTiming                                                                      \
    0x2  ///< Disable event's capability to record timing information.  May improve performance.
#define hipEventInterprocess 0x4  ///< Event can support IPC.  @warning - not supported in HIP.
#define hipEventReleaseToDevice                                                                    \
    0x40000000  /// < Use a device-scope release when recording this event.  This flag is useful to
                /// obtain more precise timings of commands between events.  The flag is a no-op on
                /// CUDA platforms.
#define hipEventReleaseToSystem                                                                    \
    0x80000000  /// < Use a system-scope release when recording this event.  This flag is
                /// useful to make non-coherent host memory visible to the host.  The flag is a
                /// no-op on CUDA platforms.
//! Flags that can be used with hipHostMalloc
#define hipHostMallocDefault 0x0
#define hipHostMallocPortable 0x1  ///< Memory is considered allocated by all contexts.
#define hipHostMallocMapped                                                                        \
    0x2  ///< Map the allocation into the address space for the current device.  The device pointer
         ///< can be obtained with #hipHostGetDevicePointer.
#define hipHostMallocWriteCombined 0x4
#define hipHostMallocNumaUser                                                                      \
    0x20000000  ///< Host memory allocation will follow numa policy set by user
#define hipHostMallocCoherent                                                                      \
    0x40000000  ///< Allocate coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific
                ///< allocation.
#define hipHostMallocNonCoherent                                                                   \
    0x80000000  ///< Allocate non-coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific
                ///< allocation.
#define hipMemAttachGlobal  0x01    ///< Memory can be accessed by any stream on any device
#define hipMemAttachHost    0x02    ///< Memory cannot be accessed by any stream on any device
#define hipMemAttachSingle  0x04    ///< Memory can only be accessed by a single stream on
                                    ///< the associated device
#define hipDeviceMallocDefault 0x0
#define hipDeviceMallocFinegrained 0x1  ///< Memory is allocated in fine grained region of device.
#define hipMallocSignalMemory 0x2       ///< Memory represents a HSA signal.
//! Flags that can be used with hipHostRegister
#define hipHostRegisterDefault 0x0   ///< Memory is Mapped and Portable
#define hipHostRegisterPortable 0x1  ///< Memory is considered registered by all contexts.
#define hipHostRegisterMapped                                                                      \
    0x2  ///< Map the allocation into the address space for the current device.  The device pointer
         ///< can be obtained with #hipHostGetDevicePointer.
#define hipHostRegisterIoMemory 0x4  ///< Not supported.
#define hipExtHostRegisterCoarseGrained 0x8  ///< Coarse Grained host memory lock
#define hipDeviceScheduleAuto 0x0  ///< Automatically select between Spin and Yield
#define hipDeviceScheduleSpin                                                                      \
    0x1  ///< Dedicate a CPU core to spin-wait.  Provides lowest latency, but burns a CPU core and
         ///< may consume more power.
#define hipDeviceScheduleYield                                                                     \
    0x2  ///< Yield the CPU to the operating system when waiting.  May increase latency, but lowers
         ///< power and is friendlier to other threads in the system.
#define hipDeviceScheduleBlockingSync 0x4
#define hipDeviceScheduleMask 0x7
#define hipDeviceMapHost 0x8
#define hipDeviceLmemResizeToMax 0x16
#define hipArrayDefault 0x00  ///< Default HIP array allocation flag
#define hipArrayLayered 0x01
#define hipArraySurfaceLoadStore 0x02
#define hipArrayCubemap 0x04
#define hipArrayTextureGather 0x08
#define hipOccupancyDefault 0x00
#define hipCooperativeLaunchMultiDeviceNoPreSync 0x01
#define hipCooperativeLaunchMultiDeviceNoPostSync 0x02
#define hipCpuDeviceId ((int)-1)
#define hipInvalidDeviceId ((int)-2)
// Flags that can be used with hipExtLaunch Set of APIs
#define hipExtAnyOrderLaunch 0x01  ///< AnyOrderLaunch of kernels
// Flags to be used with hipStreamWaitValue32 and hipStreamWaitValue64
#define hipStreamWaitValueGte 0x0
#define hipStreamWaitValueEq 0x1
#define hipStreamWaitValueAnd 0x2
#define hipStreamWaitValueNor 0x3
// Stream per thread
#define hipStreamPerThread ((hipStream_t)2) ///< Implicit stream per application thread
/*
 * @brief HIP Memory Advise values
 * @enum
 * @ingroup Enumerations
 */
typedef enum hipMemoryAdvise {
    hipMemAdviseSetReadMostly = 1,          ///< Data will mostly be read and only occassionally
                                            ///< be written to
    hipMemAdviseUnsetReadMostly = 2,        ///< Undo the effect of hipMemAdviseSetReadMostly
    hipMemAdviseSetPreferredLocation = 3,   ///< Set the preferred location for the data as
                                            ///< the specified device
    hipMemAdviseUnsetPreferredLocation = 4, ///< Clear the preferred location for the data
    hipMemAdviseSetAccessedBy = 5,          ///< Data will be accessed by the specified device,
                                            ///< so prevent page faults as much as possible
    hipMemAdviseUnsetAccessedBy = 6,        ///< Let HIP to decide on the page faulting policy
                                            ///< for the specified device
    hipMemAdviseSetCoarseGrain = 100,       ///< The default memory model is fine-grain. That allows
                                            ///< coherent operations between host and device, while
                                            ///< executing kernels. The coarse-grain can be used
                                            ///< for data that only needs to be coherent at dispatch
                                            ///< boundaries for better performance
    hipMemAdviseUnsetCoarseGrain = 101      ///< Restores cache coherency policy back to fine-grain
} hipMemoryAdvise;
/*
 * @brief HIP Coherency Mode
 * @enum
 * @ingroup Enumerations
 */
typedef enum hipMemRangeCoherencyMode {
    hipMemRangeCoherencyModeFineGrain = 0,      ///< Updates to memory with this attribute can be
                                                ///< done coherently from all devices
    hipMemRangeCoherencyModeCoarseGrain = 1,    ///< Writes to memory with this attribute can be
                                                ///< performed by a single device at a time
    hipMemRangeCoherencyModeIndeterminate = 2   ///< Memory region queried contains subregions with
                                                ///< both hipMemRangeCoherencyModeFineGrain and
                                                ///< hipMemRangeCoherencyModeCoarseGrain attributes
} hipMemRangeCoherencyMode;
/*
 * @brief HIP range attributes
 * @enum
 * @ingroup Enumerations
 */
typedef enum hipMemRangeAttribute {
    hipMemRangeAttributeReadMostly = 1,         ///< Whether the range will mostly be read and
                                                ///< only occassionally be written to
    hipMemRangeAttributePreferredLocation = 2,  ///< The preferred location of the range
    hipMemRangeAttributeAccessedBy = 3,         ///< Memory range has hipMemAdviseSetAccessedBy
                                                ///< set for the specified device
    hipMemRangeAttributeLastPrefetchLocation = 4,///< The last location to where the range was
                                                ///< prefetched
    hipMemRangeAttributeCoherencyMode = 100,    ///< Returns coherency mode
                                                ///< @ref hipMemRangeCoherencyMode for the range
} hipMemRangeAttribute;
/*
 * @brief hipJitOption
 * @enum
 * @ingroup Enumerations
 */
typedef enum hipJitOption {
    hipJitOptionMaxRegisters = 0,
    hipJitOptionThreadsPerBlock,
    hipJitOptionWallTime,
    hipJitOptionInfoLogBuffer,
    hipJitOptionInfoLogBufferSizeBytes,
    hipJitOptionErrorLogBuffer,
    hipJitOptionErrorLogBufferSizeBytes,
    hipJitOptionOptimizationLevel,
    hipJitOptionTargetFromContext,
    hipJitOptionTarget,
    hipJitOptionFallbackStrategy,
    hipJitOptionGenerateDebugInfo,
    hipJitOptionLogVerbose,
    hipJitOptionGenerateLineInfo,
    hipJitOptionCacheMode,
    hipJitOptionSm3xOpt,
    hipJitOptionFastCompile,
    hipJitOptionNumOptions
} hipJitOption;
/**
 * @warning On AMD devices and some Nvidia devices, these hints and controls are ignored.
 */
typedef enum hipFuncAttribute {
    hipFuncAttributeMaxDynamicSharedMemorySize = 8,
    hipFuncAttributePreferredSharedMemoryCarveout = 9,
    hipFuncAttributeMax
} hipFuncAttribute;
/**
 * @warning On AMD devices and some Nvidia devices, these hints and controls are ignored.
 */
typedef enum hipFuncCache_t {
    hipFuncCachePreferNone,    ///< no preference for shared memory or L1 (default)
    hipFuncCachePreferShared,  ///< prefer larger shared memory and smaller L1 cache
    hipFuncCachePreferL1,      ///< prefer larger L1 cache and smaller shared memory
    hipFuncCachePreferEqual,   ///< prefer equal size L1 cache and shared memory
} hipFuncCache_t;
/**
 * @warning On AMD devices and some Nvidia devices, these hints and controls are ignored.
 */
typedef enum hipSharedMemConfig {
    hipSharedMemBankSizeDefault,  ///< The compiler selects a device-specific value for the banking.
    hipSharedMemBankSizeFourByte,  ///< Shared mem is banked at 4-bytes intervals and performs best
                                   ///< when adjacent threads access data 4 bytes apart.
    hipSharedMemBankSizeEightByte  ///< Shared mem is banked at 8-byte intervals and performs best
                                   ///< when adjacent threads access data 4 bytes apart.
} hipSharedMemConfig;
/**
 * Struct for data in 3D
 *
 */
typedef struct dim3 {
    uint32_t x;  ///< x
    uint32_t y;  ///< y
    uint32_t z;  ///< z
#ifdef __cplusplus
    constexpr __host__ __device__ dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z){};
#endif
} dim3;
typedef struct hipLaunchParams_t {
    void* func;             ///< Device function symbol
    dim3 gridDim;           ///< Grid dimentions
    dim3 blockDim;          ///< Block dimentions
    void **args;            ///< Arguments
    size_t sharedMem;       ///< Shared memory
    hipStream_t stream;     ///< Stream identifier
} hipLaunchParams;
typedef enum hipExternalMemoryHandleType_enum {
  hipExternalMemoryHandleTypeOpaqueFd = 1,
  hipExternalMemoryHandleTypeOpaqueWin32 = 2,
  hipExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
  hipExternalMemoryHandleTypeD3D12Heap = 4,
  hipExternalMemoryHandleTypeD3D12Resource = 5,
  hipExternalMemoryHandleTypeD3D11Resource = 6,
  hipExternalMemoryHandleTypeD3D11ResourceKmt = 7,
} hipExternalMemoryHandleType;
typedef struct hipExternalMemoryHandleDesc_st {
  hipExternalMemoryHandleType type;
  union {
    int fd;
    struct {
      void *handle;
      const void *name;
    } win32;
  } handle;
  unsigned long long size;
  unsigned int flags;
} hipExternalMemoryHandleDesc;
typedef struct hipExternalMemoryBufferDesc_st {
  unsigned long long offset;
  unsigned long long size;
  unsigned int flags;
} hipExternalMemoryBufferDesc;
typedef void* hipExternalMemory_t;
typedef enum hipExternalSemaphoreHandleType_enum {
  hipExternalSemaphoreHandleTypeOpaqueFd = 1,
  hipExternalSemaphoreHandleTypeOpaqueWin32 = 2,
  hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
  hipExternalSemaphoreHandleTypeD3D12Fence = 4
} hipExternalSemaphoreHandleType;
typedef struct hipExternalSemaphoreHandleDesc_st {
  hipExternalSemaphoreHandleType type;
  union {
    int fd;
    struct {
      void* handle;
      const void* name;
    } win32;
  } handle;
  unsigned int flags;
} hipExternalSemaphoreHandleDesc;
typedef void* hipExternalSemaphore_t;
typedef struct hipExternalSemaphoreSignalParams_st {
  struct {
    struct {
      unsigned long long value;
    } fence;
    struct {
      unsigned long long key;
    } keyedMutex;
    unsigned int reserved[12];
  } params;
  unsigned int flags;
  unsigned int reserved[16];
} hipExternalSemaphoreSignalParams;
/**
 * External semaphore wait parameters, compatible with driver type
 */
typedef struct hipExternalSemaphoreWaitParams_st {
  struct {
    struct {
      unsigned long long value;
    } fence;
    struct {
      unsigned long long key;
      unsigned int timeoutMs;
    } keyedMutex;
    unsigned int reserved[10];
  } params;
  unsigned int flags;
  unsigned int reserved[16];
} hipExternalSemaphoreWaitParams;

#if __HIP_HAS_GET_PCH
/**
 * Internal use only. This API may change in the future
 * Pre-Compiled header for online compilation
 *
 */
    void __hipGetPCH(const char** pch, unsigned int*size);
#endif

/*
    * @brief HIP Devices used by current OpenGL Context.
    * @enum
    * @ingroup Enumerations
    */
typedef enum hipGLDeviceList {
    hipGLDeviceListAll = 1,           ///< All hip devices used by current OpenGL context.
    hipGLDeviceListCurrentFrame = 2,  ///< Hip devices used by current OpenGL context in current
                                    ///< frame
    hipGLDeviceListNextFrame = 3      ///< Hip devices used by current OpenGL context in next
                                    ///< frame.
} hipGLDeviceList;

/*
    * @brief HIP Access falgs for Interop resources.
    * @enum
    * @ingroup Enumerations
    */
typedef enum hipGraphicsRegisterFlags {
    hipGraphicsRegisterFlagsNone = 0,
    hipGraphicsRegisterFlagsReadOnly = 1,  ///< HIP will not write to this registered resource
    hipGraphicsRegisterFlagsWriteDiscard =
        2,  ///< HIP will only write and will not read from this registered resource
    hipGraphicsRegisterFlagsSurfaceLoadStore = 4,  ///< HIP will bind this resource to a surface
    hipGraphicsRegisterFlagsTextureGather =
        8  ///< HIP will perform texture gather operations on this registered resource
} hipGraphicsRegisterFlags;

typedef struct _hipGraphicsResource hipGraphicsResource;

typedef hipGraphicsResource* hipGraphicsResource_t;

// Doxygen end group GlobalDefs
/**  @} */
//-------------------------------------------------------------------------------------------------
// The handle allows the async commands to use the stream even if the parent hipStream_t goes
// out-of-scope.
// typedef class ihipStream_t * hipStream_t;
/*
 * Opaque structure allows the true event (pointed at by the handle) to remain "live" even if the
 * surrounding hipEvent_t goes out-of-scope. This is handy for cases where the hipEvent_t goes
 * out-of-scope but the true event is being written by some async queue or device */
// typedef struct hipEvent_t {
//    struct ihipEvent_t *_handle;
//} hipEvent_t;
/**
 *  @defgroup API HIP API
 *  @{
 *
 *  Defines the HIP API.  See the individual sections for more information.
 */
/**
 *  @defgroup Driver Initialization and Version
 *  @{
 *  This section describes the initializtion and version functions of HIP runtime API.
 *
 */
/**
 * @brief Explicitly initializes the HIP runtime.
 *
 * Most HIP APIs implicitly initialize the HIP runtime.
 * This API provides control over the timing of the initialization.
 */
// TODO-ctx - more description on error codes.
hipError_t hipInit(unsigned int flags);
/**
 * @brief Returns the approximate HIP driver version.
 *
 * @param [out] driverVersion
 *
 * @returns #hipSuccess, #hipErrorInavlidValue
 *
 * @warning The HIP feature set does not correspond to an exact CUDA SDK driver revision.
 * This function always set *driverVersion to 4 as an approximation though HIP supports
 * some features which were introduced in later CUDA SDK revisions.
 * HIP apps code should not rely on the driver revision number here and should
 * use arch feature flags to test device capabilities or conditional compilation.
 *
 * @see hipRuntimeGetVersion
 */
hipError_t hipDriverGetVersion(int* driverVersion);
/**
 * @brief Returns the approximate HIP Runtime version.
 *
 * @param [out] runtimeVersion
 *
 * @returns #hipSuccess, #hipErrorInavlidValue
 *
 * @warning The version definition of HIP runtime is different from CUDA.
 * On AMD platform, the function returns HIP runtime version,
 * while on NVIDIA platform, it returns CUDA runtime version.
 * And there is no mapping/correlation between HIP version and CUDA version.
 *
 * @see hipDriverGetVersion
 */
hipError_t hipRuntimeGetVersion(int* runtimeVersion);
/**
 * @brief Returns a handle to a compute device
 * @param [out] device
 * @param [in] ordinal
 *
 * @returns #hipSuccess, #hipErrorInavlidDevice
 */
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);
/**
 * @brief Returns the compute capability of the device
 * @param [out] major
 * @param [out] minor
 * @param [in] device
 *
 * @returns #hipSuccess, #hipErrorInavlidDevice
 */
hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device);
/**
 * @brief Returns an identifer string for the device.
 * @param [out] name
 * @param [in] len
 * @param [in] device
 *
 * @returns #hipSuccess, #hipErrorInavlidDevice
 */
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);
/**
 * @brief Returns a value for attr of link between two devices
 * @param [out] value
 * @param [in] attr
 * @param [in] srcDevice
 * @param [in] dstDevice
 *
 * @returns #hipSuccess, #hipErrorInavlidDevice
 */
hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr,
                                    int srcDevice, int dstDevice);
/**
 * @brief Returns a PCI Bus Id string for the device, overloaded to take int device ID.
 * @param [out] pciBusId
 * @param [in] len
 * @param [in] device
 *
 * @returns #hipSuccess, #hipErrorInavlidDevice
 */
hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device);
/**
 * @brief Returns a handle to a compute device.
 * @param [out] device handle
 * @param [in] PCI Bus ID
 *
 * @returns #hipSuccess, #hipErrorInavlidDevice, #hipErrorInvalidValue
 */
hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId);
/**
 * @brief Returns the total amount of memory on the device.
 * @param [out] bytes
 * @param [in] device
 *
 * @returns #hipSuccess, #hipErrorInavlidDevice
 */
hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device);
// doxygen end initialization
/**
 * @}
 */
/**
 *  @defgroup Device Device Management
 *  @{
 *  This section describes the device management functions of HIP runtime API.
 */
/**
 * @brief Waits on all active streams on current device
 *
 * When this command is invoked, the host thread gets blocked until all the commands associated
 * with streams associated with the device. HIP does not support multiple blocking modes (yet!).
 *
 * @returns #hipSuccess
 *
 * @see hipSetDevice, hipDeviceReset
 */
hipError_t hipDeviceSynchronize(void);
/**
 * @brief The state of current device is discarded and updated to a fresh state.
 *
 * Calling this function deletes all streams created, memory allocated, kernels running, events
 * created. Make sure that no other thread is using the device or streams, memory, kernels, events
 * associated with the current device.
 *
 * @returns #hipSuccess
 *
 * @see hipDeviceSynchronize
 */
hipError_t hipDeviceReset(void);
/**
 * @brief Set default device to be used for subsequent hip API calls from this thread.
 *
 * @param[in] deviceId Valid device in range 0...hipGetDeviceCount().
 *
 * Sets @p device as the default device for the calling host thread.  Valid device id's are 0...
 * (hipGetDeviceCount()-1).
 *
 * Many HIP APIs implicitly use the "default device" :
 *
 * - Any device memory subsequently allocated from this host thread (using hipMalloc) will be
 * allocated on device.
 * - Any streams or events created from this host thread will be associated with device.
 * - Any kernels launched from this host thread (using hipLaunchKernel) will be executed on device
 * (unless a specific stream is specified, in which case the device associated with that stream will
 * be used).
 *
 * This function may be called from any host thread.  Multiple host threads may use the same device.
 * This function does no synchronization with the previous or new device, and has very little
 * runtime overhead. Applications can use hipSetDevice to quickly switch the default device before
 * making a HIP runtime call which uses the default device.
 *
 * The default device is stored in thread-local-storage for each thread.
 * Thread-pool implementations may inherit the default device of the previous thread.  A good
 * practice is to always call hipSetDevice at the start of HIP coding sequency to establish a known
 * standard device.
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorDeviceAlreadyInUse
 *
 * @see hipGetDevice, hipGetDeviceCount
 */
hipError_t hipSetDevice(int deviceId);
/**
 * @brief Return the default device id for the calling host thread.
 *
 * @param [out] device *device is written with the default device
 *
 * HIP maintains an default device for each thread using thread-local-storage.
 * This device is used implicitly for HIP runtime APIs called by this thread.
 * hipGetDevice returns in * @p device the default device for the calling host thread.
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 * @see hipSetDevice, hipGetDevicesizeBytes
 */
hipError_t hipGetDevice(int* deviceId);
/**
 * @brief Return number of compute-capable devices.
 *
 * @param [output] count Returns number of compute-capable devices.
 *
 * @returns #hipSuccess, #hipErrorNoDevice
 *
 *
 * Returns in @p *count the number of devices that have ability to run compute commands.  If there
 * are no such devices, then @ref hipGetDeviceCount will return #hipErrorNoDevice. If 1 or more
 * devices can be found, then hipGetDeviceCount returns #hipSuccess.
 */
hipError_t hipGetDeviceCount(int* count);
/**
 * @brief Query for a specific device attribute.
 *
 * @param [out] pi pointer to value to return
 * @param [in] attr attribute to query
 * @param [in] deviceId which device to query for information
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 */
hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId);
/**
 * @brief Returns device properties.
 *
 * @param [out] prop written with device properties
 * @param [in]  deviceId which device to query for information
 *
 * @return #hipSuccess, #hipErrorInvalidDevice
 * @bug HCC always returns 0 for maxThreadsPerMultiProcessor
 * @bug HCC always returns 0 for regsPerBlock
 * @bug HCC always returns 0 for l2CacheSize
 *
 * Populates hipGetDeviceProperties with information for the specified device.
 */
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId);
/**
 * @brief Set L1/Shared cache partition.
 *
 * @param [in] cacheConfig
 *
 * @returns #hipSuccess, #hipErrorNotInitialized
 * Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
 * on those architectures.
 *
 */
hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig);
/**
 * @brief Set Cache configuration for a specific function
 *
 * @param [in] cacheConfig
 *
 * @returns #hipSuccess, #hipErrorNotInitialized
 * Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
 * on those architectures.
 *
 */
hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig);
/**
 * @brief Get Resource limits of current device
 *
 * @param [out] pValue
 * @param [in]  limit
 *
 * @returns #hipSuccess, #hipErrorUnsupportedLimit, #hipErrorInvalidValue
 * Note: Currently, only hipLimitMallocHeapSize is available
 *
 */
hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit);
/**
 * @brief Returns bank width of shared memory for current device
 *
 * @param [out] pConfig
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 *
 * Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 */
hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig);
/**
 * @brief Gets the flags set for current device
 *
 * @param [out] flags
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 */
hipError_t hipGetDeviceFlags(unsigned int* flags);
/**
 * @brief The bank width of shared memory on current device is set
 *
 * @param [in] config
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 *
 * Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 */
hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config);
/**
 * @brief The current device behavior is changed according the flags passed.
 *
 * @param [in] flags
 *
 * The schedule flags impact how HIP waits for the completion of a command running on a device.
 * hipDeviceScheduleSpin         : HIP runtime will actively spin in the thread which submitted the
 * work until the command completes.  This offers the lowest latency, but will consume a CPU core
 * and may increase power. hipDeviceScheduleYield        : The HIP runtime will yield the CPU to
 * system so that other tasks can use it.  This may increase latency to detect the completion but
 * will consume less power and is friendlier to other tasks in the system.
 * hipDeviceScheduleBlockingSync : On ROCm platform, this is a synonym for hipDeviceScheduleYield.
 * hipDeviceScheduleAuto         : Use a hueristic to select between Spin and Yield modes.  If the
 * number of HIP contexts is greater than the number of logical processors in the system, use Spin
 * scheduling.  Else use Yield scheduling.
 *
 *
 * hipDeviceMapHost              : Allow mapping host memory.  On ROCM, this is always allowed and
 * the flag is ignored. hipDeviceLmemResizeToMax      : @warning ROCm silently ignores this flag.
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorSetOnActiveProcess
 *
 *
 */
hipError_t hipSetDeviceFlags(unsigned flags);
/**
 * @brief Device which matches hipDeviceProp_t is returned
 *
 * @param [out] device ID
 * @param [in]  device properties pointer
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop);
/**
 * @brief Returns the link type and hop count between two devices
 *
 * @param [in] device1 Ordinal for device1
 * @param [in] device2 Ordinal for device2
 * @param [out] linktype Returns the link type (See hsa_amd_link_info_type_t) between the two devices
 * @param [out] hopcount Returns the hop count between the two devices
 *
 * Queries and returns the HSA link type and the hop count between the two specified devices.
 *
 * @returns #hipSuccess, #hipInvalidDevice, #hipErrorRuntimeOther
 */
hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype, uint32_t* hopcount);
// TODO: implement IPC apis
/**
 * @brief Gets an interprocess memory handle for an existing device memory
 *          allocation
 *
 * Takes a pointer to the base of an existing device memory allocation created
 * with hipMalloc and exports it for use in another process. This is a
 * lightweight operation and may be called multiple times on an allocation
 * without adverse effects.
 *
 * If a region of memory is freed with hipFree and a subsequent call
 * to hipMalloc returns memory with the same device address,
 * hipIpcGetMemHandle will return a unique handle for the
 * new memory.
 *
 * @param handle - Pointer to user allocated hipIpcMemHandle to return
 *                    the handle in.
 * @param devPtr - Base pointer to previously allocated device memory
 *
 * @returns
 * hipSuccess,
 * hipErrorInvalidHandle,
 * hipErrorOutOfMemory,
 * hipErrorMapFailed,
 *
 */
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);
/**
 * @brief Opens an interprocess memory handle exported from another process
 *          and returns a device pointer usable in the local process.
 *
 * Maps memory exported from another process with hipIpcGetMemHandle into
 * the current device address space. For contexts on different devices
 * hipIpcOpenMemHandle can attempt to enable peer access between the
 * devices as if the user called hipDeviceEnablePeerAccess. This behavior is
 * controlled by the hipIpcMemLazyEnablePeerAccess flag.
 * hipDeviceCanAccessPeer can determine if a mapping is possible.
 *
 * Contexts that may open hipIpcMemHandles are restricted in the following way.
 * hipIpcMemHandles from each device in a given process may only be opened
 * by one context per device per other process.
 *
 * Memory returned from hipIpcOpenMemHandle must be freed with
 * hipIpcCloseMemHandle.
 *
 * Calling hipFree on an exported memory region before calling
 * hipIpcCloseMemHandle in the importing context will result in undefined
 * behavior.
 *
 * @param devPtr - Returned device pointer
 * @param handle - hipIpcMemHandle to open
 * @param flags  - Flags for this operation. Must be specified as hipIpcMemLazyEnablePeerAccess
 *
 * @returns
 * hipSuccess,
 * hipErrorMapFailed,
 * hipErrorInvalidHandle,
 * hipErrorTooManyPeers
 *
 * @note No guarantees are made about the address returned in @p *devPtr.
 * In particular, multiple processes may not receive the same address for the same @p handle.
 *
 */
hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags);
/**
 * @brief Close memory mapped with hipIpcOpenMemHandle
 *
 * Unmaps memory returnd by hipIpcOpenMemHandle. The original allocation
 * in the exporting process as well as imported mappings in other processes
 * will be unaffected.
 *
 * Any resources used to enable peer access will be freed if this is the
 * last mapping using them.
 *
 * @param devPtr - Device pointer returned by hipIpcOpenMemHandle
 *
 * @returns
 * hipSuccess,
 * hipErrorMapFailed,
 * hipErrorInvalidHandle,
 *
 */
hipError_t hipIpcCloseMemHandle(void* devPtr);

/**
 * @brief Gets an opaque interprocess handle for an event.
 *
 * This opaque handle may be copied into other processes and opened with cudaIpcOpenEventHandle.
 * Then cudaEventRecord, cudaEventSynchronize, cudaStreamWaitEvent and cudaEventQuery may be used in
 * either process. Operations on the imported event after the exported event has been freed with hipEventDestroy
 * will result in undefined behavior.
 *
 * @param[out]  handle Pointer to cudaIpcEventHandle to return the opaque event handle
 * @param[in]   event  Event allocated with cudaEventInterprocess and cudaEventDisableTiming flags
 *
 * @returns #hipSuccess, #hipErrorInvalidConfiguration, #hipErrorInvalidValue
 *
 */
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event);

/**
 * @brief Opens an interprocess event handles.
 *
 * Opens an interprocess event handle exported from another process with cudaIpcGetEventHandle. The returned
 * hipEvent_t behaves like a locally created event with the hipEventDisableTiming flag specified. This event
 * need be freed with hipEventDestroy. Operations on the imported event after the exported event has been freed
 * with hipEventDestroy will result in undefined behavior. If the function is called within the same process where
 * handle is returned by hipIpcGetEventHandle, it will return hipErrorInvalidContext.
 *
 * @param[out]  event  Pointer to hipEvent_t to return the event
 * @param[in]   handle The opaque interprocess handle to open
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidContext
 *
 */
hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle);

// end doxygen Device
/**
 * @}
 */
/**
 *
 *  @defgroup Execution Execution Control
 *  @{
 *  This section describes the execution control functions of HIP runtime API.
 *
 */
/**
 * @brief Set attribute for a specific function
 *
 * @param [in] func;
 * @param [in] attr;
 * @param [in] value;
 *
 * @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
 *
 * Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 */
hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value);
/**
 * @brief Set Cache configuration for a specific function
 *
 * @param [in] config;
 *
 * @returns #hipSuccess, #hipErrorNotInitialized
 * Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
 * on those architectures.
 *
 */
hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t config);
/**
 * @brief Set shared memory configuation for a specific function
 *
 * @param [in] func
 * @param [in] config
 *
 * @returns #hipSuccess, #hipErrorInvalidDeviceFunction, #hipErrorInvalidValue
 *
 * Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 */
hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config);
//doxygen end execution
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Error Error Handling
 *  @{
 *  This section describes the error handling functions of HIP runtime API.
 */
/**
 * @brief Return last error returned by any HIP runtime API call and resets the stored error code to
 * #hipSuccess
 *
 * @returns return code from last HIP called from the active host thread
 *
 * Returns the last error that has been returned by any of the runtime calls in the same host
 * thread, and then resets the saved error to #hipSuccess.
 *
 * @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
 */
hipError_t hipGetLastError(void);
/**
 * @brief Return last error returned by any HIP runtime API call.
 *
 * @return #hipSuccess
 *
 * Returns the last error that has been returned by any of the runtime calls in the same host
 * thread. Unlike hipGetLastError, this function does not reset the saved error code.
 *
 * @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
 */
hipError_t hipPeekAtLastError(void);
/**
 * @brief Return name of the specified error code in text form.
 *
 * @param hip_error Error code to convert to name.
 * @return const char pointer to the NULL-terminated error name
 *
 * @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
 */
const char* hipGetErrorName(hipError_t hip_error);
/**
 * @brief Return handy text string message to explain the error which occurred
 *
 * @param hipError Error code to convert to string.
 * @return const char pointer to the NULL-terminated error string
 *
 * @warning : on HCC, this function returns the name of the error (same as hipGetErrorName)
 *
 * @see hipGetErrorName, hipGetLastError, hipPeakAtLastError, hipError_t
 */
const char* hipGetErrorString(hipError_t hipError);
// end doxygen Error
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Stream Stream Management
 *  @{
 *  This section describes the stream management functions of HIP runtime API.
 *  The following Stream APIs are not (yet) supported in HIP:
 *  - hipStreamAttachMemAsync is a nop
 */

/**
 * @brief Create an asynchronous stream.
 *
 * @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
 * newly created stream.
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
 * reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
 * the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
 * used by the stream, applicaiton must call hipStreamDestroy.
 *
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
hipError_t hipStreamCreate(hipStream_t* stream);
/**
 * @brief Create an asynchronous stream.
 *
 * @param[in, out] stream Pointer to new stream
 * @param[in ] flags to control stream creation.
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
 * reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
 * the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
 * used by the stream, applicaiton must call hipStreamDestroy. Flags controls behavior of the
 * stream.  See #hipStreamDefault, #hipStreamNonBlocking.
 *
 *
 * @see hipStreamCreate, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
/**
 * @brief Create an asynchronous stream with the specified priority.
 *
 * @param[in, out] stream Pointer to new stream
 * @param[in ] flags to control stream creation.
 * @param[in ] priority of the stream. Lower numbers represent higher priorities.
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * Create a new asynchronous stream with the specified priority.  @p stream returns an opaque handle
 * that can be used to reference the newly created stream in subsequent hipStream* commands.  The
 * stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
 * To release the memory used by the stream, applicaiton must call hipStreamDestroy. Flags controls
 * behavior of the stream.  See #hipStreamDefault, #hipStreamNonBlocking.
 *
 *
 * @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority);
/**
 * @brief Returns numerical values that correspond to the least and greatest stream priority.
 *
 * @param[in, out] leastPriority pointer in which value corresponding to least priority is returned.
 * @param[in, out] greatestPriority pointer in which value corresponding to greatest priority is returned.
 *
 * Returns in *leastPriority and *greatestPriority the numerical values that correspond to the least
 * and greatest stream priority respectively. Stream priorities follow a convention where lower numbers
 * imply greater priorities. The range of meaningful stream priorities is given by
 * [*greatestPriority, *leastPriority]. If the user attempts to create a stream with a priority value
 * that is outside the the meaningful range as specified by this API, the priority is automatically
 * clamped to within the valid range.
 */
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
/**
 * @brief Destroys the specified stream.
 *
 * @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
 * newly created stream.
 * @return #hipSuccess #hipErrorInvalidHandle
 *
 * Destroys the specified stream.
 *
 * If commands are still executing on the specified stream, some may complete execution before the
 * queue is deleted.
 *
 * The queue may be destroyed while some commands are still inflight, or may wait for all commands
 * queued to the stream before destroying it.
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamQuery, hipStreamWaitEvent,
 * hipStreamSynchronize
 */
hipError_t hipStreamDestroy(hipStream_t stream);
/**
 * @brief Return #hipSuccess if all of the operations in the specified @p stream have completed, or
 * #hipErrorNotReady if not.
 *
 * @param[in] stream stream to query
 *
 * @return #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle
 *
 * This is thread-safe and returns a snapshot of the current state of the queue.  However, if other
 * host threads are sending work to the stream, the status may change immediately after the function
 * is called.  It is typically used for debug.
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamSynchronize,
 * hipStreamDestroy
 */
hipError_t hipStreamQuery(hipStream_t stream);
/**
 * @brief Wait for all commands in stream to complete.
 *
 * @param[in] stream stream identifier.
 *
 * @return #hipSuccess, #hipErrorInvalidHandle
 *
 * This command is host-synchronous : the host will block until the specified stream is empty.
 *
 * This command follows standard null-stream semantics.  Specifically, specifying the null stream
 * will cause the command to wait for other streams on the same device to complete all pending
 * operations.
 *
 * This command honors the hipDeviceLaunchBlocking flag, which controls whether the wait is active
 * or blocking.
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamWaitEvent, hipStreamDestroy
 *
 */
hipError_t hipStreamSynchronize(hipStream_t stream);
/**
 * @brief Make the specified compute stream wait for an event
 *
 * @param[in] stream stream to make wait.
 * @param[in] event event to wait on
 * @param[in] flags control operation [must be 0]
 *
 * @return #hipSuccess, #hipErrorInvalidHandle
 *
 * This function inserts a wait operation into the specified stream.
 * All future work submitted to @p stream will wait until @p event reports completion before
 * beginning execution.
 *
 * This function only waits for commands in the current stream to complete.  Notably,, this function
 * does not impliciy wait for commands in the default stream to complete, even if the specified
 * stream is created with hipStreamNonBlocking = 0.
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamDestroy
 */
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);
/**
 * @brief Return flags associated with this stream.
 *
 * @param[in] stream stream to be queried
 * @param[in,out] flags Pointer to an unsigned integer in which the stream's flags are returned
 * @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
 *
 * @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
 *
 * Return flags associated with this stream in *@p flags.
 *
 * @see hipStreamCreateWithFlags
 */
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags);
/**
 * @brief Query the priority of a stream.
 *
 * @param[in] stream stream to be queried
 * @param[in,out] priority Pointer to an unsigned integer in which the stream's priority is returned
 * @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidHandle
 *
 * @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidHandle
 *
 * Query the priority of a stream. The priority is returned in in priority.
 *
 * @see hipStreamCreateWithFlags
 */
hipError_t hipStreamGetPriority(hipStream_t stream, int* priority);
/**
 * @brief Create an asynchronous stream with the specified CU mask.
 *
 * @param[in, out] stream Pointer to new stream
 * @param[in ] cuMaskSize Size of CU mask bit array passed in.
 * @param[in ] cuMask Bit-vector representing the CU mask. Each active bit represents using one CU.
 * The first 32 bits represent the first 32 CUs, and so on. If its size is greater than physical
 * CU number (i.e., multiProcessorCount member of hipDeviceProp_t), the extra elements are ignored.
 * It is user's responsibility to make sure the input is meaningful.
 * @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
 *
 * Create a new asynchronous stream with the specified CU mask.  @p stream returns an opaque handle
 * that can be used to reference the newly created stream in subsequent hipStream* commands.  The
 * stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
 * To release the memory used by the stream, application must call hipStreamDestroy.
 *
 *
 * @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t cuMaskSize, const uint32_t* cuMask);
/**
 * @brief Get CU mask associated with an asynchronous stream
 *
 * @param[in] stream stream to be queried
 * @param[in] cuMaskSize number of the block of memories (uint32_t *) allocated by user
 * @param[out] cuMask Pointer to a pre-allocated block of memories (uint32_t *) in which
 * the stream's CU mask is returned. The CU mask is returned in a chunck of 32 bits where
 * each active bit represents one active CU
 * @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
 *
 * @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t* cuMask);
/**
 * Stream CallBack struct
 */
typedef void (*hipStreamCallback_t)(hipStream_t stream, hipError_t status, void* userData);
/**
 * @brief Adds a callback to be called on the host after all currently enqueued
 * items in the stream have completed.  For each
 * hipStreamAddCallback call, a callback will be executed exactly once.
 * The callback will block later work in the stream until it is finished.
 * @param[in] stream   - Stream to add callback to
 * @param[in] callback - The function to call once preceding stream operations are complete
 * @param[in] userData - User specified data to be passed to the callback function
 * @param[in] flags    - Reserved for future use, must be 0
 * @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorNotSupported
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamQuery, hipStreamSynchronize,
 * hipStreamWaitEvent, hipStreamDestroy, hipStreamCreateWithPriority
 *
 */
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags);
// end doxygen Stream
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Stream Memory Operations
 *  @{
 *  This section describes Stream Memory Wait and Write functions of HIP runtime API.
 */
/**
 * @brief Enqueues a wait command to the stream.[BETA]
 *
 * @param [in] stream - Stream identifier
 * @param [in] ptr    - Pointer to memory object allocated using 'hipMallocSignalMemory' flag
 * @param [in] value  - Value to be used in compare operation
 * @param [in] flags  - Defines the compare operation, supported values are hipStreamWaitValueGte
 * hipStreamWaitValueEq, hipStreamWaitValueAnd and hipStreamWaitValueNor
 * @param [in] mask   - Mask to be applied on value at memory before it is compared with value,
 * default value is set to enable every bit
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
 * not execute until the defined wait condition is true.
 *
 * hipStreamWaitValueGte: waits until *ptr&mask >= value
 * hipStreamWaitValueEq : waits until *ptr&mask == value
 * hipStreamWaitValueAnd: waits until ((*ptr&mask) & value) != 0
 * hipStreamWaitValueNor: waits until ~((*ptr&mask) | (value&mask)) != 0
 *
 * @note when using 'hipStreamWaitValueNor', mask is applied on both 'value' and '*ptr'.
 *
 * @note Support for hipStreamWaitValue32 can be queried using 'hipDeviceGetAttribute()' and
 * 'hipDeviceAttributeCanUseStreamWaitValue' flag.
 *
 * @beta This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue64, hipStreamWriteValue64,
 * hipStreamWriteValue32, hipDeviceGetAttribute
 */
hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags,
                                uint32_t mask __dparm(0xFFFFFFFF));
/**
 * @brief Enqueues a wait command to the stream.[BETA]
 *
 * @param [in] stream - Stream identifier
 * @param [in] ptr    - Pointer to memory object allocated using 'hipMallocSignalMemory' flag
 * @param [in] value  - Value to be used in compare operation
 * @param [in] flags  - Defines the compare operation, supported values are hipStreamWaitValueGte
 * hipStreamWaitValueEq, hipStreamWaitValueAnd and hipStreamWaitValueNor.
 * @param [in] mask   - Mask to be applied on value at memory before it is compared with value
 * default value is set to enable every bit
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
 * not execute until the defined wait condition is true.
 *
 * hipStreamWaitValueGte: waits until *ptr&mask >= value
 * hipStreamWaitValueEq : waits until *ptr&mask == value
 * hipStreamWaitValueAnd: waits until ((*ptr&mask) & value) != 0
 * hipStreamWaitValueNor: waits until ~((*ptr&mask) | (value&mask)) != 0
 *
 * @note when using 'hipStreamWaitValueNor', mask is applied on both 'value' and '*ptr'.
 *
 * @note Support for hipStreamWaitValue64 can be queried using 'hipDeviceGetAttribute()' and
 * 'hipDeviceAttributeCanUseStreamWaitValue' flag.
 *
 * @beta This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue32, hipStreamWriteValue64,
 * hipStreamWriteValue32, hipDeviceGetAttribute
 */
hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags,
                                uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));
/**
 * @brief Enqueues a write command to the stream.[BETA]
 *
 * @param [in] stream - Stream identifier
 * @param [in] ptr    - Pointer to a GPU accessible memory object
 * @param [in] value  - Value to be written
 * @param [in] flags  - reserved, ignored for now, will be used in future releases
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * Enqueues a write command to the stream, write operation is performed after all earlier commands
 * on this stream have completed the execution.
 *
 * @beta This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
 * hipStreamWaitValue64
 */
hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags);
/**
 * @brief Enqueues a write command to the stream.[BETA]
 *
 * @param [in] stream - Stream identifier
 * @param [in] ptr    - Pointer to a GPU accessible memory object
 * @param [in] value  - Value to be written
 * @param [in] flags  - reserved, ignored for now, will be used in future releases
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * Enqueues a write command to the stream, write operation is performed after all earlier commands
 * on this stream have completed the execution.
 *
 * @beta This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
 * hipStreamWaitValue64
 */
hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags);
// end doxygen Stream Memory Operations
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Event Event Management
 *  @{
 *  This section describes the event management functions of HIP runtime API.
 */
/**
 * @brief Create an event with the specified flags
 *
 * @param[in,out] event Returns the newly created event.
 * @param[in] flags     Flags to control event behavior.  Valid values are #hipEventDefault,
 #hipEventBlockingSync, #hipEventDisableTiming, #hipEventInterprocess
 * #hipEventDefault : Default flag.  The event will use active synchronization and will support
 timing.  Blocking synchronization provides lowest possible latency at the expense of dedicating a
 CPU to poll on the event.
 * #hipEventBlockingSync : The event will use blocking synchronization : if hipEventSynchronize is
 called on this event, the thread will block until the event completes.  This can increase latency
 for the synchroniation but can result in lower power and more resources for other CPU threads.
 * #hipEventDisableTiming : Disable recording of timing information. Events created with this flag
 would not record profiling data and provide best performance if used for synchronization.
 * @warning On AMD platform, hipEventInterprocess support is under development.  Use of this flag
 will return an error.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
 #hipErrorLaunchFailure, #hipErrorOutOfMemory
 *
 * @see hipEventCreate, hipEventSynchronize, hipEventDestroy, hipEventElapsedTime
 */
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);
/**
 *  Create an event
 *
 * @param[in,out] event Returns the newly created event.
 *
 * @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
 * #hipErrorLaunchFailure, #hipErrorOutOfMemory
 *
 * @see hipEventCreateWithFlags, hipEventRecord, hipEventQuery, hipEventSynchronize,
 * hipEventDestroy, hipEventElapsedTime
 */
hipError_t hipEventCreate(hipEvent_t* event);
/**
 * @brief Record an event in the specified stream.
 *
 * @param[in] event event to record.
 * @param[in] stream stream in which to record event.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized,
 * #hipErrorInvalidHandle, #hipErrorLaunchFailure
 *
 * hipEventQuery() or hipEventSynchronize() must be used to determine when the event
 * transitions from "recording" (after hipEventRecord() is called) to "recorded"
 * (when timestamps are set, if requested).
 *
 * Events which are recorded in a non-NULL stream will transition to
 * from recording to "recorded" state when they reach the head of
 * the specified stream, after all previous
 * commands in that stream have completed executing.
 *
 * If hipEventRecord() has been previously called on this event, then this call will overwrite any
 * existing state in event.
 *
 * If this function is called on an event that is currently being recorded, results are undefined
 * - either outstanding recording may save state into the event, and the order is not guaranteed.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventSynchronize,
 * hipEventDestroy, hipEventElapsedTime
 *
 */
#ifdef __cplusplus
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = NULL);
#else
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
#endif
/**
 *  @brief Destroy the specified event.
 *
 *  @param[in] event Event to destroy.
 *  @returns #hipSuccess, #hipErrorNotInitialized, #hipErrorInvalidValue,
 * #hipErrorLaunchFailure
 *
 *  Releases memory associated with the event.  If the event is recording but has not completed
 * recording when hipEventDestroy() is called, the function will return immediately and the
 * completion_future resources will be released later, when the hipDevice is synchronized.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventSynchronize, hipEventRecord,
 * hipEventElapsedTime
 *
 * @returns #hipSuccess
 */
hipError_t hipEventDestroy(hipEvent_t event);
/**
 *  @brief Wait for an event to complete.
 *
 *  This function will block until the event is ready, waiting for all previous work in the stream
 * specified when event was recorded with hipEventRecord().
 *
 *  If hipEventRecord() has not been called on @p event, this function returns immediately.
 *
 *  TODO-hip- This function needs to support hipEventBlockingSync parameter.
 *
 *  @param[in] event Event on which to wait.
 *  @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized,
 * #hipErrorInvalidHandle, #hipErrorLaunchFailure
 *
 *  @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
 * hipEventElapsedTime
 */
hipError_t hipEventSynchronize(hipEvent_t event);
/**
 * @brief Return the elapsed time between two events.
 *
 * @param[out] ms : Return time between start and stop in ms.
 * @param[in]   start : Start event.
 * @param[in]   stop  : Stop event.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotReady, #hipErrorInvalidHandle,
 * #hipErrorNotInitialized, #hipErrorLaunchFailure
 *
 * Computes the elapsed time between two events. Time is computed in ms, with
 * a resolution of approximately 1 us.
 *
 * Events which are recorded in a NULL stream will block until all commands
 * on all other streams complete execution, and then record the timestamp.
 *
 * Events which are recorded in a non-NULL stream will record their timestamp
 * when they reach the head of the specified stream, after all previous
 * commands in that stream have completed executing.  Thus the time that
 * the event recorded may be significantly after the host calls hipEventRecord().
 *
 * If hipEventRecord() has not been called on either event, then #hipErrorInvalidHandle is
 * returned. If hipEventRecord() has been called on both events, but the timestamp has not yet been
 * recorded on one or both events (that is, hipEventQuery() would return #hipErrorNotReady on at
 * least one of the events), then #hipErrorNotReady is returned.
 *
 * Note, for HIP Events used in kernel dispatch using hipExtLaunchKernelGGL/hipExtLaunchKernel,
 * events passed in hipExtLaunchKernelGGL/hipExtLaunchKernel are not explicitly recorded and should
 * only be used to get elapsed time for that specific launch. In case events are used across
 * multiple dispatches, for example, start and stop events from different hipExtLaunchKernelGGL/
 * hipExtLaunchKernel calls, they will be treated as invalid unrecorded events, HIP will throw
 * error "hipErrorInvalidHandle" from hipEventElapsedTime.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
 * hipEventSynchronize
 */
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);
/**
 * @brief Query event status
 *
 * @param[in] event Event to query.
 * @returns #hipSuccess, #hipErrorNotReady, #hipErrorInvalidHandle, #hipErrorInvalidValue,
 * #hipErrorNotInitialized, #hipErrorLaunchFailure
 *
 * Query the status of the specified event.  This function will return #hipErrorNotReady if all
 * commands in the appropriate stream (specified to hipEventRecord()) have completed.  If that work
 * has not completed, or if hipEventRecord() was not called on the event, then #hipSuccess is
 * returned.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventRecord, hipEventDestroy,
 * hipEventSynchronize, hipEventElapsedTime
 */
hipError_t hipEventQuery(hipEvent_t event);
// end doxygen Events
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Memory Memory Management
 *  @{
 *  This section describes the memory management functions of HIP runtime API.
 *  The following CUDA APIs are not currently supported:
 *  - cudaMalloc3D
 *  - cudaMalloc3DArray
 *  - TODO - more 2D, 3D, array APIs here.
 *
 *
 */
/**
 *  @brief Return attributes for the specified pointer
 *
 *  @param[out] attributes for the specified pointer
 *  @param[in]  pointer to get attributes for
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @see hipGetDeviceCount, hipGetDevice, hipSetDevice, hipChooseDevice
 */
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr);

/**
 *  @brief Imports an external semaphore.
 *
 *  @param[out] extSem_out  External semaphores to be waited on
 *  @param[in] semHandleDesc Semaphore import handle descriptor
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @see
 */
hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out,
                                      const hipExternalSemaphoreHandleDesc* semHandleDesc);
/**
 *  @brief Signals a set of external semaphore objects.
 *
 *  @param[in] extSem_out  External semaphores to be waited on
 *  @param[in] paramsArray Array of semaphore parameters
 *  @param[in] numExtSems Number of semaphores to wait on
 *  @param[in] stream Stream to enqueue the wait operations in
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @see
 */
hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                            const hipExternalSemaphoreSignalParams* paramsArray,
                                            unsigned int numExtSems, hipStream_t stream);
/**
 *  @brief Waits on a set of external semaphore objects
 *
 *  @param[in] extSem_out  External semaphores to be waited on
 *  @param[in] paramsArray Array of semaphore parameters
 *  @param[in] numExtSems Number of semaphores to wait on
 *  @param[in] stream Stream to enqueue the wait operations in
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @see
 */
hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                              const hipExternalSemaphoreWaitParams* paramsArray,
                                              unsigned int numExtSems, hipStream_t stream);
/**
 *  @brief Destroys an external semaphore object and releases any references to the underlying resource. Any outstanding signals or waits must have completed before the semaphore is destroyed.
 *
 *  @param[in] extSem handle to an external memory object
 *
 *  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 *
 *  @see
 */
hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem);

/**
*  @brief Imports an external memory object.
*
*  @param[out] extMem_out  Returned handle to an external memory object
*  @param[in]  memHandleDesc Memory import handle descriptor
*
*  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
*
*  @see
*/
hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out, const hipExternalMemoryHandleDesc* memHandleDesc);
/**
*  @brief Maps a buffer onto an imported memory object.
*
*  @param[out] devPtr Returned device pointer to buffer
*  @param[in]  extMem  Handle to external memory object
*  @param[in]  bufferDesc  Buffer descriptor
*
*  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
*
*  @see
*/
hipError_t hipExternalMemoryGetMappedBuffer(void **devPtr, hipExternalMemory_t extMem, const hipExternalMemoryBufferDesc *bufferDesc);
/**
*  @brief Destroys an external memory object.
*
*  @param[in] extMem  External memory object to be destroyed
*
*  @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
*
*  @see
*/
hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem);
/**
 *  @brief Allocate memory on the default accelerator
 *
 *  @param[out] ptr Pointer to the allocated memory
 *  @param[in]  size Requested memory size
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
 *
 *  @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
 * hipHostFree, hipHostMalloc
 */
hipError_t hipMalloc(void** ptr, size_t size);
/**
 *  @brief Allocate memory on the default accelerator
 *
 *  @param[out] ptr Pointer to the allocated memory
 *  @param[in]  size Requested memory size
 *  @param[in]  flags Type of memory allocation
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory, #hipErrorInvalidValue (bad context, null *ptr)
 *
 *  @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
 * hipHostFree, hipHostMalloc
 */
hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags);
/**
 *  @brief Allocate pinned host memory [Deprecated]
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory
 *
 *  @deprecated use hipHostMalloc() instead
 */
DEPRECATED("use hipHostMalloc instead")
hipError_t hipMallocHost(void** ptr, size_t size);
/**
 *  @brief Allocate pinned host memory [Deprecated]
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory
 *
 *  @deprecated use hipHostMalloc() instead
 */
DEPRECATED("use hipHostMalloc instead")
hipError_t hipMemAllocHost(void** ptr, size_t size);
/**
 *  @brief Allocate device accessible page locked host memory
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size
 *  @param[in]  flags Type of host memory allocation
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory
 *
 *  @see hipSetDeviceFlags, hipHostFree
 */
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @addtogroup Memory Managed Memory
 *  @{
 *  @ingroup Memory
 *  This section describes the managed memory management functions of HIP runtime API.
 *
 */
/**
 * @brief Allocates memory that will be automatically managed by HIP.
 *
 * @param [out] dev_ptr - pointer to allocated device memory
 * @param [in]  size    - requested allocation size in bytes
 * @param [in]  flags   - must be either hipMemAttachGlobal or hipMemAttachHost
 *                        (defaults to hipMemAttachGlobal)
 *
 * @returns #hipSuccess, #hipErrorMemoryAllocation, #hipErrorNotSupported, #hipErrorInvalidValue
 */
hipError_t hipMallocManaged(void** dev_ptr,
                            size_t size,
                            unsigned int flags __dparm(hipMemAttachGlobal));
/**
 * @brief Prefetches memory to the specified destination device using HIP.
 *
 * @param [in] dev_ptr  pointer to be prefetched
 * @param [in] count    size in bytes for prefetching
 * @param [in] device   destination device to prefetch to
 * @param [in] stream   stream to enqueue prefetch operation
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipMemPrefetchAsync(const void* dev_ptr,
                               size_t count,
                               int device,
                               hipStream_t stream __dparm(0));
/**
 * @brief Advise about the usage of a given memory range to HIP.
 *
 * @param [in] dev_ptr  pointer to memory to set the advice for
 * @param [in] count    size in bytes of the memory range
 * @param [in] advice   advice to be applied for the specified memory range
 * @param [in] device   device to apply the advice for
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipMemAdvise(const void* dev_ptr,
                        size_t count,
                        hipMemoryAdvise advice,
                        int device);
/**
 * @brief Query an attribute of a given memory range in HIP.
 *
 * @param [in,out] data   a pointer to a memory location where the result of each
 *                        attribute query will be written to
 * @param [in] data_size  the size of data
 * @param [in] attribute  the attribute to query
 * @param [in] dev_ptr    start of the range to query
 * @param [in] count      size of the range to query
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipMemRangeGetAttribute(void* data,
                                   size_t data_size,
                                   hipMemRangeAttribute attribute,
                                   const void* dev_ptr,
                                   size_t count);
/**
 * @brief Query attributes of a given memory range in HIP.
 *
 * @param [in,out] data     a two-dimensional array containing pointers to memory locations
 *                          where the result of each attribute query will be written to
 * @param [in] data_sizes   an array, containing the sizes of each result
 * @param [in] attributes   the attribute to query
 * @param [in] num_attributes  an array of attributes to query (numAttributes and the number
 *                          of attributes in this array should match)
 * @param [in] dev_ptr      start of the range to query
 * @param [in] count        size of the range to query
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipMemRangeGetAttributes(void** data,
                                    size_t* data_sizes,
                                    hipMemRangeAttribute* attributes,
                                    size_t num_attributes,
                                    const void* dev_ptr,
                                    size_t count);
/**
 * @brief Attach memory to a stream asynchronously in HIP.
 *
 * @param [in] stream     - stream in which to enqueue the attach operation
 * @param [in] dev_ptr    - pointer to memory (must be a pointer to managed memory or
 *                          to a valid host-accessible region of system-allocated memory)
 * @param [in] length     - length of memory (defaults to zero)
 * @param [in] flags      - must be one of hipMemAttachGlobal, hipMemAttachHost or
 *                          hipMemAttachSingle (defaults to hipMemAttachSingle)
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipStreamAttachMemAsync(hipStream_t stream,
                                   void* dev_ptr,
                                   size_t length __dparm(0),
                                   unsigned int flags __dparm(hipMemAttachSingle));
// end doxygen Managed Memory
/**
 * @}
 */
/**
 *  @brief Allocate device accessible page locked host memory [Deprecated]
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size
 *  @param[in]  flags Type of host memory allocation
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory
 *
 *  @deprecated use hipHostMalloc() instead
 */
DEPRECATED("use hipHostMalloc instead")
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
/**
 *  @brief Get Device pointer from Host Pointer allocated through hipHostMalloc
 *
 *  @param[out] dstPtr Device Pointer mapped to passed host pointer
 *  @param[in]  hstPtr Host Pointer allocated through hipHostMalloc
 *  @param[in]  flags Flags to be passed for extension
 *
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorOutOfMemory
 *
 *  @see hipSetDeviceFlags, hipHostMalloc
 */
hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
/**
 *  @brief Return flags associated with host pointer
 *
 *  @param[out] flagsPtr Memory location to store flags
 *  @param[in]  hostPtr Host Pointer allocated through hipHostMalloc
 *  @return #hipSuccess, #hipErrorInvalidValue
 *
 *  @see hipHostMalloc
 */
hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr);
/**
 *  @brief Register host memory so it can be accessed from the current device.
 *
 *  @param[out] hostPtr Pointer to host memory to be registered.
 *  @param[in] sizeBytes size of the host memory
 *  @param[in] flags.  See below.
 *
 *  Flags:
 *  - #hipHostRegisterDefault   Memory is Mapped and Portable
 *  - #hipHostRegisterPortable  Memory is considered registered by all contexts.  HIP only supports
 * one context so this is always assumed true.
 *  - #hipHostRegisterMapped    Map the allocation into the address space for the current device.
 * The device pointer can be obtained with #hipHostGetDevicePointer.
 *
 *
 *  After registering the memory, use #hipHostGetDevicePointer to obtain the mapped device pointer.
 *  On many systems, the mapped device pointer will have a different value than the mapped host
 * pointer.  Applications must use the device pointer in device code, and the host pointer in device
 * code.
 *
 *  On some systems, registered memory is pinned.  On some systems, registered memory may not be
 * actually be pinned but uses OS or hardware facilities to all GPU access to the host memory.
 *
 *  Developers are strongly encouraged to register memory blocks which are aligned to the host
 * cache-line size. (typically 64-bytes but can be obtains from the CPUID instruction).
 *
 *  If registering non-aligned pointers, the application must take care when register pointers from
 * the same cache line on different devices.  HIP's coarse-grained synchronization model does not
 * guarantee correct results if different devices write to different parts of the same cache block -
 * typically one of the writes will "win" and overwrite data from the other registered memory
 * region.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory
 *
 *  @see hipHostUnregister, hipHostGetFlags, hipHostGetDevicePointer
 */
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
/**
 *  @brief Un-register host pointer
 *
 *  @param[in] hostPtr Host pointer previously registered with #hipHostRegister
 *  @return Error code
 *
 *  @see hipHostRegister
 */
hipError_t hipHostUnregister(void* hostPtr);
/**
 *  Allocates at least width (in bytes) * height bytes of linear memory
 *  Padding may occur to ensure alighnment requirements are met for the given row
 *  The change in width size due to padding will be returned in *pitch.
 *  Currently the alignment is set to 128 bytes
 *
 *  @param[out] ptr Pointer to the allocated device memory
 *  @param[out] pitch Pitch for allocation (in bytes)
 *  @param[in]  width Requested pitched allocation width (in bytes)
 *  @param[in]  height Requested pitched allocation height
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return Error code
 *
 *  @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
 * hipMalloc3DArray, hipHostMalloc
 */
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);
/**
 *  Allocates at least width (in bytes) * height bytes of linear memory
 *  Padding may occur to ensure alighnment requirements are met for the given row
 *  The change in width size due to padding will be returned in *pitch.
 *  Currently the alignment is set to 128 bytes
 *
 *  @param[out] dptr Pointer to the allocated device memory
 *  @param[out] pitch Pitch for allocation (in bytes)
 *  @param[in]  width Requested pitched allocation width (in bytes)
 *  @param[in]  height Requested pitched allocation height
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *  The intended usage of pitch is as a separate parameter of the allocation, used to compute addresses within the 2D array.
 *  Given the row and column of an array element of type T, the address is computed as:
 *  T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;
 *
 *  @return Error code
 *
 *  @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
 * hipMalloc3DArray, hipHostMalloc
 */
hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes);
/**
 *  @brief Free memory allocated by the hcc hip memory allocation API.
 *  This API performs an implicit hipDeviceSynchronize() call.
 *  If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
 *
 *  @param[in] ptr Pointer to memory to be freed
 *  @return #hipSuccess
 *  @return #hipErrorInvalidDevicePointer (if pointer is invalid, including host pointers allocated
 * with hipHostMalloc)
 *
 *  @see hipMalloc, hipMallocPitch, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D,
 * hipMalloc3DArray, hipHostMalloc
 */
hipError_t hipFree(void* ptr);
/**
 *  @brief Free memory allocated by the hcc hip host memory allocation API.  [Deprecated]
 *
 *  @param[in] ptr Pointer to memory to be freed
 *  @return #hipSuccess,
 *          #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
 hipMalloc)
 *  @deprecated use hipHostFree() instead
 */
DEPRECATED("use hipHostFree instead")
hipError_t hipFreeHost(void* ptr);
/**
 *  @brief Free memory allocated by the hcc hip host memory allocation API
 *  This API performs an implicit hipDeviceSynchronize() call.
 *  If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
 *
 *  @param[in] ptr Pointer to memory to be freed
 *  @return #hipSuccess,
 *          #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
 * hipMalloc)
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D,
 * hipMalloc3DArray, hipHostMalloc
 */
hipError_t hipHostFree(void* ptr);
/**
 *  @brief Copy data from src to dst.
 *
 *  It supports memory from host to device,
 *  device to host, device to device and host to host
 *  The src and dst must not overlap.
 *
 *  For hipMemcpy, the copy is always performed by the current device (set by hipSetDevice).
 *  For multi-gpu or peer-to-peer configurations, it is recommended to set the current device to the
 *  device where the src data is physically located. For optimal peer-to-peer copies, the copy device
 *  must be able to access the src and dst pointers (by calling hipDeviceEnablePeerAccess with copy
 *  agent as the current device and src/dest as the peerDevice argument.  if this is not done, the
 *  hipMemcpy will still work, but will perform the copy using a staging buffer on the host.
 *  Calling hipMemcpy with dst and src pointers that do not match the hipMemcpyKind results in
 *  undefined behavior.
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]  src Data being copy from
 *  @param[in]  sizeBytes Data size in bytes
 *  @param[in]  copyType Memory copy type
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknowni
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
// TODO: Add description
hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes,
                               hipMemcpyKind kind, hipStream_t stream);
/**
 *  @brief Copy data from Host to Device
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeInitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes);
/**
 *  @brief Copy data from Device to Host
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeInitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes);
/**
 *  @brief Copy data from Device to Device
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeInitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes);
/**
 *  @brief Copy data from Host to Device asynchronously
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeInitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes, hipStream_t stream);
/**
 *  @brief Copy data from Device to Host asynchronously
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeInitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream);
/**
 *  @brief Copy data from Device to Device asynchronously
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeInitialized, #hipErrorNotInitialized, #hipErrorInvalidContext,
 * #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost,
 * hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA,
 * hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD,
 * hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync,
 * hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo,
 * hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream);

/**
 *  @brief Returns a global pointer from a module.
 *  Returns in *dptr and *bytes the pointer and size of the global of name name located in module hmod.
 *  If no variable of that name exists, it returns hipErrorNotFound. Both parameters dptr and bytes are optional.
 *  If one of them is NULL, it is ignored and hipSuccess is returned.
 *
 *  @param[out]  dptr  Returned global device pointer
 *  @param[out]  bytes Returned global size in bytes
 *  @param[in]   hmod  Module to retrieve global from
 *  @param[in]   name  Name of global to retrieve
 *
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotFound, #hipErrorInvalidContext
 *
 */
hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes,
    hipModule_t hmod, const char* name);
hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol);
hipError_t hipGetSymbolSize(size_t* size, const void* symbol);
hipError_t hipMemcpyToSymbol(const void* symbol, const void* src,
                             size_t sizeBytes, size_t offset __dparm(0),
                             hipMemcpyKind kind __dparm(hipMemcpyHostToDevice));
hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src,
                                  size_t sizeBytes, size_t offset,
                                  hipMemcpyKind kind, hipStream_t stream __dparm(0));
hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol,
                               size_t sizeBytes, size_t offset __dparm(0),
                               hipMemcpyKind kind __dparm(hipMemcpyDeviceToHost));
hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol,
                                    size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind,
                                    hipStream_t stream __dparm(0));
/**
 *  @brief Copy data from src to dst asynchronously.
 *
 *  @warning If host or dest are not pinned, the memory copy will be performed synchronously.  For
 * best performance, use hipHostMalloc to allocate host memory that is transferred asynchronously.
 *
 *  @warning on HCC hipMemcpyAsync does not support overlapped H2D and D2H copies.
 *  For hipMemcpy, the copy is always performed by the device associated with the specified stream.
 *
 *  For multi-gpu or peer-to-peer configurations, it is recommended to use a stream which is a
 * attached to the device where the src data is physically located. For optimal peer-to-peer copies,
 * the copy device must be able to access the src and dst pointers (by calling
 * hipDeviceEnablePeerAccess with copy agent as the current device and src/dest as the peerDevice
 * argument.  if this is not done, the hipMemcpy will still work, but will perform the copy using a
 * staging buffer on the host.
 *
 *  @param[out] dst Data being copy to
 *  @param[in]  src Data being copy from
 *  @param[in]  sizeBytes Data size in bytes
 *  @param[in]  accelerator_view Accelerator view which the copy is being enqueued
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
 *
 *  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
 * hipMemcpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyToSymbol,
 * hipMemcpyFromSymbol, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync,
 * hipMemcpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync,
 * hipMemcpyFromSymbolAsync
 */
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream __dparm(0));
/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * byte value value.
 *
 *  @param[out] dst Data being filled
 *  @param[in]  constant value to be set
 *  @param[in]  sizeBytes Data size in bytes
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * byte value value.
 *
 *  @param[out] dst Data ptr to be filled
 *  @param[in]  constant value to be set
 *  @param[in]  number of values to be set
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count);
/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * byte value value.
 *
 * hipMemsetD8Async() is asynchronous with respect to the host, so the call may return before the
 * memset is complete. The operation can optionally be associated to a stream by passing a non-zero
 * stream argument. If stream is non-zero, the operation may overlap with operations in other
 * streams.
 *
 *  @param[out] dst Data ptr to be filled
 *  @param[in]  constant value to be set
 *  @param[in]  number of values to be set
 *  @param[in]  stream - Stream identifier
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count, hipStream_t stream __dparm(0));
/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * short value value.
 *
 *  @param[out] dst Data ptr to be filled
 *  @param[in]  constant value to be set
 *  @param[in]  number of values to be set
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count);
/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * short value value.
 *
 * hipMemsetD16Async() is asynchronous with respect to the host, so the call may return before the
 * memset is complete. The operation can optionally be associated to a stream by passing a non-zero
 * stream argument. If stream is non-zero, the operation may overlap with operations in other
 * streams.
 *
 *  @param[out] dst Data ptr to be filled
 *  @param[in]  constant value to be set
 *  @param[in]  number of values to be set
 *  @param[in]  stream - Stream identifier
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count, hipStream_t stream __dparm(0));
/**
 *  @brief Fills the memory area pointed to by dest with the constant integer
 * value for specified number of times.
 *
 *  @param[out] dst Data being filled
 *  @param[in]  constant value to be set
 *  @param[in]  number of values to be set
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count);
/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dev with the constant
 * byte value value.
 *
 *  hipMemsetAsync() is asynchronous with respect to the host, so the call may return before the
 * memset is complete. The operation can optionally be associated to a stream by passing a non-zero
 * stream argument. If stream is non-zero, the operation may overlap with operations in other
 * streams.
 *
 *  @param[out] dst Pointer to device memory
 *  @param[in]  value - Value to set for each byte of specified memory
 *  @param[in]  sizeBytes - Size in bytes to set
 *  @param[in]  stream - Stream identifier
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
 */
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream __dparm(0));
/**
 *  @brief Fills the memory area pointed to by dev with the constant integer
 * value for specified number of times.
 *
 *  hipMemsetD32Async() is asynchronous with respect to the host, so the call may return before the
 * memset is complete. The operation can optionally be associated to a stream by passing a non-zero
 * stream argument. If stream is non-zero, the operation may overlap with operations in other
 * streams.
 *
 *  @param[out] dst Pointer to device memory
 *  @param[in]  value - Value to set for each byte of specified memory
 *  @param[in]  count - number of values to be set
 *  @param[in]  stream - Stream identifier
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
 */
hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count,
                             hipStream_t stream __dparm(0));
/**
 *  @brief Fills the memory area pointed to by dst with the constant value.
 *
 *  @param[out] dst Pointer to device memory
 *  @param[in]  pitch - data size in bytes
 *  @param[in]  value - constant value to be set
 *  @param[in]  width
 *  @param[in]  height
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
 */
hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height);
/**
 *  @brief Fills asynchronously the memory area pointed to by dst with the constant value.
 *
 *  @param[in]  dst Pointer to device memory
 *  @param[in]  pitch - data size in bytes
 *  @param[in]  value - constant value to be set
 *  @param[in]  width
 *  @param[in]  height
 *  @param[in]  stream
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
 */
hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height,hipStream_t stream __dparm(0));
/**
 *  @brief Fills synchronously the memory area pointed to by pitchedDevPtr with the constant value.
 *
 *  @param[in] pitchedDevPtr
 *  @param[in]  value - constant value to be set
 *  @param[in]  extent
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
 */
hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent );
/**
 *  @brief Fills asynchronously the memory area pointed to by pitchedDevPtr with the constant value.
 *
 *  @param[in] pitchedDevPtr
 *  @param[in]  value - constant value to be set
 *  @param[in]  extent
 *  @param[in]  stream
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
 */
hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent ,hipStream_t stream __dparm(0));
/**
 * @brief Query memory info.
 * Return snapshot of free memory, and total allocatable memory on the device.
 *
 * Returns in *free a snapshot of the current free memory.
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 * @warning On HCC, the free memory only accounts for memory allocated by this process and may be
 *optimistic.
 **/
hipError_t hipMemGetInfo(size_t* free, size_t* total);
hipError_t hipMemPtrGetInfo(void* ptr, size_t* size);
/**
 *  @brief Allocate an array on the device.
 *
 *  @param[out]  array  Pointer to allocated array in device memory
 *  @param[in]   desc   Requested channel format
 *  @param[in]   width  Requested array allocation width
 *  @param[in]   height Requested array allocation height
 *  @param[in]   flags  Requested properties of allocated array
 *  @return      #hipSuccess, #hipErrorOutOfMemory
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
 */
hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc, size_t width,
                          size_t height __dparm(0), unsigned int flags __dparm(hipArrayDefault));
hipError_t hipArrayCreate(hipArray** pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray);
hipError_t hipArrayDestroy(hipArray* array);
hipError_t hipArray3DCreate(hipArray** array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray);
hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent);
/**
 *  @brief Frees an array on the device.
 *
 *  @param[in]  array  Pointer to array to free
 *  @return     #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipHostMalloc, hipHostFree
 */
hipError_t hipFreeArray(hipArray* array);
/**
 * @brief Frees a mipmapped array on the device
 *
 * @param[in] mipmappedArray - Pointer to mipmapped array to free
 *
 * @return #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray);
/**
 *  @brief Allocate an array on the device.
 *
 *  @param[out]  array  Pointer to allocated array in device memory
 *  @param[in]   desc   Requested channel format
 *  @param[in]   extent Requested array allocation width, height and depth
 *  @param[in]   flags  Requested properties of allocated array
 *  @return      #hipSuccess, #hipErrorOutOfMemory
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
 */
hipError_t hipMalloc3DArray(hipArray** array, const struct hipChannelFormatDesc* desc,
                            struct hipExtent extent, unsigned int flags);
/**
 * @brief Allocate a mipmapped array on the device
 *
 * @param[out] mipmappedArray  - Pointer to allocated mipmapped array in device memory
 * @param[in]  desc            - Requested channel format
 * @param[in]  extent          - Requested allocation size (width field in elements)
 * @param[in]  numLevels       - Number of mipmap levels to allocate
 * @param[in]  flags           - Flags for extensions
 *
 * @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
 */
hipError_t hipMallocMipmappedArray(
    hipMipmappedArray_t *mipmappedArray,
    const struct hipChannelFormatDesc* desc,
    struct hipExtent extent,
    unsigned int numLevels,
    unsigned int flags __dparm(0));
/**
 * @brief Gets a mipmap level of a HIP mipmapped array
 *
 * @param[out] levelArray     - Returned mipmap level HIP array
 * @param[in]  mipmappedArray - HIP mipmapped array
 * @param[in]  level          - Mipmap level
 *
 * @return #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipGetMipmappedArrayLevel(
    hipArray_t *levelArray,
    hipMipmappedArray_const_t mipmappedArray,
    unsigned int level);
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst    Destination memory address
 *  @param[in]   dpitch Pitch of destination memory
 *  @param[in]   src    Source memory address
 *  @param[in]   spitch Pitch of source memory
 *  @param[in]   width  Width of matrix transfer (columns in bytes)
 *  @param[in]   height Height of matrix transfer (rows)
 *  @param[in]   kind   Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind);
/**
 *  @brief Copies memory for 2D arrays.
 *  @param[in]   pCopy Parameters for the memory copy
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 *  #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
 * hipMemcpyToSymbol, hipMemcpyAsync
*/
hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy);
/**
 *  @brief Copies memory for 2D arrays.
 *  @param[in]   pCopy Parameters for the memory copy
 *  @param[in]   stream Stream to use
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
 * hipMemcpyToSymbol, hipMemcpyAsync
*/
hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t stream __dparm(0));
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst    Destination memory address
 *  @param[in]   dpitch Pitch of destination memory
 *  @param[in]   src    Source memory address
 *  @param[in]   spitch Pitch of source memory
 *  @param[in]   width  Width of matrix transfer (columns in bytes)
 *  @param[in]   height Height of matrix transfer (rows)
 *  @param[in]   kind   Type of transfer
 *  @param[in]   stream Stream to use
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream __dparm(0));
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst     Destination memory address
 *  @param[in]   wOffset Destination starting X offset
 *  @param[in]   hOffset Destination starting Y offset
 *  @param[in]   src     Source memory address
 *  @param[in]   spitch  Pitch of source memory
 *  @param[in]   width   Width of matrix transfer (columns in bytes)
 *  @param[in]   height  Height of matrix transfer (rows)
 *  @param[in]   kind    Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind);
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst     Destination memory address
 *  @param[in]   wOffset Destination starting X offset
 *  @param[in]   hOffset Destination starting Y offset
 *  @param[in]   src     Source memory address
 *  @param[in]   spitch  Pitch of source memory
 *  @param[in]   width   Width of matrix transfer (columns in bytes)
 *  @param[in]   height  Height of matrix transfer (rows)
 *  @param[in]   kind    Type of transfer
 *  @param[in]   stream    Accelerator view which the copy is being enqueued
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2DToArrayAsync(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                                   size_t spitch, size_t width, size_t height, hipMemcpyKind kind,
                                   hipStream_t stream __dparm(0));
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst     Destination memory address
 *  @param[in]   wOffset Destination starting X offset
 *  @param[in]   hOffset Destination starting Y offset
 *  @param[in]   src     Source memory address
 *  @param[in]   count   size in bytes to copy
 *  @param[in]   kind    Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind);
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst       Destination memory address
 *  @param[in]   srcArray  Source memory address
 *  @param[in]   woffset   Source starting X offset
 *  @param[in]   hOffset   Source starting Y offset
 *  @param[in]   count     Size in bytes to copy
 *  @param[in]   kind      Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind);
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst       Destination memory address
 *  @param[in]   dpitch    Pitch of destination memory
 *  @param[in]   src       Source memory address
 *  @param[in]   wOffset   Source starting X offset
 *  @param[in]   hOffset   Source starting Y offset
 *  @param[in]   width     Width of matrix transfer (columns in bytes)
 *  @param[in]   height    Height of matrix transfer (rows)
 *  @param[in]   kind      Type of transfer
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2DFromArray( void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind);
/**
 *  @brief Copies data between host and device asynchronously.
 *
 *  @param[in]   dst       Destination memory address
 *  @param[in]   dpitch    Pitch of destination memory
 *  @param[in]   src       Source memory address
 *  @param[in]   wOffset   Source starting X offset
 *  @param[in]   hOffset   Source starting Y offset
 *  @param[in]   width     Width of matrix transfer (columns in bytes)
 *  @param[in]   height    Height of matrix transfer (rows)
 *  @param[in]   kind      Type of transfer
 *  @param[in]   stream    Accelerator view which the copy is being enqueued
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2DFromArrayAsync( void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream __dparm(0));
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dst       Destination memory address
 *  @param[in]   srcArray  Source array
 *  @param[in]   srcoffset Offset in bytes of source array
 *  @param[in]   count     Size of memory copy in bytes
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpyAtoH(void* dst, hipArray* srcArray, size_t srcOffset, size_t count);
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   dstArray   Destination memory address
 *  @param[in]   dstOffset  Offset in bytes of destination array
 *  @param[in]   srcHost    Source host pointer
 *  @param[in]   count      Size of memory copy in bytes
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpyHtoA(hipArray* dstArray, size_t dstOffset, const void* srcHost, size_t count);
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   p   3D memory copy parameters
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p);
/**
 *  @brief Copies data between host and device asynchronously.
 *
 *  @param[in]   p        3D memory copy parameters
 *  @param[in]   stream   Stream to use
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 * #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream __dparm(0));
/**
 *  @brief Copies data between host and device.
 *
 *  @param[in]   pCopy   3D memory copy parameters
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 *  #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy);
/**
 *  @brief Copies data between host and device asynchronously.
 *
 *  @param[in]   pCopy    3D memory copy parameters
 *  @param[in]   stream   Stream to use
 *  @return      #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidPitchValue,
 *  #hipErrorInvalidDevicePointer, #hipErrorInvalidMemcpyDirection
 *
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream);
// doxygen end Memory
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup PeerToPeer PeerToPeer Device Memory Access
 *  @{
 *  @warning PeerToPeer support is experimental.
 *  This section describes the PeerToPeer device memory access functions of HIP runtime API.
 */
/**
 * @brief Determine if a device can access a peer's memory.
 *
 * @param [out] canAccessPeer Returns the peer access capability (0 or 1)
 * @param [in] device - device from where memory may be accessed.
 * @param [in] peerDevice - device where memory is physically located
 *
 * Returns "1" in @p canAccessPeer if the specified @p device is capable
 * of directly accessing memory physically located on peerDevice , or "0" if not.
 *
 * Returns "0" in @p canAccessPeer if deviceId == peerDeviceId, and both are valid devices : a
 * device is not a peer of itself.
 *
 * @returns #hipSuccess,
 * @returns #hipErrorInvalidDevice if deviceId or peerDeviceId are not valid devices
 */
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);
/**
 * @brief Enable direct access from current device's virtual address space to memory allocations
 * physically located on a peer device.
 *
 * Memory which already allocated on peer device will be mapped into the address space of the
 * current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
 * the address space of the current device when the memory is allocated. The peer memory remains
 * accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
 *
 *
 * @param [in] peerDeviceId
 * @param [in] flags
 *
 * Returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
 * @returns #hipErrorPeerAccessAlreadyEnabled if peer access is already enabled for this device.
 */
hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
/**
 * @brief Disable direct access from current device's virtual address space to memory allocations
 * physically located on a peer device.
 *
 * Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
 * enabled from the current device.
 *
 * @param [in] peerDeviceId
 *
 * @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
 */
hipError_t hipDeviceDisablePeerAccess(int peerDeviceId);
/**
 * @brief Get information on memory allocations.
 *
 * @param [out] pbase - BAse pointer address
 * @param [out] psize - Size of allocation
 * @param [in]  dptr- Device Pointer
 *
 * @returns #hipSuccess, #hipErrorInvalidDevicePointer
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr);
#ifndef USE_PEER_NON_UNIFIED
#define USE_PEER_NON_UNIFIED 1
#endif
#if USE_PEER_NON_UNIFIED == 1
/**
 * @brief Copies memory from one device to memory on another device.
 *
 * @param [out] dst - Destination device pointer.
 * @param [in] dstDeviceId - Destination device
 * @param [in] src - Source device pointer
 * @param [in] srcDeviceId - Source device
 * @param [in] sizeBytes - Size of memory copy in bytes
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
 */
hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId,
                         size_t sizeBytes);
/**
 * @brief Copies memory from one device to memory on another device.
 *
 * @param [out] dst - Destination device pointer.
 * @param [in] dstDevice - Destination device
 * @param [in] src - Source device pointer
 * @param [in] srcDevice - Source device
 * @param [in] sizeBytes - Size of memory copy in bytes
 * @param [in] stream - Stream identifier
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDevice
 */
hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDevice,
                              size_t sizeBytes, hipStream_t stream __dparm(0));
#endif
// doxygen end PeerToPeer
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Context Context Management
 *  @{
 *  This section describes the context management functions of HIP runtime API.
 */
/**
 *
 *  @addtogroup ContextD Context Management [Deprecated]
 *  @{
 *  @ingroup Context
 *  This section describes the deprecated context management functions of HIP runtime API.
 */
/**
 * @brief Create a context and set it as current/ default context
 *
 * @param [out] ctx
 * @param [in] flags
 * @param [in] associated device handle
 *
 * @return #hipSuccess
 *
 * @see hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxPushCurrent,
 * hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);
/**
 * @brief Destroy a HIP context.
 *
 * @param [in] ctx Context to destroy
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipCtxCreate, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,hipCtxSetCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxDestroy(hipCtx_t ctx);
/**
 * @brief Pop the current/default context and return the popped context.
 *
 * @param [out] ctx
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxSetCurrent, hipCtxGetCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxPopCurrent(hipCtx_t* ctx);
/**
 * @brief Push the context to be set as current/ default context
 *
 * @param [in] ctx
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxPushCurrent(hipCtx_t ctx);
/**
 * @brief Set the passed context as current/default
 *
 * @param [in] ctx
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetCurrent(hipCtx_t ctx);
/**
 * @brief Get the handle of the current/ default context
 *
 * @param [out] ctx
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetCurrent(hipCtx_t* ctx);
/**
 * @brief Get the handle of the device associated with current/default context
 *
 * @param [out] device
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetDevice(hipDevice_t* device);
/**
 * @brief Returns the approximate HIP api version.
 *
 * @param [in]  ctx Context to check
 * @param [out] apiVersion
 *
 * @return #hipSuccess
 *
 * @warning The HIP feature set does not correspond to an exact CUDA SDK api revision.
 * This function always set *apiVersion to 4 as an approximation though HIP supports
 * some features which were introduced in later CUDA SDK revisions.
 * HIP apps code should not rely on the api revision number here and should
 * use arch feature flags to test device capabilities or conditional compilation.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent,
 * hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion);
/**
 * @brief Set Cache configuration for a specific function
 *
 * @param [out] cacheConfiguration
 *
 * @return #hipSuccess
 *
 * @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
 * ignored on those architectures.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig);
/**
 * @brief Set L1/Shared cache partition.
 *
 * @param [in] cacheConfiguration
 *
 * @return #hipSuccess
 *
 * @warning AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is
 * ignored on those architectures.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig);
/**
 * @brief Set Shared memory bank configuration.
 *
 * @param [in] sharedMemoryConfiguration
 *
 * @return #hipSuccess
 *
 * @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config);
/**
 * @brief Get Shared memory bank configuration.
 *
 * @param [out] sharedMemoryConfiguration
 *
 * @return #hipSuccess
 *
 * @warning AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig);
/**
 * @brief Blocks until the default context has completed all preceding requested tasks.
 *
 * @return #hipSuccess
 *
 * @warning This function waits for all streams on the default context to complete execution, and
 * then returns.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSynchronize(void);
/**
 * @brief Return flags used for creating default context.
 *
 * @param [out] flags
 *
 * @returns #hipSuccess
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetFlags(unsigned int* flags);
/**
 * @brief Enables direct access to memory allocations in a peer context.
 *
 * Memory which already allocated on peer device will be mapped into the address space of the
 * current device.  In addition, all future memory allocations on peerDeviceId will be mapped into
 * the address space of the current device when the memory is allocated. The peer memory remains
 * accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
 *
 *
 * @param [in] peerCtx
 * @param [in] flags
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
 * #hipErrorPeerAccessAlreadyEnabled
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 * @warning PeerToPeer support is experimental.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags);
/**
 * @brief Disable direct access from current context's virtual address space to memory allocations
 * physically located on a peer context.Disables direct access to memory allocations in a peer
 * context and unregisters any registered allocations.
 *
 * Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been
 * enabled from the current device.
 *
 * @param [in] peerCtx
 *
 * @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 * @warning PeerToPeer support is experimental.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx);
// doxygen end Context deprecated
/**
 * @}
 */
/**
 * @brief Get the state of the primary context.
 *
 * @param [in] Device to get primary context flags for
 * @param [out] Pointer to store flags
 * @param [out] Pointer to store context state; 0 = inactive, 1 = active
 *
 * @returns #hipSuccess
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active);
/**
 * @brief Release the primary context on the GPU.
 *
 * @param [in] Device which primary context is released
 *
 * @returns #hipSuccess
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 * @warning This function return #hipSuccess though doesn't release the primaryCtx by design on
 * HIP/HCC path.
 */
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);
/**
 * @brief Retain the primary context on the GPU.
 *
 * @param [out] Returned context handle of the new context
 * @param [in] Device which primary context is released
 *
 * @returns #hipSuccess
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);
/**
 * @brief Resets the primary context on the GPU.
 *
 * @param [in] Device which primary context is reset
 *
 * @returns #hipSuccess
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);
/**
 * @brief Set flags for the primary context.
 *
 * @param [in] Device for which the primary context flags are set
 * @param [in] New flags for the device
 *
 * @returns #hipSuccess, #hipErrorContextAlreadyInUse
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,
 * hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);
// doxygen end Context Management
/**
 * @}
 */
/**
 *
 *  @defgroup Module Module Management
 *  @{
 *  This section describes the module management functions of HIP runtime API.
 *
 */
/**
 * @brief Loads code object from file into a hipModule_t
 *
 * @param [in] fname
 * @param [out] module
 *
 * @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorFileNotFound,
 * hipErrorOutOfMemory, hipErrorSharedObjectInitFailed, hipErrorNotInitialized
 *
 *
 */
hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
/**
 * @brief Frees the module
 *
 * @param [in] module
 *
 * @returns hipSuccess, hipInvalidValue
 * module is freed and the code objects associated with it are destroyed
 *
 */
hipError_t hipModuleUnload(hipModule_t module);
/**
 * @brief Function with kname will be extracted if present in module
 *
 * @param [in] module
 * @param [in] kname
 * @param [out] function
 *
 * @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorNotInitialized,
 * hipErrorNotFound,
 */
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname);
/**
 * @brief Find out attributes for a given function.
 *
 * @param [out] attr
 * @param [in] func
 *
 * @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
 */
hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr, const void* func);
/**
 * @brief Find out a specific attribute for a given function.
 *
 * @param [out] value
 * @param [in]  attrib
 * @param [in]  hfunc
 *
 * @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
 */
hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc);
/**
 * @brief returns the handle of the texture reference with the name from the module.
 *
 * @param [in] hmod
 * @param [in] name
 * @param [out] texRef
 *
 * @returns hipSuccess, hipErrorNotInitialized, hipErrorNotFound, hipErrorInvalidValue
 */
hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name);
/**
 * @brief builds module from code object which resides in host memory. Image is pointer to that
 * location.
 *
 * @param [in] image
 * @param [out] module
 *
 * @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
 */
hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
/**
 * @brief builds module from code object which resides in host memory. Image is pointer to that
 * location. Options are not used. hipModuleLoadData is called.
 *
 * @param [in] image
 * @param [out] module
 * @param [in] number of options
 * @param [in] options for JIT
 * @param [in] option values for JIT
 *
 * @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
 */
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions,
                               hipJitOption* options, void** optionValues);
/**
 * @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
 * to kernelparams or extra
 *
 * @param [in] f         Kernel to launch.
 * @param [in] gridDimX  X grid dimension specified as multiple of blockDimX.
 * @param [in] gridDimY  Y grid dimension specified as multiple of blockDimY.
 * @param [in] gridDimZ  Z grid dimension specified as multiple of blockDimZ.
 * @param [in] blockDimX X block dimensions specified in work-items
 * @param [in] blockDimY Y grid dimension specified in work-items
 * @param [in] blockDimZ Z grid dimension specified in work-items
 * @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
 * default stream is used with associated synchronization rules.
 * @param [in] kernelParams
 * @param [in] extra     Pointer to kernel arguments.   These are passed directly to the kernel and
 * must be in the memory layout and alignment expected by the kernel.
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
 *
 * @warning kernellParams argument is not yet implemented in HIP. Please use extra instead. Please
 * refer to hip_porting_driver_api.md for sample usage.
 */
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX,
                                 unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra);
/**
 * @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
 * to kernelparams or extra, where thread blocks can cooperate and synchronize as they execute
 *
 * @param [in] f         Kernel to launch.
 * @param [in] gridDim   Grid dimensions specified as multiple of blockDim.
 * @param [in] blockDim  Block dimensions specified in work-items
 * @param [in] kernelParams A list of kernel arguments
 * @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case th
 * default stream is used with associated synchronization rules.
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
 */
hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX,
                                      void** kernelParams, unsigned int sharedMemBytes,
                                      hipStream_t stream);
/**
 * @brief Launches kernels on multiple devices where thread blocks can cooperate and
 * synchronize as they execute.
 *
 * @param [in] launchParamsList         List of launch parameters, one per device.
 * @param [in] numDevices               Size of the launchParamsList array.
 * @param [in] flags                    Flags to control launch behavior.
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue, hipErrorCooperativeLaunchTooLarge
 */
hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                 int  numDevices, unsigned int  flags);
/**
 * @brief Launches kernels on multiple devices and guarantees all specified kernels are dispatched
 * on respective streams before enqueuing any other work on the specified streams from any other threads
 *
 *
 * @param [in] hipLaunchParams          List of launch parameters, one per device.
 * @param [in] numDevices               Size of the launchParamsList array.
 * @param [in] flags                    Flags to control launch behavior.
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
 */
hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList,
                                              int  numDevices, unsigned int  flags);
// doxygen end Module
/**
 * @}
 */
/**
 *
 *  @defgroup Occupancy Occupancy
 *  @{
 *  This section describes the occupancy functions of HIP runtime API.
 *
 */
/**
 * @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
 *
 * @param [out] gridSize           minimum grid size for maximum potential occupancy
 * @param [out] blockSize          block size for maximum potential occupancy
 * @param [in]  f                  kernel function for which occupancy is calulated
 * @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
 * @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
 */
//TODO - Match CUoccupancyB2DSize
hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit);
/**
 * @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
 *
 * @param [out] gridSize           minimum grid size for maximum potential occupancy
 * @param [out] blockSize          block size for maximum potential occupancy
 * @param [in]  f                  kernel function for which occupancy is calulated
 * @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
 * @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
 * @param [in]  flags            Extra flags for occupancy calculation (only default supported)
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
 */
//TODO - Match CUoccupancyB2DSize
hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit, unsigned int  flags);
/**
 * @brief Returns occupancy for a device function.
 *
 * @param [out] numBlocks        Returned occupancy
 * @param [in]  func             Kernel function (hipFunction) for which occupancy is calulated
 * @param [in]  blockSize        Block size the kernel is intended to be launched with
 * @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
 */
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
   int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk);
/**
 * @brief Returns occupancy for a device function.
 *
 * @param [out] numBlocks        Returned occupancy
 * @param [in]  f                Kernel function(hipFunction_t) for which occupancy is calulated
 * @param [in]  blockSize        Block size the kernel is intended to be launched with
 * @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
 * @param [in]  flags            Extra flags for occupancy calculation (only default supported)
 */
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
   int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags);
/**
 * @brief Returns occupancy for a device function.
 *
 * @param [out] numBlocks        Returned occupancy
 * @param [in]  func             Kernel function for which occupancy is calulated
 * @param [in]  blockSize        Block size the kernel is intended to be launched with
 * @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
 */
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
   int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk);
/**
 * @brief Returns occupancy for a device function.
 *
 * @param [out] numBlocks        Returned occupancy
 * @param [in]  f                Kernel function for which occupancy is calulated
 * @param [in]  blockSize        Block size the kernel is intended to be launched with
 * @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
 * @param [in]  flags            Extra flags for occupancy calculation (currently ignored)
 */
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
   int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags __dparm(hipOccupancyDefault));
/**
 * @brief determine the grid and block sizes to achieves maximum occupancy for a kernel
 *
 * @param [out] gridSize           minimum grid size for maximum potential occupancy
 * @param [out] blockSize          block size for maximum potential occupancy
 * @param [in]  f                  kernel function for which occupancy is calulated
 * @param [in]  dynSharedMemPerBlk dynamic shared memory usage (in bytes) intended for each block
 * @param [in]  blockSizeLimit     the maximum block size for the kernel, use 0 for no limit
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorInvalidValue
 */
hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                             const void* f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit);
// doxygen end Occupancy
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Profiler Profiler Control[Deprecated]
 *  @{
 *  This section describes the profiler control functions of HIP runtime API.
 *
 *  @warning The cudaProfilerInitialize API format for "configFile" is not supported.
 *
 */
// TODO - expand descriptions:
/**
 * @brief Start recording of profiling information
 * When using this API, start the profiler with profiling disabled.  (--startdisabled)
 * @warning : hipProfilerStart API is under development.
 */
DEPRECATED("use roctracer/rocTX instead")
hipError_t hipProfilerStart();
/**
 * @brief Stop recording of profiling information.
 * When using this API, start the profiler with profiling disabled.  (--startdisabled)
 * @warning : hipProfilerStop API is under development.
 */
DEPRECATED("use roctracer/rocTX instead")
hipError_t hipProfilerStop();
// doxygen end profiler
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Clang Launch API to support the triple-chevron syntax
 *  @{
 *  This section describes the API to support the triple-chevron syntax.
 */
/**
 * @brief Configure a kernel launch.
 *
 * @param [in] gridDim   grid dimension specified as multiple of blockDim.
 * @param [in] blockDim  block dimensions specified in work-items
 * @param [in] sharedMem Amount of dynamic shared memory to allocate for this kernel. The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
 * default stream is used with associated synchronization rules.
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
 *
 */
hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dparm(0), hipStream_t stream __dparm(0));
/**
 * @brief Set a kernel argument.
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
 *
 * @param [in] arg    Pointer the argument in host memory.
 * @param [in] size   Size of the argument.
 * @param [in] offset Offset of the argument on the argument stack.
 *
 */
hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset);
/**
 * @brief Launch a kernel.
 *
 * @param [in] func Kernel to launch.
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
 *
 */
hipError_t hipLaunchByPtr(const void* func);
/**
 * @brief Push configuration of a kernel launch.
 *
 * @param [in] gridDim   grid dimension specified as multiple of blockDim.
 * @param [in] blockDim  block dimensions specified in work-items
 * @param [in] sharedMem Amount of dynamic shared memory to allocate for this kernel. The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
 * default stream is used with associated synchronization rules.
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
 *
 */
hipError_t __hipPushCallConfiguration(dim3 gridDim,
                                      dim3 blockDim,
                                      size_t sharedMem __dparm(0),
                                      hipStream_t stream __dparm(0));
/**
 * @brief Pop configuration of a kernel launch.
 *
 * @param [out] gridDim   grid dimension specified as multiple of blockDim.
 * @param [out] blockDim  block dimensions specified in work-items
 * @param [out] sharedMem Amount of dynamic shared memory to allocate for this kernel.  The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [out] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
 * default stream is used with associated synchronization rules.
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
 *
 */
hipError_t __hipPopCallConfiguration(dim3 *gridDim,
                                     dim3 *blockDim,
                                     size_t *sharedMem,
                                     hipStream_t *stream);
/**
 * @brief C compliant kernel launch API
 *
 * @param [in] function_address - kernel stub function pointer.
 * @param [in] numBlocks - number of blocks
 * @param [in] dimBlocks - dimension of a block
 * @param [in] args - kernel arguments
 * @param [in] sharedMemBytes - Amount of dynamic shared memory to allocate for this kernel. The
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream - Stream where the kernel should be dispatched.  May be 0, in which case th
 *  default stream is used with associated synchronization rules.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, hipInvalidDevice
 *
 */
hipError_t hipLaunchKernel(const void* function_address,
                           dim3 numBlocks,
                           dim3 dimBlocks,
                           void** args,
                           size_t sharedMemBytes __dparm(0),
                           hipStream_t stream __dparm(0));
/**
 * Copies memory for 2D arrays.
 *
 * @param pCopy           - Parameters for the memory copy
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy);
//TODO: Move this to hip_ext.h
hipError_t hipExtLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                              void** args, size_t sharedMemBytes, hipStream_t stream,
                              hipEvent_t startEvent, hipEvent_t stopEvent, int flags);
// doxygen end Clang launch
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Texture Texture Management
 *  @{
 *  This section describes the texture management functions of HIP runtime API.
 */
hipError_t hipBindTextureToMipmappedArray(
    const textureReference* tex,
    hipMipmappedArray_const_t mipmappedArray,
    const hipChannelFormatDesc* desc);
hipError_t hipGetTextureReference(
    const textureReference** texref,
    const void* symbol);
hipError_t hipCreateTextureObject(
    hipTextureObject_t* pTexObject,
    const hipResourceDesc* pResDesc,
    const hipTextureDesc* pTexDesc,
    const struct hipResourceViewDesc* pResViewDesc);
hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject);
hipError_t hipGetChannelDesc(
    hipChannelFormatDesc* desc,
    hipArray_const_t array);
hipError_t hipGetTextureObjectResourceDesc(
    hipResourceDesc* pResDesc,
    hipTextureObject_t textureObject);
hipError_t hipGetTextureObjectResourceViewDesc(
    struct hipResourceViewDesc* pResViewDesc,
    hipTextureObject_t textureObject);
hipError_t hipGetTextureObjectTextureDesc(
    hipTextureDesc* pTexDesc,
    hipTextureObject_t textureObject);
hipError_t hipTexRefSetAddressMode(
    textureReference* texRef,
    int dim,
    enum hipTextureAddressMode am);
hipError_t hipTexRefSetArray(
    textureReference* tex,
    hipArray_const_t array,
    unsigned int flags);
hipError_t hipTexRefSetFilterMode(
    textureReference* texRef,
    enum hipTextureFilterMode fm);
hipError_t hipTexRefSetFlags(
    textureReference* texRef,
    unsigned int Flags);
hipError_t hipTexRefSetFormat(
    textureReference* texRef,
    hipArray_Format fmt,
    int NumPackedComponents);
hipError_t hipTexObjectCreate(
    hipTextureObject_t* pTexObject,
    const HIP_RESOURCE_DESC* pResDesc,
    const HIP_TEXTURE_DESC* pTexDesc,
    const HIP_RESOURCE_VIEW_DESC* pResViewDesc);
hipError_t hipTexObjectDestroy(
    hipTextureObject_t texObject);
hipError_t hipTexObjectGetResourceDesc(
    HIP_RESOURCE_DESC* pResDesc,
    hipTextureObject_t texObject);
hipError_t hipTexObjectGetResourceViewDesc(
    HIP_RESOURCE_VIEW_DESC* pResViewDesc,
    hipTextureObject_t texObject);
hipError_t hipTexObjectGetTextureDesc(
    HIP_TEXTURE_DESC* pTexDesc,
    hipTextureObject_t texObject);

/**
 *
 *  @addtogroup TexturD Texture Management [Deprecated]
 *  @{
 *  @ingroup Texture
 *  This section describes the deprecated texture management functions of HIP runtime API.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipBindTexture(
    size_t* offset,
    const textureReference* tex,
    const void* devPtr,
    const hipChannelFormatDesc* desc,
    size_t size __dparm(UINT_MAX));
DEPRECATED(DEPRECATED_MSG)
hipError_t hipBindTexture2D(
    size_t* offset,
    const textureReference* tex,
    const void* devPtr,
    const hipChannelFormatDesc* desc,
    size_t width,
    size_t height,
    size_t pitch);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipBindTextureToArray(
    const textureReference* tex,
    hipArray_const_t array,
    const hipChannelFormatDesc* desc);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipGetTextureAlignmentOffset(
    size_t* offset,
    const textureReference* texref);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipUnbindTexture(const textureReference* tex);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetAddress(
    hipDeviceptr_t* dev_ptr,
    const textureReference* texRef);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetAddressMode(
    enum hipTextureAddressMode* pam,
    const textureReference* texRef,
    int dim);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetFilterMode(
    enum hipTextureFilterMode* pfm,
    const textureReference* texRef);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetFlags(
    unsigned int* pFlags,
    const textureReference* texRef);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetFormat(
    hipArray_Format* pFormat,
    int* pNumChannels,
    const textureReference* texRef);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetMaxAnisotropy(
    int* pmaxAnsio,
    const textureReference* texRef);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetMipmapFilterMode(
    enum hipTextureFilterMode* pfm,
    const textureReference* texRef);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetMipmapLevelBias(
    float* pbias,
    const textureReference* texRef);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetMipmapLevelClamp(
    float* pminMipmapLevelClamp,
    float* pmaxMipmapLevelClamp,
    const textureReference* texRef);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefGetMipMappedArray(
    hipMipmappedArray_t* pArray,
    const textureReference* texRef);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetAddress(
    size_t* ByteOffset,
    textureReference* texRef,
    hipDeviceptr_t dptr,
    size_t bytes);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetAddress2D(
    textureReference* texRef,
    const HIP_ARRAY_DESCRIPTOR* desc,
    hipDeviceptr_t dptr,
    size_t Pitch);
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetMaxAnisotropy(
    textureReference* texRef,
    unsigned int maxAniso);
// doxygen end deprecated texture management
/**
 * @}
 */

// The following are not supported.
/**
 *
 *  @addtogroup TextureU Texture Management [Not supported]
 *  @{
 *  @ingroup Texture
 *  This section describes the texture management functions currently unsupported in HIP runtime.
 */
DEPRECATED(DEPRECATED_MSG)
hipError_t hipTexRefSetBorderColor(
    textureReference* texRef,
    float* pBorderColor);
hipError_t hipTexRefSetMipmapFilterMode(
    textureReference* texRef,
    enum hipTextureFilterMode fm);
hipError_t hipTexRefSetMipmapLevelBias(
    textureReference* texRef,
    float bias);
hipError_t hipTexRefSetMipmapLevelClamp(
    textureReference* texRef,
    float minMipMapLevelClamp,
    float maxMipMapLevelClamp);
hipError_t hipTexRefSetMipmappedArray(
    textureReference* texRef,
    struct hipMipmappedArray* mipmappedArray,
    unsigned int Flags);
hipError_t hipMipmappedArrayCreate(
    hipMipmappedArray_t* pHandle,
    HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
    unsigned int numMipmapLevels);
hipError_t hipMipmappedArrayDestroy(
    hipMipmappedArray_t hMipmappedArray);
hipError_t hipMipmappedArrayGetLevel(
    hipArray_t* pLevelArray,
    hipMipmappedArray_t hMipMappedArray,
    unsigned int level);
// doxygen end Texture management unsupported
/**
 * @}
 */

// doxygen end Texture management
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Runtime Runtime Compilation
 *  @{
 *  This section describes the runtime compilation functions of HIP runtime API.
 *
 */
// This group is for HIPrtc

// doxygen end Runtime
/**
 * @}
 */

/**
 *
 *  @defgroup Callback Callback Activity APIs
 *  @{
 *  This section describes the callback/Activity of HIP runtime API.
 */
hipError_t hipRegisterApiCallback(uint32_t id, void* fun, void* arg);
hipError_t hipRemoveApiCallback(uint32_t id);
hipError_t hipRegisterActivityCallback(uint32_t id, void* fun, void* arg);
hipError_t hipRemoveActivityCallback(uint32_t id);
const char* hipApiName(uint32_t id);
const char* hipKernelNameRef(const hipFunction_t f);
const char* hipKernelNameRefByPtr(const void* hostFunction, hipStream_t stream);
int hipGetStreamDeviceId(hipStream_t stream);

// doxygen end Callback
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Graph Management
 *  @{
 *  This section describes the graph management types & functions of HIP runtime API.
 */

/**
 * An opaque value that represents a hip graph
 */
typedef struct ihipGraph* hipGraph_t;
/**
 * An opaque value that represents a hip graph node
 */
typedef struct hipGraphNode* hipGraphNode_t;
/**
 * An opaque value that represents a hip graph Exec
 */
typedef struct hipGraphExec* hipGraphExec_t;

/**
 * @brief hipGraphNodeType
 * @enum
 *
 */
typedef enum hipGraphNodeType {
  hipGraphNodeTypeKernel = 1,             ///< GPU kernel node
  hipGraphNodeTypeMemcpy = 2,             ///< Memcpy 3D node
  hipGraphNodeTypeMemset = 3,             ///< Memset 1D node
  hipGraphNodeTypeHost = 4,               ///< Host (executable) node
  hipGraphNodeTypeGraph = 5,              ///< Node which executes an embedded graph
  hipGraphNodeTypeEmpty = 6,              ///< Empty (no-op) node
  hipGraphNodeTypeWaitEvent = 7,          ///< External event wait node
  hipGraphNodeTypeEventRecord = 8,        ///< External event record node
  hipGraphNodeTypeMemcpy1D = 9,           ///< Memcpy 1D node
  hipGraphNodeTypeMemcpyFromSymbol = 10,  ///< MemcpyFromSymbol node
  hipGraphNodeTypeMemcpyToSymbol = 11,    ///< MemcpyToSymbol node
  hipGraphNodeTypeCount
} hipGraphNodeType;

typedef void (*hipHostFn_t)(void* userData);
typedef struct hipHostNodeParams {
  hipHostFn_t fn;
  void* userData;
} hipHostNodeParams;
typedef struct hipKernelNodeParams {
  dim3 blockDim;
  void** extra;
  void* func;
  dim3 gridDim;
  void** kernelParams;
  unsigned int sharedMemBytes;
} hipKernelNodeParams;
typedef struct hipMemsetParams {
  void* dst;
  unsigned int elementSize;
  size_t height;
  size_t pitch;
  unsigned int value;
  size_t width;
} hipMemsetParams;

/**
 * @brief hipGraphExecUpdateResult
 * @enum
 *
 */
typedef enum hipGraphExecUpdateResult {
  hipGraphExecUpdateSuccess = 0x0,  ///< The update succeeded
  hipGraphExecUpdateError = 0x1,  ///< The update failed for an unexpected reason which is described
                                  ///< in the return value of the function
  hipGraphExecUpdateErrorTopologyChanged = 0x2,  ///< The update failed because the topology changed
  hipGraphExecUpdateErrorNodeTypeChanged = 0x3,  ///< The update failed because a node type changed
  hipGraphExecUpdateErrorFunctionChanged = 
      0x4,  ///< The update failed because the function of a kernel node changed
  hipGraphExecUpdateErrorParametersChanged =
      0x5,  ///< The update failed because the parameters changed in a way that is not supported
  hipGraphExecUpdateErrorNotSupported =
      0x6,  ///< The update failed because something about the node is not supported
  hipGraphExecUpdateErrorUnsupportedFunctionChange = 0x7
} hipGraphExecUpdateResult;

typedef enum hipStreamCaptureMode {
  hipStreamCaptureModeGlobal = 0,
  hipStreamCaptureModeThreadLocal,
  hipStreamCaptureModeRelaxed
} hipStreamCaptureMode;
typedef enum hipStreamCaptureStatus {
  hipStreamCaptureStatusNone = 0,    ///< Stream is not capturing
  hipStreamCaptureStatusActive,      ///< Stream is actively capturing
  hipStreamCaptureStatusInvalidated  ///< Stream is part of a capture sequence that has been
                                     ///< invalidated, but not terminated
} hipStreamCaptureStatus;

typedef enum hipStreamUpdateCaptureDependenciesFlags {
  hipStreamAddCaptureDependencies = 0,  ///< Add new nodes to the dependency set
  hipStreamSetCaptureDependencies,      ///< Replace the dependency set with the new nodes
} hipStreamUpdateCaptureDependenciesFlags;

/**
 * @brief Begins graph capture on a stream.
 *
 * @param [in] stream - Stream to initiate capture.
 * @param [in] mode - Controls the interaction of this capture sequence with other API calls that
 * are not safe.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode);

/**
 * @brief Ends capture on a stream, returning the captured graph.
 *
 * @param [in] stream - Stream to end capture.
 * @param [out] pGraph - returns the graph captured.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph);

/**
 * @brief Get capture status of a stream.
 *
 * @param [in] stream - Stream under capture.
 * @param [out] pCaptureStatus - returns current status of the capture.
 * @param [out] pId - unique ID of the capture.
 *
 * @returns #hipSuccess, #hipErrorStreamCaptureImplicit
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId);

/**
 * @brief Get stream's capture state
 *
 * @param [in] stream - Stream under capture.
 * @param [out] captureStatus_out - returns current status of the capture.
 * @param [out] id_out - unique ID of the capture.
 * @param [in] graph_out - returns the graph being captured into.
 * @param [out] dependencies_out - returns pointer to an array of nodes.
 * @param [out] numDependencies_out - returns size of the array returned in dependencies_out.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out __dparm(0),
                                      hipGraph_t* graph_out __dparm(0),
                                      const hipGraphNode_t** dependencies_out __dparm(0),
                                      size_t* numDependencies_out __dparm(0));

/**
 * @brief Get stream's capture state
 *
 * @param [in] stream - Stream under capture.
 * @param [out] pCaptureStatus - returns current status of the capture.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorStreamCaptureImplicit
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus);

/**
 * @brief Update the set of dependencies in a capturing stream
 *
 * @param [in] stream - Stream under capture.
 * @param [in] dependencies - pointer to an array of nodes to Add/Replace.
 * @param [in] numDependencies - size of the array in dependencies.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorIllegalState
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies,
                                              size_t numDependencies,
                                              unsigned int flags __dparm(0));

/**
 * @brief Creates a graph
 *
 * @param [out] pGraph - pointer to graph to create.
 * @param [in] flags - flags for graph creation, must be 0.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags);

/**
 * @brief Destroys a graph
 *
 * @param [in] graph - instance of graph to destroy.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphDestroy(hipGraph_t graph);

/**
 * @brief Adds dependency edges to a graph.
 *
 * @param [in] graph - instance of the graph to add dependencies.
 * @param [in] from - pointer to the graph nodes with dependenties to add from.
 * @param [in] to - pointer to the graph nodes to add dependenties to.
 * @param [in] numDependencies - the number of dependencies to add.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                   const hipGraphNode_t* to, size_t numDependencies);

/**
 * @brief Removes dependency edges from a graph.
 *
 * @param [in] graph - instance of the graph to remove dependencies.
 * @param [in] from - Array of nodes that provide the dependencies.
 * @param [in] to - Array of dependent nodes.
 * @param [in] numDependencies - the number of dependencies to remove.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                      const hipGraphNode_t* to, size_t numDependencies);

/**
 * @brief Returns a graph's dependency edges.
 *
 * @param [in] graph - instance of the graph to get the edges from.
 * @param [out] from - pointer to the graph nodes to return edge endpoints.
 * @param [out] to - pointer to the graph nodes to return edge endpoints.
 * @param [out] numEdges - returns number of edges.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * from and to may both be NULL, in which case this function only returns the number of edges in
 * numEdges. Otherwise, numEdges entries will be filled in. If numEdges is higher than the actual
 * number of edges, the remaining entries in from and to will be set to NULL, and the number of
 * edges actually returned will be written to numEdges
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to,
                            size_t* numEdges);

/**
 * @brief Returns graph nodes.
 *
 * @param [in] graph - instance of graph to get the nodes.
 * @param [out] nodes - pointer to return the  graph nodes.
 * @param [out] numNodes - returns number of graph nodes.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * nodes may be NULL, in which case this function will return the number of nodes in numNodes.
 * Otherwise, numNodes entries will be filled in. If numNodes is higher than the actual number of
 * nodes, the remaining entries in nodes will be set to NULL, and the number of nodes actually
 * obtained will be returned in numNodes.
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes);

/**
 * @brief Returns graph's root nodes.
 *
 * @param [in] graph - instance of the graph to get the nodes.
 * @param [out] pRootNodes - pointer to return the graph's root nodes.
 * @param [out] pNumRootNodes - returns the number of graph's root nodes.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * pRootNodes may be NULL, in which case this function will return the number of root nodes in
 * pNumRootNodes. Otherwise, pNumRootNodes entries will be filled in. If pNumRootNodes is higher
 * than the actual number of root nodes, the remaining entries in pRootNodes will be set to NULL,
 * and the number of nodes actually obtained will be returned in pNumRootNodes.
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes,
                                size_t* pNumRootNodes);

/**
 * @brief Returns a node's dependencies.
 *
 * @param [in] node - graph node to get the dependencies from.
 * @param [out] pDependencies - pointer to to return the dependencies.
 * @param [out] pNumDependencies -  returns the number of graph node dependencies.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * pDependencies may be NULL, in which case this function will return the number of dependencies in
 * pNumDependencies. Otherwise, pNumDependencies entries will be filled in. If pNumDependencies is
 * higher than the actual number of dependencies, the remaining entries in pDependencies will be set
 * to NULL, and the number of nodes actually obtained will be returned in pNumDependencies.
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies,
                                       size_t* pNumDependencies);

/**
 * @brief Returns a node's dependent nodes.
 *
 * @param [in] node - graph node to get the Dependent nodes from.
 * @param [out] pDependentNodes - pointer to return the graph dependent nodes.
 * @param [out] pNumDependentNodes - returns the number of graph node dependent nodes.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * DependentNodes may be NULL, in which case this function will return the number of dependent nodes
 * in pNumDependentNodes. Otherwise, pNumDependentNodes entries will be filled in. If
 * pNumDependentNodes is higher than the actual number of dependent nodes, the remaining entries in
 * pDependentNodes will be set to NULL, and the number of nodes actually obtained will be returned
 * in pNumDependentNodes.
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes,
                                         size_t* pNumDependentNodes);

/**
 * @brief Returns a node's type.
 *
 * @param [in] node - instance of the graph to add dependencies.
 * @param [out] pType - pointer to the return the type
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType);

/**
 * @brief Remove a node from the graph.
 *
 * @param [in] node - graph node to remove
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphDestroyNode(hipGraphNode_t node);

/**
 * @brief Clones a graph.
 *
 * @param [out] pGraphClone - Returns newly created cloned graph.
 * @param [in] originalGraph - original graph to clone from.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph);

/**
 * @brief Finds a cloned version of a node.
 *
 * @param [out] pNode - Returns the cloned node.
 * @param [in] originalNode - original node handle.
 * @param [in] clonedGraph - Cloned graph to query.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode,
                                   hipGraph_t clonedGraph);

/**
 * @brief Creates an executable graph from a graph
 *
 * @param [out] pGraphExec - pointer to instantiated executable graph that is created.
 * @param [in] graph - instance of graph to instantiate.
 * @param [out] pErrorNode - pointer to error node in case error occured in graph instantiation,
 *  it could modify the correponding node.
 * @param [out] pLogBuffer - pointer to log buffer.
 * @param [out] bufferSize - the size of log buffer.
 *
 * @returns #hipSuccess, #hipErrorOutOfMemory
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                               hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);

/**
 * @brief Creates an executable graph from a graph.
 *
 * @param [out] pGraphExec - pointer to instantiated executable graph that is created.
 * @param [in] graph - instance of graph to instantiate.
 * @param [in] flags - Flags to control instantiation.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 */
hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                        unsigned long long flags);

/**
 * @brief launches an executable graph in a stream
 *
 * @param [in] graphExec - instance of executable graph to launch.
 * @param [in] stream - instance of stream in which to launch executable graph.
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream);

/**
 * @brief Destroys an executable graph
 *
 * @param [in] pGraphExec - instance of executable graph to destry.
 *
 * @returns #hipSuccess.
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec);

// Check whether an executable graph can be updated with a graph and perform the update if possible.
/**
 * @brief Check whether an executable graph can be updated with a graph and perform the update if  *
 * possible.
 *
 * @param [in] hGraphExec - instance of executable graph to update.
 * @param [in] hGraph - graph that contains the updated parameters.
 * @param [in] hErrorNode_out -  node which caused the permissibility check to forbid the update.
 * @param [in] updateResult_out - Whether the graph update was permitted.
 * @returns #hipSuccess, #hipErrorGraphExecUpdateFailure
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                              hipGraphNode_t* hErrorNode_out,
                              hipGraphExecUpdateResult* updateResult_out);

/**
 * @brief Creates a kernel execution node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to graph node to create.
 * @param [in] graph - instance of graph to add the created node.
 * @param [in] pDependencies - pointer to the dependencies on the kernel execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] pNodeParams - pointer to the parameters to the kernel execution node on the GPU.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidDeviceFunction
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipKernelNodeParams* pNodeParams);

/**
 * @brief Gets kernel node's parameters.
 *
 * @param [in] node - instance of the node to get parameters from.
 * @param [out] pNodeParams - pointer to the parameters
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams* pNodeParams);

/**
 * @brief Sets a kernel node's parameters.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - const pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node, const hipKernelNodeParams* pNodeParams);

/**
 * @brief Sets the parameters for a kernel node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - const pointer to the kernel node parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipKernelNodeParams* pNodeParams);

/**
 * @brief Creates a memcpy node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to graph node to create.
 * @param [in] graph - instance of graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] pCopyParams - const pointer to the parameters for the memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemcpy3DParms* pCopyParams);
/**
 * @brief Gets a memcpy node's parameters.
 *
 * @param [in] node - instance of the node to get parameters from.
 * @param [out] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams);

/**
 * @brief Sets a memcpy node's parameters.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - const pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams);

/**
 * @brief Sets the parameters for a memcpy node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - const pointer to the kernel node parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           hipMemcpy3DParms* pNodeParams);

/**
 * @brief Creates a 1D memcpy node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to graph node to create.
 * @param [in] graph - instance of graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] dst - pointer to memory address to the destination.
 * @param [in] src - pointer to memory address to the source.
 * @param [in] count - the size of the memory to copy.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   void* dst, const void* src, size_t count, hipMemcpyKind kind);

/**
 * @brief Sets a memcpy node's parameters to perform a 1-dimensional copy.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] dst - pointer to memory address to the destination.
 * @param [in] src - pointer to memory address to the source.
 * @param [in] count - the size of the memory to copy.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst, const void* src,
                                         size_t count, hipMemcpyKind kind);

/**
 * @brief Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional
 * copy.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] dst - pointer to memory address to the destination.
 * @param [in] src - pointer to memory address to the source.
 * @param [in] count - the size of the memory to copy.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                             void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind);

/**
 * @brief Creates a memcpy node to copy from a symbol on the device and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to graph node to create.
 * @param [in] graph - instance of graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] dst - pointer to memory address to the destination.
 * @param [in] symbol - Device symbol address.
 * @param [in] count - the size of the memory to copy.
 * @param [in] offset - Offset from start of symbol in bytes.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                           const hipGraphNode_t* pDependencies,
                                           size_t numDependencies, void* dst, const void* symbol,
                                           size_t count, size_t offset, hipMemcpyKind kind);

/**
 * @brief Sets a memcpy node's parameters to copy from a symbol on the device.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] dst - pointer to memory address to the destination.
 * @param [in] symbol - Device symbol address.
 * @param [in] count - the size of the memory to copy.
 * @param [in] offset - Offset from start of symbol in bytes.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst, const void* symbol,
                                                 size_t count, size_t offset, hipMemcpyKind kind);

/**
 * @brief Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the
 * * device.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] dst - pointer to memory address to the destination.
 * @param [in] symbol - Device symbol address.
 * @param [in] count - the size of the memory to copy.
 * @param [in] offset - Offset from start of symbol in bytes.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                     void* dst, const void* symbol, size_t count,
                                                     size_t offset, hipMemcpyKind kind);

/**
 * @brief Creates a memcpy node to copy to a symbol on the device and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to graph node to create.
 * @param [in] graph - instance of graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memcpy execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] symbol - Device symbol address.
 * @param [in] src - pointer to memory address of the src.
 * @param [in] count - the size of the memory to copy.
 * @param [in] offset - Offset from start of symbol in bytes.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies, const void* symbol,
                                         const void* src, size_t count, size_t offset,
                                         hipMemcpyKind kind);

/**
 * @brief Sets a memcpy node's parameters to copy to a symbol on the device.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] symbol - Device symbol address.
 * @param [in] src - pointer to memory address of the src.
 * @param [in] count - the size of the memory to copy.
 * @param [in] offset - Offset from start of symbol in bytes.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void* symbol,
                                               const void* src, size_t count, size_t offset,
                                               hipMemcpyKind kind);


/**
 * @brief Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the
 * device.
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] symbol - Device symbol address.
 * @param [in] src - pointer to memory address of the src.
 * @param [in] count - the size of the memory to copy.
 * @param [in] offset - Offset from start of symbol in bytes.
 * @param [in] kind - the type of memory copy.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                   const void* symbol, const void* src,
                                                   size_t count, size_t offset, hipMemcpyKind kind);

/**
 * @brief Creates a memset node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create.
 * @param [in] graph - instance of the graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] pMemsetParams - const pointer to the parameters for the memory set.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemsetParams* pMemsetParams);

/**
 * @brief Gets a memset node's parameters.
 *
 * @param [in] node - instane of the node to get parameters from.
 * @param [out] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams* pNodeParams);

/**
 * @brief Sets a memset node's parameters.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams* pNodeParams);

/**
 * @brief Sets the parameters for a memset node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipMemsetParams* pNodeParams);

/**
 * @brief Creates a host execution node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create.
 * @param [in] graph - instance of the graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] pNodeParams -pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                               const hipGraphNode_t* pDependencies, size_t numDependencies,
                               const hipHostNodeParams* pNodeParams);

/**
 * @brief Returns a host node's parameters.
 *
 * @param [in] node - instane of the node to get parameters from.
 * @param [out] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams);

/**
 * @brief Sets a host node's parameters.
 *
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams* pNodeParams);

/**
 * @brief Sets the parameters for a host node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - instance of the node to set parameters to.
 * @param [in] pNodeParams - pointer to the parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                         const hipHostNodeParams* pNodeParams);

/**
 * @brief Creates a child graph node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create.
 * @param [in] graph - instance of the graph to add the created node.
 * @param [in] pDependencies - const pointer to the dependencies on the memset execution node.
 * @param [in] numDependencies - the number of the dependencies.
 * @param [in] childGraph - the graph to clone into this node
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* pDependencies, size_t numDependencies,
                                     hipGraph_t childGraph);

/**
 * @brief Gets a handle to the embedded graph of a child graph node.
 *
 * @param [in] node - instane of the node to get child graph.
 * @param [out] pGraph - pointer to get the graph.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph);

/**
 * @brief Updates node parameters in the child graph node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] node - node from the graph which was used to instantiate graphExec.
 * @param [in] childGraph - child graph with updated parameters.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                               hipGraph_t childGraph);

/**
 * @brief Creates an empty node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
 * @param [in] graph - instane of the graph the node is add to.
 * @param [in] pDependencies - const pointer to the node dependenties.
 * @param [in] numDependencies - the number of dependencies.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                const hipGraphNode_t* pDependencies, size_t numDependencies);


/**
 * @brief Creates an event record node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
 * @param [in] graph - instane of the graph the node to be added.
 * @param [in] pDependencies - const pointer to the node dependenties.
 * @param [in] numDependencies - the number of dependencies.
 * @param [in] event - Event for the node.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                      const hipGraphNode_t* pDependencies, size_t numDependencies,
                                      hipEvent_t event);

/**
 * @brief Returns the event associated with an event record node.
 *
 * @param [in] node -  instane of the node to get event from.
 * @param [out] event_out - Pointer to return the event.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out);

/**
 * @brief Sets an event record node's event.
 *
 * @param [in] node - instane of the node to set event to.
 * @param [in] event - pointer to the event.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event);

/**
 * @brief Sets the event for an event record node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] hNode - node from the graph which was used to instantiate graphExec.
 * @param [in] event - pointer to the event.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               hipEvent_t event);

/**
 * @brief Creates an event wait node and adds it to a graph.
 *
 * @param [out] pGraphNode - pointer to the graph node to create and add to the graph.
 * @param [in] graph - instane of the graph the node to be added.
 * @param [in] pDependencies - const pointer to the node dependenties.
 * @param [in] numDependencies - the number of dependencies.
 * @param [in] event - Event for the node.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    hipEvent_t event);


/**
 * @brief Returns the event associated with an event wait node.
 *
 * @param [in] node -  instane of the node to get event from.
 * @param [out] event_out - Pointer to return the event.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out);

/**
 * @brief Sets an event wait node's event.
 *
 * @param [in] node - instane of the node to set event to.
 * @param [in] event - pointer to the event.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event);

/**
 * @brief Sets the event for an event record node in the given graphExec.
 *
 * @param [in] hGraphExec - instance of the executable graph with the node.
 * @param [in] hNode - node from the graph which was used to instantiate graphExec.
 * @param [in] event - pointer to the event.
 * @returns #hipSuccess, #hipErrorInvalidValue
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 */
hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                             hipEvent_t event);

// doxygen end graph API
/**
 * @}
 */

#ifdef __cplusplus
} /* extern "c" */
#endif
#ifdef __cplusplus
#if defined(__clang__) && defined(__HIP__)
template <typename T>
static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
    T f, size_t dynSharedMemPerBlk = 0, int blockSizeLimit = 0) {
    return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, reinterpret_cast<const void*>(f),dynSharedMemPerBlk,blockSizeLimit);
}
template <typename T>
static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize,
    T f, size_t dynSharedMemPerBlk = 0, int blockSizeLimit = 0, unsigned int  flags = 0 ) {
    return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, reinterpret_cast<const void*>(f),dynSharedMemPerBlk,blockSizeLimit);
}
#endif // defined(__clang__) && defined(__HIP__)
template <typename T>
hipError_t hipGetSymbolAddress(void** devPtr, const T &symbol) {
  return ::hipGetSymbolAddress(devPtr, (const void *)&symbol);
}
template <typename T>
hipError_t hipGetSymbolSize(size_t* size, const T &symbol) {
  return ::hipGetSymbolSize(size, (const void *)&symbol);
}
template <typename T>
hipError_t hipMemcpyToSymbol(const T& symbol, const void* src, size_t sizeBytes,
                             size_t offset __dparm(0),
                             hipMemcpyKind kind __dparm(hipMemcpyHostToDevice)) {
  return ::hipMemcpyToSymbol((const void*)&symbol, src, sizeBytes, offset, kind);
}
template <typename T>
hipError_t hipMemcpyToSymbolAsync(const T& symbol, const void* src, size_t sizeBytes, size_t offset,
                                  hipMemcpyKind kind, hipStream_t stream __dparm(0)) {
  return ::hipMemcpyToSymbolAsync((const void*)&symbol, src, sizeBytes, offset, kind, stream);
}
template <typename T>
hipError_t hipMemcpyFromSymbol(void* dst, const T &symbol,
                               size_t sizeBytes, size_t offset __dparm(0),
                               hipMemcpyKind kind __dparm(hipMemcpyDeviceToHost)) {
  return ::hipMemcpyFromSymbol(dst, (const void*)&symbol, sizeBytes, offset, kind);
}
template <typename T>
hipError_t hipMemcpyFromSymbolAsync(void* dst, const T& symbol, size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind, hipStream_t stream __dparm(0)) {
  return ::hipMemcpyFromSymbolAsync(dst, (const void*)&symbol, sizeBytes, offset, kind, stream);
}
template <class T>
inline hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, T f, int blockSize, size_t dynSharedMemPerBlk) {
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks, reinterpret_cast<const void*>(f), blockSize, dynSharedMemPerBlk);
}
template <class T>
inline hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, T f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags) {
    return hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, reinterpret_cast<const void*>(f), blockSize, dynSharedMemPerBlk, flags);
}
template <typename F>
inline hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                                    F kernel, size_t dynSharedMemPerBlk, uint32_t blockSizeLimit) {
return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize,(hipFunction_t)kernel, dynSharedMemPerBlk, blockSizeLimit);
}
template <class T>
inline hipError_t hipLaunchCooperativeKernel(T f, dim3 gridDim, dim3 blockDim,
                                             void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream) {
    return hipLaunchCooperativeKernel(reinterpret_cast<const void*>(f), gridDim,
                                      blockDim, kernelParams, sharedMemBytes, stream);
}
template <class T>
inline hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                        unsigned int  numDevices, unsigned int  flags = 0) {
    return hipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
}
template <class T>
inline hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                     unsigned int  numDevices, unsigned int  flags = 0) {
    return hipExtLaunchMultiKernelMultiDevice(launchParamsList, numDevices, flags);
}
hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject, const hipResourceDesc* pResDesc);
hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject);
template <class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipBindTexture(size_t* offset, const struct texture<T, dim, readMode>& tex,
                                        const void* devPtr, size_t size = UINT_MAX) {
    return hipBindTexture(offset, &tex, devPtr, &tex.channelDesc, size);
}
template <class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t
    hipBindTexture(size_t* offset, const struct texture<T, dim, readMode>& tex, const void* devPtr,
                   const struct hipChannelFormatDesc& desc, size_t size = UINT_MAX) {
    return hipBindTexture(offset, &tex, devPtr, &desc, size);
}
template<class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipBindTexture2D(
    size_t *offset,
    const struct texture<T, dim, readMode> &tex,
    const void *devPtr,
    size_t width,
    size_t height,
    size_t pitch)
{
    return hipBindTexture2D(offset, &tex, devPtr, &tex.channelDesc, width, height, pitch);
}
template<class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipBindTexture2D(
  size_t *offset,
  const struct texture<T, dim, readMode> &tex,
  const void *devPtr,
  const struct hipChannelFormatDesc &desc,
  size_t width,
  size_t height,
  size_t pitch)
{
  return hipBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch);
}
template<class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipBindTextureToArray(
    const struct texture<T, dim, readMode> &tex,
    hipArray_const_t array)
{
    struct hipChannelFormatDesc desc;
    hipError_t err = hipGetChannelDesc(&desc, array);
    return (err == hipSuccess) ? hipBindTextureToArray(&tex, array, &desc) : err;
}
template<class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipBindTextureToArray(
    const struct texture<T, dim, readMode> &tex,
    hipArray_const_t array,
    const struct hipChannelFormatDesc &desc)
{
    return hipBindTextureToArray(&tex, array, &desc);
}
template<class T, int dim, enum hipTextureReadMode readMode>
static inline hipError_t hipBindTextureToMipmappedArray(
    const struct texture<T, dim, readMode> &tex,
    hipMipmappedArray_const_t mipmappedArray)
{
    struct hipChannelFormatDesc desc;
    hipArray_t levelArray;
    hipError_t err = hipGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0);
    if (err != hipSuccess) {
        return err;
    }
    err = hipGetChannelDesc(&desc, levelArray);
    return (err == hipSuccess) ? hipBindTextureToMipmappedArray(&tex, mipmappedArray, &desc) : err;
}
template<class T, int dim, enum hipTextureReadMode readMode>
static inline hipError_t hipBindTextureToMipmappedArray(
    const struct texture<T, dim, readMode> &tex,
    hipMipmappedArray_const_t mipmappedArray,
    const struct hipChannelFormatDesc &desc)
{
    return hipBindTextureToMipmappedArray(&tex, mipmappedArray, &desc);
}
template<class T, int dim, enum hipTextureReadMode readMode>
DEPRECATED(DEPRECATED_MSG)
static inline hipError_t hipUnbindTexture(
    const struct texture<T, dim, readMode> &tex)
{
    return hipUnbindTexture(&tex);
}


#endif // __cplusplus

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup GL Interop
 *  @{
 *  This section describes Stream Memory Wait and Write functions of HIP runtime API.
 */
typedef unsigned int GLuint;

// Queries devices associated with GL Context.
hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices,
                           unsigned int hipDeviceCount, hipGLDeviceList deviceList);
// Registers a GL Buffer for interop and returns corresponding graphics resource.
hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer,
                                       unsigned int flags);
// Maps a graphics resource for hip access.
hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources,
                                   hipStream_t stream  __dparm(0) );
// Gets device accessible address of a graphics resource.
hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size,
                                               hipGraphicsResource_t resource);
// Unmaps a graphics resource for hip access.
hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources,
                                     hipStream_t stream  __dparm(0));
// Unregisters a graphics resource.
hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource);
// doxygen end GL Interop
/**
 * @}
 */

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
// doxygen end HIP API
/**
 *   @}
 */

#elif !(defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)) && (defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__))
#include "hip/nvidia_detail/nvidia_hip_runtime_api.h"
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

#if USE_PROF_API
#include <hip/amd_detail/hip_prof_str.h>
#endif
