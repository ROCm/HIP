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

//#pragma once
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_RUNTIME_API_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_RUNTIME_API_H
/**
 *  @file  amd_detail/hip_runtime_api.h
 *  @brief Contains C function APIs for HIP runtime. This file does not use any HCC builtin or
 * special language extensions (-hc mode) ; those functions in hip_runtime.h.
 */
#include <stdint.h>
#include <stddef.h>

#ifndef GENERIC_GRID_LAUNCH
#define GENERIC_GRID_LAUNCH 1
#endif

#include <hip/amd_detail/host_defines.h>
#include <hip/amd_detail/driver_types.h>
#include <hip/amd_detail/hip_texture_types.h>
#include <hip/amd_detail/hip_surface_types.h>

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
    0x80000000  /// < Use a system-scope release that when recording this event.  This flag is
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
    hipMemAdviseUnsetAccessedBy = 6         ///< Let the Unified Memory subsystem decide on
                                            ///< the page faulting policy for the specified device
} hipMemoryAdvise;

/*
 * @brief HIP range attributes
 * @enum
 * @ingroup Enumerations
 */
typedef enum hipMemRangeAttribute {
    hipMemRangeAttributeReadMostly = 1,         ///< Whether the range will mostly be read and
                                                ///< only occassionally be written to
    hipMemRangeAttributePreferredLocation = 2,  ///< The preferred location of the range
    hipMemRangeAttributeAccessedBy = 3,         ///< Memory range has cudaMemAdviseSetAccessedBy
                                                ///< set for specified device
    hipMemRangeAttributeLastPrefetchLocation = 4,///< The last location to which the range was prefetched
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

#if __HIP_HAS_GET_PCH
/**
 * Internal use only. This API may change in the future
 * Pre-Compiled header for online compilation
 *
 */
    void __hipGetPCH(const char** pch, unsigned int*size);
#endif


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
 * @warning On HIP/HCC path this function returns HIP runtime patch version however on
 * HIP/NVCC path this function return CUDA runtime version.
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


hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event);
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
 *  - cudaStreamAttachMemAsync
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
 * cudaStreamAddCallback call, a callback will be executed exactly once.
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
 * @brief Enqueues a wait command to the stream.
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
 * @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue64, hipStreamWriteValue64,
 * hipStreamWriteValue32, hipDeviceGetAttribute
 */

hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags,
                                uint32_t mask __dparm(0xFFFFFFFF));

/**
 * @brief Enqueues a wait command to the stream.
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
 * @see hipExtMallocWithFlags, hipFree, hipStreamWaitValue32, hipStreamWriteValue64,
 * hipStreamWriteValue32, hipDeviceGetAttribute
 */

hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags,
                                uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));

/**
 * @brief Enqueues a write command to the stream.
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
 * @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
 * hipStreamWaitValue64
 */
hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags);

/**
 * @brief Enqueues a write command to the stream.
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
 * @see hipExtMallocWithFlags, hipFree, hipStreamWriteValue32, hipStreamWaitValue32,
 * hipStreamWaitValue64
 */
hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags);


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
 *  @addtogroup MemoryM Managed Memory (ROCm HMM)
 *  @{
 *  @ingroup Memory
 *  This section describes the managed memory management functions of HIP runtime API.
 *
 */

/**
 * @brief Allocates memory that will be automatically managed by AMD HMM.
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
 * @brief Prefetches memory to the specified destination device using AMD HMM.
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
 * @brief Advise about the usage of a given memory range to AMD HMM.
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
 * @brief Query an attribute of a given memory range in AMD HMM.
 *
 * @param [in/out] data   a pointer to a memory location where the result of each
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
 * @brief Query attributes of a given memory range in AMD HMM.
 *
 * @param [in/out] data     a two-dimensional array containing pointers to memory locations
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
 * @brief Attach memory to a stream asynchronously in AMD HMM.
 *
 * @param [in] stream     - stream in which to enqueue the attach operation
 * @param [in] dev_ptr    - pointer to memory (must be a pointer to managed memory or
 *                          to a valid host-accessible region of system-allocated memory)
 * @param [in] length     - length of memory (defaults to zero)
 * @param [in] flags      - must be one of cudaMemAttachGlobal, cudaMemAttachHost or
 *                          cudaMemAttachSingle (defaults to cudaMemAttachSingle)
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipStreamAttachMemAsync(hipStream_t stream,
                                   hipDeviceptr_t* dev_ptr,
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
 * @param [in] hipLaunchParams          List of launch parameters, one per device.
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
 *  @defgroup Textur Texture Management
 *  @{
 *  This section describes the texture management functions of HIP runtime API.
 */

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

// doxygen end deprecated texture management
/**
 * @}
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

hipError_t hipTexRefGetAddress(
    hipDeviceptr_t* dev_ptr,
    const textureReference* texRef);

hipError_t hipTexRefGetAddressMode(
    enum hipTextureAddressMode* pam,
    const textureReference* texRef,
    int dim);

hipError_t hipTexRefGetFilterMode(
    enum hipTextureFilterMode* pfm,
    const textureReference* texRef);

hipError_t hipTexRefGetFlags(
    unsigned int* pFlags,
    const textureReference* texRef);

hipError_t hipTexRefGetFormat(
    hipArray_Format* pFormat,
    int* pNumChannels,
    const textureReference* texRef);

hipError_t hipTexRefGetMaxAnisotropy(
    int* pmaxAnsio,
    const textureReference* texRef);

hipError_t hipTexRefGetMipmapFilterMode(
    enum hipTextureFilterMode* pfm,
    const textureReference* texRef);

hipError_t hipTexRefGetMipmapLevelBias(
    float* pbias,
    const textureReference* texRef);

hipError_t hipTexRefGetMipmapLevelClamp(
    float* pminMipmapLevelClamp,
    float* pmaxMipmapLevelClamp,
    const textureReference* texRef);

hipError_t hipTexRefGetMipMappedArray(
    hipMipmappedArray_t* pArray,
    const textureReference* texRef);

hipError_t hipTexRefSetAddress(
    size_t* ByteOffset,
    textureReference* texRef,
    hipDeviceptr_t dptr,
    size_t bytes);

hipError_t hipTexRefSetAddress2D(
    textureReference* texRef,
    const HIP_ARRAY_DESCRIPTOR* desc,
    hipDeviceptr_t dptr,
    size_t Pitch);

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

hipError_t hipTexRefSetMaxAnisotropy(
    textureReference* texRef,
    unsigned int maxAniso);

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

// doxygen end Texture management
/**
 * @}
 */

// The following are not supported.
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

/**
 * Callback/Activity API
 */
hipError_t hipRegisterApiCallback(uint32_t id, void* fun, void* arg);
hipError_t hipRemoveApiCallback(uint32_t id);
hipError_t hipRegisterActivityCallback(uint32_t id, void* fun, void* arg);
hipError_t hipRemoveActivityCallback(uint32_t id);
const char* hipApiName(uint32_t id);
const char* hipKernelNameRef(const hipFunction_t f);
const char* hipKernelNameRefByPtr(const void* hostFunction, hipStream_t stream);
int hipGetStreamDeviceId(hipStream_t stream);

#ifdef __cplusplus
} /* extern "c" */
#endif


#if USE_PROF_API
#include <hip/amd_detail/hip_prof_str.h>
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

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

// doxygen end HIP API
/**
 *   @}
 */

#endif // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_RUNTIME_API_H
