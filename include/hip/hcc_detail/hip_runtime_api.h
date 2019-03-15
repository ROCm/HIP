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
#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_API_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_API_H
/**
 *  @file  hcc_detail/hip_runtime_api.h
 *  @brief Contains C function APIs for HIP runtime. This file does not use any HCC builtin or
 * special language extensions (-hc mode) ; those functions in hip_runtime.h.
 */
#include <stdint.h>
#include <stddef.h>

#ifndef GENERIC_GRID_LAUNCH
#define GENERIC_GRID_LAUNCH 1
#endif

#ifndef __HIP_VDI__
#define __HIP_VDI__ 0
#endif

#include <hip/hcc_detail/host_defines.h>
#include <hip/hip_runtime_api.h>
#include <hip/hcc_detail/driver_types.h>
#include <hip/hcc_detail/hip_texture_types.h>
#include <hip/hcc_detail/hip_surface_types.h>

#if !__HIP_VDI__
#include <hsa/hsa.h>
#include <hip/hcc_detail/program_state.hpp>
#endif

#if defined(_MSC_VER)
#define DEPRECATED(msg) __declspec(deprecated(msg))
#else // !defined(_MSC_VER)
#define DEPRECATED(msg) __attribute__ ((deprecated(msg)))
#endif // !defined(_MSC_VER)

#define DEPRECATED_MSG "This API is marked as deprecated and may not be supported in future releases.For more details please refer https://github.com/ROCm-Developer-Tools/HIP/tree/master/docs/markdown/hip_deprecated_api_list"

#if defined(__HCC__) && (__hcc_workweek__ < 16155)
#error("This version of HIP requires a newer version of HCC.");
#endif

#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#define HIP_LAUNCH_PARAM_END ((void*)0x03)

#ifdef __cplusplus
  #include <mutex>
  #include <string>
  #include <unordered_map>
  #include <vector>

  #define __dparm(x) \
          = x
#else
  #define __dparm(x)
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

typedef struct ihipStream_t* hipStream_t;

// TODO: IPC implementation

#define hipIpcMemLazyEnablePeerAccess 0

#define HIP_IPC_HANDLE_SIZE 64

typedef struct hipIpcMemHandle_st {
    char reserved[HIP_IPC_HANDLE_SIZE];
} hipIpcMemHandle_t;

// TODO: IPC event handle currently unsupported
struct ihipIpcEventHandle_t;
typedef struct ihipIpcEventHandle_t* hipIpcEventHandle_t;


// END TODO

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
#define hipHostMallocCoherent                                                                      \
    0x40000000  ///< Allocate coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific
                ///< allocation.
#define hipHostMallocNonCoherent                                                                   \
    0x80000000  ///< Allocate non-coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific
                ///< allocation.


//! Flags that can be used with hipHostRegister
#define hipHostRegisterDefault 0x0   ///< Memory is Mapped and Portable
#define hipHostRegisterPortable 0x1  ///< Memory is considered registered by all contexts.
#define hipHostRegisterMapped                                                                      \
    0x2  ///< Map the allocation into the address space for the current device.  The device pointer
         ///< can be obtained with #hipHostGetDevicePointer.
#define hipHostRegisterIoMemory 0x4  ///< Not supported.


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
    dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z){};
#endif
} dim3;


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
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Device Device Management
 *  @{
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
 * @returns #hipSuccess
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
 * @returns #hipSuccess, #hipErrorInitializationError
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
 * @returns #hipSuccess, #hipErrorInitializationError
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
 * @brief Set Cache configuration for a specific function
 *
 * @param [in] config;
 *
 * @returns #hipSuccess, #hipErrorInitializationError
 * Note: AMD devices and some Nvidia GPUS do not support reconfigurable cache.  This hint is ignored
 * on those architectures.
 *
 */
hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t config);

/**
 * @brief Returns bank width of shared memory for current device
 *
 * @param [out] pConfig
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInitializationError
 *
 * Note: AMD devices and some Nvidia GPUS do not support shared cache banking, and the hint is
 * ignored on those architectures.
 *
 */
hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig);


/**
 * @brief The bank width of shared memory on current device is set
 *
 * @param [in] config
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInitializationError
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

// end doxygen Device
/**
 * @}
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Error Error Handling
 *  @{
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
 *
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
 * @return #hipSuccess #hipErrorInvalidResourceHandle
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
 * @return #hipSuccess, #hipErrorNotReady, #hipErrorInvalidResourceHandle
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
 * @return #hipSuccess, #hipErrorInvalidResourceHandle
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
 * @return #hipSuccess, #hipErrorInvalidResourceHandle
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
 * @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidResourceHandle
 *
 * @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidResourceHandle
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
 * @return #hipSuccess, #hipErrorInvalidValue, #hipErrorInvalidResourceHandle
 *
 * @returns #hipSuccess #hipErrorInvalidValue #hipErrorInvalidResourceHandle
 *
 * Query the priority of a stream. The priority is returned in in priority.
 *
 * @see hipStreamCreateWithFlags
 */
hipError_t hipStreamGetPriority(hipStream_t stream, int* priority);


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
 * @return #hipSuccess, #hipErrorInvalidResourceHandle, #hipErrorNotSupported
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
 *  @defgroup Event Event Management
 *  @{
 */

/**
 * @brief Create an event with the specified flags
 *
 * @param[in,out] event Returns the newly created event.
 * @param[in] flags     Flags to control event behavior.  Valid values are #hipEventDefault,
 #hipEventBlockingSync, #hipEventDisableTiming, #hipEventInterprocess

 * #hipEventDefault : Default flag.  The event will use active synchronization and will support
 timing.  Blocking synchronization provides lowest possible latency at the expense of dedicating a
 CPU to poll on the eevent.
 * #hipEventBlockingSync : The event will use blocking synchronization : if hipEventSynchronize is
 called on this event, the thread will block until the event completes.  This can increase latency
 for the synchroniation but can result in lower power and more resources for other CPU threads.
 * #hipEventDisableTiming : Disable recording of timing information.  On ROCM platform, timing
 information is always recorded and this flag has no performance benefit.

 * @warning On HCC platform, hipEventInterprocess support is under development.  Use of this flag
 will return an error.
 *
 * @returns #hipSuccess, #hipErrorInitializationError, #hipErrorInvalidValue,
 #hipErrorLaunchFailure, #hipErrorMemoryAllocation
 *
 * @see hipEventCreate, hipEventSynchronize, hipEventDestroy, hipEventElapsedTime
 */
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);


/**
 *  Create an event
 *
 * @param[in,out] event Returns the newly created event.
 *
 * @returns #hipSuccess, #hipErrorInitializationError, #hipErrorInvalidValue,
 * #hipErrorLaunchFailure, #hipErrorMemoryAllocation
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
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInitializationError,
 * #hipErrorInvalidResourceHandle, #hipErrorLaunchFailure
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
 * If this function is called on a an event that is currently being recorded, results are undefined
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
 *  @returns #hipSuccess, #hipErrorInitializationError, #hipErrorInvalidValue,
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
 *  TODO-hcc - This function needs to support hipEventBlockingSync parameter.
 *
 *  @param[in] event Event on which to wait.
 *  @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInitializationError,
 * #hipErrorInvalidResourceHandle, #hipErrorLaunchFailure
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
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotReady, #hipErrorInvalidResourceHandle,
 * #hipErrorInitializationError, #hipErrorLaunchFailure
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
 * If hipEventRecord() has not been called on either event, then #hipErrorInvalidResourceHandle is
 * returned. If hipEventRecord() has been called on both events, but the timestamp has not yet been
 * recorded on one or both events (that is, hipEventQuery() would return #hipErrorNotReady on at
 * least one of the events), then #hipErrorNotReady is returned.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
 * hipEventSynchronize
 */
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);


/**
 * @brief Query event status
 *
 * @param[in] event Event to query.
 * @returns #hipSuccess, #hipErrorNotReady, #hipErrorInvalidResourceHandle, #hipErrorInvalidValue,
 * #hipErrorInitializationError, #hipErrorLaunchFailure
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
 *
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
 *  @return #hipSuccess, #hipErrorMemoryAllocation, #hipErrorInvalidValue (bad context, null *ptr)
 *
 *  @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
 * hipHostFree, hipHostMalloc
 */
hipError_t hipMalloc(void** ptr, size_t size);

/**
 *  @brief Allocate pinned host memory [Deprecated]
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorMemoryAllocation
 *
 *  @deprecated use hipHostMalloc() instead
 */
DEPRECATED("use hipHostMalloc instead")
hipError_t hipMallocHost(void** ptr, size_t size);

/**
 *  @brief Allocate device accessible page locked host memory
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size
 *  @param[in]  flags Type of host memory allocation
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorMemoryAllocation
 *
 *  @see hipSetDeviceFlags, hipHostFree
 */
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);

/**
 *  @brief Allocate device accessible page locked host memory [Deprecated]
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size
 *  @param[in]  flags Type of host memory allocation
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *
 *  @return #hipSuccess, #hipErrorMemoryAllocation
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
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryAllocation
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
 *  @return #hipSuccess, #hipErrorMemoryAllocation
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
 * device where the src data is physically located. For optimal peer-to-peer copies, the copy device
 * must be able to access the src and dst pointers (by calling hipDeviceEnablePeerAccess with copy
 * agent as the current device and src/dest as the peerDevice argument.  if this is not done, the
 * hipMemcpy will still work, but will perform the copy using a staging buffer on the host.
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

#if !__HIP_VDI__
__attribute__((visibility("hidden")))
hipError_t hipModuleGetGlobal(void**, size_t*, hipModule_t, const char*);


/**
 *  @brief Copies the memory address of symbol @p symbolName to @p devPtr
 *
 * @param[in]  symbolName - Symbol on device
 * @param[out] devPtr - Pointer to a pointer to the memory referred to by the symbol
 * @return #hipSuccess, #hipErrorNotInitialized, #hipErrorNotFound
 *
 *  @see hipGetSymbolSize, hipMemcpyToSymbol, hipMemcpyFromSymbol, hipMemcpyToSymbolAsync,
 * hipMemcpyFromSymbolAsync
 */
inline
__attribute__((visibility("hidden")))
hipError_t hipGetSymbolAddress(void** devPtr, const void* symbolName) {
    //HIP_INIT_API(hipGetSymbolAddress, devPtr, symbolName);

    size_t size = 0;
    return hipModuleGetGlobal(devPtr, &size, 0, (const char*)symbolName);
}


/**
 *  @brief Copies the size of symbol @p symbolName to @p size
 *
 * @param[in]  symbolName - Symbol on device
 * @param[out] size - Pointer to the size of the symbol
 * @return #hipSuccess, #hipErrorNotInitialized, #hipErrorNotFound
 *
 *  @see hipGetSymbolSize, hipMemcpyToSymbol, hipMemcpyFromSymbol, hipMemcpyToSymbolAsync,
 * hipMemcpyFromSymbolAsync
 */
inline
__attribute__((visibility("hidden")))
hipError_t hipGetSymbolSize(size_t* size, const void* symbolName) {
    // HIP_INIT_API(hipGetSymbolSize, size, symbolName);

    void* devPtr = nullptr;
    return hipModuleGetGlobal(&devPtr, size, 0, (const char*)symbolName);
}

#if defined(__cplusplus)
} // extern "C"
#endif

namespace hip_impl {
hipError_t hipMemcpyToSymbol(void*, const void*, size_t, size_t, hipMemcpyKind,
                             const char*);
} // Namespace hip_impl.

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Copies @p sizeBytes bytes from the memory area pointed to by @p src to the memory area
 * pointed to by @p offset bytes from the start of symbol @p symbol.
 *
 *  The memory areas may not overlap. Symbol can either be a variable that resides in global or
 * constant memory space, or it can be a character string, naming a variable that resides in global
 * or constant memory space. Kind can be either hipMemcpyHostToDevice or hipMemcpyDeviceToDevice
 *  TODO: cudaErrorInvalidSymbol and cudaErrorInvalidMemcpyDirection is not supported, use
 * hipErrorUnknown for now.
 *
 *  @param[in]  symbolName - Symbol destination on device
 *  @param[in]  src - Data being copy from
 *  @param[in]  sizeBytes - Data size in bytes
 *  @param[in]  offset - Offset from start of symbol in bytes
 *  @param[in]  kind - Type of transfer
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
 *
 *  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
 * hipMemcpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyFromSymbol,
 * hipMemcpyAsync, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync,
 * hipMemcpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync,
 * hipMemcpyFromSymbolAsync
 */
inline
__attribute__((visibility("hidden")))
hipError_t hipMemcpyToSymbol(const void* symbolName, const void* src,
                             size_t sizeBytes, size_t offset __dparm(0),
                             hipMemcpyKind kind __dparm(hipMemcpyHostToDevice)) {
    if (!symbolName) return hipErrorInvalidSymbol;

    hipDeviceptr_t dst = NULL;
    hipGetSymbolAddress(&dst, (const char*)symbolName);

    return hip_impl::hipMemcpyToSymbol(dst, src, sizeBytes, offset, kind,
                                       (const char*)symbolName);
}

#if defined(__cplusplus)
} // extern "C"
#endif

namespace hip_impl {
hipError_t hipMemcpyToSymbolAsync(void*, const void*, size_t, size_t,
                                  hipMemcpyKind, hipStream_t, const char*);
hipError_t hipMemcpyFromSymbol(void*, const void*, size_t, size_t,
                               hipMemcpyKind, const char*);
hipError_t hipMemcpyFromSymbolAsync(void*, const void*, size_t, size_t,
                                    hipMemcpyKind, hipStream_t, const char*);
} // Namespace hip_impl.

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Copies @p sizeBytes bytes from the memory area pointed to by @p src to the memory area
 * pointed to by @p offset bytes from the start of symbol @p symbol
 *
 *  The memory areas may not overlap. Symbol can either be a variable that resides in global or
 * constant memory space, or it can be a character string, naming a variable that resides in global
 * or constant memory space. Kind can be either hipMemcpyHostToDevice or hipMemcpyDeviceToDevice
 *  hipMemcpyToSymbolAsync() is asynchronous with respect to the host, so the call may return before
 * copy is complete.
 *  TODO: cudaErrorInvalidSymbol and cudaErrorInvalidMemcpyDirection is not supported, use
 * hipErrorUnknown for now.
 *
 *  @param[in]  symbolName - Symbol destination on device
 *  @param[in]  src - Data being copy from
 *  @param[in]  sizeBytes - Data size in bytes
 *  @param[in]  offset - Offset from start of symbol in bytes
 *  @param[in]  kind - Type of transfer
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
 *
 *  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray,
 * hipMemcpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyFromSymbol,
 * hipMemcpyAsync, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync,
 * hipMemcpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync,
 * hipMemcpyFromSymbolAsync
 */
inline
__attribute__((visibility("hidden")))
hipError_t hipMemcpyToSymbolAsync(const void* symbolName, const void* src,
                                  size_t sizeBytes, size_t offset,
                                  hipMemcpyKind kind, hipStream_t stream __dparm(0)) {
    if (!symbolName) return hipErrorInvalidSymbol;

    hipDeviceptr_t dst = NULL;
    hipGetSymbolAddress(&dst, symbolName);

    return hip_impl::hipMemcpyToSymbolAsync(dst, src, sizeBytes, offset, kind,
                                            stream,
                                            (const char*)symbolName);
}

inline
__attribute__((visibility("hidden")))
hipError_t hipMemcpyFromSymbol(void* dst, const void* symbolName,
                               size_t sizeBytes, size_t offset __dparm(0),
                               hipMemcpyKind kind __dparm(hipMemcpyDeviceToHost)) {
    if (!symbolName) return hipErrorInvalidSymbol;

    hipDeviceptr_t src = NULL;
    hipGetSymbolAddress(&src, symbolName);

    return hip_impl::hipMemcpyFromSymbol(dst, src, sizeBytes, offset, kind,
                                         (const char*)symbolName);
}

inline
__attribute__((visibility("hidden")))
hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbolName,
                                    size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind,
                                    hipStream_t stream __dparm(0)) {
    if (!symbolName) return hipErrorInvalidSymbol;

    hipDeviceptr_t src = NULL;
    hipGetSymbolAddress(&src, symbolName);

    return hip_impl::hipMemcpyFromSymbolAsync(dst, src, sizeBytes, offset, kind,
                                              stream,
                                              (const char*)symbolName);
}

#endif // __HIP_VDI__
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
 *  @param[in]  sizeBytes Data size in bytes
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorNotInitialized
 */
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t sizeBytes);

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
 *  @return      #hipSuccess, #hipErrorMemoryAllocation
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
 */
hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc, size_t width,
                          size_t height __dparm(0), unsigned int flags __dparm(hipArrayDefault));
hipError_t hipArrayCreate(hipArray** pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray);

hipError_t hipArray3DCreate(hipArray** array, const HIP_ARRAY_DESCRIPTOR* pAllocateArray);

hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent);

/**
 *  @brief Frees an array on the device.
 *
 *  @param[in]  array  Pointer to array to free
 *  @return     #hipSuccess, #hipErrorInvalidValue, #hipErrorInitializationError
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipHostMalloc, hipHostFree
 */
hipError_t hipFreeArray(hipArray* array);

/**
 *  @brief Allocate an array on the device.
 *
 *  @param[out]  array  Pointer to allocated array in device memory
 *  @param[in]   desc   Requested channel format
 *  @param[in]   extent Requested array allocation width, height and depth
 *  @param[in]   flags  Requested properties of allocated array
 *  @return      #hipSuccess, #hipErrorMemoryAllocation
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipFreeArray, hipHostMalloc, hipHostFree
 */

hipError_t hipMalloc3DArray(hipArray** array, const struct hipChannelFormatDesc* desc,
                            struct hipExtent extent, unsigned int flags);
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
hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy);

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
 *  @see hipMemcpy, hipMemcpyToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind);

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
 *  @see hipMemcpy, hipMemcpy2DToArray, hipMemcpy2D, hipMemcpyFromArray, hipMemcpyToSymbol,
 * hipMemcpyAsync
 */
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
hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind);

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

// doxygen end Memory
/**
 * @}
 */


/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup PeerToPeer Device Memory Access
 *  @{
 *
 *  @warning PeerToPeer support is experimental.
 *
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
 *  @defgroup Driver Initialization and Version
 *  @{
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
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Context Management
 *  @{
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
 * @bried Find out attributes for a given function.
 *
 * @param [out] attr
 * @param [in] func
 *
 * @returns hipSuccess, hipErrorInvalidDeviceFunction
 */

hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func);

struct Agent_global {

    Agent_global() : name(nullptr), address(nullptr), byte_cnt(0) {}
    Agent_global(const char* name, hipDeviceptr_t address, uint32_t byte_cnt) 
      : name(nullptr), address(address), byte_cnt(byte_cnt) {
      if (name)
        this->name = strdup(name);
    }

    Agent_global& operator=(Agent_global&& t) {
      if (this == &t) return *this;

      if (name) free(name);
      name = t.name;
      address = t.address;
      byte_cnt = t.byte_cnt;

      t.name = nullptr;
      t.address = nullptr;
      t.byte_cnt = 0;

      return *this;
    }

    Agent_global(Agent_global&& t) 
      : name(nullptr), address(nullptr), byte_cnt(0) {
      *this = std::move(t);
    }

    // not needed, delete them to prevent bugs
    Agent_global(const Agent_global&) = delete;
    Agent_global& operator=(Agent_global& t) = delete;

    ~Agent_global() { if (name) free(name); }
    
    char* name;
    hipDeviceptr_t address;
    uint32_t byte_cnt;
};

#if !__HIP_VDI__
#if defined(__cplusplus)
} // extern "C"
#endif

namespace hip_impl {
hsa_executable_t executable_for(hipModule_t);
const char* hash_for(hipModule_t);

template<typename ForwardIterator>
std::pair<hipDeviceptr_t, std::size_t> read_global_description(
    ForwardIterator f, ForwardIterator l, const char* name) {
    const auto it = std::find_if(f, l, [=](const Agent_global& x) {
        return strcmp(x.name, name) == 0;
    });

    return it == l ?
        std::make_pair(nullptr, 0u) : std::make_pair(it->address, it->byte_cnt);
}

std::vector<Agent_global> read_agent_globals(hsa_agent_t agent,
                                             hsa_executable_t executable);
hsa_agent_t this_agent();

inline
__attribute__((visibility("hidden")))
hipError_t read_agent_global_from_module(hipDeviceptr_t* dptr, size_t* bytes,
                                         hipModule_t hmod, const char* name) {
    // the key of the map would the hash of code object associated with the
    // hipModule_t instance
    static std::unordered_map<
        std::string, std::vector<Agent_global>> agent_globals;
    std::string key(hash_for(hmod));

    if (agent_globals.count(key) == 0) {
        static std::mutex mtx;
        std::lock_guard<std::mutex> lck{mtx};

        if (agent_globals.count(key) == 0) {
            agent_globals.emplace(
                key, read_agent_globals(this_agent(), executable_for(hmod)));
        }
    }

    const auto it0 = agent_globals.find(key);
    if (it0 == agent_globals.cend()) {
        hip_throw(
            std::runtime_error{"agent_globals data structure corrupted."});
    }

    std::tie(*dptr, *bytes) = read_global_description(it0->second.cbegin(),
                                                      it0->second.cend(), name);

    return *dptr ? hipSuccess : hipErrorNotFound;
}

inline
__attribute__((visibility("hidden")))
hipError_t read_agent_global_from_process(hipDeviceptr_t* dptr, size_t* bytes,
                                          const char* name) {
    static std::unordered_map<
        hsa_agent_t, std::vector<Agent_global>> agent_globals;
    static std::once_flag f;

    std::call_once(f, []() {
        for (auto&& agent_executables : executables()) {
            std::vector<Agent_global> tmp0;
            for (auto&& executable : agent_executables.second) {
                auto tmp1 = read_agent_globals(agent_executables.first,
                                               executable);

                tmp0.insert(tmp0.end(), make_move_iterator(tmp1.begin()),
                            make_move_iterator(tmp1.end()));
            }
            agent_globals.emplace(agent_executables.first, move(tmp0));
        }
    });

    const auto it = agent_globals.find(this_agent());

    if (it == agent_globals.cend()) return hipErrorNotInitialized;

    std::tie(*dptr, *bytes) = read_global_description(it->second.cbegin(),
                                                      it->second.cend(), name);

    return *dptr ? hipSuccess : hipErrorNotFound;
}
} // Namespace hip_impl.

#if defined(__cplusplus)
extern "C" {
#endif
/**
 * @brief returns device memory pointer and size of the kernel present in the module with symbol @p
 * name
 *
 * @param [out] dptr
 * @param [out] bytes
 * @param [in] hmod
 * @param [in] name
 *
 * @returns hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
 */
inline
__attribute__((visibility("hidden")))
hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes,
                              hipModule_t hmod, const char* name) {
    if (!dptr || !bytes) return hipErrorInvalidValue;

    if (!name) return hipErrorNotInitialized;

    const auto r = hmod ?
        hip_impl::read_agent_global_from_module(dptr, bytes, hmod, name) :
        hip_impl::read_agent_global_from_process(dptr, bytes, name);

    return r;
}
#endif // __HIP_VDI__

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
 * @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel.  The
 * kernel can access this with HIP_DYNAMIC_SHARED.
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

// doxygen end Version Management
/**
 * @}
 */


/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Profiler Control
 *  @{
 *
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
hipError_t hipProfilerStart();


/**
 * @brief Stop recording of profiling information.
 * When using this API, start the profiler with profiling disabled.  (--startdisabled)
 * @warning : hipProfilerStop API is under development.
 */
hipError_t hipProfilerStop();


/**
 * @}
 */

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
 * hipErrorInvalidResourceHandle,
 * hipErrorMemoryAllocation,
 * hipErrorMapBufferObjectFailed,
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
 * hipErrorMapBufferObjectFailed,
 * hipErrorInvalidResourceHandle,
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
 * hipErrorMapBufferObjectFailed,
 * hipErrorInvalidResourceHandle,
 *
 */
hipError_t hipIpcCloseMemHandle(void* devPtr);


// hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);
// hipError_t hipIpcCloseMemHandle(void *devPtr);
// // hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle);
// hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags);


/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Clang Launch API to support the triple-chevron syntax
 *  @{
 */

/**
 * @brief Configure a kernel launch.
 *
 * @param [in] gridDim   grid dimension specified as multiple of blockDim.
 * @param [in] blockDim  block dimensions specified in work-items
 * @param [in] sharedMem Amount of dynamic shared memory to allocate for this kernel.  The
 * kernel can access this with HIP_DYNAMIC_SHARED.
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
 * @}
 */


#ifdef __cplusplus
} /* extern "c" */
#endif

#include <hip/hcc_detail/hip_prof_api.h>

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Callback/Activity API
 */
hipError_t hipRegisterApiCallback(uint32_t id, void* fun, void* arg);
hipError_t hipRemoveApiCallback(uint32_t id);
hipError_t hipRegisterActivityCallback(uint32_t id, void* fun, void* arg);
hipError_t hipRemoveActivityCallback(uint32_t id);
static inline const char* hipApiName(const uint32_t& id) { return hip_api_name(id); }
const char* hipKernelNameRef(hipFunction_t f);
#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __cplusplus

hipError_t hipBindTexture(size_t* offset, textureReference* tex, const void* devPtr,
                          const hipChannelFormatDesc* desc, size_t size = UINT_MAX);

hipError_t ihipBindTextureImpl(int dim, enum hipTextureReadMode readMode, size_t* offset,
                               const void* devPtr, const struct hipChannelFormatDesc* desc,
                               size_t size, textureReference* tex);

/*
 * @brief hipBindTexture Binds size bytes of the memory area pointed to by @p devPtr to the texture
 *reference tex.
 *
 * @p desc describes how the memory is interpreted when fetching values from the texture. The @p
 *offset parameter is an optional byte offset as with the low-level hipBindTexture() function. Any
 *memory previously bound to tex is unbound.
 *
 *  @param[in]  offset - Offset in bytes
 *  @param[out]  tex - texture to bind
 *  @param[in]  devPtr - Memory area on device
 *  @param[in]  desc - Channel format
 *  @param[in]  size - Size of the memory area pointed to by devPtr
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
 **/
template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t hipBindTexture(size_t* offset, struct texture<T, dim, readMode>& tex, const void* devPtr,
                          const struct hipChannelFormatDesc& desc, size_t size = UINT_MAX) {
    return ihipBindTextureImpl(dim, readMode, offset, devPtr, &desc, size, &tex);
}

/*
 * @brief hipBindTexture Binds size bytes of the memory area pointed to by @p devPtr to the texture
 *reference tex.
 *
 * @p desc describes how the memory is interpreted when fetching values from the texture. The @p
 *offset parameter is an optional byte offset as with the low-level hipBindTexture() function. Any
 *memory previously bound to tex is unbound.
 *
 *  @param[in]  offset - Offset in bytes
 *  @param[in]  tex - texture to bind
 *  @param[in]  devPtr - Memory area on device
 *  @param[in]  size - Size of the memory area pointed to by devPtr
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
 **/
template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t hipBindTexture(size_t* offset, struct texture<T, dim, readMode>& tex, const void* devPtr,
                          size_t size = UINT_MAX) {
    return ihipBindTextureImpl(dim, readMode, offset, devPtr, &(tex.channelDesc), size, &tex);
}

// C API
hipError_t hipBindTexture2D(size_t* offset, textureReference* tex, const void* devPtr,
                            const hipChannelFormatDesc* desc, size_t width, size_t height,
                            size_t pitch);

hipError_t ihipBindTexture2DImpl(int dim, enum hipTextureReadMode readMode, size_t* offset,
                                 const void* devPtr, const struct hipChannelFormatDesc* desc,
                                 size_t width, size_t height, textureReference* tex);

template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t hipBindTexture2D(size_t* offset, struct texture<T, dim, readMode>& tex,
                            const void* devPtr, size_t width, size_t height, size_t pitch) {
    return ihipBindTexture2DImpl(dim, readMode, offset, devPtr, &(tex.channelDesc), width, height,
                                 &tex);
}

template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t hipBindTexture2D(size_t* offset, struct texture<T, dim, readMode>& tex,
                            const void* devPtr, const struct hipChannelFormatDesc& desc,
                            size_t width, size_t height, size_t pitch) {
    return ihipBindTexture2DImpl(dim, readMode, offset, devPtr, &desc, width, height, &tex);
}

// C API
hipError_t hipBindTextureToArray(textureReference* tex, hipArray_const_t array,
                                 const hipChannelFormatDesc* desc);

hipError_t ihipBindTextureToArrayImpl(int dim, enum hipTextureReadMode readMode,
                                      hipArray_const_t array,
                                      const struct hipChannelFormatDesc& desc,
                                      textureReference* tex);

template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t hipBindTextureToArray(struct texture<T, dim, readMode>& tex, hipArray_const_t array) {
    return ihipBindTextureToArrayImpl(dim, readMode, array, tex.channelDesc, &tex);
}

template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t hipBindTextureToArray(struct texture<T, dim, readMode>& tex, hipArray_const_t array,
                                 const struct hipChannelFormatDesc& desc) {
    return ihipBindTextureToArrayImpl(dim, readMode, array, desc, &tex);
}

template <class T, int dim, enum hipTextureReadMode readMode>
inline static hipError_t hipBindTextureToArray(struct texture<T, dim, readMode> *tex,
                                               hipArray_const_t array,
                                               const struct hipChannelFormatDesc* desc) {
    return ihipBindTextureToArrayImpl(dim, readMode, array, *desc, tex);
}

// C API
hipError_t hipBindTextureToMipmappedArray(const textureReference* tex,
                                          hipMipmappedArray_const_t mipmappedArray,
                                          const hipChannelFormatDesc* desc);

template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t hipBindTextureToMipmappedArray(const texture<T, dim, readMode>& tex,
                                          hipMipmappedArray_const_t mipmappedArray) {
    return hipSuccess;
}

template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t hipBindTextureToMipmappedArray(const texture<T, dim, readMode>& tex,
                                          hipMipmappedArray_const_t mipmappedArray,
                                          const hipChannelFormatDesc& desc) {
    return hipSuccess;
}

/*
 * @brief Unbinds the textuer bound to @p tex
 *
 *  @param[in]  tex - texture to unbind
 *
 *  @return #hipSuccess
 **/
hipError_t hipUnbindTexture(const textureReference* tex);

extern hipError_t ihipUnbindTextureImpl(const hipTextureObject_t& textureObject);

template <class T, int dim, enum hipTextureReadMode readMode>
hipError_t hipUnbindTexture(struct texture<T, dim, readMode>& tex) {
    return ihipUnbindTextureImpl(tex.textureObject);
}

hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array);
hipError_t hipGetTextureAlignmentOffset(size_t* offset, const textureReference* texref);
hipError_t hipGetTextureReference(const textureReference** texref, const void* symbol);

hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const hipResourceViewDesc* pResViewDesc);

hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject);

hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject);
hipError_t hipGetTextureObjectResourceViewDesc(hipResourceViewDesc* pResViewDesc,
                                               hipTextureObject_t textureObject);
hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc,
                                          hipTextureObject_t textureObject);
hipError_t hipTexRefSetArray(textureReference* tex, hipArray_const_t array, unsigned int flags);

hipError_t hipTexRefSetAddressMode(textureReference* tex, int dim, hipTextureAddressMode am);

hipError_t hipTexRefSetFilterMode(textureReference* tex, hipTextureFilterMode fm);

hipError_t hipTexRefSetFlags(textureReference* tex, unsigned int flags);

hipError_t hipTexRefSetFormat(textureReference* tex, hipArray_Format fmt, int NumPackedComponents);

hipError_t hipTexRefSetAddress(size_t* offset, textureReference* tex, hipDeviceptr_t devPtr,
                               size_t size);

hipError_t hipTexRefSetAddress2D(textureReference* tex, const HIP_ARRAY_DESCRIPTOR* desc,
                                 hipDeviceptr_t devPtr, size_t pitch);

hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject, const hipResourceDesc* pResDesc);

hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject);

// doxygen end Texture
/**
 * @}
 */


#endif


/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup HCC_Specific HCC-Specific Accessors
 *  @{
 *
 * The following calls are only supported when compiler HIP with HCC.
 * To produce portable code, use of these calls must be guarded #ifdef checks:
 * @code
 * #ifdef __HCC__
 *  hc::accelerator acc;
    hipError_t err = hipHccGetAccelerator(deviceId, &acc)
 * #endif
 * @endcode
 *
 */

// end-group HCC_Specific
/**
 * @}
 */


// doxygen end HIP API
/**
 *   @}
 */

#endif
