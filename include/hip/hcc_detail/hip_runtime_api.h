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

//#pragma once
#ifndef HIP_RUNTIME_API_H
#define HIP_RUNTIME_API_H
/**
 *  @file  hcc_detail/hip_runtime_api.h
 *  @brief Contains C function APIs for HIP runtime. This file does not use any HCC builtin or special language extensions (-hc mode) ; those functions in hip_runtime.h.
 */

#include <stdint.h>
#include <stddef.h>

#include <hip/hcc_detail/host_defines.h>
#include <hip/hip_runtime_api.h>
//#include "hip/hip_hcc.h"

#if defined (__HCC__) &&  (__hcc_workweek__ < 16155)
#error("This version of HIP requires a newer version of HCC.");
#endif

// Structure definitions:
#ifdef __cplusplus
extern "C" {
#endif

//---
//API-visible structures
typedef struct ihipCtx_t    *hipCtx_t;

// Note many APIs also use integer deviceIds as an alternative to the device pointer:
typedef struct ihipDevice_t *hipDevice_t;

typedef struct ihipStream_t *hipStream_t;

typedef struct ihipModule_t *hipModule_t;

typedef struct ihipFunction_t *hipFunction_t;

typedef void* hipDeviceptr_t;

typedef struct ihipEvent_t *hipEvent_t;

enum hipLimit_t
{
    hipLimitMallocHeapSize = 0x02,
};

/**
 * @addtogroup GlobalDefs More
 * @{
 */
//! Flags that can be used with hipStreamCreateWithFlags
#define hipStreamDefault            0x00 ///< Default stream creation flags. These are used with hipStreamCreate().
#define hipStreamNonBlocking        0x01 ///< Stream does not implicitly synchronize with null stream


//! Flags that can be used with hipEventCreateWithFlags:
#define hipEventDefault             0x0  ///< Default flags
#define hipEventBlockingSync        0x1  ///< Waiting will yield CPU.  Power-friendly and usage-friendly but may increase latency.
#define hipEventDisableTiming       0x2  ///< Disable event's capability to record timing information.  May improve performance.
#define hipEventInterprocess        0x4  ///< Event can support IPC.  @warning - not supported in HIP.


//! Flags that can be used with hipHostMalloc
#define hipHostMallocDefault        0x0
#define hipHostMallocPortable       0x1
#define hipHostMallocMapped         0x2
#define hipHostMallocWriteCombined  0x4

//! Flags that can be used with hipHostRegister
#define hipHostRegisterDefault      0x0  ///< Memory is Mapped and Portable
#define hipHostRegisterPortable     0x1  ///< Memory is considered registered by all contexts.  HIP only supports one context so this is always assumed true.
#define hipHostRegisterMapped       0x2  ///< Map the allocation into the address space for the current device.  The device pointer can be obtained with #hipHostGetDevicePointer.
#define hipHostRegisterIoMemory     0x4  ///< Not supported.


#define hipDeviceScheduleAuto       0x0  ///< Automatically select between Spin and Yield
#define hipDeviceScheduleSpin       0x1  ///< Dedicate a CPU core to spin-wait.  Provides lowest latency, but burns a CPU core and may consume more power.
#define hipDeviceScheduleYield      0x2  ///< Yield the CPU to the operating system when waiting.  May increase latency, but lowers power and is friendlier to other threads in the system.
#define hipDeviceScheduleBlockingSync 0x4
#define hipDeviceScheduleMask       0x7

#define hipDeviceMapHost            0x8
#define hipDeviceLmemResizeToMax    0x16


/**
 * @warning On AMD devices and recent Nvidia devices, these hints and controls are ignored.
 */
typedef enum hipFuncCache {
    hipFuncCachePreferNone, ///< no preference for shared memory or L1 (default)
    hipFuncCachePreferShared, ///< prefer larger shared memory and smaller L1 cache
    hipFuncCachePreferL1, ///< prefer larger L1 cache and smaller shared memory
    hipFuncCachePreferEqual, ///< prefer equal size L1 cache and shared memory
} hipFuncCache;


/**
 * @warning On AMD devices and recent Nvidia devices, these hints and controls are ignored.
 */
typedef enum hipSharedMemConfig {
    hipSharedMemBankSizeDefault,   ///< The compiler selects a device-specific value for the banking.
    hipSharedMemBankSizeFourByte,  ///< Shared mem is banked at 4-bytes intervals and performs best when adjacent threads access data 4 bytes apart.
    hipSharedMemBankSizeEightByte  ///< Shared mem is banked at 8-byte intervals and performs best when adjacent threads access data 4 bytes apart.
} hipSharedMemConfig;



/**
 * Struct for data in 3D
 *
 */
typedef struct dim3 {
  uint32_t x;                 ///< x
  uint32_t y;                 ///< y
  uint32_t z;                 ///< z
#ifdef __cplusplus
  dim3(uint32_t _x=1, uint32_t _y=1, uint32_t _z=1) : x(_x), y(_y), z(_z) {};
#endif
} dim3;


/**
 * Memory copy types
 *
 */
typedef enum hipMemcpyKind {
   hipMemcpyHostToHost = 0    ///< Host-to-Host Copy
  ,hipMemcpyHostToDevice = 1  ///< Host-to-Device Copy
  ,hipMemcpyDeviceToHost = 2  ///< Device-to-Host Copy
  ,hipMemcpyDeviceToDevice =3 ///< Device-to-Device Copy
  ,hipMemcpyDefault = 4,      ///< Runtime will automatically determine copy-kind based on virtual addresses.
} hipMemcpyKind;




// Doxygen end group GlobalDefs
/**  @} */


//-------------------------------------------------------------------------------------------------


// The handle allows the async commands to use the stream even if the parent hipStream_t goes out-of-scope.
//typedef class ihipStream_t * hipStream_t;


/*
 * Opaque structure allows the true event (pointed at by the handle) to remain "live" even if the surrounding hipEvent_t goes out-of-scope.
 * This is handy for cases where the hipEvent_t goes out-of-scope but the true event is being written by some async queue or device */
//typedef struct hipEvent_t {
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
 * Calling this function deletes all streams created, memory allocated, kernels running, events created.
 * Make sure that no other thread is using the device or streams, memory, kernels, events associated with the current device.
 *
 * @returns #hipSuccess
 *
 * @see hipDeviceSynchronize
 */
hipError_t hipDeviceReset(void) ;


/**
 * @brief Set default device to be used for subsequent hip API calls from this thread.
 *
 * @param[in] deviceId Valid device in range 0...hipGetDeviceCount().
 *
 * Sets @p device as the default device for the calling host thread.  Valid device id's are 0... (hipGetDeviceCount()-1).
 *
 * Many HIP APIs implicitly use the "default device" :
 *
 * - Any device memory subsequently allocated from this host thread (using hipMalloc) will be allocated on device.
 * - Any streams or events created from this host thread will be associated with device.
 * - Any kernels launched from this host thread (using hipLaunchKernel) will be executed on device (unless a specific stream is specified,
 * in which case the device associated with that stream will be used).
 *
 * This function may be called from any host thread.  Multiple host threads may use the same device.
 * This function does no synchronization with the previous or new device, and has very little runtime overhead.
 * Applications can use hipSetDevice to quickly switch the default device before making a HIP runtime call which uses the default device.
 *
 * The default device is stored in thread-local-storage for each thread.
 * Thread-pool implementations may inherit the default device of the previous thread.  A good practice is to always call hipSetDevice
 * at the start of HIP coding sequency to establish a known standard device.
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
hipError_t hipGetDevice(int *deviceId);


/**
 * @brief Return number of compute-capable devices.
 *
 * @param [output] count Returns number of compute-capable devices.
 *
 * @returns #hipSuccess, #hipErrorNoDevice
 *
 *
 * Returns in @p *count the number of devices that have ability to run compute commands.  If there are no such devices, then @ref hipGetDeviceCount will return #hipErrorNoDevice.
 * If 1 or more devices can be found, then hipGetDeviceCount returns #hipSuccess.
 */
hipError_t hipGetDeviceCount(int *count);

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
 * Note: AMD devices and recent Nvidia GPUS do not support reconfigurable cache.  This hint is ignored on those architectures.
 *
 */
hipError_t hipDeviceSetCacheConfig ( hipFuncCache cacheConfig );


/**
 * @brief Set Cache configuration for a specific function
 *
 * @param [in] cacheConfig
 *
 * @returns #hipSuccess, #hipErrorInitializationError
 * Note: AMD devices and recent Nvidia GPUS do not support reconfigurable cache.  This hint is ignored on those architectures.
 *
 */
hipError_t hipDeviceGetCacheConfig ( hipFuncCache *cacheConfig );

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
hipError_t hipDeviceGetLimit(size_t *pValue, hipLimit_t limit);


/**
 * @brief Set Cache configuration for a specific function
 *
 * @param [in] config;
 *
 * @returns #hipSuccess, #hipErrorInitializationError
 * Note: AMD devices and recent Nvidia GPUS do not support reconfigurable cache.  This hint is ignored on those architectures.
 *
 */
hipError_t hipFuncSetCacheConfig ( hipFuncCache config );

/**
 * @brief Returns bank width of shared memory for current device
 *
 * @param [out] pConfig
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInitializationError
 *
 * Note: AMD devices and recent Nvidia GPUS do not support shared cache banking, and the hint is ignored on those architectures.
 *
 */
hipError_t hipDeviceGetSharedMemConfig ( hipSharedMemConfig * pConfig );


/**
 * @brief The bank width of shared memory on current device is set
 *
 * @param [in] config
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInitializationError
 *
 * Note: AMD devices and recent Nvidia GPUS do not support shared cache banking, and the hint is ignored on those architectures.
 *
 */
hipError_t hipDeviceSetSharedMemConfig ( hipSharedMemConfig config );

/**
 * @brief The current device behavior is changed according the flags passed.
 *
 * @param [in] flags
 *
 * The schedule flags impact how HIP waits for the completion of a command running on a device.
 * hipDeviceScheduleSpin         : HIP runtime will actively spin in the thread which submitted the work until the command completes.  This offers the lowest latency, but will consume a CPU core and may increase power.
 * hipDeviceScheduleYield        : The HIP runtime will yield the CPU to system so that other tasks can use it.  This may increase latency to detect the completion but will consume less power and is friendlier to other tasks in the system.
 * hipDeviceScheduleBlockingSync : On ROCm platform, this is a synonym for hipDeviceScheduleYield.
 * hipDeviceScheduleAuto         : Use a hueristic to select between Spin and Yield modes.  If the number of HIP contexts is greater than the number of logical processors in the system, use Spin scheduling.  Else use Yield scheduling.
 *
 *
 * hipDeviceMapHost              : Allow mapping host memory.  On ROCM, this is always allowed and the flag is ignored.
 * hipDeviceLmemResizeToMax      : @warning ROCm silently ignores this flag.
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorSetOnActiveProcess
 *
 *
*/
hipError_t hipSetDeviceFlags ( unsigned flags);

/**
 * @brief Device which matches hipDeviceProp_t is returned
 *
 * @param [out] device ID
 * @param [in]  device properties pointer
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 */
hipError_t hipChooseDevice(int *device, hipDeviceProp_t* prop);

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
 * @brief Return last error returned by any HIP runtime API call and resets the stored error code to #hipSuccess
 *
 * @returns return code from last HIP called from the active host thread
 *
 * Returns the last error that has been returned by any of the runtime calls in the same host thread,
 * and then resets the saved error to #hipSuccess.
 *
 * @see hipGetErrorString, hipGetLastError, hipPeakAtLastError, hipError_t
 */
hipError_t hipGetLastError(void);


/**
 * @brief Return last error returned by any HIP runtime API call.
 *
 * @return #hipSuccess
 *
 * Returns the last error that has been returned by any of the runtime calls in the same host thread.
 * Unlike hipGetLastError, this function does not reset the saved error code.
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
const char *hipGetErrorName(hipError_t hip_error);


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
const char *hipGetErrorString(hipError_t hipError);

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
 *  - cudaStreamAddCallback
 *  - cudaStreamAttachMemAsync
 *  - cudaStreamCreateWithPriority
 *  - cudaStreamGetPriority
 *  - cudaStreamWaitEvent
 */


/**
 * @brief Create an asynchronous stream.
 *
 * @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the newly created stream.
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to reference the newly
 * created stream in subsequent hipStream* commands.  The stream is allocated on the heap and will remain allocated
 * even if the handle goes out-of-scope.  To release the memory used by the stream, applicaiton must call hipStreamDestroy.
 *
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipStreamCreateWithFlags, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
hipError_t hipStreamCreate(hipStream_t *stream);


/**
 * @brief Create an asynchronous stream.
 *
 * @param[in, out] stream Pointer to new stream
 * @param[in ] flags to control stream creation.
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to reference the newly
 * created stream in subsequent hipStream* commands.  The stream is allocated on the heap and will remain allocated
 * even if the handle goes out-of-scope.  To release the memory used by the stream, applicaiton must call hipStreamDestroy.
 * Flags controls behavior of the stream.  See #hipStreamDefault, #hipStreamNonBlocking.
 *
 *
 * @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */

hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags);


/**
 * @brief Destroys the specified stream.
 *
 * @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the newly created stream.
 * @return #hipSuccess #hipErrorInvalidResourceHandle
 *
 * Destroys the specified stream.
 *
 * If commands are still executing on the specified stream, some may complete execution before the queue is deleted.
 *
 * The queue may be destroyed while some commands are still inflight, or may wait for all commands queued to the stream
 * before destroying it.
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamQuery, hipStreamWaitEvent, hipStreamSynchronize
 */
hipError_t hipStreamDestroy(hipStream_t stream);


/**
 * @brief Return #hipSuccess if all of the operations in the specified @p stream have completed, or #hipErrorNotReady if not.
 *
 * @param[in] stream stream to query
 *
 * @return #hipSuccess, #hipErrorNotReady, #hipErrorInvalidResourceHandle
 *
 * This is thread-safe and returns a snapshot of the current state of the queue.  However, if other host threads are sending work to the stream,
 * the status may change immediately after the function is called.  It is typically used for debug.
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamWaitEvent, hipStreamSynchronize, hipStreamDestroy
 */
hipError_t hipStreamQuery(hipStream_t stream);


/**
 * @brief Wait for all commands in stream to complete.
 *
 * @param[in] stream stream identifier.
 *
 * @return #hipSuccess, #hipErrorInvalidResourceHandle
 *
 * If the null stream is specified, this command blocks until all
 * This command honors the hipDeviceLaunchBlocking flag, which controls whether the wait is active or blocking.
 * This command is host-synchronous : the host will block until the stream is empty.
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamWaitEvent, hipStreamDestroy
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
 * All future work submitted to @p stream will wait until @p event reports completion before beginning execution.
 * This function is host-asynchronous and the function may return before the wait has completed.
 *
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamSynchronize, hipStreamDestroy
 *
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
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags);

/**
 * Stream CallBack struct
 */
typedef void(* hipStreamCallback_t)(hipStream_t stream,  hipError_t status, void* userData);

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
 * @see hipStreamCreate, hipStreamCreateWithFlags, hipStreamQuery, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 *
 */
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void *userData, unsigned int flags);


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
 * @param[in] flags     Flags to control event behavior.  Valid values are #hipEventDefault, #hipEventBlockingSync, #hipEventDisableTiming, #hipEventInterprocess

 * #hipEventDefault : Default flag.  The event will use active synchronization and will support timing.  Blocking synchronization provides lowest possible latency at the expense of dedicating a CPU to poll on the eevent.
 * #hipEventBlockingSync : The event will use blocking synchronization : if hipEventSynchronize is called on this event, the thread will block until the event completes.  This can increase latency for the synchroniation but can result in lower power and more resources for other CPU threads.
 * #hipEventDisableTiming : Disable recording of timing information.  On ROCM platform, timing information is always recorded and this flag has no performance benefit.

 * @warning On HCC platform, hipEventInterprocess support is under development.  Use of this flag will return an error.
 *
 * @returns #hipSuccess, #hipErrorInitializationError, #hipErrorInvalidValue, #hipErrorLaunchFailure, #hipErrorMemoryAllocation
 *
 * @see hipEventCreate, hipEventSynchronize, hipEventDestroy, hipEventElapsedTime
 */
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);


/**
 *  Create an event
 *
 * @param[in,out] event Returns the newly created event.
 *
 * @returns #hipSuccess, #hipErrorInitializationError, #hipErrorInvalidValue, #hipErrorLaunchFailure, #hipErrorMemoryAllocation
 *
 * @see hipEventCreateWithFlags, hipEventRecord, hipEventQuery, hipEventSynchronize, hipEventDestroy, hipEventElapsedTime
 */
hipError_t hipEventCreate(hipEvent_t* event);


/**
 * @brief Record an event in the specified stream.
 *
 * @param[in] event event to record.
 * @param[in] stream stream in which to record event.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInitializationError, #hipErrorInvalidResourceHandle, #hipErrorLaunchFailure
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
 * If hipEventRecord() has been previously called aon event, then this call will overwrite any existing state in event.
 *
 * If this function is called on a an event that is currently being recorded, results are undefined - either
 * outstanding recording may save state into the event, and the order is not guaranteed.  This shoul be avoided.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventSynchronize, hipEventDestroy, hipEventElapsedTime
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
 *  @returns #hipSuccess, #hipErrorInitializationError, #hipErrorInvalidValue, #hipErrorLaunchFailure
 *
 *  Releases memory associated with the event.  If the event is recording but has not completed recording when hipEventDestroy() is called,
 *  the function will return immediately and the completion_future resources will be released later, when the hipDevice is synchronized.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventSynchronize, hipEventRecord, hipEventElapsedTime
 *
 * @returns #hipSuccess
 */
hipError_t hipEventDestroy(hipEvent_t event);


/**
 *  @brief Wait for an event to complete.
 *
 *  This function will block until the event is ready, waiting for all previous work in the stream specified when event was recorded with hipEventRecord().
 *
 *  If hipEventRecord() has not been called on @p event, this function returns immediately.
 *
 *  TODO-hcc - This function needs to support hipEventBlockingSync parameter.
 *
 *  @param[in] event Event on which to wait.
 *  @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorInitializationError, #hipErrorInvalidResourceHandle, #hipErrorLaunchFailure
 *
 *  @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord, hipEventElapsedTime
 */
hipError_t hipEventSynchronize(hipEvent_t event);


/**
 * @brief Return the elapsed time between two events.
 *
 * @param[out] ms : Return time between start and stop in ms.
 * @param[in]   start : Start event.
 * @param[in]   stop  : Stop event.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotReady, #hipErrorInvalidResourceHandle, #hipErrorInitializationError, #hipErrorLaunchFailure
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
 * If hipEventRecord() has not been called on either event, then #hipErrorInvalidResourceHandle is returned.
 * If hipEventRecord() has been called on both events, but the timestamp has not yet been recorded on one or
 * both events (that is, hipEventQuery() would return #hipErrorNotReady on at least one of the events), then
 * #hipErrorNotReady is returned.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord, hipEventSynchronize
 */
hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop);


/**
 * @brief Query event status
 *
 * @param[in] event Event to query.
 * @returns #hipSuccess, #hipErrorNotReady, #hipErrorInvalidResourceHandle, #hipErrorInvalidValue, #hipErrorInitializationError, #hipErrorLaunchFailure
 *
 * Query the status of the specified event.  This function will return #hipErrorNotReady if all commands
 * in the appropriate stream (specified to hipEventRecord()) have completed.  If that work has not completed,
 * or if hipEventRecord() was not called on the event, then #hipSuccess is returned.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventRecord, hipEventDestroy, hipEventSynchronize, hipEventElapsedTime
 */
hipError_t hipEventQuery(hipEvent_t event) ;


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
hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, void* ptr);

/**
 *  @brief Allocate memory on the default accelerator
 *
 *  @param[out] ptr Pointer to the allocated memory
 *  @param[in]  size Requested memory size
 *
 *  @return #hipSuccess
 *
 *  @see hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray, hipHostFree, hipHostMalloc
 */
hipError_t hipMalloc(void** ptr, size_t size) ;

/**
 *  @brief Allocate pinned host memory [Deprecated]
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size
 *
 *  @return #hipSuccess, #hipErrorMemoryAllocation
 *
 *  @deprecated use hipHostMalloc() instead
 */
hipError_t hipMallocHost(void** ptr, size_t size) __attribute__((deprecated("use hipHostMalloc instead"))) ;

/**
 *  @brief Allocate device accessible page locked host memory
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size
 *  @param[in]  flags Type of host memory allocation
 *
 *  @return #hipSuccess, #hipErrorMemoryAllocation
 *
 *  @see hipSetDeviceFlags, hipHostFree
 */
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) ;

/**
 *  @brief Allocate device accessible page locked host memory [Deprecated]
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size
 *  @param[in]  flags Type of host memory allocation
 *
 *  @return #hipSuccess, #hipErrorMemoryAllocation
 *
 *  @deprecated use hipHostMalloc() instead
 */
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) __attribute__((deprecated("use hipHostMalloc instead"))) ;

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
hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags) ;

/**
 *  @brief Return flags associated with host pointer
 *
 *  @param[out] flagsPtr Memory location to store flags
 *  @param[in]  hostPtr Host Pointer allocated through hipHostMalloc
 *  @return #hipSuccess, #hipErrorInvalidValue
 *
 *  @see hipHostMalloc
 */
hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) ;

/**
 *  @brief Register host memory so it can be accessed from the current device.
 *
 *  @param[out] hostPtr Pointer to host memory to be registered.
 *  @param[in] sizeBytes size of the host memory
 *  @param[in] flags.  See below.
 *
 *  Flags:
 *  - #hipHostRegisterDefault   Memory is Mapped and Portable
 *  - #hipHostRegisterPortable  Memory is considered registered by all contexts.  HIP only supports one context so this is always assumed true.
 *  - #hipHostRegisterMapped    Map the allocation into the address space for the current device.  The device pointer can be obtained with #hipHostGetDevicePointer.
 *
 *
 *  After registering the memory, use #hipHostGetDevicePointer to obtain the mapped device pointer.
 *  On many systems, the mapped device pointer will have a different value than the mapped host pointer.  Applications
 *  must use the device pointer in device code, and the host pointer in device code.
 *
 *  On some systems, registered memory is pinned.  On some systems, registered memory may not be actually be pinned
 *  but uses OS or hardware facilities to all GPU access to the host memory.
 *
 *  Developers are strongly encouraged to register memory blocks which are aligned to the host cache-line size.
 *  (typically 64-bytes but can be obtains from the CPUID instruction).
 *
 *  If registering non-aligned pointers, the application must take care when register pointers from the same cache line
 *  on different devices.  HIP's coarse-grained synchronization model does not guarantee correct results if different
 *  devices write to different parts of the same cache block - typically one of the writes will "win" and overwrite data
 *  from the other registered memory region.
 *
 *  @return #hipSuccess, #hipErrorMemoryAllocation
 *
 *  @see hipHostUnregister, hipHostGetFlags, hipHostGetDevicePointer
 */
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) ;

/**
 *  @brief Un-register host pointer
 *
 *  @param[in] hostPtr Host pointer previously registered with #hipHostRegister
 *  @return Error code
 *
 *  @see hipHostRegister
 */
hipError_t hipHostUnregister(void* hostPtr) ;

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
 *  @return Error code
 *
 *  @see hipMalloc, hipFree, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D, hipMalloc3DArray, hipHostMalloc
 */

hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);

/**
 *  @brief Free memory allocated by the hcc hip memory allocation API.
 *  This API performs an implicit hipDeviceSynchronize() call.
 *  If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
 *
 *  @param[in] ptr Pointer to memory to be freed
 *  @return #hipSuccess
 *  @return #hipErrorInvalidDevicePointer (if pointer is invalid, including host pointers allocated with hipHostMalloc)
 *
 *  @see hipMalloc, hipMallocPitch, hipMallocArray, hipFreeArray, hipHostFree, hipMalloc3D, hipMalloc3DArray, hipHostMalloc
 */
hipError_t hipFree(void* ptr);

/**
 *  @brief Free memory allocated by the hcc hip host memory allocation API.  [Deprecated]
 *
 *  @param[in] ptr Pointer to memory to be freed
 *  @return #hipSuccess,
 *          #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with hipMalloc)

 *  @deprecated use hipHostFree() instead
 */
hipError_t hipFreeHost(void* ptr) __attribute__((deprecated("use hipHostFree instead")));

/**
 *  @brief Free memory allocated by the hcc hip host memory allocation API
 *  This API performs an implicit hipDeviceSynchronize() call.
 *  If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
 *
 *  @param[in] ptr Pointer to memory to be freed
 *  @return #hipSuccess,
 *          #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with hipMalloc)
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray, hipHostMalloc
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
 *  For multi-gpu or peer-to-peer configurations, it is recommended to set the current device to the device where the src data is physically located.
 *  For optimal peer-to-peer copies, the copy device must be able to access the src and dst pointers (by calling hipDeviceEnablePeerAccess with copy agent as the
 *  current device and src/dest as the peerDevice argument.  if this is not done, the hipMemcpy will still work, but will perform the copy using a staging buffer
 *  on the host.
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]  src Data being copy from
 *  @param[in]  sizeBytes Data size in bytes
 *  @param[in]  copyType Memory copy type
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknowni
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA, hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo, hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

/**
 *  @brief Copy data from Host to Device
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeInitialized, #hipErrorNotInitialized, #hipErrorInvalidContext, #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA, hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo, hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes);

/**
 *  @brief Copy data from Device to Host
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeInitialized, #hipErrorNotInitialized, #hipErrorInvalidContext, #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA, hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo, hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes);

/**
 *  @brief Copy data from Device to Device
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeInitialized, #hipErrorNotInitialized, #hipErrorInvalidContext, #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA, hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo, hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes);

/**
 *  @brief Copy data from Host to Device asynchronously
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeInitialized, #hipErrorNotInitialized, #hipErrorInvalidContext, #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA, hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo, hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes, hipStream_t stream);

/**
 *  @brief Copy data from Device to Host asynchronously
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeInitialized, #hipErrorNotInitialized, #hipErrorInvalidContext, #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA, hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo, hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream);

/**
 *  @brief Copy data from Device to Device asynchronously
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]   src Data being copy from
 *  @param[in]   sizeBytes Data size in bytes
 *
 *  @return #hipSuccess, #hipErrorDeInitialized, #hipErrorNotInitialized, #hipErrorInvalidContext, #hipErrorInvalidValue
 *
 *  @see hipArrayCreate, hipArrayDestroy, hipArrayGetDescriptor, hipMemAlloc, hipMemAllocHost, hipMemAllocPitch, hipMemcpy2D, hipMemcpy2DAsync, hipMemcpy2DUnaligned, hipMemcpyAtoA, hipMemcpyAtoD, hipMemcpyAtoH, hipMemcpyAtoHAsync, hipMemcpyDtoA, hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync, hipMemcpyHtoA, hipMemcpyHtoAAsync, hipMemcpyHtoDAsync, hipMemFree, hipMemFreeHost, hipMemGetAddressRange, hipMemGetInfo, hipMemHostAlloc, hipMemHostGetDevicePointer
 */
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream);


/**
 *  @brief Copies @p sizeBytes bytes from the memory area pointed to by @p src to the memory area pointed to by @p offset bytes from the start of symbol @p symbol.
 *
 *  The memory areas may not overlap. Symbol can either be a variable that resides in global or constant memory space, or it can be a character string,
 *  naming a variable that resides in global or constant memory space. Kind can be either hipMemcpyHostToDevice or hipMemcpyDeviceToDevice
 *  TODO: cudaErrorInvalidSymbol and cudaErrorInvalidMemcpyDirection is not supported, use hipErrorUnknown for now.
 *
 *  @param[in]  symbolName - Symbol destination on device
 *  @param[in]  src - Data being copy from
 *  @param[in]  sizeBytes - Data size in bytes
 *  @param[in]  offset - Offset from start of symbol in bytes
 *  @param[in]  kind - Type of transfer
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
 *
 *  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyFromSymbol, hipMemcpyAsync, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync, hipMemcpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync, hipMemcpyFromSymbolAsync
 */
hipError_t hipMemcpyToSymbol(const char* symbolName, const void *src, size_t sizeBytes, size_t offset, hipMemcpyKind kind);


/**
 *  @brief Copies @p sizeBytes bytes from the memory area pointed to by @p src to the memory area pointed to by @p offset bytes from the start of symbol @p symbol
 *
 *  The memory areas may not overlap. Symbol can either be a variable that resides in global or constant memory space, or it can be a character string,
 *  naming a variable that resides in global or constant memory space. Kind can be either hipMemcpyHostToDevice or hipMemcpyDeviceToDevice
 *  hipMemcpyToSymbolAsync() is asynchronous with respect to the host, so the call may return before copy is complete.
 *  TODO: cudaErrorInvalidSymbol and cudaErrorInvalidMemcpyDirection is not supported, use hipErrorUnknown for now.
 *
 *  @param[in]  symbolName - Symbol destination on device
 *  @param[in]  src - Data being copy from
 *  @param[in]  sizeBytes - Data size in bytes
 *  @param[in]  offset - Offset from start of symbol in bytes
 *  @param[in]  kind - Type of transfer
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
 *
 *  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyFromSymbol, hipMemcpyAsync, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync, hipMemcpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync, hipMemcpyFromSymbolAsync
 */
hipError_t hipMemcpyToSymbolAsync(const char* symbolName, const void *src, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream);



/**
 *  @brief Copy data from src to dst asynchronously.
 *
 *  @warning If host or dest are not pinned, the memory copy will be performed synchronously.  For best performance, use hipHostMalloc to
 *  allocate host memory that is transferred asynchronously.
 *
 *  @warning on HCC hipMemcpyAsync does not support overlapped H2D and D2H copies.
 *  For hipMemcpy, the copy is always performed by the device associated with the specified stream.
 *
 *  For multi-gpu or peer-to-peer configurations, it is recommended to use a stream which is a attached to the device where the src data is physically located.
 *  For optimal peer-to-peer copies, the copy device must be able to access the src and dst pointers (by calling hipDeviceEnablePeerAccess with copy agent as the
 *  current device and src/dest as the peerDevice argument.  if this is not done, the hipMemcpy will still work, but will perform the copy using a staging buffer
 *  on the host.
 *
 *  @param[out] dst Data being copy to
 *  @param[in]  src Data being copy from
 *  @param[in]  sizeBytes Data size in bytes
 *  @param[in]  accelerator_view Accelerator view which the copy is being enqueued
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree, #hipErrorUnknown
 *
 *  @see hipMemcpy, hipMemcpy2D, hipMemcpyToArray, hipMemcpy2DToArray, hipMemcpyFromArray, hipMemcpy2DFromArray, hipMemcpyArrayToArray, hipMemcpy2DArrayToArray, hipMemcpyToSymbol, hipMemcpyFromSymbol, hipMemcpy2DAsync, hipMemcpyToArrayAsync, hipMemcpy2DToArrayAsync, hipMemcpyFromArrayAsync, hipMemcpy2DFromArrayAsync, hipMemcpyToSymbolAsync, hipMemcpyFromSymbolAsync
 */
#if __cplusplus
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream=0);
#else
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream);
#endif

/**
 *  @brief Copy data from src to dst asynchronously.
 *
 * It supports memory from host to device,
 *  device to host, device to device and host to host.
 *
 *  @param[out] dst Data being copy to
 *  @param[in]  src Data being copy from
 *  @param[in]  sizeBytes Data size in bytes
 *  @param[in]  accelerator_view Accelerator view which the copy is being enqueued
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
 */
hipError_t hipMemset(void* dst, int  value, size_t sizeBytes );


/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dev with the constant byte value value.
 *
 *  hipMemsetAsync() is asynchronous with respect to the host, so the call may return before the memset is complete.
 *  The operation can optionally be associated to a stream by passing a non-zero stream argument.
 *  If stream is non-zero, the operation may overlap with operations in other streams.
 *
 *  @param[out] dst Pointer to device memory
 *  @param[in]  value - Value to set for each byte of specified memory
 *  @param[in]  sizeBytes - Size in bytes to set
 *  @param[in]  stream - Stream identifier
 *  @return #hipSuccess, #hipErrorInvalidValue, #hipErrorMemoryFree
 */
#if __cplusplus
hipError_t hipMemsetAsync(void* dst, int  value, size_t sizeBytes, hipStream_t stream = 0 );
#else
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream);
#endif

/**
 * @brief Query memory info.
 * Return snapshot of free memory, and total allocatable memory on the device.
 *
 * Returns in *free a snapshot of the current free memory.
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue
 * @warning On HCC, the free memory only accounts for memory allocated by this process and may be optimistic.
 **/
hipError_t hipMemGetInfo  (size_t * free, size_t * total)   ;

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
 * Returns "0" in @p canAccessPeer if deviceId == peerDeviceId, and both are valid devices : a device is not a peer of itself.
 *
 * @returns #hipSuccess,
 * @returns #hipErrorInvalidDevice if deviceId or peerDeviceId are not valid devices
 * @warning PeerToPeer support is experimental.
 */
hipError_t hipDeviceCanAccessPeer (int* canAccessPeer, int deviceId, int peerDeviceId);


/**
 * @brief Enable direct access from current device's virtual address space to memory allocations physically located on a peer device.
 *
 * Memory which already allocated on peer device will be mapped into the address space of the current device.  In addition, all
 * future memory allocations on peerDeviceId will be mapped into the address space of the current device when the memory is allocated.
 * The peer memory remains accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
 *
 *
 * @param [in] peerDeviceId
 * @param [in] flags
 *
 * Returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue,
 * @returns #hipErrorPeerAccessAlreadyEnabled if peer access is already enabled for this device.
 * @warning PeerToPeer support is experimental.
 */
hipError_t  hipDeviceEnablePeerAccess (int  peerDeviceId, unsigned int flags);


/**
 * @brief Disable direct access from current device's virtual address space to memory allocations physically located on a peer device.
 *
 * Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been enabled from the current device.
 *
 * @param [in] peerDeviceId
 *
 * @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
 * @warning PeerToPeer support is experimental.
 */
hipError_t  hipDeviceDisablePeerAccess (int peerDeviceId);


#ifdef PEER_NON_UNIFIED
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
 * @warning PeerToPeer support is experimental.
 */
hipError_t hipMemcpyPeer (void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t sizeBytes);

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
#if __cplusplus
hipError_t hipMemcpyPeerAsync ( void* dst, int  dstDeviceId, const void* src, int  srcDevice, size_t sizeBytes, hipStream_t stream=0 );
#else
hipError_t hipMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t sizeBytes, hipStream_t stream);
#endif
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
hipError_t hipInit(unsigned int flags) ;


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
 * @see hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags,  hipDevice_t device);

/**
 * @brief Destroy a HIP context.
 *
 * @param [in] ctx Context to destroy
 *
 * @returns #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipCtxCreate, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent,hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
 */
hipError_t hipCtxDestroy(hipCtx_t ctx);

/**
 * @brief Pop the current/default context and return the popped context.
 *
 * @param [out] ctx
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxSetCurrent, hipCtxGetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipCtxPopCurrent(hipCtx_t* ctx);

/**
 * @brief Push the context to be set as current/ default context
 *
 * @param [in] ctx
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
 */
hipError_t hipCtxPushCurrent(hipCtx_t ctx);

/**
 * @brief Set the passed context as current/default
 *
 * @param [in] ctx
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize , hipCtxGetDevice
 */
hipError_t hipCtxSetCurrent(hipCtx_t ctx);

/**
 * @brief Get the handle of the current/ default context
 *
 * @param [out] ctx
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipCtxGetCurrent(hipCtx_t* ctx);

/**
 * @brief Get the handle of the device associated with current/default context
 *
 * @param [out] device
 *
 * @returns #hipSuccess, #hipErrorInvalidContext
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize
 */

hipError_t hipCtxGetDevice(hipDevice_t *device);

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
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetDevice, hipCtxGetFlags, hipCtxPopCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipCtxGetApiVersion (hipCtx_t ctx,int *apiVersion);

/**
 * @brief Set Cache configuration for a specific function
 *
 * @param [out] cacheConfiguration
 *
 * @return #hipSuccess
 *
 * @warning AMD devices and recent Nvidia GPUS do not support reconfigurable cache.  This hint is ignored on those architectures.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipCtxGetCacheConfig ( hipFuncCache *cacheConfig );

/**
 * @brief Set L1/Shared cache partition.
 *
 * @param [in] cacheConfiguration
 *
 * @return #hipSuccess
 *
 * @warning AMD devices and recent Nvidia GPUS do not support reconfigurable cache.  This hint is ignored on those architectures.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipCtxSetCacheConfig ( hipFuncCache cacheConfig );

/**
 * @brief Set Shared memory bank configuration.
 *
 * @param [in] sharedMemoryConfiguration
 *
 * @return #hipSuccess
 *
 * @warning AMD devices and recent Nvidia GPUS do not support shared cache banking, and the hint is ignored on those architectures.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipCtxSetSharedMemConfig ( hipSharedMemConfig config );

/**
 * @brief Get Shared memory bank configuration.
 *
 * @param [out] sharedMemoryConfiguration
 *
 * @return #hipSuccess
 *
 * @warning AMD devices and recent Nvidia GPUS do not support shared cache banking, and the hint is ignored on those architectures.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 */
hipError_t hipCtxGetSharedMemConfig ( hipSharedMemConfig * pConfig );

/**
 * @brief Blocks until the default context has completed all preceding requested tasks.
 *
 * @return #hipSuccess
 *
 * @warning This function waits for all streams on the default context to complete execution, and then returns.
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxGetDevice
*/
hipError_t hipCtxSynchronize ( void );

/**
 * @brief Return flags used for creating default context.
 *
 * @param [out] flags
 *
 * @returns #hipSuccess
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxGetCurrent, hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
*/
hipError_t hipCtxGetFlags ( unsigned int* flags );

/**
 * @brief Enables direct access to memory allocations in a peer context.
 *
 * Memory which already allocated on peer device will be mapped into the address space of the current device.  In addition, all
 * future memory allocations on peerDeviceId will be mapped into the address space of the current device when the memory is allocated.
 * The peer memory remains accessible from the current device until a call to hipDeviceDisablePeerAccess or hipDeviceReset.
 *
 *
 * @param [in] peerCtx
 * @param [in] flags
 *
 * @returns #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidValue, #hipErrorPeerAccessAlreadyEnabled
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 * @warning PeerToPeer support is experimental.
 */
hipError_t  hipCtxEnablePeerAccess (hipCtx_t peerCtx, unsigned int flags);

/**
 * @brief Disable direct access from current context's virtual address space to memory allocations physically located on a peer context.Disables direct access to memory allocations in a peer context and unregisters any registered allocations.
 *
 * Returns hipErrorPeerAccessNotEnabled if direct access to memory on peerDevice has not yet been enabled from the current device.
 *
 * @param [in] peerCtx
 *
 * @returns #hipSuccess, #hipErrorPeerAccessNotEnabled
 *
 * @see hipCtxCreate, hipCtxDestroy, hipCtxGetFlags, hipCtxPopCurrent, hipCtxGetCurrent, hipCtxSetCurrent, hipCtxPushCurrent, hipCtxSetCacheConfig, hipCtxSynchronize, hipCtxGetDevice
 * @warning PeerToPeer support is experimental.
 */
hipError_t  hipCtxDisablePeerAccess (hipCtx_t peerCtx);

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
hipError_t hipDeviceGet(hipDevice_t *device, int ordinal);

/**
 * @brief Returns the compute capability of the device
 * @param [out] major
 * @param [out] minor
 * @param [in] device
 *
 * @returns #hipSuccess, #hipErrorInavlidDevice
 */
hipError_t hipDeviceComputeCapability(int *major,int *minor,hipDevice_t device);

/**
 * @brief Returns an identifer string for the device.
 * @param [out] name
 * @param [in] len
 * @param [in] device
 *
 * @returns #hipSuccess, #hipErrorInavlidDevice
 */
hipError_t hipDeviceGetName(char *name,int len,hipDevice_t device);

/**
 * @brief Returns a PCI Bus Id string for the device.
 * @param [out] pciBusId
 * @param [in] len
 * @param [in] device
 *
 * @returns #hipSuccess, #hipErrorInavlidDevice
 */
hipError_t hipDeviceGetPCIBusId (int *pciBusId,int len,hipDevice_t device);

/**
 * @brief Returns the total amount of memory on the device.
 * @param [out] bytes
 * @param [in] device
 *
 * @returns #hipSuccess, #hipErrorInavlidDevice
 */
hipError_t hipDeviceTotalMem (size_t *bytes,hipDevice_t device);

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
hipError_t hipDriverGetVersion(int *driverVersion) ;

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
hipError_t hipRuntimeGetVersion(int *runtimeVersion) ;

/**
 * @brief Loads code object from file into a hipModule_t
 *
 * @param [in] fname
 * @param [out] module
 *
 * @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorFileNotFound, hipErrorOutOfMemory, hipErrorSharedObjectInitFailed, hipErrorNotInitialized
 *
 *
 */
hipError_t hipModuleLoad(hipModule_t *module, const char *fname);

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
 * @returns hipSuccess, hipErrorInvalidValue, hipErrorInvalidContext, hipErrorNotInitialized, hipErrorNotFound,
 */
hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module, const char *kname);

/**
 * @brief returns device memory pointer and size of the kernel present in the module with symbol @p name
 *
 * @param [out] dptr
 * @param [out[ bytes
 * @param [in] hmod
 * @param [in] name
 *
 * @returns hipSuccess, hipErrorInvalidValue, hipErrorNotInitialized
 */
hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes, hipModule_t hmod, const char *name);


/**
 * @brief builds module from code object which resides in host memory. Image is pointer to that location.
 *
 * @param [in] image
 * @param [out] module
 *
 * @returns hipSuccess, hipErrorNotInitialized, hipErrorOutOfMemory, hipErrorNotInitialized
 */
hipError_t hipModuleLoadData(hipModule_t *module, const void *image);


/**
 * @brief launches kernel f with launch parameters and shared memory on stream with arguments passed to kernelparams or extra
 *
 * @param [in[ f
 * @param [in] gridDimX
 * @param [in] gridDimY
 * @param [in] gridDimZ
 * @param [in] blockDimX
 * @param [in] blockDimY
 * @param [in] blockDimZ
 * @param [in] sharedMemBytes
 * @param [in] stream
 * @param [in] kernelParams
 * @param [in] extraa
 *
 * The function takes the above arguments and run the kernel in hipFunction_t f.  with launch parameters specified in gridDimX, gridDimY, gridDimZ,  blockDimX, blockDimY and blockDimmZ. The amount of shared memory is specificed and can be used with HIP_DYNAMIC_SHARED. The arguemt extra is used to pass in the arguments for the kernel.
 * @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
 *
 * @warning kernellParams argument is not yet implemented in HIP. Please use extra instead. Please refer to hip_porting_driver_api.md for sample usage.
 */
hipError_t hipModuleLaunchKernel(hipFunction_t f,
                              unsigned int gridDimX,
                              unsigned int gridDimY,
                              unsigned int gridDimZ,
                              unsigned int blockDimX,
                              unsigned int blockDimY,
                              unsigned int blockDimZ,
                              unsigned int sharedMemBytes,
                              hipStream_t stream,
                              void **kernelParams,
                              void **extra) ;

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
 * @warning : hipProfilerStart API is under development.
 */
hipError_t hipProfilerStart();


/**
 * @brief Stop recording of profiling information.
 * @warning : hipProfilerStop API is under development.
 */
hipError_t hipProfilerStop();


/**
 * @}
 */




#ifdef __cplusplus
} /* extern "c" */
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
