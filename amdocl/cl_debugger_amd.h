/* Copyright (c) 2014-present Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef __CL_DEBUGGER_AMD_H
#define __CL_DEBUGGER_AMD_H

#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl_platform.h>
#endif

/******************************************
* Private AMD extension cl_dbg *
******************************************/
#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/

#define CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD -80
#define CL_DEBUGGER_REGISTER_FAILURE_AMD -81
#define CL_TRAP_HANDLER_NOT_DEFINED_AMD -82
#define CL_EVENT_TIMEOUT_AMD -83


typedef uintptr_t cl_dbg_event_amd;  //! debug event

/*!  \brief  Trap Handler Type
 *
 *   The trap handler for each support type.
 */
enum cl_dbg_trap_type_amd {
  CL_DBG_DEBUG_TRAP = 0,  //! HW debug
  CL_DBG_MAX_TRAP
};

/*!  \brief  Wave actions used to control the wave execution on the hardware
 *
 *   The wave action enumerations are used to specify the desired
 *   behavior when calling the wave control function. Overall, there are
 *   five types of operations that can be specified.
 */
enum cl_dbg_waves_action_amd {
  CL_DBG_WAVES_DONT_USE_ZERO = 0,  //! NOT USED
  CL_DBG_WAVES_HALT = 1,           //! halt wave
  CL_DBG_WAVES_RESUME = 2,         //! resume wave
  CL_DBG_WAVES_KILL = 3,           //! kill wave
  CL_DBG_WAVES_DEBUG = 4,          //! debug wave
  CL_DBG_WAVES_TRAP = 5,           //! trap
  CL_DBG_WAVES_MAX
};

/*!  \brief  Host actions when encountering an exception in the kernel.
 *
 *   The host action enumeration is used to specify the desired host
 *   response in the event thatn a device kernel exception is encountered.
 */
enum cl_dbg_host_action_amd {
  CL_DBG_HOST_IGNORE = 1,  //! ignore the kernel exception
  CL_DBG_HOST_EXIT = 2,    //! exit the host application on a kernel exception
  CL_DBG_HOST_NOTIFY = 4   //! report the kernel exception
};

/*!  \brief  Mode of the wave action when calling the wave control function
 *
 *   The wave mode enumerations are used to specify the desired
 *   broadcast level when calling the wave control function.
 */
enum cl_dbg_wave_mode_amd {
  CL_DBG_WAVEMODE_SINGLE = 0,        //! send command to single wave
  CL_DBG_WAVEMODE_BROADCAST = 2,     //! send command to wave with match VMID
  CL_DBG_WAVEMODE_BROADCAST_CU = 3,  //! send command to wave with match VMID with specific CU
  CL_DBG_WAVEMODE_MAX
};

/*!  \brief  Enumeration of address watch mode
 *
 *   This enumeration indicates the different modes of address watch.
 */
enum cl_dbg_address_watch_mode_amd {
  CL_DBG_ADDR_WATCH_MODE_READ = 0,     //! Read operations only
  CL_DBG_ADDR_WATCH_MODE_NONREAD = 1,  //! Write or Atomic operations only
  CL_DBG_ADDR_WATCH_MODE_ATOMIC = 2,   //! Atomic Operations only
  CL_DBG_ADDR_WATCH_MODE_ALL = 3,      //! Read, Write or Atomic operations
  CL_DBG_ADDR_WATCH_MODE_MAX           //! Number of address watch modes
};

/*!  \brief  Dispatch exception policy descriptor
 *
 *   The dispatch exception policy descriptor is used to define the
 *   expected exception policy in the event an exception is encountered
 *   on the associated dispatch.
 */
typedef struct _cl_dbg_exception_policy_amd {
  cl_uint exceptionMask;               //! exception mask
  cl_dbg_waves_action_amd waveAction;  //! wave action
  cl_dbg_host_action_amd hostAction;   //! host action
  cl_dbg_wave_mode_amd waveMode;       //! wave mode
} cl_dbg_exception_policy_amd;

/*!  \brief  Kernel execution mode
 *
 *   This structure is used to control the kernel execution mode. The
 *   following aspects are included in this structure:
 *   1. Regular execution or debug mode (0: regular execution (default),
 *                                       1: debug mode)
 *   2. SQ debugger mode on/off
 *   3. Disable L1 scalar cache (0: enable (default), 1: disable)
 *   4. Disable L1 vector cache (0: enable (default), 1: disable)
 *   5. Disable L2 cache (0: enable (default), 1: disable)
 *   6. Num of CUs reserved for display (0 (default), 7: max)
 */
typedef struct _cl_dbg_kernel_exec_mode_amd {
  union {
    struct {
      cl_uint monitorMode : 1;
      cl_uint gpuSingleStepMode : 1;
      cl_uint disableL1Scalar : 1;
      cl_uint disableL1Vector : 1;
      cl_uint disableL2Cache : 1;
      cl_uint reservedCuNum : 3;
      cl_uint reserved : 24;
    };
    cl_uint ui32All;
  };
} cl_dbg_kernel_exec_mode_amd;

/*!  \brief  GPU cache mask
 *
 *   This structure is used to specify the GPU cache to be flushed/invalidated
 */
typedef struct _cl_dbg_gpu_cache_mask_amd {
  union {
    struct {
      cl_uint sqICache : 1;  //! instruction cache
      cl_uint sqKCache : 1;  //! data cache
      cl_uint tcL1 : 1;      //! tcL1 cache
      cl_uint tcL2 : 1;      //! tcL2 cache
      cl_uint reserved : 28;
    };
    cl_uint ui32All;
  };
} cl_dbg_gpu_cache_mask_amd;

/*!  \brief  Dispatch Debug Info
 *
 *   This structure is used to store the scratch and global memory descriptors
 */
typedef struct _cl_dispatch_debug_info_amd {
  cl_uint scratchMemoryDescriptor[4];  //! Scratch memory descriptors
  cl_uint globalMemoryDescriptor[4];   //! Global memory descriptors
} cl_dispatch_debug_info_amd;

/*!  \brief AQL Packet Info
 *
 *   This structure is used to store AQL packet informatin for kernel dispatch
 */
typedef struct _cl_aql_packet_info_amd {
  cl_uint trapReservedVgprIndex;    //! VGPR index reserved for trap
                                    //!   value is -1 when kernel was not compiled
                                    //!   in debug mode.
  cl_uint scratchBufferWaveOffset;  //! scratch buffer wave offset
                                    //!   value is -1 when kernel was not compiled
                                    //!   in debug mode  or scratch buffer is not enabled
  void* pointerToIsaBuffer;         //! Pointer to buffer containing ISA
  size_t sizeOfIsaBuffer;           //! Size of the ISA buffer

  cl_uint numberOfVgprs;           //! Number of VGPRs used by the kernel
  cl_uint numberOfSgprs;           //! Number of SGPRs used by the kernel
  size_t sizeOfStaticGroupMemory;  //! Static local memory used by the kernel
} cl_aql_packet_info_amd;

/*!  \brief  Wave address
 *
 *   This structure specifies the wave for the SQ control command
 */
typedef struct _cl_dbg_wave_addr_amd {
  cl_uint shaderEngine : 2;  //! Shader engine
  cl_uint shaderArray : 1;   //! Shader array
  cl_uint computeUnit : 4;   //! Compute unit
  cl_uint simd : 2;          //! SIMD id
  cl_uint wave : 4;          //! Wave id
  cl_uint vmid : 4;          //! VMID
  cl_uint reserved : 15;

} cl_dbg_wave_addr_amd;

/*!  \brief Pre-dispatch call back function signature
 *
 *   This is the signature of the call back fuction before the kernel
 *   dispatch. The call back function is to indicate the start of the
 *   the kernel launch. It is used by the debugger.
 */
typedef void* (*cl_PreDispatchCallBackFunctionAMD)(cl_device_id device, void* ocl_event_handle,
                                                   const void* aql_packet, void* acl_binary,
                                                   void* user_args);

/*!  \brief Post-dispatch call back function signature
 *
 *    This is the signature of the call back fuction after the kernel
 *    dispatch. The call back function is to indicate the completion of
 *    the the kernel launch. It is used by the debugger.
 */
typedef void* (*cl_PostDispatchCallBackFunctionAMD)(cl_device_id device, cl_ulong event,
                                                    void* user_args);

/*! \brief Set up the dispatch call back function pointers
 *
 *  \param device  specifies the device to be used
 *
 *  \param preDispatchFunction  is the function to be called before dispatching the kernel
 *
 *  \param postDispatchFunction  is the function to be called after kernel execution
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgSetCallBackFunctionsAMD(
    cl_device_id /* device */, cl_PreDispatchCallBackFunctionAMD /* preDispatchFunction */,
    cl_PostDispatchCallBackFunctionAMD /* postDispatchFunction */
    ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Set up the arguments of the dispatch call back function
 *
 *  \param device  specifies the device to be used
 *
 *  \param preDispatchArgs  is the arguments for the pre-dispatch callback function
 *
 *  \param postDispatchArgs  is the arguments for the post-dispatch callback function
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgSetCallBackArgumentsAMD(cl_device_id /* device */,
                                                                      void* /* preDispatchArgs */,
                                                                      void* /* postDispatchArgs */
                                                                      ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Invalidate all cache on the device.
 *
 *  \param device  specifies the device to be used
 *
 *  \param mask    is the mask to specify which cache to be flush/invalidate
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgFlushCacheAMD(cl_device_id /* device */,
                                                            cl_dbg_gpu_cache_mask_amd /* mask */
                                                            ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Set up an exception policy in the trap handler object
 *
 *  \param device  specifies the device to be used
 *
 *  \param policy  specifies the exception policy, which includes the exception mask,
 *                 wave action, host action, wave mode.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the policy is not specified (NULL)
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgSetExceptionPolicyAMD(
    cl_device_id /* device */, cl_dbg_exception_policy_amd* /* policy */
    ) CL_API_SUFFIX__VERSION_2_0;

/*! \brief Get the exception policy in the trap handler object
 *
 *  \param device  specifies the device to be used
 *
 *  \param policy  is a pointer to the memory where the policy is returned
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the policy storage is not specified
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgGetExceptionPolicyAMD(
    cl_device_id /* device */, cl_dbg_exception_policy_amd* /* policy */
    ) CL_API_SUFFIX__VERSION_2_0;

/*! \brief Set up the kernel execution mode in the trap handler object
 *
 *  \param device  specifies the device to be used
 *
 *  \param mode    specifies the kernel execution mode, which indicate whether single
 *                 step mode is used, how many CUs are reserved.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the mode is not specified, ie, has a NULL value
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgSetKernelExecutionModeAMD(
    cl_device_id /* device */, cl_dbg_kernel_exec_mode_amd* /* mode */
    ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Get the kernel execution mode in the trap handler object
 *
 *  \param device  specifies the device to be used
 *
 *  \param mode    is a pointer to the memory where the exectuion mode is returned
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the mode storage is not specified, ie, has a NULL value
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgGetKernelExecutionModeAMD(
    cl_device_id /* device */, cl_dbg_kernel_exec_mode_amd* /* mode */
    ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Create a debug event
 *
 *  \param device  specifies the device to be used
 *
 *  \param autoReset   is the auto reset flag
 *
 *  \param pDebugEvent returns the debug event to be used for exception notification
 *
 *  \param pEventId    is the event ID, which is not used at this moment
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the pDebugEvent value is NULL
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 *  - CL_OUT_OF_RESOURCES if fails to create the event
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgCreateEventAMD(cl_device_id /* device */,
                                                             bool /* autoReset */,
                                                             cl_dbg_event_amd* /* pDebugEvent */,
                                                             cl_uint* /* pEventId */
                                                             ) CL_API_SUFFIX__VERSION_2_0;

/*! \brief Wait for a debug event to be signaled
 *
 *  \param device  specifies the device to be used
 *
 *  \param pDebugEvent is the debug event to be waited for
 *
 *  \param pEventId    is the event ID, which is not used at this moment
 *
 *  \param timeOut     is the duration for waiting
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the pDebugEvent value is NULL
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 *  - CL_EVENT_TIMEOUT_AMD if timeout occurs
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgWaitEventAMD(cl_device_id /* device */,
                                                           cl_dbg_event_amd /* pDebugEvent */,
                                                           cl_uint /* pEventId */,
                                                           cl_uint /* timeOut */
                                                           ) CL_API_SUFFIX__VERSION_2_0;

/*! \brief Destroy a debug event
 *
 *  \param device  specifies the device to be used
 *
 *  \param pDebugEvent is the debug event to be waited for
 *
 *  \param pEventId    is the event ID, which is not used at this moment
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the pDebugEvent value is NULL
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgDestroyEventAMD(cl_device_id /* device */,
                                                              cl_dbg_event_amd* /* pDebugEvent */,
                                                              cl_uint* /* pEventId */
                                                              ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Register the debugger on a device
 *
 *  \param context specifies the context for the debugger
 *
 *  \param device  specifies the device to be used
 *
 *  \param pMessageStorge specifies the memory for trap message passing between KMD and OCL runtime
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_CONTEXT if the context is not valid
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the pMEssageStorge value is NULL
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 *  - CL_OUT_OF_RESOURCES if a host queue cannot be created for the debugger
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgRegisterDebuggerAMD(
    cl_context /* context */, cl_device_id /* device */, volatile void* /* pMessageStorage */
    ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Unregister the debugger on a device
 *
 *  \param device  specifies the device to be used
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgUnregisterDebuggerAMD(cl_device_id /* device */
                                                                    ) CL_API_SUFFIX__VERSION_2_0;

/*! \brief Setup the pointer of the acl_binary to be used by the debugger
 *
 *  \param device  specifies the device to be used
 *
 *  \param aclBinary  specifies the ACL binary to be used
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the aclBinary is not provided
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgSetAclBinaryAMD(cl_device_id /* device */,
                                                              void* /* aclBinary */
                                                              ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Control the execution of wavefront on the GPU
 *
 *  \param device       specifies the device to be used
 *
 *  \param action       specifies the wave action - halt, resume, kill, debug
 *
 *  \param mode         specifies the wave mode
 *
 *  \param trapID       specifies the trap ID, which should be 0x7
 *
 *  \param waveAddress  specifies the wave address for the wave control
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the waveMsg is not provided, invalid action or mode value
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgWaveControlAMD(cl_device_id /* device */,
                                                             cl_dbg_waves_action_amd /* action */,
                                                             cl_dbg_wave_mode_amd /* mode */,
                                                             cl_uint /* trapId */,
                                                             cl_dbg_wave_addr_amd /* waveAddress */
                                                             ) CL_API_SUFFIX__VERSION_2_0;

/*! \brief Set watch points on memory address ranges to generate exception events
 *
 *  \param device          specifies the device to be used
 *
 *  \param numWatchPoints  specifies the number of watch points
 *
 *  \param watchMode       is the array of watch mode for the watch points
 *
 *  \param watchAddress    is the array of watch address for the watch points
 *
 *  \param watchMask       is the array of mask for the watch points
 *
 *  \param watchEvent      is the array of event for the watch points
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the number of points <= 0, or other parameters is not specified
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgAddressWatchAMD(
    cl_device_id /* device */, cl_uint /* numWatchPoints */,
    cl_dbg_address_watch_mode_amd* /* watchMode */, void** /* watchAddress */,
    cl_ulong* /* watchMask */, cl_dbg_event_amd* /* watchEvent */
    ) CL_API_SUFFIX__VERSION_2_0;

/*! \brief Get the packaet information for kernel execution
 *
 *  \param device  specifies the device to be used
 *
 *  \param aqlCodeInfo   specifies the kernel code and its size
 *
 *  \param packetInfo    points to the memory for the packet information to be returned
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgGetAqlPacketInfoAMD(
    cl_device_id /* device */, const void* /* aqlCodeInfo */,
    cl_aql_packet_info_amd* /* packetInfo */
    ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Get the dispatch debug information
 *
 *  \param device  specifies the device to be used
 *
 *  \param debugInfo  points to the memory for the debug information to be returned
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgGetDispatchDebugInfoAMD(
    cl_device_id /* device */, cl_dispatch_debug_info_amd* /* debugInfo */
    ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Map the video memory for the kernel code to allow host access
 *
 *  \param device          specifies the device to be used
 *
 *  \param aqlCodeAddress  is the memory points to the returned host memory address for the kernel
 * code
 *
 *  \param aqlCodeSize     returns the size of the kernel code
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgMapKernelCodeAMD(cl_device_id /* device */,
                                                               void* /* aqlCodeInfo */
                                                               ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Unmap the video memory for the kernel code
 *
 *  \param device          specifies the device to be used (no needed, just to be consistent)
 *
 *  \param aqlCodeAddress  is the memory points to the mapped memory address for the kernel code
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgUnmapKernelCodeAMD(cl_device_id /* device */,
                                                                 cl_ulong* /* aqlCodeAddress */
                                                                 ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Map the shader scratch ring's video memory to allow CPU access
 *
 *  \param device  specifies the device to be used
 *
 *  \param scratchRingAddr  is the memory points to the returned host memory address for scratch
 * ring
 *
 *  \param scratchRingSize  returns the size of the scratch ring
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgMapScratchRingAMD(cl_device_id /* device */,
                                                                cl_ulong* /* scratchRingAddr */,
                                                                cl_uint* /* scratchRingSize */
                                                                ) CL_API_SUFFIX__VERSION_2_0;

/*! \brief Unmap the shader scratch ring's video memory
 *
 *  \param device           specifies the device to be used (no needed, just to be consistent)
 *
 *  \param scratchRingAddr  is the memory points to the mapped memory address for scratch ring
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgUnmapScratchRingAMD(cl_device_id /* device */,
                                                                  cl_ulong* /* scratchRingAddr */
                                                                  ) CL_API_SUFFIX__VERSION_2_0;

/*! \brief Get the memory object associated with the kernel parameter
 *
 *  \param device     specifies the device to be used
 *
 *  \param paramIdx   is the index of of the kernel argument
 *
 *  \param paramMem   is pointer of the memory associated with the kernel argument to be returned
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if the paramIdx is less than zero, or the paramMem has NULL value
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 *  - CL_INVALID_KERNEL_ARGS if it fails to get the memory object for the kernel argument
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgGetKernelParamMemAMD(cl_device_id /* devicepointer */,
                                                                   cl_uint /* paramIdx */,
                                                                   cl_mem* /* paramMem */
                                                                   ) CL_API_SUFFIX__VERSION_2_0;

/*! \brief Set value of a global memory object
 *
 *  \param device      specifies the device to be used
 *
 *  \param memObject   is the memory object handle to be assigned the value specified in srcMem.
 *
 *  \param offset      is offset of the memory object
 *
 *  \param srcMem      points to the memory which contains the values to be assigned to the memory
 *
 *  \param size        size (in bytes) of the srcMem
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if memObj or srcPtr has NULL value, size <= 0 or offset < 0
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgSetGlobalMemoryAMD(cl_device_id /* device */,
                                                                 cl_mem /* memObject */,
                                                                 cl_uint /* offset */,
                                                                 void* /* srcMem */,
                                                                 cl_uint /* size */
                                                                 ) CL_API_SUFFIX__VERSION_2_0;


/*! \brief Install the trap handler of a given type
 *
 *  \param device      specifies the device to be used
 *
 *  \param trapType    is the type of trap handler
 *
 *  \param trapHandler is the pointer of trap handler (TBA)
 *
 *  \param trapBuffer  is the pointer of trap handler buffer (TMA)
 *
 *  \param trapHandlerSize   size (in bytes) of the trap handler
 *
 *  \param trapBufferSize    size (in bytes) of the trap handler buffer
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the event occurs before the timeout
 *  - CL_INVALID_DEVICE if the device is not valid
 *  - CL_INVALID_VALUE if trapHandler is NULL or trapHandlerSize <= 0
 *  - CL_HWDBG_MANAGER_NOT_AVAILABLE_AMD if there is no HW DEBUG manager
 */
extern CL_API_ENTRY cl_int CL_API_CALL clHwDbgInstallTrapAMD(cl_device_id /* device */,
                                                             cl_dbg_trap_type_amd /* trapType */,
                                                             cl_mem /* trapHandler */,
                                                             cl_mem /* trapBuffer */
                                                             ) CL_API_SUFFIX__VERSION_2_0;


#ifdef __cplusplus
} /*extern "C"*/
#endif /*__cplusplus*/

#endif /*__CL_DEBUGGER_AMD_H*/
