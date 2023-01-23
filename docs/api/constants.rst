*********
Constants
*********

Enumerations
============

.. contents:: List of HIP constants
   :local:
   :backlinks: top

.. doxygenenum:: hipDeviceAttribute_t
..
   todo:: hipComputeMode
.. doxygenenum:: hipMemoryAdvise
.. doxygenenum:: hipMemRangeCoherencyMode
.. doxygenenum:: hipMemRangeAttribute
.. doxygenenum:: hipMemPoolAttr
.. doxygenenum:: hipMemLocationType
.. doxygenenum:: hipMemAllocationType
.. doxygenenum:: hipMemAllocationHandleType
..
   todo:: hipJitOption
.. doxygenenum:: hipFuncAttribute
.. doxygenenum:: hipFuncCache_t
.. doxygenenum:: hipSharedMemConfig
..
   todo:: hipExternalMemoryHandleType
..
   todo:: hipExternalSemaphoreHandleType
.. doxygenenum:: hipGLDeviceList
.. doxygenenum:: hipGraphicsRegisterFlags
.. doxygenenum:: hipGraphNodeType
..
   todo:: hipKernelNodeAttrID
..
   todo:: hipAccessProperty
.. doxygenenum:: hipGraphExecUpdateResult
..
   todo:: hipStreamCaptureMode
.. doxygenenum:: hipStreamCaptureStatus
.. doxygenenum:: hipStreamUpdateCaptureDependenciesFlags
.. doxygenenum:: hipGraphMemAttributeType
.. doxygenenum:: hipUserObjectFlags
.. doxygenenum:: hipUserObjectRetainFlags
.. doxygenenum:: hipGraphInstantiateFlags
.. doxygenenum:: hipMemAllocationGranularity_flags
.. doxygenenum:: hipMemHandleType
.. doxygenenum:: hipMemOperationType
.. doxygenenum:: hipArraySparseSubresourceType


Structures
==========

.. doxygenstruct:: hipMemLocation
.. doxygenstruct:: hipMemAccessDesc
.. doxygenstruct:: hipMemPoolProps
.. doxygenstruct:: hipMemPoolPtrExportData
.. doxygenstruct:: dim3
.. doxygenstruct:: hipLaunchParams_t
.. doxygenstruct:: hipExternalMemoryHandleDesc_st
.. doxygenstruct:: hipExternalMemoryBufferDesc_st
.. doxygenstruct:: hipExternalSemaphoreHandleDesc_st
.. doxygenstruct:: hipExternalSemaphoreSignalParams_st
.. doxygenstruct:: hipExternalSemaphoreWaitParams_st
.. doxygenstruct:: hipHostNodeParams
.. doxygenstruct:: hipKernelNodeParams
.. doxygenstruct:: hipMemsetParams
.. doxygenstruct:: hipAccessPolicyWindow
.. doxygenstruct:: hipKernelNodeAttrValue
.. doxygenstruct:: hipMemAllocationProp
.. doxygenstruct:: hipArrayMapInfo

Macros
======

.. doxygendefine:: hipStreamDefault
.. doxygendefine:: hipStreamNonBlocking
.. doxygendefine:: hipEventDefault
.. doxygendefine:: hipEventBlockingSync
.. doxygendefine:: hipEventDisableTiming
.. doxygendefine:: hipEventInterprocess
.. doxygendefine:: hipEventReleaseToDevice
.. doxygendefine:: hipEventReleaseToSystem
.. doxygendefine:: hipHostMallocDefault
.. doxygendefine:: hipHostMallocPortable
.. doxygendefine:: hipHostMallocMapped
.. doxygendefine:: hipHostMallocWriteCombined
.. doxygendefine:: hipHostMallocNumaUser
.. doxygendefine:: hipHostMallocCoherent 
.. doxygendefine:: hipHostMallocNonCoherent
.. doxygendefine:: hipMemAttachGlobal
.. doxygendefine:: hipMemAttachHost
.. doxygendefine:: hipMemAttachSingle
..
   todo:: hipDeviceMallocDefault
.. doxygendefine:: hipDeviceMallocFinegrained
.. doxygendefine:: hipMallocSignalMemory
.. doxygendefine:: hipHostRegisterDefault
.. doxygendefine:: hipHostRegisterPortable
.. doxygendefine:: hipHostRegisterMapped
.. doxygendefine:: hipHostRegisterIoMemory
.. doxygendefine:: hipExtHostRegisterCoarseGrained
.. doxygendefine:: hipDeviceScheduleAuto
.. doxygendefine:: hipDeviceScheduleSpin
.. doxygendefine:: hipDeviceScheduleYield
..
   todo:: hipDeviceScheduleBlockingSync
..
   todo:: hipDeviceScheduleMask
..
   todo:: hipDeviceMapHost
..
   todo:: hipDeviceLmemResizeToMax
.. doxygendefine:: hipArrayDefault
..
   todo:: hipArrayLayered
..
   todo:: hipArraySurfaceLoadStore
..
   todo:: hipArrayCubemap
..
   todo:: hipArrayTextureGather
..
   todo:: hipOccupancyDefault
..
   todo:: hipCooperativeLaunchMultiDeviceNoPreSync
..
   todo:: hipCooperativeLaunchMultiDeviceNoPostSync
..
   todo:: hipCpuDeviceId
..
   todo:: hipInvalidDeviceId
.. doxygendefine:: hipExtAnyOrderLaunch
..
   todo:: hipStreamWaitValueGte
..
   todo:: hipStreamWaitValueEq
..
   todo:: hipStreamWaitValueAnd
..
   todo:: hipStreamWaitValueNor
.. doxygendefine:: hipStreamPerThread
