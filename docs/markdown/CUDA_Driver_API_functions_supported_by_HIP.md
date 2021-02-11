# CUDA Driver API supported by HIP

## **1. CUDA Driver Data Types**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`CUDA_ARRAY3D_2DARRAY`|  | 5.0 |  |  |  |  |  | 
|`CUDA_ARRAY3D_COLOR_ATTACHMENT`| 10.0 |  |  |  |  |  |  | 
|`CUDA_ARRAY3D_CUBEMAP`|  |  |  | `hipArrayCubemap` | 1.7.0 |  |  | 
|`CUDA_ARRAY3D_DEPTH_TEXTURE`|  |  |  |  |  |  |  | 
|`CUDA_ARRAY3D_DESCRIPTOR`|  |  |  | `HIP_ARRAY3D_DESCRIPTOR` | 2.7.0 |  |  | 
|`CUDA_ARRAY3D_DESCRIPTOR_st`|  |  |  | `HIP_ARRAY3D_DESCRIPTOR` | 2.7.0 |  |  | 
|`CUDA_ARRAY3D_LAYERED`|  |  |  | `hipArrayLayered` | 1.7.0 |  |  | 
|`CUDA_ARRAY3D_SPARSE`| 11.1 |  |  |  |  |  |  | 
|`CUDA_ARRAY3D_SURFACE_LDST`|  |  |  | `hipArraySurfaceLoadStore` | 1.7.0 |  |  | 
|`CUDA_ARRAY3D_TEXTURE_GATHER`|  |  |  | `hipArrayTextureGather` | 1.7.0 |  |  | 
|`CUDA_ARRAY_DESCRIPTOR`|  |  |  | `HIP_ARRAY_DESCRIPTOR` | 1.7.0 |  |  | 
|`CUDA_ARRAY_DESCRIPTOR_st`|  |  |  | `HIP_ARRAY_DESCRIPTOR` | 1.7.0 |  |  | 
|`CUDA_ARRAY_DESCRIPTOR_v1`|  |  |  | `HIP_ARRAY_DESCRIPTOR` | 1.7.0 |  |  | 
|`CUDA_ARRAY_DESCRIPTOR_v1_st`|  |  |  | `HIP_ARRAY_DESCRIPTOR` | 1.7.0 |  |  | 
|`CUDA_ARRAY_SPARSE_PROPERTIES`| 11.1 |  |  |  |  |  |  | 
|`CUDA_ARRAY_SPARSE_PROPERTIES_st`| 11.1 |  |  |  |  |  |  | 
|`CUDA_CB`|  |  |  |  |  |  |  | 
|`CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC`| 9.0 |  |  | `hipCooperativeLaunchMultiDeviceNoPostSync` | 3.2.0 |  |  | 
|`CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC`| 9.0 |  |  | `hipCooperativeLaunchMultiDeviceNoPreSync` | 3.2.0 |  |  | 
|`CUDA_ERROR_ALREADY_ACQUIRED`|  |  |  | `hipErrorAlreadyAcquired` | 1.6.0 |  |  | 
|`CUDA_ERROR_ALREADY_MAPPED`|  |  |  | `hipErrorAlreadyMapped` | 1.6.0 |  |  | 
|`CUDA_ERROR_ARRAY_IS_MAPPED`|  |  |  | `hipErrorArrayIsMapped` | 1.6.0 |  |  | 
|`CUDA_ERROR_ASSERT`|  |  |  | `hipErrorAssert` | 1.9.0 |  |  | 
|`CUDA_ERROR_CAPTURED_EVENT`| 10.0 |  |  |  |  |  |  | 
|`CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`| 10.1 |  |  |  |  |  |  | 
|`CUDA_ERROR_CONTEXT_ALREADY_CURRENT`|  | 3.2 |  | `hipErrorContextAlreadyCurrent` | 1.6.0 |  |  | 
|`CUDA_ERROR_CONTEXT_ALREADY_IN_USE`|  |  |  | `hipErrorContextAlreadyInUse` | 1.6.0 |  |  | 
|`CUDA_ERROR_CONTEXT_IS_DESTROYED`|  |  |  |  |  |  |  | 
|`CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`| 9.0 |  |  | `hipErrorCooperativeLaunchTooLarge` | 3.2.0 |  |  | 
|`CUDA_ERROR_DEINITIALIZED`|  |  |  | `hipErrorDeinitialized` | 1.6.0 |  |  | 
|`CUDA_ERROR_DEVICE_NOT_LICENSED`| 11.1 |  |  |  |  |  |  | 
|`CUDA_ERROR_ECC_UNCORRECTABLE`|  |  |  | `hipErrorECCNotCorrectable` | 1.6.0 |  |  | 
|`CUDA_ERROR_FILE_NOT_FOUND`|  |  |  | `hipErrorFileNotFound` | 1.6.0 |  |  | 
|`CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE`| 10.2 |  |  |  |  |  |  | 
|`CUDA_ERROR_HARDWARE_STACK_ERROR`|  |  |  |  |  |  |  | 
|`CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`|  |  |  | `hipErrorHostMemoryAlreadyRegistered` | 1.6.0 |  |  | 
|`CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED`|  |  |  | `hipErrorHostMemoryNotRegistered` | 1.6.0 |  |  | 
|`CUDA_ERROR_ILLEGAL_ADDRESS`|  |  |  | `hipErrorIllegalAddress` | 1.6.0 |  |  | 
|`CUDA_ERROR_ILLEGAL_INSTRUCTION`|  |  |  |  |  |  |  | 
|`CUDA_ERROR_ILLEGAL_STATE`| 10.0 |  |  |  |  |  |  | 
|`CUDA_ERROR_INVALID_ADDRESS_SPACE`|  |  |  |  |  |  |  | 
|`CUDA_ERROR_INVALID_CONTEXT`|  |  |  | `hipErrorInvalidContext` | 1.6.0 |  |  | 
|`CUDA_ERROR_INVALID_DEVICE`|  |  |  | `hipErrorInvalidDevice` | 1.6.0 |  |  | 
|`CUDA_ERROR_INVALID_GRAPHICS_CONTEXT`|  |  |  | `hipErrorInvalidGraphicsContext` | 1.6.0 |  |  | 
|`CUDA_ERROR_INVALID_HANDLE`|  |  |  | `hipErrorInvalidHandle` | 1.6.0 |  |  | 
|`CUDA_ERROR_INVALID_IMAGE`|  |  |  | `hipErrorInvalidImage` | 1.6.0 |  |  | 
|`CUDA_ERROR_INVALID_PC`|  |  |  |  |  |  |  | 
|`CUDA_ERROR_INVALID_PTX`|  |  |  | `hipErrorInvalidKernelFile` | 1.6.0 |  |  | 
|`CUDA_ERROR_INVALID_SOURCE`|  |  |  | `hipErrorInvalidSource` | 1.6.0 |  |  | 
|`CUDA_ERROR_INVALID_VALUE`|  |  |  | `hipErrorInvalidValue` | 1.6.0 |  |  | 
|`CUDA_ERROR_JIT_COMPILATION_DISABLED`| 11.2 |  |  |  |  |  |  | 
|`CUDA_ERROR_JIT_COMPILER_NOT_FOUND`| 9.0 |  |  |  |  |  |  | 
|`CUDA_ERROR_LAUNCH_FAILED`|  |  |  | `hipErrorLaunchFailure` | 1.6.0 |  |  | 
|`CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING`|  |  |  |  |  |  |  | 
|`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`|  |  |  | `hipErrorLaunchOutOfResources` | 1.6.0 |  |  | 
|`CUDA_ERROR_LAUNCH_TIMEOUT`|  |  |  | `hipErrorLaunchTimeOut` | 1.6.0 |  |  | 
|`CUDA_ERROR_MAP_FAILED`|  |  |  | `hipErrorMapFailed` | 1.6.0 |  |  | 
|`CUDA_ERROR_MISALIGNED_ADDRESS`|  |  |  |  |  |  |  | 
|`CUDA_ERROR_NOT_FOUND`|  |  |  | `hipErrorNotFound` | 1.6.0 |  |  | 
|`CUDA_ERROR_NOT_INITIALIZED`|  |  |  | `hipErrorNotInitialized` | 1.6.0 |  |  | 
|`CUDA_ERROR_NOT_MAPPED`|  |  |  | `hipErrorNotMapped` | 1.6.0 |  |  | 
|`CUDA_ERROR_NOT_MAPPED_AS_ARRAY`|  |  |  | `hipErrorNotMappedAsArray` | 1.6.0 |  |  | 
|`CUDA_ERROR_NOT_MAPPED_AS_POINTER`|  |  |  | `hipErrorNotMappedAsPointer` | 1.6.0 |  |  | 
|`CUDA_ERROR_NOT_PERMITTED`|  |  |  |  |  |  |  | 
|`CUDA_ERROR_NOT_READY`|  |  |  | `hipErrorNotReady` | 1.6.0 |  |  | 
|`CUDA_ERROR_NOT_SUPPORTED`|  |  |  | `hipErrorNotSupported` | 1.6.0 |  |  | 
|`CUDA_ERROR_NO_BINARY_FOR_GPU`|  |  |  | `hipErrorNoBinaryForGpu` | 1.6.0 |  |  | 
|`CUDA_ERROR_NO_DEVICE`|  |  |  | `hipErrorNoDevice` | 1.6.0 |  |  | 
|`CUDA_ERROR_NVLINK_UNCORRECTABLE`| 8.0 |  |  |  |  |  |  | 
|`CUDA_ERROR_OPERATING_SYSTEM`|  |  |  | `hipErrorOperatingSystem` | 1.6.0 |  |  | 
|`CUDA_ERROR_OUT_OF_MEMORY`|  |  |  | `hipErrorOutOfMemory` | 1.6.0 |  |  | 
|`CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED`|  |  |  | `hipErrorPeerAccessAlreadyEnabled` | 1.6.0 |  |  | 
|`CUDA_ERROR_PEER_ACCESS_NOT_ENABLED`|  |  |  | `hipErrorPeerAccessNotEnabled` | 1.6.0 |  |  | 
|`CUDA_ERROR_PEER_ACCESS_UNSUPPORTED`|  |  |  | `hipErrorPeerAccessUnsupported` | 1.6.0 |  |  | 
|`CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE`|  |  |  | `hipErrorSetOnActiveProcess` | 1.6.0 |  |  | 
|`CUDA_ERROR_PROFILER_ALREADY_STARTED`|  | 5.0 |  | `hipErrorProfilerAlreadyStarted` | 1.6.0 |  |  | 
|`CUDA_ERROR_PROFILER_ALREADY_STOPPED`|  | 5.0 |  | `hipErrorProfilerAlreadyStopped` | 1.6.0 |  |  | 
|`CUDA_ERROR_PROFILER_DISABLED`|  |  |  | `hipErrorProfilerDisabled` | 1.6.0 |  |  | 
|`CUDA_ERROR_PROFILER_NOT_INITIALIZED`|  | 5.0 |  | `hipErrorProfilerNotInitialized` | 1.6.0 |  |  | 
|`CUDA_ERROR_SHARED_OBJECT_INIT_FAILED`|  |  |  | `hipErrorSharedObjectInitFailed` | 1.6.0 |  |  | 
|`CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND`|  |  |  | `hipErrorSharedObjectSymbolNotFound` | 1.6.0 |  |  | 
|`CUDA_ERROR_STREAM_CAPTURE_IMPLICIT`| 10.0 |  |  |  |  |  |  | 
|`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`| 10.0 |  |  |  |  |  |  | 
|`CUDA_ERROR_STREAM_CAPTURE_ISOLATION`| 10.0 |  |  |  |  |  |  | 
|`CUDA_ERROR_STREAM_CAPTURE_MERGE`| 10.0 |  |  |  |  |  |  | 
|`CUDA_ERROR_STREAM_CAPTURE_UNJOINED`| 10.0 |  |  |  |  |  |  | 
|`CUDA_ERROR_STREAM_CAPTURE_UNMATCHED`| 10.0 |  |  |  |  |  |  | 
|`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`| 10.0 |  |  |  |  |  |  | 
|`CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD`| 10.1 |  |  |  |  |  |  | 
|`CUDA_ERROR_STUB_LIBRARY`| 11.1 |  |  |  |  |  |  | 
|`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`| 10.1 |  |  |  |  |  |  | 
|`CUDA_ERROR_SYSTEM_NOT_READY`| 10.0 |  |  |  |  |  |  | 
|`CUDA_ERROR_TIMEOUT`| 10.2 |  |  |  |  |  |  | 
|`CUDA_ERROR_TOO_MANY_PEERS`|  |  |  |  |  |  |  | 
|`CUDA_ERROR_UNKNOWN`|  |  |  | `hipErrorUnknown` | 1.6.0 |  |  | 
|`CUDA_ERROR_UNMAP_FAILED`|  |  |  | `hipErrorUnmapFailed` | 1.6.0 |  |  | 
|`CUDA_ERROR_UNSUPPORTED_LIMIT`|  |  |  | `hipErrorUnsupportedLimit` | 1.6.0 |  |  | 
|`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`| 11.1 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_MEMORY_BUFFER_DESC`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_MEMORY_DEDICATED`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_MEMORY_HANDLE_DESC`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC`| 10.2 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st`| 10.0 |  |  |  |  |  |  | 
|`CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC`| 10.2 |  |  |  |  |  |  | 
|`CUDA_EXT_SEM_SIGNAL_NODE_PARAMS`| 11.2 |  |  |  |  |  |  | 
|`CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st`| 11.2 |  |  |  |  |  |  | 
|`CUDA_EXT_SEM_WAIT_NODE_PARAMS`| 11.2 |  |  |  |  |  |  | 
|`CUDA_EXT_SEM_WAIT_NODE_PARAMS_st`| 11.2 |  |  |  |  |  |  | 
|`CUDA_HOST_NODE_PARAMS`| 10.0 |  |  |  |  |  |  | 
|`CUDA_HOST_NODE_PARAMS_st`| 10.0 |  |  |  |  |  |  | 
|`CUDA_KERNEL_NODE_PARAMS`| 10.0 |  |  |  |  |  |  | 
|`CUDA_KERNEL_NODE_PARAMS_st`| 10.0 |  |  |  |  |  |  | 
|`CUDA_LAUNCH_PARAMS`| 9.0 |  |  | `hipLaunchParams` | 2.6.0 |  |  | 
|`CUDA_LAUNCH_PARAMS_st`| 9.0 |  |  | `hipLaunchParams` | 2.6.0 |  |  | 
|`CUDA_MEMCPY2D`|  |  |  | `hip_Memcpy2D` | 1.7.0 |  |  | 
|`CUDA_MEMCPY2D_st`|  |  |  | `hip_Memcpy2D` | 1.7.0 |  |  | 
|`CUDA_MEMCPY2D_v1`|  |  |  | `hip_Memcpy2D` | 1.7.0 |  |  | 
|`CUDA_MEMCPY2D_v1_st`|  |  |  | `hip_Memcpy2D` | 1.7.0 |  |  | 
|`CUDA_MEMCPY3D`|  |  |  | `HIP_MEMCPY3D` | 3.2.0 |  |  | 
|`CUDA_MEMCPY3D_PEER`|  |  |  |  |  |  |  | 
|`CUDA_MEMCPY3D_PEER_st`|  |  |  |  |  |  |  | 
|`CUDA_MEMCPY3D_st`|  |  |  | `HIP_MEMCPY3D` | 3.2.0 |  |  | 
|`CUDA_MEMCPY3D_v1`|  |  |  | `HIP_MEMCPY3D` | 3.2.0 |  |  | 
|`CUDA_MEMCPY3D_v1_st`|  |  |  | `HIP_MEMCPY3D` | 3.2.0 |  |  | 
|`CUDA_MEMSET_NODE_PARAMS`| 10.0 |  |  |  |  |  |  | 
|`CUDA_MEMSET_NODE_PARAMS_st`| 10.0 |  |  |  |  |  |  | 
|`CUDA_NVSCISYNC_ATTR_SIGNAL`| 10.2 |  |  |  |  |  |  | 
|`CUDA_NVSCISYNC_ATTR_WAIT`| 10.2 |  |  |  |  |  |  | 
|`CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS`| 11.1 |  |  |  |  |  |  | 
|`CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum`| 11.1 |  |  |  |  |  |  | 
|`CUDA_POINTER_ATTRIBUTE_P2P_TOKENS`|  |  |  |  |  |  |  | 
|`CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st`|  |  |  |  |  |  |  | 
|`CUDA_RESOURCE_DESC`|  |  |  | `HIP_RESOURCE_DESC` | 3.5.0 |  |  | 
|`CUDA_RESOURCE_DESC_st`|  |  |  | `HIP_RESOURCE_DESC_st` | 3.5.0 |  |  | 
|`CUDA_RESOURCE_VIEW_DESC`|  |  |  | `HIP_RESOURCE_VIEW_DESC` | 3.5.0 |  |  | 
|`CUDA_RESOURCE_VIEW_DESC_st`|  |  |  | `HIP_RESOURCE_VIEW_DESC_st` | 3.5.0 |  |  | 
|`CUDA_SUCCESS`|  |  |  | `hipSuccess` | 1.5.0 |  |  | 
|`CUDA_TEXTURE_DESC`|  |  |  | `HIP_TEXTURE_DESC` | 3.5.0 |  |  | 
|`CUDA_TEXTURE_DESC_st`|  |  |  | `HIP_TEXTURE_DESC_st` | 3.5.0 |  |  | 
|`CUDA_VERSION`|  |  |  |  |  |  |  | 
|`CUGLDeviceList`|  |  |  |  |  |  |  | 
|`CUGLDeviceList_enum`|  |  |  |  |  |  |  | 
|`CUGLmap_flags`|  |  |  |  |  |  |  | 
|`CUGLmap_flags_enum`|  |  |  |  |  |  |  | 
|`CU_ACCESS_PROPERTY_NORMAL`| 11.0 |  |  |  |  |  |  | 
|`CU_ACCESS_PROPERTY_PERSISTING`| 11.0 |  |  |  |  |  |  | 
|`CU_ACCESS_PROPERTY_STREAMING`| 11.0 |  |  |  |  |  |  | 
|`CU_AD_FORMAT_FLOAT`|  |  |  | `HIP_AD_FORMAT_FLOAT` | 1.7.0 |  |  | 
|`CU_AD_FORMAT_HALF`|  |  |  | `HIP_AD_FORMAT_HALF` | 1.7.0 |  |  | 
|`CU_AD_FORMAT_NV12`| 11.2 |  |  |  |  |  |  | 
|`CU_AD_FORMAT_SIGNED_INT16`|  |  |  | `HIP_AD_FORMAT_SIGNED_INT16` | 1.7.0 |  |  | 
|`CU_AD_FORMAT_SIGNED_INT32`|  |  |  | `HIP_AD_FORMAT_SIGNED_INT32` | 1.7.0 |  |  | 
|`CU_AD_FORMAT_SIGNED_INT8`|  |  |  | `HIP_AD_FORMAT_SIGNED_INT8` | 1.7.0 |  |  | 
|`CU_AD_FORMAT_UNSIGNED_INT16`|  |  |  | `HIP_AD_FORMAT_UNSIGNED_INT16` | 1.7.0 |  |  | 
|`CU_AD_FORMAT_UNSIGNED_INT32`|  |  |  | `HIP_AD_FORMAT_UNSIGNED_INT32` | 1.7.0 |  |  | 
|`CU_AD_FORMAT_UNSIGNED_INT8`|  |  |  | `HIP_AD_FORMAT_UNSIGNED_INT8` | 1.7.0 |  |  | 
|`CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL`| 11.1 |  |  |  |  |  |  | 
|`CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL`| 11.1 |  |  |  |  |  |  | 
|`CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL`| 11.1 |  |  |  |  |  |  | 
|`CU_COMPUTEMODE_DEFAULT`|  |  |  | `hipComputeModeDefault` | 1.9.0 |  |  | 
|`CU_COMPUTEMODE_EXCLUSIVE`|  |  | 8.0 | `hipComputeModeExclusive` | 1.9.0 |  |  | 
|`CU_COMPUTEMODE_EXCLUSIVE_PROCESS`|  |  |  | `hipComputeModeExclusiveProcess` | 2.0.0 |  |  | 
|`CU_COMPUTEMODE_PROHIBITED`|  |  |  | `hipComputeModeProhibited` | 1.9.0 |  |  | 
|`CU_CTX_BLOCKING_SYNC`|  | 4.0 |  | `hipDeviceScheduleBlockingSync` | 1.6.0 |  |  | 
|`CU_CTX_FLAGS_MASK`|  |  |  |  |  |  |  | 
|`CU_CTX_LMEM_RESIZE_TO_MAX`|  |  |  | `hipDeviceLmemResizeToMax` | 1.6.0 |  |  | 
|`CU_CTX_MAP_HOST`|  |  |  | `hipDeviceMapHost` | 1.6.0 |  |  | 
|`CU_CTX_SCHED_AUTO`|  |  |  | `hipDeviceScheduleAuto` | 1.6.0 |  |  | 
|`CU_CTX_SCHED_BLOCKING_SYNC`|  |  |  | `hipDeviceScheduleBlockingSync` | 1.6.0 |  |  | 
|`CU_CTX_SCHED_MASK`|  |  |  | `hipDeviceScheduleMask` | 1.6.0 |  |  | 
|`CU_CTX_SCHED_SPIN`|  |  |  | `hipDeviceScheduleSpin` | 1.6.0 |  |  | 
|`CU_CTX_SCHED_YIELD`|  |  |  | `hipDeviceScheduleYield` | 1.6.0 |  |  | 
|`CU_CUBEMAP_FACE_NEGATIVE_X`|  |  |  |  |  |  |  | 
|`CU_CUBEMAP_FACE_NEGATIVE_Y`|  |  |  |  |  |  |  | 
|`CU_CUBEMAP_FACE_NEGATIVE_Z`|  |  |  |  |  |  |  | 
|`CU_CUBEMAP_FACE_POSITIVE_X`|  |  |  |  |  |  |  | 
|`CU_CUBEMAP_FACE_POSITIVE_Y`|  |  |  |  |  |  |  | 
|`CU_CUBEMAP_FACE_POSITIVE_Z`|  |  |  |  |  |  |  | 
|`CU_D3D10_DEVICE_LIST_ALL`|  |  |  |  |  |  |  | 
|`CU_D3D10_DEVICE_LIST_CURRENT_FRAME`|  |  |  |  |  |  |  | 
|`CU_D3D10_DEVICE_LIST_NEXT_FRAME`|  |  |  |  |  |  |  | 
|`CU_D3D10_MAPRESOURCE_FLAGS_NONE`|  |  |  |  |  |  |  | 
|`CU_D3D10_MAPRESOURCE_FLAGS_READONLY`|  |  |  |  |  |  |  | 
|`CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD`|  |  |  |  |  |  |  | 
|`CU_D3D10_REGISTER_FLAGS_ARRAY`|  |  |  |  |  |  |  | 
|`CU_D3D10_REGISTER_FLAGS_NONE`|  |  |  |  |  |  |  | 
|`CU_D3D11_DEVICE_LIST_ALL`|  |  |  |  |  |  |  | 
|`CU_D3D11_DEVICE_LIST_CURRENT_FRAME`|  |  |  |  |  |  |  | 
|`CU_D3D11_DEVICE_LIST_NEXT_FRAME`|  |  |  |  |  |  |  | 
|`CU_D3D9_DEVICE_LIST_ALL`|  |  |  |  |  |  |  | 
|`CU_D3D9_DEVICE_LIST_CURRENT_FRAME`|  |  |  |  |  |  |  | 
|`CU_D3D9_DEVICE_LIST_NEXT_FRAME`|  |  |  |  |  |  |  | 
|`CU_D3D9_MAPRESOURCE_FLAGS_NONE`|  |  |  |  |  |  |  | 
|`CU_D3D9_MAPRESOURCE_FLAGS_READONLY`|  |  |  |  |  |  |  | 
|`CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD`|  |  |  |  |  |  |  | 
|`CU_D3D9_REGISTER_FLAGS_ARRAY`|  |  |  |  |  |  |  | 
|`CU_D3D9_REGISTER_FLAGS_NONE`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES`| 9.2 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY`|  |  |  | `hipDeviceAttributeCanMapHostMemory` | 2.10.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER`|  | 5.0 |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS`| 9.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM`| 9.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS`| 9.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR`| 9.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_CLOCK_RATE`|  |  |  | `hipDeviceAttributeClockRate` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`|  |  |  | `hipDeviceAttributeComputeCapabilityMajor` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`|  |  |  | `hipDeviceAttributeComputeCapabilityMinor` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_COMPUTE_MODE`|  |  |  | `hipDeviceAttributeComputeMode` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED`| 8.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS`|  |  |  | `hipDeviceAttributeConcurrentKernels` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`| 8.0 |  |  | `hipDeviceAttributeConcurrentManagedAccess` | 3.10.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`| 9.0 |  |  | `hipDeviceAttributeCooperativeLaunch` | 2.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH`| 9.0 |  |  | `hipDeviceAttributeCooperativeMultiDeviceLaunch` | 2.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST`| 9.2 |  |  | `hipDeviceAttributeDirectManagedMemAccessFromHost` | 3.10.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_ECC_ENABLED`|  |  |  | `hipDeviceAttributeEccEnabled` | 2.10.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED`| 11.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH`|  |  |  | `hipDeviceAttributeMemoryBusWidth` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED`| 11.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_GPU_OVERLAP`|  | 5.0 |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED`| 10.2 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED`| 10.2 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED`| 10.2 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED`| 8.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED`| 9.2 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_INTEGRATED`|  |  |  | `hipDeviceAttributeIntegrated` | 1.9.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT`|  |  |  | `hipDeviceAttributeKernelExecTimeout` | 2.10.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`|  |  |  | `hipDeviceAttributeL2CacheSize` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY`|  |  |  | `hipDeviceAttributeManagedMemory` | 3.10.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH`|  | 11.2 |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH`|  |  |  | `hipDeviceAttributeMaxTexture1DWidth` | 2.7.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT`|  | 5.0 |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES`|  | 5.0 |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH`|  | 5.0 |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT`|  |  |  | `hipDeviceAttributeMaxTexture2DHeight` | 2.7.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH`|  |  |  | `hipDeviceAttributeMaxTexture2DWidth` | 2.7.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH`|  |  |  | `hipDeviceAttributeMaxTexture3DDepth` | 2.7.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT`|  |  |  | `hipDeviceAttributeMaxTexture3DHeight` | 2.7.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH`|  |  |  | `hipDeviceAttributeMaxTexture3DWidth` | 2.7.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE`| 11.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR`| 11.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X`|  |  |  | `hipDeviceAttributeMaxBlockDimX` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y`|  |  |  | `hipDeviceAttributeMaxBlockDimY` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z`|  |  |  | `hipDeviceAttributeMaxBlockDimZ` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X`|  |  |  | `hipDeviceAttributeMaxGridDimX` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y`|  |  |  | `hipDeviceAttributeMaxGridDimY` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z`|  |  |  | `hipDeviceAttributeMaxGridDimZ` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE`| 11.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_PITCH`|  |  |  | `hipDeviceAttributeMaxPitch` | 2.10.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK`|  |  |  | `hipDeviceAttributeMaxRegistersPerBlock` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`|  |  |  | `hipDeviceAttributeMaxSharedMemoryPerBlock` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`| 9.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`|  |  |  | `hipDeviceAttributeMaxSharedMemoryPerMultiprocessor` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`|  |  |  | `hipDeviceAttributeMaxThreadsPerBlock` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR`|  |  |  | `hipDeviceAttributeMaxThreadsPerMultiProcessor` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE`|  |  |  | `hipDeviceAttributeMemoryClockRate` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED`| 11.2 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`|  |  |  | `hipDeviceAttributeMultiprocessorCount` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD`|  |  |  | `hipDeviceAttributeIsMultiGpuBoard` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`| 8.0 |  |  | `hipDeviceAttributePageableMemoryAccess` | 3.10.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`| 9.2 |  |  | `hipDeviceAttributePageableMemoryAccessUsesHostPageTables` | 3.10.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_PCI_BUS_ID`|  |  |  | `hipDeviceAttributePciBusId` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID`|  |  |  | `hipDeviceAttributePciDeviceId` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED`| 11.1 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK`|  | 5.0 |  | `hipDeviceAttributeMaxRegistersPerBlock` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK`| 11.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK`|  | 5.0 |  | `hipDeviceAttributeMaxSharedMemoryPerBlock` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO`| 8.0 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED`| 11.1 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_TCC_DRIVER`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT`|  |  |  | `hipDeviceAttributeTextureAlignment` | 2.10.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT`|  |  |  |  | 3.2.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED`| 11.2 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY`|  |  |  | `hipDeviceAttributeTotalConstantMemory` | 1.6.0 |  |  | 
|`CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING`|  |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED`| 10.2 | 11.2 |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`| 11.2 |  |  |  |  |  |  | 
|`CU_DEVICE_ATTRIBUTE_WARP_SIZE`|  |  |  | `hipDeviceAttributeWarpSize` | 1.6.0 |  |  | 
|`CU_DEVICE_CPU`| 8.0 |  |  | `hipCpuDeviceId` | 3.7.0 |  |  | 
|`CU_DEVICE_INVALID`| 8.0 |  |  | `hipInvalidDeviceId` | 3.7.0 |  |  | 
|`CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED`| 10.1 | 10.1 |  | `hipDevP2PAttrHipArrayAccessSupported` | 3.8.0 |  |  | 
|`CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED`| 8.0 |  |  | `hipDevP2PAttrAccessSupported` | 3.8.0 |  |  | 
|`CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED`| 9.2 | 10.0 | 10.1 | `hipDevP2PAttrHipArrayAccessSupported` | 3.8.0 |  |  | 
|`CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED`| 10.0 |  |  | `hipDevP2PAttrHipArrayAccessSupported` | 3.8.0 |  |  | 
|`CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED`| 8.0 |  |  | `hipDevP2PAttrNativeAtomicSupported` | 3.8.0 |  |  | 
|`CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK`| 8.0 |  |  | `hipDevP2PAttrPerformanceRank` | 3.8.0 |  |  | 
|`CU_EGL_COLOR_FORMAT_A`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_ABGR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_ARGB`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_AYUV`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_AYUV_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER10_BGGR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER10_GBRG`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER10_GRBG`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER10_RGGB`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER12_BGGR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER12_GBRG`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER12_GRBG`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER12_RGGB`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER14_BGGR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER14_GBRG`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER14_GRBG`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER14_RGGB`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER20_BGGR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER20_GBRG`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER20_GRBG`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER20_RGGB`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER_BGGR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER_GBRG`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER_GRBG`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR`| 9.2 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG`| 9.2 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG`| 9.2 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB`| 9.2 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BAYER_RGGB`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BGR`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_BGRA`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_L`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_MAX`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_R`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_RG`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_RGB`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_RGBA`|  |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_UYVY_422`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_UYVY_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_VYUY_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV420_PLANAR`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV422_PLANAR`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV444_PLANAR`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUVA_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUV_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUYV_422`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YUYV_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVU420_PLANAR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVU422_PLANAR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVU444_PLANAR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_COLOR_FORMAT_YVYU_ER`| 9.1 |  |  |  |  |  |  | 
|`CU_EGL_FRAME_TYPE_ARRAY`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_FRAME_TYPE_PITCH`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_RESOURCE_LOCATION_SYSMEM`| 9.0 |  |  |  |  |  |  | 
|`CU_EGL_RESOURCE_LOCATION_VIDMEM`| 9.0 |  |  |  |  |  |  | 
|`CU_EVENT_BLOCKING_SYNC`|  |  |  | `hipEventBlockingSync` | 1.6.0 |  |  | 
|`CU_EVENT_DEFAULT`|  |  |  | `hipEventDefault` | 1.6.0 |  |  | 
|`CU_EVENT_DISABLE_TIMING`|  |  |  | `hipEventDisableTiming` | 1.6.0 |  |  | 
|`CU_EVENT_INTERPROCESS`|  |  |  | `hipEventInterprocess` | 1.6.0 |  |  | 
|`CU_EVENT_RECORD_DEFAULT`| 11.1 |  |  |  |  |  |  | 
|`CU_EVENT_RECORD_EXTERNAL`| 11.1 |  |  |  |  |  |  | 
|`CU_EVENT_WAIT_DEFAULT`| 11.1 |  |  |  |  |  |  | 
|`CU_EVENT_WAIT_EXTERNAL`| 11.1 |  |  |  |  |  |  | 
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE`| 10.2 |  |  |  |  |  |  | 
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT`| 10.2 |  |  |  |  |  |  | 
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP`| 10.0 |  |  |  |  |  |  | 
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE`| 10.0 |  |  |  |  |  |  | 
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF`| 10.2 |  |  |  |  |  |  | 
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD`| 10.0 |  |  |  |  |  |  | 
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32`| 10.0 |  |  |  |  |  |  | 
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT`| 10.0 |  |  |  |  |  |  | 
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE`| 10.2 |  |  |  |  |  |  | 
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX`| 10.2 |  |  |  |  |  |  | 
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT`| 10.2 |  |  |  |  |  |  | 
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE`| 10.0 |  |  |  |  |  |  | 
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC`| 10.2 |  |  |  |  |  |  | 
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD`| 10.0 |  |  |  |  |  |  | 
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32`| 10.0 |  |  |  |  |  |  | 
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT`| 10.0 |  |  |  |  |  |  | 
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD`| 11.2 |  |  |  |  |  |  | 
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32`| 11.2 |  |  |  |  |  |  | 
|`CU_FUNC_ATTRIBUTE_BINARY_VERSION`|  |  |  | `HIP_FUNC_ATTRIBUTE_BINARY_VERSION` | 2.8.0 |  |  | 
|`CU_FUNC_ATTRIBUTE_CACHE_MODE_CA`|  |  |  | `HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA` | 2.8.0 |  |  | 
|`CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES`|  |  |  | `HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES` | 2.8.0 |  |  | 
|`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`|  |  |  | `HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` | 2.8.0 |  |  | 
|`CU_FUNC_ATTRIBUTE_MAX`|  |  |  | `HIP_FUNC_ATTRIBUTE_MAX` | 2.8.0 |  |  | 
|`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`| 9.0 |  |  | `HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` | 2.8.0 |  |  | 
|`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`|  |  |  | `HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` | 2.8.0 |  |  | 
|`CU_FUNC_ATTRIBUTE_NUM_REGS`|  |  |  | `HIP_FUNC_ATTRIBUTE_NUM_REGS` | 2.8.0 |  |  | 
|`CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`| 9.0 |  |  | `HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT` | 2.8.0 |  |  | 
|`CU_FUNC_ATTRIBUTE_PTX_VERSION`|  |  |  | `HIP_FUNC_ATTRIBUTE_PTX_VERSION` | 2.8.0 |  |  | 
|`CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`|  |  |  | `HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES` | 2.8.0 |  |  | 
|`CU_FUNC_CACHE_PREFER_EQUAL`|  |  |  | `hipFuncCachePreferEqual` | 1.6.0 |  |  | 
|`CU_FUNC_CACHE_PREFER_L1`|  |  |  | `hipFuncCachePreferL1` | 1.6.0 |  |  | 
|`CU_FUNC_CACHE_PREFER_NONE`|  |  |  | `hipFuncCachePreferNone` | 1.6.0 |  |  | 
|`CU_FUNC_CACHE_PREFER_SHARED`|  |  |  | `hipFuncCachePreferShared` | 1.6.0 |  |  | 
|`CU_GL_DEVICE_LIST_ALL`|  |  |  |  |  |  |  | 
|`CU_GL_DEVICE_LIST_CURRENT_FRAME`|  |  |  |  |  |  |  | 
|`CU_GL_DEVICE_LIST_NEXT_FRAME`|  |  |  |  |  |  |  | 
|`CU_GL_MAP_RESOURCE_FLAGS_NONE`|  |  |  |  |  |  |  | 
|`CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY`|  |  |  |  |  |  |  | 
|`CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD`|  |  |  |  |  |  |  | 
|`CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE`|  |  |  |  |  |  |  | 
|`CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY`|  |  |  |  |  |  |  | 
|`CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD`|  |  |  |  |  |  |  | 
|`CU_GRAPHICS_REGISTER_FLAGS_NONE`|  |  |  |  |  |  |  | 
|`CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY`|  |  |  |  |  |  |  | 
|`CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST`|  |  |  |  |  |  |  | 
|`CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER`|  |  |  |  |  |  |  | 
|`CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD`|  |  |  |  |  |  |  | 
|`CU_GRAPH_EXEC_UPDATE_ERROR`| 10.2 |  |  |  |  |  |  | 
|`CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED`| 10.2 |  |  |  |  |  |  | 
|`CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED`| 10.2 |  |  |  |  |  |  | 
|`CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED`| 10.2 |  |  |  |  |  |  | 
|`CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED`| 10.2 |  |  |  |  |  |  | 
|`CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED`| 10.2 |  |  |  |  |  |  | 
|`CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE`| 11.2 |  |  |  |  |  |  | 
|`CU_GRAPH_EXEC_UPDATE_SUCCESS`| 10.2 |  |  |  |  |  |  | 
|`CU_GRAPH_NODE_TYPE_COUNT`| 10.0 |  | 11.0 |  |  |  |  | 
|`CU_GRAPH_NODE_TYPE_EMPTY`| 10.0 |  |  |  |  |  |  | 
|`CU_GRAPH_NODE_TYPE_EVENT_RECORD`| 11.1 |  |  |  |  |  |  | 
|`CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL`| 11.2 |  |  |  |  |  |  | 
|`CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT`| 11.2 |  |  |  |  |  |  | 
|`CU_GRAPH_NODE_TYPE_GRAPH`| 10.0 |  |  |  |  |  |  | 
|`CU_GRAPH_NODE_TYPE_HOST`| 10.0 |  |  |  |  |  |  | 
|`CU_GRAPH_NODE_TYPE_KERNEL`| 10.0 |  |  |  |  |  |  | 
|`CU_GRAPH_NODE_TYPE_MEMCPY`| 10.0 |  |  |  |  |  |  | 
|`CU_GRAPH_NODE_TYPE_MEMSET`| 10.0 |  |  |  |  |  |  | 
|`CU_GRAPH_NODE_TYPE_WAIT_EVENT`| 11.1 |  |  |  |  |  |  | 
|`CU_IPC_HANDLE_SIZE`|  |  |  | `HIP_IPC_HANDLE_SIZE` | 1.6.0 |  |  | 
|`CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS`|  |  |  | `hipIpcMemLazyEnablePeerAccess` | 1.6.0 |  |  | 
|`CU_JIT_CACHE_MODE`|  |  |  | `hipJitOptionCacheMode` | 1.6.0 |  |  | 
|`CU_JIT_CACHE_OPTION_CA`|  |  |  |  |  |  |  | 
|`CU_JIT_CACHE_OPTION_CG`|  |  |  |  |  |  |  | 
|`CU_JIT_CACHE_OPTION_NONE`|  |  |  |  |  |  |  | 
|`CU_JIT_ERROR_LOG_BUFFER`|  |  |  | `hipJitOptionErrorLogBuffer` | 1.6.0 |  |  | 
|`CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`|  |  |  | `hipJitOptionErrorLogBufferSizeBytes` | 1.6.0 |  |  | 
|`CU_JIT_FALLBACK_STRATEGY`|  |  |  | `hipJitOptionFallbackStrategy` | 1.6.0 |  |  | 
|`CU_JIT_FAST_COMPILE`|  |  |  | `hipJitOptionFastCompile` | 1.6.0 |  |  | 
|`CU_JIT_GENERATE_DEBUG_INFO`|  |  |  | `hipJitOptionGenerateDebugInfo` | 1.6.0 |  |  | 
|`CU_JIT_GENERATE_LINE_INFO`|  |  |  | `hipJitOptionGenerateLineInfo` | 1.6.0 |  |  | 
|`CU_JIT_GLOBAL_SYMBOL_ADDRESSES`|  |  |  |  |  |  |  | 
|`CU_JIT_GLOBAL_SYMBOL_COUNT`|  |  |  |  |  |  |  | 
|`CU_JIT_GLOBAL_SYMBOL_NAMES`|  |  |  |  |  |  |  | 
|`CU_JIT_INFO_LOG_BUFFER`|  |  |  | `hipJitOptionInfoLogBuffer` | 1.6.0 |  |  | 
|`CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES`|  |  |  | `hipJitOptionInfoLogBufferSizeBytes` | 1.6.0 |  |  | 
|`CU_JIT_INPUT_CUBIN`|  |  |  |  |  |  |  | 
|`CU_JIT_INPUT_FATBINARY`|  |  |  |  |  |  |  | 
|`CU_JIT_INPUT_LIBRARY`|  |  |  |  |  |  |  | 
|`CU_JIT_INPUT_OBJECT`|  |  |  |  |  |  |  | 
|`CU_JIT_INPUT_PTX`|  |  |  |  |  |  |  | 
|`CU_JIT_LOG_VERBOSE`|  |  |  | `hipJitOptionLogVerbose` | 1.6.0 |  |  | 
|`CU_JIT_MAX_REGISTERS`|  |  |  | `hipJitOptionMaxRegisters` | 1.6.0 |  |  | 
|`CU_JIT_NEW_SM3X_OPT`|  |  |  | `hipJitOptionSm3xOpt` | 1.6.0 |  |  | 
|`CU_JIT_NUM_INPUT_TYPES`|  |  |  |  |  |  |  | 
|`CU_JIT_NUM_OPTIONS`|  |  |  | `hipJitOptionNumOptions` | 1.6.0 |  |  | 
|`CU_JIT_OPTIMIZATION_LEVEL`|  |  |  | `hipJitOptionOptimizationLevel` | 1.6.0 |  |  | 
|`CU_JIT_TARGET`|  |  |  | `hipJitOptionTarget` | 1.6.0 |  |  | 
|`CU_JIT_TARGET_FROM_CUCONTEXT`|  |  |  | `hipJitOptionTargetFromContext` | 1.6.0 |  |  | 
|`CU_JIT_THREADS_PER_BLOCK`|  |  |  | `hipJitOptionThreadsPerBlock` | 1.6.0 |  |  | 
|`CU_JIT_WALL_TIME`|  |  |  | `hipJitOptionWallTime` | 1.6.0 |  |  | 
|`CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW`| 11.0 |  |  |  |  |  |  | 
|`CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE`| 11.0 |  |  |  |  |  |  | 
|`CU_LAUNCH_PARAM_BUFFER_POINTER`|  |  |  | `HIP_LAUNCH_PARAM_BUFFER_POINTER` | 1.6.0 |  |  | 
|`CU_LAUNCH_PARAM_BUFFER_SIZE`|  |  |  | `HIP_LAUNCH_PARAM_BUFFER_SIZE` | 1.6.0 |  |  | 
|`CU_LAUNCH_PARAM_END`|  |  |  | `HIP_LAUNCH_PARAM_END` | 1.6.0 |  |  | 
|`CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT`|  |  |  |  |  |  |  | 
|`CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH`|  |  |  |  |  |  |  | 
|`CU_LIMIT_MALLOC_HEAP_SIZE`|  |  |  | `hipLimitMallocHeapSize` | 1.6.0 |  |  | 
|`CU_LIMIT_MAX`|  |  |  |  |  |  |  | 
|`CU_LIMIT_MAX_L2_FETCH_GRANULARITY`| 10.0 |  |  |  |  |  |  | 
|`CU_LIMIT_PERSISTING_L2_CACHE_SIZE`| 11.0 |  |  |  |  |  |  | 
|`CU_LIMIT_PRINTF_FIFO_SIZE`|  |  |  |  |  |  |  | 
|`CU_LIMIT_STACK_SIZE`|  |  |  |  |  |  |  | 
|`CU_MEMHOSTALLOC_DEVICEMAP`|  |  |  | `hipHostMallocMapped` | 1.6.0 |  |  | 
|`CU_MEMHOSTALLOC_PORTABLE`|  |  |  | `hipHostMallocPortable` | 1.6.0 |  |  | 
|`CU_MEMHOSTALLOC_WRITECOMBINED`|  |  |  | `hipHostMallocWriteCombined` | 1.6.0 |  |  | 
|`CU_MEMHOSTREGISTER_DEVICEMAP`|  |  |  | `hipHostRegisterMapped` | 1.6.0 |  |  | 
|`CU_MEMHOSTREGISTER_IOMEMORY`| 7.5 |  |  | `hipHostRegisterIoMemory` | 1.6.0 |  |  | 
|`CU_MEMHOSTREGISTER_PORTABLE`|  |  |  | `hipHostRegisterPortable` | 1.6.0 |  |  | 
|`CU_MEMHOSTREGISTER_READ_ONLY`| 11.1 |  |  |  |  |  |  | 
|`CU_MEMORYTYPE_ARRAY`|  |  |  | `hipMemoryTypeArray` | 1.7.0 |  |  | 
|`CU_MEMORYTYPE_DEVICE`|  |  |  | `hipMemoryTypeDevice` | 1.6.0 |  |  | 
|`CU_MEMORYTYPE_HOST`|  |  |  | `hipMemoryTypeHost` | 1.6.0 |  |  | 
|`CU_MEMORYTYPE_UNIFIED`|  |  |  | `hipMemoryTypeUnified` | 1.6.0 |  |  | 
|`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`| 11.2 |  |  |  |  |  |  | 
|`CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES`| 11.2 |  |  |  |  |  |  | 
|`CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC`| 11.2 |  |  |  |  |  |  | 
|`CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES`| 11.2 |  |  |  |  |  |  | 
|`CU_MEM_ACCESS_FLAGS_PROT_MAX`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_ACCESS_FLAGS_PROT_NONE`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_ACCESS_FLAGS_PROT_READ`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_ACCESS_FLAGS_PROT_READWRITE`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_ADVISE_SET_ACCESSED_BY`| 8.0 |  |  | `hipMemAdviseSetAccessedBy` | 3.7.0 |  |  | 
|`CU_MEM_ADVISE_SET_PREFERRED_LOCATION`| 8.0 |  |  | `hipMemAdviseSetPreferredLocation` | 3.7.0 |  |  | 
|`CU_MEM_ADVISE_SET_READ_MOSTLY`| 8.0 |  |  | `hipMemAdviseSetReadMostly` | 3.7.0 |  |  | 
|`CU_MEM_ADVISE_UNSET_ACCESSED_BY`| 8.0 |  |  | `hipMemAdviseUnsetAccessedBy` | 3.7.0 |  |  | 
|`CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION`| 8.0 |  |  | `hipMemAdviseUnsetPreferredLocation` | 3.7.0 |  |  | 
|`CU_MEM_ADVISE_UNSET_READ_MOSTLY`| 8.0 |  |  | `hipMemAdviseUnsetReadMostly` | 3.7.0 |  |  | 
|`CU_MEM_ALLOCATION_TYPE_INVALID`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_ALLOCATION_TYPE_MAX`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_ALLOCATION_TYPE_PINNED`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_ALLOC_GRANULARITY_MINIMUM`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_ALLOC_GRANULARITY_RECOMMENDED`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_ATTACH_GLOBAL`|  |  |  | `hipMemAttachGlobal` | 2.5.0 |  |  | 
|`CU_MEM_ATTACH_HOST`|  |  |  | `hipMemAttachHost` | 2.5.0 |  |  | 
|`CU_MEM_ATTACH_SINGLE`|  |  |  | `hipMemAttachSingle` | 3.7.0 |  |  | 
|`CU_MEM_CREATE_USAGE_TILE_POOL`| 11.1 |  |  |  |  |  |  | 
|`CU_MEM_HANDLE_TYPE_GENERIC`| 11.1 |  |  |  |  |  |  | 
|`CU_MEM_HANDLE_TYPE_MAX`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_HANDLE_TYPE_NONE`| 11.2 |  |  |  |  |  |  | 
|`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_HANDLE_TYPE_WIN32`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_HANDLE_TYPE_WIN32_KMT`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_LOCATION_TYPE_DEVICE`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_LOCATION_TYPE_INVALID`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_LOCATION_TYPE_MAX`| 10.2 |  |  |  |  |  |  | 
|`CU_MEM_OPERATION_TYPE_MAP`| 11.1 |  |  |  |  |  |  | 
|`CU_MEM_OPERATION_TYPE_UNMAP`| 11.1 |  |  |  |  |  |  | 
|`CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY`| 8.0 |  |  | `hipMemRangeAttributeAccessedBy` | 3.7.0 |  |  | 
|`CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION`| 8.0 |  |  | `hipMemRangeAttributeLastPrefetchLocation` | 3.7.0 |  |  | 
|`CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION`| 8.0 |  |  | `hipMemRangeAttributePreferredLocation` | 3.7.0 |  |  | 
|`CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY`| 8.0 |  |  | `hipMemRangeAttributeReadMostly` | 3.7.0 |  |  | 
|`CU_OCCUPANCY_DEFAULT`|  |  |  | `hipOccupancyDefault` | 3.2.0 |  |  | 
|`CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE`|  |  |  |  |  |  |  | 
|`CU_PARAM_TR_DEFAULT`|  |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_ACCESS_FLAGS`| 11.1 |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE`| 11.1 |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ`| 11.1 |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE`| 11.1 |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES`| 10.2 |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_BUFFER_ID`|  |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_CONTEXT`|  |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL`| 9.2 |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_DEVICE_POINTER`|  |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_HOST_POINTER`|  |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE`| 11.0 |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE`| 10.2 |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_IS_MANAGED`|  |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_MAPPED`| 10.2 |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_MEMORY_TYPE`|  |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_P2P_TOKENS`|  |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_RANGE_SIZE`| 10.2 |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_RANGE_START_ADDR`| 10.2 |  |  |  |  |  |  | 
|`CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`|  |  |  |  |  |  |  | 
|`CU_PREFER_BINARY`|  |  |  |  |  |  |  | 
|`CU_PREFER_PTX`|  |  |  |  |  |  |  | 
|`CU_RESOURCE_TYPE_ARRAY`|  |  |  | `HIP_RESOURCE_TYPE_ARRAY` | 3.5.0 |  |  | 
|`CU_RESOURCE_TYPE_LINEAR`|  |  |  | `HIP_RESOURCE_TYPE_LINEAR` | 3.5.0 |  |  | 
|`CU_RESOURCE_TYPE_MIPMAPPED_ARRAY`|  |  |  | `HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY` | 3.5.0 |  |  | 
|`CU_RESOURCE_TYPE_PITCH2D`|  |  |  | `HIP_RESOURCE_TYPE_PITCH2D` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_FLOAT_1X16`|  |  |  | `HIP_RES_VIEW_FORMAT_FLOAT_1X16` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_FLOAT_1X32`|  |  |  | `HIP_RES_VIEW_FORMAT_FLOAT_1X32` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_FLOAT_2X16`|  |  |  | `HIP_RES_VIEW_FORMAT_FLOAT_2X16` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_FLOAT_2X32`|  |  |  | `HIP_RES_VIEW_FORMAT_FLOAT_2X32` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_FLOAT_4X16`|  |  |  | `HIP_RES_VIEW_FORMAT_FLOAT_4X16` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_FLOAT_4X32`|  |  |  | `HIP_RES_VIEW_FORMAT_FLOAT_4X32` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_NONE`|  |  |  | `HIP_RES_VIEW_FORMAT_NONE` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_SIGNED_BC4`|  |  |  | `HIP_RES_VIEW_FORMAT_SIGNED_BC4` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_SIGNED_BC5`|  |  |  | `HIP_RES_VIEW_FORMAT_SIGNED_BC5` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_SIGNED_BC6H`|  |  |  | `HIP_RES_VIEW_FORMAT_SIGNED_BC6H` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_SINT_1X16`|  |  |  | `HIP_RES_VIEW_FORMAT_SINT_1X16` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_SINT_1X32`|  |  |  | `HIP_RES_VIEW_FORMAT_SINT_1X32` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_SINT_1X8`|  |  |  | `HIP_RES_VIEW_FORMAT_SINT_1X8` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_SINT_2X16`|  |  |  | `HIP_RES_VIEW_FORMAT_SINT_2X16` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_SINT_2X32`|  |  |  | `HIP_RES_VIEW_FORMAT_SINT_2X32` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_SINT_2X8`|  |  |  | `HIP_RES_VIEW_FORMAT_SINT_2X8` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_SINT_4X16`|  |  |  | `HIP_RES_VIEW_FORMAT_SINT_4X16` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_SINT_4X32`|  |  |  | `HIP_RES_VIEW_FORMAT_SINT_4X32` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_SINT_4X8`|  |  |  | `HIP_RES_VIEW_FORMAT_SINT_4X8` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UINT_1X16`|  |  |  | `HIP_RES_VIEW_FORMAT_UINT_1X16` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UINT_1X32`|  |  |  | `HIP_RES_VIEW_FORMAT_UINT_1X32` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UINT_1X8`|  |  |  | `HIP_RES_VIEW_FORMAT_UINT_1X8` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UINT_2X16`|  |  |  | `HIP_RES_VIEW_FORMAT_UINT_2X16` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UINT_2X32`|  |  |  | `HIP_RES_VIEW_FORMAT_UINT_2X32` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UINT_2X8`|  |  |  | `HIP_RES_VIEW_FORMAT_UINT_2X8` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UINT_4X16`|  |  |  | `HIP_RES_VIEW_FORMAT_UINT_4X16` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UINT_4X32`|  |  |  | `HIP_RES_VIEW_FORMAT_UINT_4X32` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UINT_4X8`|  |  |  | `HIP_RES_VIEW_FORMAT_UINT_4X8` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC1`|  |  |  | `HIP_RES_VIEW_FORMAT_UNSIGNED_BC1` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC2`|  |  |  | `HIP_RES_VIEW_FORMAT_UNSIGNED_BC2` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC3`|  |  |  | `HIP_RES_VIEW_FORMAT_UNSIGNED_BC3` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC4`|  |  |  | `HIP_RES_VIEW_FORMAT_UNSIGNED_BC4` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC5`|  |  |  | `HIP_RES_VIEW_FORMAT_UNSIGNED_BC5` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC6H`|  |  |  | `HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H` | 3.5.0 |  |  | 
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC7`|  |  |  | `HIP_RES_VIEW_FORMAT_UNSIGNED_BC7` | 3.5.0 |  |  | 
|`CU_SHAREDMEM_CARVEOUT_DEFAULT`| 9.0 |  |  |  |  |  |  | 
|`CU_SHAREDMEM_CARVEOUT_MAX_L1`| 9.0 |  |  |  |  |  |  | 
|`CU_SHAREDMEM_CARVEOUT_MAX_SHARED`| 9.0 |  |  |  |  |  |  | 
|`CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE`|  |  |  | `hipSharedMemBankSizeDefault` | 1.6.0 |  |  | 
|`CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE`|  |  |  | `hipSharedMemBankSizeEightByte` | 1.6.0 |  |  | 
|`CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE`|  |  |  | `hipSharedMemBankSizeFourByte` | 1.6.0 |  |  | 
|`CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW`| 11.0 |  |  |  |  |  |  | 
|`CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY`| 11.0 |  |  |  |  |  |  | 
|`CU_STREAM_CAPTURE_MODE_GLOBAL`| 10.1 |  |  |  |  |  |  | 
|`CU_STREAM_CAPTURE_MODE_RELAXED`| 10.1 |  |  |  |  |  |  | 
|`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`| 10.1 |  |  |  |  |  |  | 
|`CU_STREAM_CAPTURE_STATUS_ACTIVE`| 10.0 |  |  |  |  |  |  | 
|`CU_STREAM_CAPTURE_STATUS_INVALIDATED`| 10.0 |  |  |  |  |  |  | 
|`CU_STREAM_CAPTURE_STATUS_NONE`| 10.0 |  |  |  |  |  |  | 
|`CU_STREAM_DEFAULT`|  |  |  | `hipStreamDefault` | 1.6.0 |  |  | 
|`CU_STREAM_LEGACY`|  |  |  |  |  |  |  | 
|`CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES`| 8.0 |  |  |  |  |  |  | 
|`CU_STREAM_MEM_OP_WAIT_VALUE_32`| 8.0 |  |  |  |  |  |  | 
|`CU_STREAM_MEM_OP_WAIT_VALUE_64`| 9.0 |  |  |  |  |  |  | 
|`CU_STREAM_MEM_OP_WRITE_VALUE_32`| 8.0 |  |  |  |  |  |  | 
|`CU_STREAM_MEM_OP_WRITE_VALUE_64`| 9.0 |  |  |  |  |  |  | 
|`CU_STREAM_NON_BLOCKING`|  |  |  | `hipStreamNonBlocking` | 1.6.0 |  |  | 
|`CU_STREAM_PER_THREAD`|  |  |  |  |  |  |  | 
|`CU_STREAM_WAIT_VALUE_AND`| 8.0 |  |  |  |  |  |  | 
|`CU_STREAM_WAIT_VALUE_EQ`| 8.0 |  |  |  |  |  |  | 
|`CU_STREAM_WAIT_VALUE_FLUSH`| 8.0 |  |  |  |  |  |  | 
|`CU_STREAM_WAIT_VALUE_GEQ`| 8.0 |  |  |  |  |  |  | 
|`CU_STREAM_WRITE_VALUE_DEFAULT`| 8.0 |  |  |  |  |  |  | 
|`CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER`| 8.0 |  |  |  |  |  |  | 
|`CU_SYNC_POLICY_AUTO`| 11.0 |  |  |  |  |  |  | 
|`CU_SYNC_POLICY_BLOCKING_SYNC`| 11.0 |  |  |  |  |  |  | 
|`CU_SYNC_POLICY_SPIN`| 11.0 |  |  |  |  |  |  | 
|`CU_SYNC_POLICY_YIELD`| 11.0 |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_10`|  |  | 9.0 |  |  |  |  | 
|`CU_TARGET_COMPUTE_11`|  |  | 9.0 |  |  |  |  | 
|`CU_TARGET_COMPUTE_12`|  |  | 9.0 |  |  |  |  | 
|`CU_TARGET_COMPUTE_13`|  |  | 9.0 |  |  |  |  | 
|`CU_TARGET_COMPUTE_20`|  |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_21`|  |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_30`|  |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_32`|  |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_35`|  |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_37`|  |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_50`|  |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_52`|  |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_53`| 8.0 |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_60`| 8.0 |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_61`| 8.0 |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_62`| 8.0 |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_70`| 9.0 |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_72`| 10.1 |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_73`| 9.1 |  | 10.0 |  |  |  |  | 
|`CU_TARGET_COMPUTE_75`| 9.1 |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_80`| 11.0 |  |  |  |  |  |  | 
|`CU_TARGET_COMPUTE_86`| 11.1 |  |  |  |  |  |  | 
|`CU_TRSA_OVERRIDE_FORMAT`|  |  |  | `HIP_TRSA_OVERRIDE_FORMAT` | 1.7.0 |  |  | 
|`CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION`| 11.0 |  |  |  |  |  |  | 
|`CU_TRSF_NORMALIZED_COORDINATES`|  |  |  | `HIP_TRSF_NORMALIZED_COORDINATES` | 1.7.0 |  |  | 
|`CU_TRSF_READ_AS_INTEGER`|  |  |  | `HIP_TRSF_READ_AS_INTEGER` | 1.7.0 |  |  | 
|`CU_TRSF_SRGB`|  |  |  | `HIP_TRSF_SRGB` | 3.2.0 |  |  | 
|`CU_TR_ADDRESS_MODE_BORDER`|  |  |  | `HIP_TR_ADDRESS_MODE_BORDER` | 3.5.0 |  |  | 
|`CU_TR_ADDRESS_MODE_CLAMP`|  |  |  | `HIP_TR_ADDRESS_MODE_CLAMP` | 3.5.0 |  |  | 
|`CU_TR_ADDRESS_MODE_MIRROR`|  |  |  | `HIP_TR_ADDRESS_MODE_MIRROR` | 3.5.0 |  |  | 
|`CU_TR_ADDRESS_MODE_WRAP`|  |  |  | `HIP_TR_ADDRESS_MODE_WRAP` | 3.5.0 |  |  | 
|`CU_TR_FILTER_MODE_LINEAR`|  |  |  | `HIP_TR_FILTER_MODE_LINEAR` | 3.5.0 |  |  | 
|`CU_TR_FILTER_MODE_POINT`|  |  |  | `HIP_TR_FILTER_MODE_POINT` | 3.5.0 |  |  | 
|`CUaccessPolicyWindow`| 11.0 |  |  |  |  |  |  | 
|`CUaccessPolicyWindow_st`| 11.0 |  |  |  |  |  |  | 
|`CUaccessProperty`| 11.0 |  |  |  |  |  |  | 
|`CUaccessProperty_enum`| 11.0 |  |  |  |  |  |  | 
|`CUaddress_mode`|  |  |  | `HIPaddress_mode` | 3.5.0 |  |  | 
|`CUaddress_mode_enum`|  |  |  | `HIPaddress_mode_enum` | 3.5.0 |  |  | 
|`CUarray`|  |  |  | `hipArray *` |  |  |  | 
|`CUarrayMapInfo`| 11.1 |  |  |  |  |  |  | 
|`CUarrayMapInfo_st`| 11.1 |  |  |  |  |  |  | 
|`CUarraySparseSubresourceType`| 11.1 |  |  |  |  |  |  | 
|`CUarraySparseSubresourceType_enum`| 11.1 |  |  |  |  |  |  | 
|`CUarray_cubemap_face`|  |  |  |  |  |  |  | 
|`CUarray_cubemap_face_enum`|  |  |  |  |  |  |  | 
|`CUarray_format`|  |  |  | `hipArray_Format` | 1.7.0 |  |  | 
|`CUarray_format_enum`|  |  |  | `hipArray_Format` | 1.7.0 |  |  | 
|`CUarray_st`|  |  |  | `hipArray` | 1.7.0 |  |  | 
|`CUcomputemode`|  |  |  | `hipComputeMode` | 1.9.0 |  |  | 
|`CUcomputemode_enum`|  |  |  | `hipComputeMode` | 1.9.0 |  |  | 
|`CUcontext`|  |  |  | `hipCtx_t` | 1.6.0 |  |  | 
|`CUctx_flags`|  |  |  |  |  |  |  | 
|`CUctx_flags_enum`|  |  |  |  |  |  |  | 
|`CUctx_st`|  |  |  | `ihipCtx_t` | 1.6.0 |  |  | 
|`CUd3d10DeviceList`|  |  |  |  |  |  |  | 
|`CUd3d10DeviceList_enum`|  |  |  |  |  |  |  | 
|`CUd3d10map_flags`|  |  |  |  |  |  |  | 
|`CUd3d10map_flags_enum`|  |  |  |  |  |  |  | 
|`CUd3d10register_flags`|  |  |  |  |  |  |  | 
|`CUd3d10register_flags_enum`|  |  |  |  |  |  |  | 
|`CUd3d11DeviceList`|  |  |  |  |  |  |  | 
|`CUd3d11DeviceList_enum`|  |  |  |  |  |  |  | 
|`CUd3d9DeviceList`|  |  |  |  |  |  |  | 
|`CUd3d9DeviceList_enum`|  |  |  |  |  |  |  | 
|`CUd3d9map_flags`|  |  |  |  |  |  |  | 
|`CUd3d9map_flags_enum`|  |  |  |  |  |  |  | 
|`CUd3d9register_flags`|  |  |  |  |  |  |  | 
|`CUd3d9register_flags_enum`|  |  |  |  |  |  |  | 
|`CUdevice`|  |  |  | `hipDevice_t` | 1.6.0 |  |  | 
|`CUdevice_P2PAttribute`| 8.0 |  |  | `hipDeviceP2PAttr` | 3.8.0 |  |  | 
|`CUdevice_P2PAttribute_enum`| 8.0 |  |  | `hipDeviceP2PAttr` | 3.8.0 |  |  | 
|`CUdevice_attribute`|  |  |  | `hipDeviceAttribute_t` | 1.6.0 |  |  | 
|`CUdevice_attribute_enum`|  |  |  | `hipDeviceAttribute_t` | 1.6.0 |  |  | 
|`CUdeviceptr`|  |  |  | `hipDeviceptr_t` | 1.7.0 |  |  | 
|`CUdeviceptr_v1`|  |  |  | `hipDeviceptr_t` | 1.7.0 |  |  | 
|`CUdevprop`|  |  |  |  |  |  |  | 
|`CUdevprop_st`|  |  |  |  |  |  |  | 
|`CUeglColorFormat`| 9.0 |  |  |  |  |  |  | 
|`CUeglColorFormate_enum`| 9.0 |  |  |  |  |  |  | 
|`CUeglFrameType`| 9.0 |  |  |  |  |  |  | 
|`CUeglFrameType_enum`| 9.0 |  |  |  |  |  |  | 
|`CUeglResourceLocationFlags`| 9.0 |  |  |  |  |  |  | 
|`CUeglResourceLocationFlags_enum`| 9.0 |  |  |  |  |  |  | 
|`CUeglStreamConnection`| 9.0 |  |  |  |  |  |  | 
|`CUeglStreamConnection_st`| 9.0 |  |  |  |  |  |  | 
|`CUevent`|  |  |  | `hipEvent_t` | 1.6.0 |  |  | 
|`CUevent_flags`|  |  |  |  |  |  |  | 
|`CUevent_flags_enum`|  |  |  |  |  |  |  | 
|`CUevent_record_flags`| 11.1 |  |  |  |  |  |  | 
|`CUevent_record_flags_enum`| 11.1 |  |  |  |  |  |  | 
|`CUevent_st`|  |  |  | `ihipEvent_t` | 1.6.0 |  |  | 
|`CUevent_wait_flags`| 11.1 |  |  |  |  |  |  | 
|`CUevent_wait_flags_enum`|  |  |  |  |  |  |  | 
|`CUextMemory_st`| 10.0 |  |  |  |  |  |  | 
|`CUextSemaphore_st`| 10.0 |  |  |  |  |  |  | 
|`CUexternalMemory`| 10.0 |  |  |  |  |  |  | 
|`CUexternalMemoryHandleType`| 10.0 |  |  |  |  |  |  | 
|`CUexternalMemoryHandleType_enum`| 10.0 |  |  |  |  |  |  | 
|`CUexternalSemaphore`| 10.0 |  |  |  |  |  |  | 
|`CUexternalSemaphoreHandleType`| 10.0 |  |  |  |  |  |  | 
|`CUexternalSemaphoreHandleType_enum`| 10.0 |  |  |  |  |  |  | 
|`CUfilter_mode`|  |  |  | `HIPfilter_mode` | 3.5.0 |  |  | 
|`CUfilter_mode_enum`|  |  |  | `HIPfilter_mode_enum` | 3.5.0 |  |  | 
|`CUfunc_cache`|  |  |  | `hipFuncCache_t` | 1.6.0 |  |  | 
|`CUfunc_cache_enum`|  |  |  | `hipFuncCache_t` | 1.6.0 |  |  | 
|`CUfunc_st`|  |  |  | `ihipModuleSymbol_t` | 1.6.0 |  |  | 
|`CUfunction`|  |  |  | `hipFunction_t` | 1.6.0 |  |  | 
|`CUfunction_attribute`|  |  |  | `hipFunction_attribute` | 2.8.0 |  |  | 
|`CUfunction_attribute_enum`|  |  |  | `hipFunction_attribute` | 2.8.0 |  |  | 
|`CUgraph`| 10.0 |  |  |  |  |  |  | 
|`CUgraphExec`| 10.0 |  |  |  |  |  |  | 
|`CUgraphExecUpdateResult`| 10.2 |  |  |  |  |  |  | 
|`CUgraphExecUpdateResult_enum`| 10.2 |  |  |  |  |  |  | 
|`CUgraphExec_st`| 10.0 |  |  |  |  |  |  | 
|`CUgraphNode`| 10.0 |  |  |  |  |  |  | 
|`CUgraphNodeType`| 10.0 |  |  |  |  |  |  | 
|`CUgraphNodeType_enum`| 10.0 |  |  |  |  |  |  | 
|`CUgraphNode_st`| 10.0 |  |  |  |  |  |  | 
|`CUgraph_st`| 10.0 |  |  |  |  |  |  | 
|`CUgraphicsMapResourceFlags`|  |  |  |  |  |  |  | 
|`CUgraphicsMapResourceFlags_enum`|  |  |  |  |  |  |  | 
|`CUgraphicsRegisterFlags`|  |  |  |  |  |  |  | 
|`CUgraphicsRegisterFlags_enum`|  |  |  |  |  |  |  | 
|`CUgraphicsResource`|  |  |  |  |  |  |  | 
|`CUgraphicsResource_st`|  |  |  |  |  |  |  | 
|`CUhostFn`| 10.0 |  |  |  |  |  |  | 
|`CUipcEventHandle`|  |  |  | `hipIpcEventHandle_t` | 1.6.0 |  |  | 
|`CUipcEventHandle_st`|  |  |  | `hipIpcEventHandle_st` | 3.5.0 |  |  | 
|`CUipcMemHandle`|  |  |  | `hipIpcMemHandle_t` | 1.6.0 |  |  | 
|`CUipcMemHandle_st`|  |  |  | `hipIpcMemHandle_st` | 1.6.0 |  |  | 
|`CUipcMem_flags`|  |  |  |  |  |  |  | 
|`CUipcMem_flags_enum`|  |  |  |  |  |  |  | 
|`CUjitInputType`|  |  |  |  |  |  |  | 
|`CUjitInputType_enum`|  |  |  |  |  |  |  | 
|`CUjit_cacheMode`|  |  |  |  |  |  |  | 
|`CUjit_cacheMode_enum`|  |  |  |  |  |  |  | 
|`CUjit_fallback`|  |  |  |  |  |  |  | 
|`CUjit_fallback_enum`|  |  |  |  |  |  |  | 
|`CUjit_option`|  |  |  | `hipJitOption` | 1.6.0 |  |  | 
|`CUjit_option_enum`|  |  |  | `hipJitOption` | 1.6.0 |  |  | 
|`CUjit_target`|  |  |  |  |  |  |  | 
|`CUjit_target_enum`|  |  |  |  |  |  |  | 
|`CUkernelNodeAttrID`| 11.0 |  |  |  |  |  |  | 
|`CUkernelNodeAttrID_enum`| 11.0 |  |  |  |  |  |  | 
|`CUkernelNodeAttrValue`| 11.0 |  |  |  |  |  |  | 
|`CUkernelNodeAttrValue_union`| 11.0 |  |  |  |  |  |  | 
|`CUlimit`|  |  |  | `hipLimit_t` | 1.6.0 |  |  | 
|`CUlimit_enum`|  |  |  | `hipLimit_t` | 1.6.0 |  |  | 
|`CUmemAccessDesc`| 10.2 |  |  |  |  |  |  | 
|`CUmemAccessDesc_st`| 10.2 |  |  |  |  |  |  | 
|`CUmemAccess_flags`| 10.2 |  |  |  |  |  |  | 
|`CUmemAccess_flags_enum`| 10.2 |  |  |  |  |  |  | 
|`CUmemAllocationGranularity_flags`| 10.2 |  |  |  |  |  |  | 
|`CUmemAllocationGranularity_flags_enum`| 10.2 |  |  |  |  |  |  | 
|`CUmemAllocationHandleType`| 10.2 |  |  |  |  |  |  | 
|`CUmemAllocationHandleType_enum`| 10.2 |  |  |  |  |  |  | 
|`CUmemAllocationProp`| 10.2 |  |  |  |  |  |  | 
|`CUmemAllocationProp_st`| 10.2 |  |  |  |  |  |  | 
|`CUmemAllocationType`| 10.2 |  |  |  |  |  |  | 
|`CUmemAllocationType_enum`| 10.2 |  |  |  |  |  |  | 
|`CUmemAttach_flags`|  |  |  |  |  |  |  | 
|`CUmemAttach_flags_enum`|  |  |  |  |  |  |  | 
|`CUmemGenericAllocationHandle`| 10.2 |  |  |  |  |  |  | 
|`CUmemHandleType`| 11.1 |  |  |  |  |  |  | 
|`CUmemHandleType_enum`| 11.1 |  |  |  |  |  |  | 
|`CUmemLocation`| 10.2 |  |  |  |  |  |  | 
|`CUmemLocationType`| 10.2 |  |  |  |  |  |  | 
|`CUmemLocationType_enum`| 10.2 |  |  |  |  |  |  | 
|`CUmemLocation_st`| 10.2 |  |  |  |  |  |  | 
|`CUmemOperationType`| 11.1 |  |  |  |  |  |  | 
|`CUmemOperationType_enum`| 11.1 |  |  |  |  |  |  | 
|`CUmemPoolHandle_st`| 11.2 |  |  |  |  |  |  | 
|`CUmemPoolProps`| 11.2 |  |  |  |  |  |  | 
|`CUmemPoolProps_st`| 11.2 |  |  |  |  |  |  | 
|`CUmemPoolPtrExportData`| 11.2 |  |  |  |  |  |  | 
|`CUmemPoolPtrExportData_st`| 11.2 |  |  |  |  |  |  | 
|`CUmemPool_attribute`| 11.2 |  |  |  |  |  |  | 
|`CUmemPool_attribute_enum`| 11.2 |  |  |  |  |  |  | 
|`CUmem_advise`| 8.0 |  |  | `hipMemoryAdvise` | 3.7.0 |  |  | 
|`CUmem_advise_enum`| 8.0 |  |  | `hipMemoryAdvise` | 3.7.0 |  |  | 
|`CUmem_range_attribute`| 8.0 |  |  | `hipMemRangeAttribute` | 3.7.0 |  |  | 
|`CUmem_range_attribute_enum`| 8.0 |  |  | `hipMemRangeAttribute` | 3.7.0 |  |  | 
|`CUmemoryPool`| 11.2 |  |  |  |  |  |  | 
|`CUmemorytype`|  |  |  | `hipMemoryType` | 1.6.0 |  |  | 
|`CUmemorytype_enum`|  |  |  | `hipMemoryType` | 1.6.0 |  |  | 
|`CUmipmappedArray`|  |  |  | `hipMipmappedArray_t` | 1.7.0 |  |  | 
|`CUmipmappedArray_st`|  |  |  | `hipMipmappedArray` | 1.7.0 |  |  | 
|`CUmod_st`|  |  |  | `ihipModule_t` | 1.6.0 |  |  | 
|`CUmodule`|  |  |  | `hipModule_t` | 1.6.0 |  |  | 
|`CUoccupancyB2DSize`|  |  |  |  |  |  |  | 
|`CUoccupancy_flags`|  |  |  |  |  |  |  | 
|`CUoccupancy_flags_enum`|  |  |  |  |  |  |  | 
|`CUpointer_attribute`|  |  |  |  |  |  |  | 
|`CUpointer_attribute_enum`|  |  |  |  |  |  |  | 
|`CUresourceViewFormat`|  |  |  | `HIPresourceViewFormat` | 3.5.0 |  |  | 
|`CUresourceViewFormat_enum`|  |  |  | `HIPresourceViewFormat_enum` | 3.5.0 |  |  | 
|`CUresourcetype`|  |  |  | `HIPresourcetype` | 3.5.0 |  |  | 
|`CUresourcetype_enum`|  |  |  | `HIPresourcetype_enum` | 3.5.0 |  |  | 
|`CUresult`|  |  |  | `hipError_t` | 1.5.0 |  |  | 
|`CUshared_carveout`| 9.0 |  |  |  |  |  |  | 
|`CUshared_carveout_enum`| 9.0 |  |  |  |  |  |  | 
|`CUsharedconfig`|  |  |  | `hipSharedMemConfig` | 1.6.0 |  |  | 
|`CUsharedconfig_enum`|  |  |  | `hipSharedMemConfig` | 1.6.0 |  |  | 
|`CUstream`|  |  |  | `hipStream_t` | 1.5.0 |  |  | 
|`CUstreamAttrID`| 11.0 |  |  |  |  |  |  | 
|`CUstreamAttrID_enum`| 11.0 |  |  |  |  |  |  | 
|`CUstreamAttrValue`| 11.0 |  |  |  |  |  |  | 
|`CUstreamAttrValue_union`| 11.0 |  |  |  |  |  |  | 
|`CUstreamBatchMemOpParams`| 8.0 |  |  |  |  |  |  | 
|`CUstreamBatchMemOpParams_union`| 8.0 |  |  |  |  |  |  | 
|`CUstreamBatchMemOpType`| 8.0 |  |  |  |  |  |  | 
|`CUstreamBatchMemOpType_enum`| 8.0 |  |  |  |  |  |  | 
|`CUstreamCallback`|  |  |  | `hipStreamCallback_t` | 1.6.0 |  |  | 
|`CUstreamCaptureMode`| 10.1 |  |  |  |  |  |  | 
|`CUstreamCaptureMode_enum`| 10.1 |  |  |  |  |  |  | 
|`CUstreamCaptureStatus`| 10.0 |  |  |  |  |  |  | 
|`CUstreamCaptureStatus_enum`| 10.0 |  |  |  |  |  |  | 
|`CUstreamWaitValue_flags`| 8.0 |  |  |  |  |  |  | 
|`CUstreamWaitValue_flags_enum`| 8.0 |  |  |  |  |  |  | 
|`CUstreamWriteValue_flags`| 8.0 |  |  |  |  |  |  | 
|`CUstreamWriteValue_flags_enum`| 8.0 |  |  |  |  |  |  | 
|`CUstream_flags`|  |  |  |  |  |  |  | 
|`CUstream_flags_enum`|  |  |  |  |  |  |  | 
|`CUstream_st`|  |  |  | `ihipStream_t` | 1.5.0 |  |  | 
|`CUsurfObject`|  |  |  |  |  |  |  | 
|`CUsurfref`|  |  |  |  |  |  |  | 
|`CUsurfref_st`|  |  |  |  |  |  |  | 
|`CUsynchronizationPolicy`| 11.0 |  |  |  |  |  |  | 
|`CUsynchronizationPolicy_enum`| 11.0 |  |  |  |  |  |  | 
|`CUtexObject`|  |  |  | `hipTextureObject_t` | 1.7.0 |  |  | 
|`CUtexref`|  |  |  |  |  |  |  | 
|`CUtexref_st`|  |  |  | `textureReference` | 1.6.0 |  |  | 
|`CUuuid`|  |  |  |  |  |  |  | 
|`CUuuid_st`|  |  |  |  |  |  |  | 
|`__CUDACC__`|  |  |  | `__HIPCC__` | 1.6.0 |  |  | 
|`cudaError_enum`|  |  |  | `hipError_t` | 1.5.0 |  |  | 

## **2. Error Handling**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuGetErrorName`|  |  |  |  |  |  |  | 
|`cuGetErrorString`|  |  |  |  |  |  |  | 

## **3. Initialization**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuInit`|  |  |  | `hipInit` | 1.6.0 |  |  | 

## **4. Version Management**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuDriverGetVersion`|  |  |  | `hipDriverGetVersion` | 1.6.0 |  |  | 

## **5. Device Management**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuDeviceGet`|  |  |  | `hipDeviceGet` | 1.6.0 |  |  | 
|`cuDeviceGetAttribute`|  |  |  | `hipDeviceGetAttribute` | 1.6.0 |  |  | 
|`cuDeviceGetCount`|  |  |  | `hipGetDeviceCount` | 1.6.0 |  |  | 
|`cuDeviceGetDefaultMemPool`| 11.2 |  |  |  |  |  |  | 
|`cuDeviceGetLuid`| 10.0 |  |  |  |  |  |  | 
|`cuDeviceGetMemPool`| 11.2 |  |  |  |  |  |  | 
|`cuDeviceGetName`|  |  |  | `hipDeviceGetName` | 1.6.0 |  |  | 
|`cuDeviceGetNvSciSyncAttributes`| 10.2 |  |  |  |  |  |  | 
|`cuDeviceGetTexture1DLinearMaxWidth`| 11.1 |  |  |  |  |  |  | 
|`cuDeviceGetUuid`| 9.2 |  |  |  |  |  |  | 
|`cuDeviceSetMemPool`| 11.2 |  |  |  |  |  |  | 
|`cuDeviceTotalMem`|  |  |  | `hipDeviceTotalMem` | 1.6.0 |  |  | 
|`cuDeviceTotalMem_v2`|  |  |  | `hipDeviceTotalMem` | 1.6.0 |  |  | 

## **6. Device Management [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuDeviceComputeCapability`|  | 9.2 |  | `hipDeviceComputeCapability` | 1.6.0 |  |  | 
|`cuDeviceGetProperties`|  | 9.2 |  |  |  |  |  | 

## **7. Primary Context Management**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuDevicePrimaryCtxGetState`|  |  |  | `hipDevicePrimaryCtxGetState` | 1.9.0 |  |  | 
|`cuDevicePrimaryCtxRelease`|  |  |  | `hipDevicePrimaryCtxRelease` | 1.9.0 |  |  | 
|`cuDevicePrimaryCtxRelease_v2`| 11.0 |  |  | `hipDevicePrimaryCtxRelease` | 1.9.0 |  |  | 
|`cuDevicePrimaryCtxReset`|  |  |  | `hipDevicePrimaryCtxReset` | 1.9.0 |  |  | 
|`cuDevicePrimaryCtxReset_v2`| 11.0 |  |  | `hipDevicePrimaryCtxReset` | 1.9.0 |  |  | 
|`cuDevicePrimaryCtxRetain`|  |  |  | `hipDevicePrimaryCtxRetain` | 1.9.0 |  |  | 
|`cuDevicePrimaryCtxSetFlags`|  |  |  | `hipDevicePrimaryCtxSetFlags` | 1.9.0 |  |  | 
|`cuDevicePrimaryCtxSetFlags_v2`| 11.0 |  |  | `hipDevicePrimaryCtxSetFlags` | 1.9.0 |  |  | 

## **8. Context Management**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuCtxCreate`|  |  |  | `hipCtxCreate` | 1.6.0 | 1.9.0 |  | 
|`cuCtxCreate_v2`|  |  |  | `hipCtxCreate` | 1.6.0 | 1.9.0 |  | 
|`cuCtxDestroy`|  |  |  | `hipCtxDestroy` | 1.6.0 | 1.9.0 |  | 
|`cuCtxDestroy_v2`|  |  |  | `hipCtxDestroy` | 1.6.0 | 1.9.0 |  | 
|`cuCtxGetApiVersion`|  |  |  | `hipCtxGetApiVersion` | 1.9.0 | 1.9.0 |  | 
|`cuCtxGetCacheConfig`|  |  |  | `hipCtxGetCacheConfig` | 1.9.0 | 1.9.0 |  | 
|`cuCtxGetCurrent`|  |  |  | `hipCtxGetCurrent` | 1.6.0 | 1.9.0 |  | 
|`cuCtxGetDevice`|  |  |  | `hipCtxGetDevice` | 1.6.0 | 1.9.0 |  | 
|`cuCtxGetFlags`|  |  |  | `hipCtxGetFlags` | 1.9.0 | 1.9.0 |  | 
|`cuCtxGetLimit`|  |  |  | `hipDeviceGetLimit` | 1.6.0 |  |  | 
|`cuCtxGetSharedMemConfig`|  |  |  | `hipCtxGetSharedMemConfig` | 1.9.0 | 1.9.0 |  | 
|`cuCtxGetStreamPriorityRange`|  |  |  | `hipDeviceGetStreamPriorityRange` | 2.0.0 |  |  | 
|`cuCtxPopCurrent`|  |  |  | `hipCtxPopCurrent` | 1.6.0 | 1.9.0 |  | 
|`cuCtxPopCurrent_v2`|  |  |  | `hipCtxPopCurrent` | 1.6.0 | 1.9.0 |  | 
|`cuCtxPushCurrent`|  |  |  | `hipCtxPushCurrent` | 1.6.0 | 1.9.0 |  | 
|`cuCtxPushCurrent_v2`|  |  |  | `hipCtxPushCurrent` | 1.6.0 | 1.9.0 |  | 
|`cuCtxResetPersistingL2Cache`| 11.0 |  |  |  |  |  |  | 
|`cuCtxSetCacheConfig`|  |  |  | `hipCtxSetCacheConfig` | 1.9.0 | 1.9.0 |  | 
|`cuCtxSetCurrent`|  |  |  | `hipCtxSetCurrent` | 1.6.0 | 1.9.0 |  | 
|`cuCtxSetLimit`|  |  |  |  |  |  |  | 
|`cuCtxSetSharedMemConfig`|  |  |  | `hipCtxSetSharedMemConfig` | 1.9.0 | 1.9.0 |  | 
|`cuCtxSynchronize`|  |  |  | `hipCtxSynchronize` | 1.9.0 | 1.9.0 |  | 

## **9. Context Management [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuCtxAttach`|  |  |  |  |  |  |  | 
|`cuCtxDetach`|  |  |  |  |  |  |  | 

## **10. Module Management**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuLinkAddData`|  |  |  |  |  |  |  | 
|`cuLinkAddData_v2`|  |  |  |  |  |  |  | 
|`cuLinkAddFile`|  |  |  |  |  |  |  | 
|`cuLinkAddFile_v2`|  |  |  |  |  |  |  | 
|`cuLinkComplete`|  |  |  |  |  |  |  | 
|`cuLinkCreate`|  |  |  |  |  |  |  | 
|`cuLinkCreate_v2`|  |  |  |  |  |  |  | 
|`cuLinkDestroy`|  |  |  |  |  |  |  | 
|`cuModuleGetFunction`|  |  |  | `hipModuleGetFunction` | 1.6.0 |  |  | 
|`cuModuleGetGlobal`|  |  |  | `hipModuleGetGlobal` | 1.6.0 |  |  | 
|`cuModuleGetGlobal_v2`|  |  |  | `hipModuleGetGlobal` | 1.6.0 |  |  | 
|`cuModuleGetSurfRef`|  |  |  |  |  |  |  | 
|`cuModuleGetTexRef`|  |  |  | `hipModuleGetTexRef` | 1.7.0 |  |  | 
|`cuModuleLoad`|  |  |  | `hipModuleLoad` | 1.6.0 |  |  | 
|`cuModuleLoadData`|  |  |  | `hipModuleLoadData` | 1.6.0 |  |  | 
|`cuModuleLoadDataEx`|  |  |  | `hipModuleLoadDataEx` | 1.6.0 |  |  | 
|`cuModuleLoadFatBinary`|  |  |  |  |  |  |  | 
|`cuModuleUnload`|  |  |  | `hipModuleUnload` | 1.6.0 |  |  | 

## **11. Memory Management**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuArray3DCreate`|  |  |  | `hipArray3DCreate` | 1.7.1 |  |  | 
|`cuArray3DCreate_v2`|  |  |  | `hipArray3DCreate` | 1.7.1 |  |  | 
|`cuArray3DGetDescriptor`|  |  |  |  |  |  |  | 
|`cuArray3DGetDescriptor_v2`|  |  |  |  |  |  |  | 
|`cuArrayCreate`|  |  |  | `hipArrayCreate` | 1.9.0 |  |  | 
|`cuArrayCreate_v2`|  |  |  | `hipArrayCreate` | 1.9.0 |  |  | 
|`cuArrayDestroy`|  |  |  |  |  |  |  | 
|`cuArrayGetDescriptor`|  |  |  |  |  |  |  | 
|`cuArrayGetDescriptor_v2`|  |  |  |  |  |  |  | 
|`cuArrayGetPlane`| 11.2 |  |  |  |  |  |  | 
|`cuArrayGetSparseProperties`| 11.1 |  |  |  |  |  |  | 
|`cuDeviceGetByPCIBusId`|  |  |  | `hipDeviceGetByPCIBusId` | 1.6.0 |  |  | 
|`cuDeviceGetPCIBusId`|  |  |  | `hipDeviceGetPCIBusId` | 1.6.0 |  |  | 
|`cuIpcCloseMemHandle`|  |  |  | `hipIpcCloseMemHandle` | 1.6.0 |  |  | 
|`cuIpcGetEventHandle`|  |  |  | `hipIpcGetEventHandle` | 1.6.0 |  |  | 
|`cuIpcGetMemHandle`|  |  |  | `hipIpcGetMemHandle` | 1.6.0 |  |  | 
|`cuIpcOpenEventHandle`|  |  |  | `hipIpcOpenEventHandle` | 1.6.0 |  |  | 
|`cuIpcOpenMemHandle`|  |  |  | `hipIpcOpenMemHandle` | 1.6.0 |  |  | 
|`cuMemAlloc`|  |  |  | `hipMalloc` | 1.5.0 |  |  | 
|`cuMemAllocHost`|  |  |  | `hipHostMalloc` | 1.6.0 |  |  | 
|`cuMemAllocHost_v2`|  |  |  | `hipHostMalloc` | 1.6.0 |  |  | 
|`cuMemAllocManaged`|  |  |  | `hipMallocManaged` | 2.5.0 |  |  | 
|`cuMemAllocPitch`|  |  |  | `hipMemAllocPitch` | 3.0.0 |  |  | 
|`cuMemAllocPitch_v2`|  |  |  | `hipMemAllocPitch` | 3.0.0 |  |  | 
|`cuMemAlloc_v2`|  |  |  | `hipMalloc` | 1.5.0 |  |  | 
|`cuMemFree`|  |  |  | `hipFree` | 1.5.0 |  |  | 
|`cuMemFreeHost`|  |  |  | `hipHostFree` | 1.6.0 |  |  | 
|`cuMemFree_v2`|  |  |  | `hipFree` | 1.5.0 |  |  | 
|`cuMemGetAddressRange`|  |  |  | `hipMemGetAddressRange` | 1.9.0 |  |  | 
|`cuMemGetAddressRange_v2`|  |  |  | `hipMemGetAddressRange` | 1.9.0 |  |  | 
|`cuMemGetInfo`|  |  |  | `hipMemGetInfo` | 1.6.0 |  |  | 
|`cuMemGetInfo_v2`|  |  |  | `hipMemGetInfo` | 1.6.0 |  |  | 
|`cuMemHostAlloc`|  |  |  | `hipHostMalloc` | 1.6.0 |  |  | 
|`cuMemHostGetDevicePointer`|  |  |  | `hipHostGetDevicePointer` | 1.6.0 |  |  | 
|`cuMemHostGetDevicePointer_v2`|  |  |  | `hipHostGetDevicePointer` | 1.6.0 |  |  | 
|`cuMemHostGetFlags`|  |  |  | `hipHostGetFlags` | 1.6.0 |  |  | 
|`cuMemHostRegister`|  |  |  | `hipHostRegister` | 1.6.0 |  |  | 
|`cuMemHostRegister_v2`|  |  |  | `hipHostRegister` | 1.6.0 |  |  | 
|`cuMemHostUnregister`|  |  |  | `hipHostUnregister` | 1.6.0 |  |  | 
|`cuMemcpy`|  |  |  |  |  |  |  | 
|`cuMemcpy2D`|  |  |  | `hipMemcpyParam2D` | 1.7.0 |  |  | 
|`cuMemcpy2DAsync`|  |  |  | `hipMemcpyParam2DAsync` | 2.8.0 |  |  | 
|`cuMemcpy2DAsync_v2`|  |  |  | `hipMemcpyParam2DAsync` | 2.8.0 |  |  | 
|`cuMemcpy2DUnaligned`|  |  |  |  |  |  |  | 
|`cuMemcpy2DUnaligned_v2`|  |  |  |  |  |  |  | 
|`cuMemcpy2D_v2`|  |  |  | `hipMemcpyParam2D` | 1.7.0 |  |  | 
|`cuMemcpy3D`|  |  |  | `hipDrvMemcpy3D` | 3.5.0 |  |  | 
|`cuMemcpy3DAsync`|  |  |  | `hipDrvMemcpy3DAsync` | 3.5.0 |  |  | 
|`cuMemcpy3DAsync_v2`|  |  |  | `hipDrvMemcpy3DAsync` | 3.5.0 |  |  | 
|`cuMemcpy3DPeer`|  |  |  |  |  |  |  | 
|`cuMemcpy3DPeerAsync`|  |  |  |  |  |  |  | 
|`cuMemcpy3D_v2`|  |  |  | `hipDrvMemcpy3D` | 3.5.0 |  |  | 
|`cuMemcpyAsync`|  |  |  |  |  |  |  | 
|`cuMemcpyAtoA`|  |  |  |  |  |  |  | 
|`cuMemcpyAtoA_v2`|  |  |  |  |  |  |  | 
|`cuMemcpyAtoD`|  |  |  |  |  |  |  | 
|`cuMemcpyAtoD_v2`|  |  |  |  |  |  |  | 
|`cuMemcpyAtoH`|  |  |  | `hipMemcpyAtoH` | 1.9.0 |  |  | 
|`cuMemcpyAtoHAsync`|  |  |  |  |  |  |  | 
|`cuMemcpyAtoHAsync_v2`|  |  |  |  |  |  |  | 
|`cuMemcpyAtoH_v2`|  |  |  | `hipMemcpyAtoH` | 1.9.0 |  |  | 
|`cuMemcpyDtoA`|  |  |  |  |  |  |  | 
|`cuMemcpyDtoA_v2`|  |  |  |  |  |  |  | 
|`cuMemcpyDtoD`|  |  |  | `hipMemcpyDtoD` | 1.6.0 |  |  | 
|`cuMemcpyDtoDAsync`|  |  |  | `hipMemcpyDtoDAsync` | 1.6.0 |  |  | 
|`cuMemcpyDtoDAsync_v2`|  |  |  | `hipMemcpyDtoDAsync` | 1.6.0 |  |  | 
|`cuMemcpyDtoD_v2`|  |  |  | `hipMemcpyDtoD` | 1.6.0 |  |  | 
|`cuMemcpyDtoH`|  |  |  | `hipMemcpyDtoH` | 1.6.0 |  |  | 
|`cuMemcpyDtoHAsync`|  |  |  | `hipMemcpyDtoHAsync` | 1.6.0 |  |  | 
|`cuMemcpyDtoHAsync_v2`|  |  |  | `hipMemcpyDtoHAsync` | 1.6.0 |  |  | 
|`cuMemcpyDtoH_v2`|  |  |  | `hipMemcpyDtoH` | 1.6.0 |  |  | 
|`cuMemcpyHtoA`|  |  |  | `hipMemcpyHtoA` | 1.9.0 |  |  | 
|`cuMemcpyHtoAAsync`|  |  |  |  |  |  |  | 
|`cuMemcpyHtoAAsync_v2`|  |  |  |  |  |  |  | 
|`cuMemcpyHtoA_v2`|  |  |  | `hipMemcpyHtoA` | 1.9.0 |  |  | 
|`cuMemcpyHtoD`|  |  |  | `hipMemcpyHtoD` | 1.6.0 |  |  | 
|`cuMemcpyHtoDAsync`|  |  |  | `hipMemcpyHtoDAsync` | 1.6.0 |  |  | 
|`cuMemcpyHtoDAsync_v2`|  |  |  | `hipMemcpyHtoDAsync` | 1.6.0 |  |  | 
|`cuMemcpyHtoD_v2`|  |  |  | `hipMemcpyHtoD` | 1.6.0 |  |  | 
|`cuMemcpyPeer`|  |  |  |  |  |  |  | 
|`cuMemcpyPeerAsync`|  |  |  |  |  |  |  | 
|`cuMemsetD16`|  |  |  | `hipMemsetD16` | 3.0.0 |  |  | 
|`cuMemsetD16Async`|  |  |  | `hipMemsetD16Async` | 3.0.0 |  |  | 
|`cuMemsetD16_v2`|  |  |  | `hipMemsetD16` | 3.0.0 |  |  | 
|`cuMemsetD2D16`|  |  |  |  |  |  |  | 
|`cuMemsetD2D16Async`|  |  |  |  |  |  |  | 
|`cuMemsetD2D16_v2`|  |  |  |  |  |  |  | 
|`cuMemsetD2D32`|  |  |  |  |  |  |  | 
|`cuMemsetD2D32Async`|  |  |  |  |  |  |  | 
|`cuMemsetD2D32_v2`|  |  |  |  |  |  |  | 
|`cuMemsetD2D8`|  |  |  |  |  |  |  | 
|`cuMemsetD2D8Async`|  |  |  |  |  |  |  | 
|`cuMemsetD2D8_v2`|  |  |  |  |  |  |  | 
|`cuMemsetD32`|  |  |  | `hipMemsetD32` | 2.3.0 |  |  | 
|`cuMemsetD32Async`|  |  |  | `hipMemsetD32Async` | 2.3.0 |  |  | 
|`cuMemsetD32_v2`|  |  |  | `hipMemsetD32` | 2.3.0 |  |  | 
|`cuMemsetD8`|  |  |  | `hipMemsetD8` | 1.6.0 |  |  | 
|`cuMemsetD8Async`|  |  |  | `hipMemsetD8Async` | 3.0.0 |  |  | 
|`cuMemsetD8_v2`|  |  |  | `hipMemsetD8` | 1.6.0 |  |  | 
|`cuMipmappedArrayCreate`|  |  |  | `hipMipmappedArrayCreate` | 3.5.0 |  |  | 
|`cuMipmappedArrayDestroy`|  |  |  | `hipMipmappedArrayDestroy` | 3.5.0 |  |  | 
|`cuMipmappedArrayGetLevel`|  |  |  | `hipMipmappedArrayGetLevel` | 3.5.0 |  |  | 

## **12. Virtual Memory Management**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuMemAddressFree`| 10.2 |  |  |  |  |  |  | 
|`cuMemAddressReserve`| 10.2 |  |  |  |  |  |  | 
|`cuMemCreate`| 10.2 |  |  |  |  |  |  | 
|`cuMemExportToShareableHandle`| 10.2 |  |  |  |  |  |  | 
|`cuMemGetAccess`| 10.2 |  |  |  |  |  |  | 
|`cuMemGetAllocationGranularity`| 10.2 |  |  |  |  |  |  | 
|`cuMemGetAllocationPropertiesFromHandle`| 10.2 |  |  |  |  |  |  | 
|`cuMemImportFromShareableHandle`| 10.2 |  |  |  |  |  |  | 
|`cuMemMap`| 10.2 |  |  |  |  |  |  | 
|`cuMemMapArrayAsync`| 11.1 |  |  |  |  |  |  | 
|`cuMemRelease`| 10.2 |  |  |  |  |  |  | 
|`cuMemRetainAllocationHandle`| 11.0 |  |  |  |  |  |  | 
|`cuMemSetAccess`| 10.2 |  |  |  |  |  |  | 
|`cuMemUnmap`| 10.2 |  |  |  |  |  |  | 

## **13. Stream Ordered Memory Allocator**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuMemAllocAsync`| 11.2 |  |  |  |  |  |  | 
|`cuMemAllocFromPoolAsync`| 11.2 |  |  |  |  |  |  | 
|`cuMemFreeAsync`| 11.2 |  |  |  |  |  |  | 
|`cuMemPoolCreate`| 11.2 |  |  |  |  |  |  | 
|`cuMemPoolDestroy`| 11.2 |  |  |  |  |  |  | 
|`cuMemPoolExportPointer`| 11.2 |  |  |  |  |  |  | 
|`cuMemPoolExportToShareableHandle`| 11.2 |  |  |  |  |  |  | 
|`cuMemPoolGetAccess`| 11.2 |  |  |  |  |  |  | 
|`cuMemPoolGetAttribute`| 11.2 |  |  |  |  |  |  | 
|`cuMemPoolImportFromShareableHandle`| 11.2 |  |  |  |  |  |  | 
|`cuMemPoolImportPointer`| 11.2 |  |  |  |  |  |  | 
|`cuMemPoolSetAccess`| 11.2 |  |  |  |  |  |  | 
|`cuMemPoolSetAttribute`| 11.2 |  |  |  |  |  |  | 
|`cuMemPoolTrimTo`| 11.2 |  |  |  |  |  |  | 

## **14. Unified Addressing**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuMemAdvise`| 8.0 |  |  | `hipMemAdvise` | 3.7.0 |  |  | 
|`cuMemPrefetchAsync`| 8.0 |  |  |  |  |  |  | 
|`cuMemRangeGetAttribute`| 8.0 |  |  | `hipMemRangeGetAttribute` | 3.7.0 |  |  | 
|`cuMemRangeGetAttributes`| 8.0 |  |  | `hipMemRangeGetAttributes` | 3.7.0 |  |  | 
|`cuPointerGetAttribute`|  |  |  |  |  |  |  | 
|`cuPointerGetAttributes`|  |  |  |  |  |  |  | 
|`cuPointerSetAttribute`|  |  |  |  |  |  |  | 

## **15. Stream Management**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuStreamAddCallback`|  |  |  | `hipStreamAddCallback` | 1.6.0 |  |  | 
|`cuStreamAttachMemAsync`|  |  |  | `hipStreamAttachMemAsync` | 3.7.0 |  |  | 
|`cuStreamBeginCapture`| 10.0 |  |  |  |  |  |  | 
|`cuStreamBeginCapture_ptsz`| 10.1 |  |  |  |  |  |  | 
|`cuStreamBeginCapture_v2`| 10.1 |  |  |  |  |  |  | 
|`cuStreamCopyAttributes`| 11.0 |  |  |  |  |  |  | 
|`cuStreamCreate`|  |  |  | `hipStreamCreateWithFlags` | 1.6.0 |  |  | 
|`cuStreamCreateWithPriority`|  |  |  | `hipStreamCreateWithPriority` | 2.0.0 |  |  | 
|`cuStreamDestroy`|  |  |  | `hipStreamDestroy` | 1.6.0 |  |  | 
|`cuStreamDestroy_v2`|  |  |  | `hipStreamDestroy` | 1.6.0 |  |  | 
|`cuStreamEndCapture`| 10.0 |  |  |  |  |  |  | 
|`cuStreamGetAttribute`| 11.0 |  |  |  |  |  |  | 
|`cuStreamGetCaptureInfo`| 10.1 |  |  |  |  |  |  | 
|`cuStreamGetCtx`| 9.2 |  |  |  |  |  |  | 
|`cuStreamGetFlags`|  |  |  | `hipStreamGetFlags` | 1.6.0 |  |  | 
|`cuStreamGetPriority`|  |  |  | `hipStreamGetPriority` | 2.0.0 |  |  | 
|`cuStreamIsCapturing`| 10.0 |  |  |  |  |  |  | 
|`cuStreamQuery`|  |  |  | `hipStreamQuery` | 1.6.0 |  |  | 
|`cuStreamSetAttribute`| 11.0 |  |  |  |  |  |  | 
|`cuStreamSynchronize`|  |  |  | `hipStreamSynchronize` | 1.6.0 |  |  | 
|`cuStreamWaitEvent`|  |  |  | `hipStreamWaitEvent` | 1.6.0 |  |  | 
|`cuThreadExchangeStreamCaptureMode`| 10.1 |  |  |  |  |  |  | 

## **16. Event Management**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuEventCreate`|  |  |  | `hipEventCreateWithFlags` | 1.6.0 |  |  | 
|`cuEventDestroy`|  |  |  | `hipEventDestroy` | 1.6.0 |  |  | 
|`cuEventDestroy_v2`|  |  |  | `hipEventDestroy` | 1.6.0 |  |  | 
|`cuEventElapsedTime`|  |  |  | `hipEventElapsedTime` | 1.6.0 |  |  | 
|`cuEventQuery`|  |  |  | `hipEventQuery` | 1.6.0 |  |  | 
|`cuEventRecord`|  |  |  | `hipEventRecord` | 1.6.0 |  |  | 
|`cuEventRecordWithFlags`| 11.1 |  |  |  |  |  |  | 
|`cuEventSynchronize`|  |  |  | `hipEventSynchronize` | 1.6.0 |  |  | 

## **17. External Resource Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuDestroyExternalMemory`| 10.0 |  |  |  |  |  |  | 
|`cuDestroyExternalSemaphore`| 10.0 |  |  |  |  |  |  | 
|`cuExternalMemoryGetMappedBuffer`| 10.0 |  |  |  |  |  |  | 
|`cuExternalMemoryGetMappedMipmappedArray`| 10.0 |  |  |  |  |  |  | 
|`cuImportExternalMemory`| 10.0 |  |  |  |  |  |  | 
|`cuImportExternalSemaphore`| 10.0 |  |  |  |  |  |  | 
|`cuSignalExternalSemaphoresAsync`| 10.0 |  |  |  |  |  |  | 
|`cuWaitExternalSemaphoresAsync`| 10.0 |  |  |  |  |  |  | 

## **18. Stream Memory Operations**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuStreamBatchMemOp`| 8.0 |  |  |  |  |  |  | 
|`cuStreamWaitValue32`| 8.0 |  |  |  |  |  |  | 
|`cuStreamWaitValue64`| 9.0 |  |  |  |  |  |  | 
|`cuStreamWriteValue32`| 8.0 |  |  |  |  |  |  | 
|`cuStreamWriteValue64`| 9.0 |  |  |  |  |  |  | 

## **19. Execution Control**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuFuncGetAttribute`|  |  |  | `hipFuncGetAttribute` | 2.8.0 |  |  | 
|`cuFuncGetModule`| 11.0 |  |  |  |  |  |  | 
|`cuFuncSetAttribute`| 9.0 |  |  |  |  |  |  | 
|`cuFuncSetCacheConfig`|  |  |  |  |  |  |  | 
|`cuFuncSetSharedMemConfig`|  |  |  |  |  |  |  | 
|`cuLaunchCooperativeKernel`| 9.0 |  |  |  |  |  |  | 
|`cuLaunchCooperativeKernelMultiDevice`| 9.0 |  |  |  |  |  |  | 
|`cuLaunchHostFunc`| 10.0 |  |  |  |  |  |  | 
|`cuLaunchKernel`|  |  |  | `hipModuleLaunchKernel` | 1.6.0 |  |  | 

## **20. Execution Control [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuFuncSetBlockShape`|  | 9.2 |  |  |  |  |  | 
|`cuFuncSetSharedSize`|  | 9.2 |  |  |  |  |  | 
|`cuLaunch`|  | 9.2 |  |  |  |  |  | 
|`cuLaunchGrid`|  | 9.2 |  |  |  |  |  | 
|`cuLaunchGridAsync`|  | 9.2 |  |  |  |  |  | 
|`cuParamSetSize`|  | 9.2 |  |  |  |  |  | 
|`cuParamSetTexRef`|  | 9.2 |  |  |  |  |  | 
|`cuParamSetf`|  | 9.2 |  |  |  |  |  | 
|`cuParamSeti`|  | 9.2 |  |  |  |  |  | 
|`cuParamSetv`|  | 9.2 |  |  |  |  |  | 

## **21. Graph Management**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuGraphAddChildGraphNode`| 10.0 |  |  |  |  |  |  | 
|`cuGraphAddDependencies`| 10.0 |  |  |  |  |  |  | 
|`cuGraphAddEmptyNode`| 10.0 |  |  |  |  |  |  | 
|`cuGraphAddEventRecordNode`| 11.1 |  |  |  |  |  |  | 
|`cuGraphAddEventWaitNode`| 11.1 |  |  |  |  |  |  | 
|`cuGraphAddExternalSemaphoresSignalNode`| 11.2 |  |  |  |  |  |  | 
|`cuGraphAddExternalSemaphoresWaitNode`| 11.2 |  |  |  |  |  |  | 
|`cuGraphAddHostNode`| 10.0 |  |  |  |  |  |  | 
|`cuGraphAddKernelNode`| 10.0 |  |  |  |  |  |  | 
|`cuGraphAddMemcpyNode`| 10.0 |  |  |  |  |  |  | 
|`cuGraphAddMemsetNode`| 10.0 |  |  |  |  |  |  | 
|`cuGraphChildGraphNodeGetGraph`| 10.0 |  |  |  |  |  |  | 
|`cuGraphClone`| 10.0 |  |  |  |  |  |  | 
|`cuGraphCreate`| 10.0 |  |  |  |  |  |  | 
|`cuGraphDestroy`| 10.0 |  |  |  |  |  |  | 
|`cuGraphDestroyNode`| 10.0 |  |  |  |  |  |  | 
|`cuGraphEventRecordNodeGetEvent`| 11.1 |  |  |  |  |  |  | 
|`cuGraphEventRecordNodeSetEvent`| 11.1 |  |  |  |  |  |  | 
|`cuGraphEventWaitNodeGetEvent`| 11.1 |  |  |  |  |  |  | 
|`cuGraphEventWaitNodeSetEvent`| 11.1 |  |  |  |  |  |  | 
|`cuGraphExecChildGraphNodeSetParams`| 11.1 |  |  |  |  |  |  | 
|`cuGraphExecDestroy`| 10.0 |  |  |  |  |  |  | 
|`cuGraphExecEventRecordNodeSetEvent`| 11.1 |  |  |  |  |  |  | 
|`cuGraphExecEventWaitNodeSetEvent`| 11.1 |  |  |  |  |  |  | 
|`cuGraphExecExternalSemaphoresSignalNodeSetParams`| 11.2 |  |  |  |  |  |  | 
|`cuGraphExecExternalSemaphoresWaitNodeSetParams`| 11.2 |  |  |  |  |  |  | 
|`cuGraphExecHostNodeSetParams`| 10.2 |  |  |  |  |  |  | 
|`cuGraphExecKernelNodeSetParams`| 10.1 |  |  |  |  |  |  | 
|`cuGraphExecMemcpyNodeSetParams`| 10.2 |  |  |  |  |  |  | 
|`cuGraphExecMemsetNodeSetParams`| 10.2 |  |  |  |  |  |  | 
|`cuGraphExecUpdate`| 10.2 |  |  |  |  |  |  | 
|`cuGraphExternalSemaphoresSignalNodeGetParams`| 11.2 |  |  |  |  |  |  | 
|`cuGraphExternalSemaphoresSignalNodeSetParams`| 11.2 |  |  |  |  |  |  | 
|`cuGraphExternalSemaphoresWaitNodeGetParams`| 11.2 |  |  |  |  |  |  | 
|`cuGraphExternalSemaphoresWaitNodeSetParams`| 11.2 |  |  |  |  |  |  | 
|`cuGraphGetEdges`| 10.0 |  |  |  |  |  |  | 
|`cuGraphGetNodes`| 10.0 |  |  |  |  |  |  | 
|`cuGraphGetRootNodes`| 10.0 |  |  |  |  |  |  | 
|`cuGraphHostNodeGetParams`| 10.0 |  |  |  |  |  |  | 
|`cuGraphHostNodeSetParams`| 10.0 |  |  |  |  |  |  | 
|`cuGraphInstantiate`| 10.0 |  |  |  |  |  |  | 
|`cuGraphInstantiate_v2`| 11.0 |  |  |  |  |  |  | 
|`cuGraphKernelNodeCopyAttributes`| 11.0 |  |  |  |  |  |  | 
|`cuGraphKernelNodeGetAttribute`| 11.0 |  |  |  |  |  |  | 
|`cuGraphKernelNodeGetParams`| 10.0 |  |  |  |  |  |  | 
|`cuGraphKernelNodeSetAttribute`| 11.0 |  |  |  |  |  |  | 
|`cuGraphKernelNodeSetParams`| 10.0 |  |  |  |  |  |  | 
|`cuGraphLaunch`| 10.0 |  |  |  |  |  |  | 
|`cuGraphMemcpyNodeGetParams`| 10.0 |  |  |  |  |  |  | 
|`cuGraphMemcpyNodeSetParams`| 10.0 |  |  |  |  |  |  | 
|`cuGraphMemsetNodeGetParams`| 10.0 |  |  |  |  |  |  | 
|`cuGraphMemsetNodeSetParams`| 10.0 |  |  |  |  |  |  | 
|`cuGraphNodeFindInClone`| 10.0 |  |  |  |  |  |  | 
|`cuGraphNodeGetDependencies`| 10.0 |  |  |  |  |  |  | 
|`cuGraphNodeGetDependentNodes`| 10.0 |  |  |  |  |  |  | 
|`cuGraphNodeGetType`| 10.0 |  |  |  |  |  |  | 
|`cuGraphRemoveDependencies`| 10.0 |  |  |  |  |  |  | 
|`cuGraphUpload`| 11.1 |  |  |  |  |  |  | 

## **22. Occupancy**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuOccupancyAvailableDynamicSMemPerBlock`| 11.0 |  |  |  |  |  |  | 
|`cuOccupancyMaxActiveBlocksPerMultiprocessor`|  |  |  | `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` | 3.5.0 |  |  | 
|`cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`|  |  |  | `hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` | 3.5.0 |  |  | 
|`cuOccupancyMaxPotentialBlockSize`|  |  |  | `hipModuleOccupancyMaxPotentialBlockSize` | 3.5.0 |  |  | 
|`cuOccupancyMaxPotentialBlockSizeWithFlags`|  |  |  | `hipModuleOccupancyMaxPotentialBlockSizeWithFlags` | 3.5.0 |  |  | 

## **23. Texture Reference Management [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuTexRefCreate`|  | 11.0 |  |  |  |  |  | 
|`cuTexRefDestroy`|  | 11.0 |  |  |  |  |  | 
|`cuTexRefGetAddress`|  | 11.0 |  | `hipTexRefGetAddress` | 3.0.0 |  |  | 
|`cuTexRefGetAddressMode`|  | 11.0 |  | `hipTexRefGetAddressMode` | 3.0.0 |  |  | 
|`cuTexRefGetAddress_v2`|  | 11.0 |  | `hipTexRefGetAddress` | 3.0.0 |  |  | 
|`cuTexRefGetArray`|  | 11.0 |  | `hipTexRefGetArray` | 3.0.0 |  |  | 
|`cuTexRefGetBorderColor`| 8.0 | 11.0 |  |  |  |  |  | 
|`cuTexRefGetFilterMode`|  | 11.0 |  | `hipTexRefGetFilterMode` | 3.5.0 |  |  | 
|`cuTexRefGetFlags`|  | 11.0 |  | `hipTexRefGetFlags` | 3.5.0 |  |  | 
|`cuTexRefGetFormat`|  | 11.0 |  | `hipTexRefGetFormat` | 3.5.0 |  |  | 
|`cuTexRefGetMaxAnisotropy`|  | 11.0 |  | `hipTexRefGetMaxAnisotropy` | 3.5.0 |  |  | 
|`cuTexRefGetMipmapFilterMode`|  | 11.0 |  | `hipTexRefGetMipmapFilterMode` | 3.5.0 |  |  | 
|`cuTexRefGetMipmapLevelBias`|  | 11.0 |  | `hipTexRefGetMipmapLevelBias` | 3.5.0 |  |  | 
|`cuTexRefGetMipmapLevelClamp`|  | 11.0 |  | `hipTexRefGetMipmapLevelClamp` | 3.5.0 |  |  | 
|`cuTexRefGetMipmappedArray`|  | 11.0 |  | `hipTexRefGetMipMappedArray` | 3.5.0 |  |  | 
|`cuTexRefSetAddress`|  | 11.0 |  | `hipTexRefSetAddress` | 1.7.0 |  |  | 
|`cuTexRefSetAddress2D`|  | 11.0 |  | `hipTexRefSetAddress2D` | 1.7.0 |  |  | 
|`cuTexRefSetAddress2D_v2`|  |  |  | `hipTexRefSetAddress2D` | 1.7.0 |  |  | 
|`cuTexRefSetAddress2D_v3`|  |  |  | `hipTexRefSetAddress2D` | 1.7.0 |  |  | 
|`cuTexRefSetAddressMode`|  | 11.0 |  | `hipTexRefSetAddressMode` | 1.9.0 |  |  | 
|`cuTexRefSetAddress_v2`|  | 11.0 |  | `hipTexRefSetAddress` | 1.7.0 |  |  | 
|`cuTexRefSetArray`|  | 11.0 |  | `hipTexRefSetArray` | 1.9.0 |  |  | 
|`cuTexRefSetBorderColor`| 8.0 | 11.0 |  | `hipTexRefSetBorderColor` | 3.5.0 |  |  | 
|`cuTexRefSetFilterMode`|  | 11.0 |  | `hipTexRefSetFilterMode` | 1.9.0 |  |  | 
|`cuTexRefSetFlags`|  | 11.0 |  | `hipTexRefSetFlags` | 1.9.0 |  |  | 
|`cuTexRefSetFormat`|  | 11.0 |  | `hipTexRefSetFormat` | 1.9.0 |  |  | 
|`cuTexRefSetMaxAnisotropy`|  | 11.0 |  | `hipTexRefSetMaxAnisotropy` | 3.5.0 |  |  | 
|`cuTexRefSetMipmapFilterMode`|  | 11.0 |  | `hipTexRefSetMipmapFilterMode` | 3.5.0 |  |  | 
|`cuTexRefSetMipmapLevelBias`|  | 11.0 |  | `hipTexRefSetMipmapLevelBias` | 3.5.0 |  |  | 
|`cuTexRefSetMipmapLevelClamp`|  | 11.0 |  | `hipTexRefSetMipmapLevelClamp` | 3.5.0 |  |  | 
|`cuTexRefSetMipmappedArray`|  | 11.0 |  | `hipTexRefSetMipmappedArray` | 3.5.0 |  |  | 

## **24. Surface Reference Management [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuSurfRefGetArray`|  | 11.0 |  |  |  |  |  | 
|`cuSurfRefSetArray`|  | 11.0 |  |  |  |  |  | 

## **25. Texture Object Management**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuTexObjectCreate`|  |  |  | `hipTexObjectCreate` | 3.5.0 |  |  | 
|`cuTexObjectDestroy`|  |  |  | `hipTexObjectDestroy` | 3.5.0 |  |  | 
|`cuTexObjectGetResourceDesc`|  |  |  | `hipTexObjectGetResourceDesc` | 3.5.0 |  |  | 
|`cuTexObjectGetResourceViewDesc`|  |  |  | `hipTexObjectGetResourceViewDesc` | 3.5.0 |  |  | 
|`cuTexObjectGetTextureDesc`|  |  |  | `hipTexObjectGetTextureDesc` | 3.5.0 |  |  | 

## **26. Surface Object Management**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuSurfObjectCreate`|  |  |  |  |  |  |  | 
|`cuSurfObjectDestroy`|  |  |  |  |  |  |  | 
|`cuSurfObjectGetResourceDesc`|  |  |  |  |  |  |  | 

## **27. Peer Context Memory Access**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuCtxDisablePeerAccess`|  |  |  | `hipCtxDisablePeerAccess` | 1.6.0 | 1.9.0 |  | 
|`cuCtxEnablePeerAccess`|  |  |  | `hipCtxEnablePeerAccess` | 1.6.0 | 1.9.0 |  | 
|`cuDeviceCanAccessPeer`|  |  |  | `hipDeviceCanAccessPeer` | 1.9.0 |  |  | 
|`cuDeviceGetP2PAttribute`| 8.0 |  |  | `hipDeviceGetP2PAttribute` | 3.8.0 |  |  | 

## **28. Graphics Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuGraphicsMapResources`|  |  |  |  |  |  |  | 
|`cuGraphicsResourceGetMappedMipmappedArray`|  |  |  |  |  |  |  | 
|`cuGraphicsResourceGetMappedPointer`|  |  |  |  |  |  |  | 
|`cuGraphicsResourceGetMappedPointer_v2`|  |  |  |  |  |  |  | 
|`cuGraphicsResourceSetMapFlags`|  |  |  |  |  |  |  | 
|`cuGraphicsResourceSetMapFlags_v2`|  |  |  |  |  |  |  | 
|`cuGraphicsSubResourceGetMappedArray`|  |  |  |  |  |  |  | 
|`cuGraphicsUnmapResources`|  |  |  |  |  |  |  | 
|`cuGraphicsUnregisterResource`|  |  |  |  |  |  |  | 

## **29. Profiler Control [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuProfilerInitialize`|  | 11.0 |  |  |  |  |  | 

## **30. Profiler Control**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuProfilerStart`|  |  |  | `hipProfilerStart` | 1.6.0 | 3.0.0 |  | 
|`cuProfilerStop`|  |  |  | `hipProfilerStop` | 1.6.0 | 3.0.0 |  | 

## **31. OpenGL Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuGLCtxCreate`|  | 9.2 |  |  |  |  |  | 
|`cuGLGetDevices`|  |  |  |  |  |  |  | 
|`cuGLInit`|  | 9.2 |  |  |  |  |  | 
|`cuGLMapBufferObject`|  | 9.2 |  |  |  |  |  | 
|`cuGLMapBufferObjectAsync`|  | 9.2 |  |  |  |  |  | 
|`cuGLRegisterBufferObject`|  | 9.2 |  |  |  |  |  | 
|`cuGLSetBufferObjectMapFlags`|  | 9.2 |  |  |  |  |  | 
|`cuGLUnmapBufferObject`|  | 9.2 |  |  |  |  |  | 
|`cuGLUnmapBufferObjectAsync`|  | 9.2 |  |  |  |  |  | 
|`cuGLUnregisterBufferObject`|  | 9.2 |  |  |  |  |  | 
|`cuGraphicsGLRegisterBuffer`|  |  |  |  |  |  |  | 
|`cuGraphicsGLRegisterImage`|  |  |  |  |  |  |  | 
|`cuWGLGetDevice`|  |  |  |  |  |  |  | 

## **32. VDPAU Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuGraphicsVDPAURegisterOutputSurface`|  |  |  |  |  |  |  | 
|`cuGraphicsVDPAURegisterVideoSurface`|  |  |  |  |  |  |  | 
|`cuVDPAUCtxCreate`|  |  |  |  |  |  |  | 
|`cuVDPAUGetDevice`|  |  |  |  |  |  |  | 

## **33. EGL Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuEGLStreamConsumerAcquireFrame`| 9.1 |  |  |  |  |  |  | 
|`cuEGLStreamConsumerConnect`| 9.1 |  |  |  |  |  |  | 
|`cuEGLStreamConsumerConnectWithFlags`| 9.1 |  |  |  |  |  |  | 
|`cuEGLStreamConsumerDisconnect`| 9.1 |  |  |  |  |  |  | 
|`cuEGLStreamConsumerReleaseFrame`| 9.1 |  |  |  |  |  |  | 
|`cuEGLStreamProducerConnect`| 9.1 |  |  |  |  |  |  | 
|`cuEGLStreamProducerDisconnect`| 9.1 |  |  |  |  |  |  | 
|`cuEGLStreamProducerPresentFrame`| 9.1 |  |  |  |  |  |  | 
|`cuEGLStreamProducerReturnFrame`| 9.1 |  |  |  |  |  |  | 
|`cuEventCreateFromEGLSync`| 9.1 |  |  |  |  |  |  | 
|`cuGraphicsEGLRegisterImage`| 9.1 |  |  |  |  |  |  | 
|`cuGraphicsResourceGetMappedEglFrame`| 9.1 |  |  |  |  |  |  | 

## **34. Direct3D 9 Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuD3D9CtxCreate`|  |  |  |  |  |  |  | 
|`cuD3D9CtxCreateOnDevice`|  |  |  |  |  |  |  | 
|`cuD3D9GetDevice`|  |  |  |  |  |  |  | 
|`cuD3D9GetDevices`|  |  |  |  |  |  |  | 
|`cuD3D9GetDirect3DDevice`|  |  |  |  |  |  |  | 
|`cuD3D9MapResources`|  | 9.2 |  |  |  |  |  | 
|`cuD3D9RegisterResource`|  | 9.2 |  |  |  |  |  | 
|`cuD3D9ResourceGetMappedArray`|  | 9.2 |  |  |  |  |  | 
|`cuD3D9ResourceGetMappedPitch`|  | 9.2 |  |  |  |  |  | 
|`cuD3D9ResourceGetMappedPointer`|  | 9.2 |  |  |  |  |  | 
|`cuD3D9ResourceGetMappedSize`|  | 9.2 |  |  |  |  |  | 
|`cuD3D9ResourceGetSurfaceDimensions`|  | 9.2 |  |  |  |  |  | 
|`cuD3D9ResourceSetMapFlags`|  | 9.2 |  |  |  |  |  | 
|`cuD3D9UnmapResources`|  | 9.2 |  |  |  |  |  | 
|`cuD3D9UnregisterResource`|  | 9.2 |  |  |  |  |  | 
|`cuGraphicsD3D9RegisterResource`|  |  |  |  |  |  |  | 

## **35. Direct3D 10 Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuD3D10CtxCreate`|  | 9.2 |  |  |  |  |  | 
|`cuD3D10CtxCreateOnDevice`|  | 9.2 |  |  |  |  |  | 
|`cuD3D10GetDevice`|  |  |  |  |  |  |  | 
|`cuD3D10GetDevices`|  |  |  |  |  |  |  | 
|`cuD3D10GetDirect3DDevice`|  | 9.2 |  |  |  |  |  | 
|`cuD3D10MapResources`|  | 9.2 |  |  |  |  |  | 
|`cuD3D10RegisterResource`|  | 9.2 |  |  |  |  |  | 
|`cuD3D10ResourceGetMappedArray`|  | 9.2 |  |  |  |  |  | 
|`cuD3D10ResourceGetMappedPitch`|  | 9.2 |  |  |  |  |  | 
|`cuD3D10ResourceGetMappedPointer`|  | 9.2 |  |  |  |  |  | 
|`cuD3D10ResourceGetMappedSize`|  | 9.2 |  |  |  |  |  | 
|`cuD3D10ResourceGetSurfaceDimensions`|  | 9.2 |  |  |  |  |  | 
|`cuD3D10ResourceSetMapFlags`|  | 9.2 |  |  |  |  |  | 
|`cuD3D10UnmapResources`|  | 9.2 |  |  |  |  |  | 
|`cuD3D10UnregisterResource`|  | 9.2 |  |  |  |  |  | 
|`cuGraphicsD3D10RegisterResource`|  |  |  |  |  |  |  | 

## **36. Direct3D 11 Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** | **A** | **D** | **R** |
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cuD3D11CtxCreate`|  | 9.2 |  |  |  |  |  | 
|`cuD3D11CtxCreateOnDevice`|  | 9.2 |  |  |  |  |  | 
|`cuD3D11GetDevice`|  |  |  |  |  |  |  | 
|`cuD3D11GetDevices`|  |  |  |  |  |  |  | 
|`cuD3D11GetDirect3DDevice`|  | 9.2 |  |  |  |  |  | 
|`cuGraphicsD3D11RegisterResource`|  |  |  |  |  |  |  | 


\*A - Added; D - Deprecated; R - Removed