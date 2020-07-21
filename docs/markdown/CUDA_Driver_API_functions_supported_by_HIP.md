# CUDA Driver API functions supported by HIP

## **1. Data types used by CUDA driver**

| **type**     |   **CUDA**                                                         |   **HIP**                                                  |**CUDA version\***|
|-------------:|:-------------------------------------------------------------------|:-----------------------------------------------------------|:----------------:|
| struct       |`CUDA_ARRAY3D_DESCRIPTOR`                                           |`HIP_ARRAY3D_DESCRIPTOR`                                    |
| struct       |`CUDA_ARRAY3D_DESCRIPTOR_v1`                                        |`HIP_ARRAY3D_DESCRIPTOR`                                    | 11.0             |
| typedef      |`CUDA_ARRAY3D_DESCRIPTOR_st`                                        |`HIP_ARRAY3D_DESCRIPTOR`                                    |
| typedef      |`CUDA_ARRAY3D_DESCRIPTOR_v1_st`                                     |`HIP_ARRAY3D_DESCRIPTOR`                                    | 11.0             |
| struct       |`CUDA_ARRAY_DESCRIPTOR`                                             |`HIP_ARRAY_DESCRIPTOR`                                      |
| struct       |`CUDA_ARRAY_DESCRIPTOR_v1`                                          |`HIP_ARRAY_DESCRIPTOR`                                      | 11.0             |
| typedef      |`CUDA_ARRAY_DESCRIPTOR_st`                                          |`HIP_ARRAY_DESCRIPTOR`                                      |
| typedef      |`CUDA_ARRAY_DESCRIPTOR_v1_st`                                       |`HIP_ARRAY_DESCRIPTOR`                                      | 11.0             |
| struct       |`CUDA_MEMCPY2D`                                                     |`hip_Memcpy2D`                                              |
| struct       |`CUDA_MEMCPY2D_v1`                                                  |`hip_Memcpy2D`                                              | 11.0             |
| typedef      |`CUDA_MEMCPY2D_st`                                                  |`hip_Memcpy2D`                                              |
| typedef      |`CUDA_MEMCPY2D_v1_st`                                               |`hip_Memcpy2D`                                              | 11.0             |
| struct       |`CUDA_MEMCPY3D`                                                     |`HIP_MEMCPY3D`                                              |
| struct       |`CUDA_MEMCPY3D_v1`                                                  |`HIP_MEMCPY3D`                                              | 11.0             |
| typedef      |`CUDA_MEMCPY3D_st`                                                  |`HIP_MEMCPY3D`                                              |
| typedef      |`CUDA_MEMCPY3D_v1_st`                                               |`HIP_MEMCPY3D`                                              | 11.0             |
| struct       |`CUDA_MEMCPY3D_PEER`                                                |                                                            |
| typedef      |`CUDA_MEMCPY3D_PEER_st`                                             |                                                            |
| struct       |`CUDA_POINTER_ATTRIBUTE_P2P_TOKENS`                                 |                                                            |
| typedef      |`CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st`                              |                                                            |
| struct       |`CUDA_RESOURCE_DESC`                                                |                                                            |
| typedef      |`CUDA_RESOURCE_DESC_st`                                             |                                                            |
| struct       |`CUDA_RESOURCE_VIEW_DESC`                                           |                                                            |
| typedef      |`CUDA_RESOURCE_VIEW_DESC_st`                                        |                                                            |
| struct       |`CUDA_TEXTURE_DESC`                                                 |                                                            |
| typedef      |`CUDA_TEXTURE_DESC_st`                                              |                                                            |
| struct       |`CUdevprop`                                                         |                                                            |
| typedef      |`CUdevprop_st`                                                      |                                                            |
| struct       |`CUipcEventHandle`                                                  |`ihipIpcEventHandle_t`                                      |
| typedef      |`CUipcEventHandle_st`                                               |`ihipIpcEventHandle_t`                                      |
| struct       |`CUipcMemHandle`                                                    |`hipIpcMemHandle_t`                                         |
| typedef      |`CUipcMemHandle_st`                                                 |`hipIpcMemHandle_st`                                        |
| union        |`CUstreamBatchMemOpParams`                                          |                                                            | 8.0              |
| typedef      |`CUstreamBatchMemOpParams_union`                                    |                                                            | 8.0              |
| enum         |***`CUaddress_mode`***                                              |***`hipTextureAddressMode`***                               |
| typedef      |***`CUaddress_mode_enum`***                                         |***`hipTextureAddressMode`***                               |
|            0 |*`CU_TR_ADDRESS_MODE_WRAP`*                                         |*`hipAddressModeWrap`*                                      |
|            1 |*`CU_TR_ADDRESS_MODE_CLAMP`*                                        |*`hipAddressModeClamp`*                                     |
|            2 |*`CU_TR_ADDRESS_MODE_MIRROR`*                                       |*`hipAddressModeMirror`*                                    |
|            3 |*`CU_TR_ADDRESS_MODE_BORDER`*                                       |*`hipAddressModeBorder`*                                    |
| enum         |***`CUarray_cubemap_face`***                                        |                                                            |
| typedef      |***`CUarray_cubemap_face_enum`***                                   |                                                            |
|         0x00 |*`CU_CUBEMAP_FACE_POSITIVE_X`*                                      |                                                            |
|         0x01 |*`CU_CUBEMAP_FACE_NEGATIVE_X`*                                      |                                                            |
|         0x02 |*`CU_CUBEMAP_FACE_POSITIVE_Y`*                                      |                                                            |
|         0x03 |*`CU_CUBEMAP_FACE_NEGATIVE_Y`*                                      |                                                            |
|         0x04 |*`CU_CUBEMAP_FACE_POSITIVE_Z`*                                      |                                                            |
|         0x05 |*`CU_CUBEMAP_FACE_NEGATIVE_Z`*                                      |                                                            |
| enum         |***`CUarray_format`***                                              |***`hipArray_format`***                                     |
| typedef      |***`CUarray_format_enum`***                                         |***`hipArray_format`***                                     |
|         0x01 |*`CU_AD_FORMAT_UNSIGNED_INT8`*                                      |*`HIP_AD_FORMAT_UNSIGNED_INT8`*                             |
|         0x02 |*`CU_AD_FORMAT_UNSIGNED_INT16`*                                     |*`HIP_AD_FORMAT_UNSIGNED_INT16`*                            |
|         0x03 |*`CU_AD_FORMAT_UNSIGNED_INT32`*                                     |*`HIP_AD_FORMAT_UNSIGNED_INT32`*                            |
|         0x08 |*`CU_AD_FORMAT_SIGNED_INT8`*                                        |*`HIP_AD_FORMAT_SIGNED_INT8`*                               |
|         0x09 |*`CU_AD_FORMAT_SIGNED_INT16`*                                       |*`HIP_AD_FORMAT_SIGNED_INT16`*                              |
|         0x0a |*`CU_AD_FORMAT_SIGNED_INT32`*                                       |*`HIP_AD_FORMAT_SIGNED_INT32`*                              |
|         0x10 |*`CU_AD_FORMAT_HALF`*                                               |*`HIP_AD_FORMAT_HALF`*                                      |
|         0x20 |*`CU_AD_FORMAT_FLOAT`*                                              |*`HIP_AD_FORMAT_FLOAT`*                                     |
| enum         |***`CUctx_flags`***                                                 |                                                            |
| typedef      |***`CUctx_flags_enum`***                                            |                                                            |
|         0x00 |*`CU_CTX_SCHED_AUTO`*                                               |`hipDeviceScheduleAuto`                                     |
|         0x01 |*`CU_CTX_SCHED_SPIN`*                                               |`hipDeviceScheduleSpin`                                     |
|         0x02 |*`CU_CTX_SCHED_YIELD`*                                              |`hipDeviceScheduleYield`                                    |
|         0x04 |*`CU_CTX_SCHED_BLOCKING_SYNC`*                                      |`hipDeviceScheduleBlockingSync`                             |
|         0x04 |*`CU_CTX_BLOCKING_SYNC`*                                            |`hipDeviceScheduleBlockingSync`                             |
|         0x07 |*`CU_CTX_SCHED_MASK`*                                               |`hipDeviceScheduleMask`                                     |
|         0x08 |*`CU_CTX_MAP_HOST`*                                                 |`hipDeviceMapHost`                                          |
|         0x10 |*`CU_CTX_LMEM_RESIZE_TO_MAX`*                                       |`hipDeviceLmemResizeToMax`                                  |
|         0x1f |*`CU_CTX_FLAGS_MASK`*                                               |                                                            |
| enum         |***`CUdevice_attribute`***                                          |***`hipDeviceAttribute_t`***                                |
| typedef      |***`CUdevice_attribute_enum`***                                     |***`hipDeviceAttribute_t`***                                |
|            1 |*`CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`*                       |*`hipDeviceAttributeMaxThreadsPerBlock`*                    |
|            2 |*`CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X`*                             |*`hipDeviceAttributeMaxBlockDimX`*                          |
|            3 |*`CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y`*                             |*`hipDeviceAttributeMaxBlockDimY`*                          |
|            4 |*`CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z`*                             |*`hipDeviceAttributeMaxBlockDimZ`*                          |
|            5 |*`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X`*                              |*`hipDeviceAttributeMaxGridDimX`*                           |
|            6 |*`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y`*                              |*`hipDeviceAttributeMaxGridDimY`*                           |
|            7 |*`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z`*                              |*`hipDeviceAttributeMaxGridDimZ`*                           |
|            8 |*`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`*                 |*`hipDeviceAttributeMaxSharedMemoryPerBlock`*               |
|            8 |*`CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK`*                     |*`hipDeviceAttributeMaxSharedMemoryPerBlock`*               |
|            9 |*`CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY`*                       |*`hipDeviceAttributeTotalConstantMemory`*                   |
|           10 |*`CU_DEVICE_ATTRIBUTE_WARP_SIZE`*                                   |*`hipDeviceAttributeWarpSize`*                              |
|           11 |*`CU_DEVICE_ATTRIBUTE_MAX_PITCH`*                                   |*`hipDeviceAttributeMaxPitch`*                              |
|           12 |*`CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK`*                     |*`hipDeviceAttributeMaxRegistersPerBlock`*                  |
|           12 |*`CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK`*                         |*`hipDeviceAttributeMaxRegistersPerBlock`*                  |
|           13 |*`CU_DEVICE_ATTRIBUTE_CLOCK_RATE`*                                  |*`hipDeviceAttributeClockRate`*                             |
|           14 |*`CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT`*                           |*`hipDeviceAttributeTextureAlignment`*                      |
|           15 |*`CU_DEVICE_ATTRIBUTE_GPU_OVERLAP`*                                 |                                                            |
|           16 |*`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`*                        |*`hipDeviceAttributeMultiprocessorCount`*                   |
|           17 |*`CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT`*                         |*`hipDeviceAttributeKernelExecTimeout`*                     |
|           18 |*`CU_DEVICE_ATTRIBUTE_INTEGRATED`*                                  |*`hipDeviceAttributeIntegrated`*                            |
|           19 |*`CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY`*                         |*`hipDeviceAttributeCanMapHostMemory`*                      |
|           20 |*`CU_DEVICE_ATTRIBUTE_COMPUTE_MODE`*                                |*`hipDeviceAttributeComputeMode`*                           |
|           21 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH`*                     |                                                            |
|           22 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH`*                     |                                                            |
|           23 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT`*                    |                                                            |
|           24 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH`*                     |                                                            |
|           25 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT`*                    |                                                            |
|           26 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH`*                     |                                                            |
|           27 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH`*             |                                                            |
|           28 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT`*            |                                                            |
|           29 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS`*            |                                                            |
|           27 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH`*               |                                                            |
|           28 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT`*              |                                                            |
|           29 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES`*           |                                                            |
|           30 |*`CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT`*                           |                                                            |
|           31 |*`CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS`*                          |*`hipDeviceAttributeConcurrentKernels`*                     |
|           32 |*`CU_DEVICE_ATTRIBUTE_ECC_ENABLED`*                                 |*`hipDeviceAttributeEccEnabled`*                            |
|           33 |*`CU_DEVICE_ATTRIBUTE_PCI_BUS_ID`*                                  |*`hipDeviceAttributePciBusId`*                              |
|           34 |*`CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID`*                               |*`hipDeviceAttributePciDeviceId`*                           |
|           35 |*`CU_DEVICE_ATTRIBUTE_TCC_DRIVER`*                                  |                                                            |
|           36 |*`CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE`*                           |*`hipDeviceAttributeMemoryClockRate`*                       |
|           37 |*`CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH`*                     |*`hipDeviceAttributeMemoryBusWidth`*                        |
|           38 |*`CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`*                               |*`hipDeviceAttributeL2CacheSize`*                           |
|           39 |*`CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR`*              |*`hipDeviceAttributeMaxThreadsPerMultiProcessor`*           |
|           40 |*`CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT`*                          |                                                            |
|           41 |*`CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING`*                          |                                                            |
|           42 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH`*             |                                                            |
|           43 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS`*            |                                                            |
|           44 |*`CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER`*                            |                                                            |
|           45 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH`*              |                                                            |
|           46 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT`*             |                                                            |
|           47 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE`*           |                                                            |
|           48 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE`*          |                                                            |
|           49 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE`*           |                                                            |
|           50 |*`CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID`*                               |                                                            |
|           51 |*`CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT`*                     |                                                            |
|           52 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH`*                |                                                            |
|           53 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH`*        |                                                            |
|           54 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS`*       |                                                            |
|           55 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH`*                     |                                                            |
|           56 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH`*                     |                                                            |
|           57 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT`*                    |                                                            |
|           58 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH`*                     |                                                            |
|           59 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT`*                    |                                                            |
|           60 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH`*                     |                                                            |
|           61 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH`*             |                                                            |
|           62 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS`*            |                                                            |
|           63 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH`*             |                                                            |
|           64 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT`*            |                                                            |
|           65 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS`*            |                                                            |
|           66 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH`*                |                                                            |
|           67 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH`*        |                                                            |
|           68 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS`*       |                                                            |
|           69 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH`*              |                                                            |
|           70 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH`*              |                                                            |
|           71 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT`*             |                                                            |
|           72 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH`*              |                                                            |
|           73 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH`*           |                                                            |
|           74 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT`*          |                                                            |
|           75 |*`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`*                    |*`hipDeviceAttributeComputeCapabilityMajor`*                |
|           76 |*`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`*                    |*`hipDeviceAttributeComputeCapabilityMinor`*                |
|           77 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH`*           |                                                            |
|           78 |*`CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED`*                 |                                                            |
|           79 |*`CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED`*                   |                                                            |
|           80 |*`CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED`*                    |                                                            |
|           81 |*`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`*        |*`hipDeviceAttributeMaxSharedMemoryPerMultiprocessor`*      |
|           82 |*`CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR`*            |                                                            |
|           83 |*`CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY`*                              |                                                            |
|           84 |*`CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD`*                             |*`hipDeviceAttributeIsMultiGpuBoard`*                       |
|           85 |*`CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID`*                    |                                                            |
|           86 |*`CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED`*                |                                                            | 8.0              |
|           87 |*`CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO`*       |                                                            | 8.0              |
|           88 |*`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`*                      |                                                            | 8.0              |
|           89 |*`CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`*                   |                                                            | 8.0              |
|           90 |*`CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED`*                |                                                            | 8.0              |
|           91 |*`CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM`*     |                                                            | 8.0              |
|           92 |*`CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS`*                      |                                                            | 9.0              |
|           93 |*`CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS`*               |                                                            | 9.0              |
|           94 |*`CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR`*               |                                                            | 9.0              |
|           95 |*`CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`*                          |*`hipDeviceAttributeCooperativeLaunch`*                     | 9.0              |
|           96 |*`CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH`*             |*`hipDeviceAttributeCooperativeMultiDeviceLaunch`*          | 9.0              |
|           97 |*`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`*           |                                                            | 9.0              |
|           98 |*`CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES`*                     |                                                            | 9.2              |
|           99 |*`CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED`*                     |                                                            | 9.2              |
|          100 |*`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`*|                                                            | 9.2              |
|          101 |*`CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST`*         |                                                            | 9.2              |
|          102 |*`CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED`*        |                                                            | 10.2             |
|          103 |*`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED`* |                                                            | 10.2             |
|          104 |*`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED`*          |                                                            | 10.2             |
|          105 |*`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED`*      |                                                            | 10.2             |
|          106 |*`CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR`*               |                                                            | 11.0             |
|          107 |*`CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED`*               |                                                            | 11.0             |
|          108 |*`CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE`*                |                                                            | 11.0             |
|          109 |*`CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE`*               |                                                            | 11.0             |
|          110 |*`CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED`*     |                                                            | 11.0             |
|          111 |*`CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK`*            |                                                            | 11.0             |
|          112 |*`CU_DEVICE_ATTRIBUTE_MAX`*                                         |                                                            |
| enum         |***`CUevent_flags`***                                               |                                                            |
| typedef      |***`CUevent_flags_enum`***                                          |                                                            |
|         0x00 |*`CU_EVENT_DEFAULT`*                                                |*`hipEventDefault`*                                         |
|         0x01 |*`CU_EVENT_BLOCKING_SYNC`*                                          |*`hipEventBlockingSync`*                                    |
|         0x02 |*`CU_EVENT_DISABLE_TIMING`*                                         |*`hipEventDisableTiming`*                                   |
|         0x04 |*`CU_EVENT_INTERPROCESS`*                                           |*`hipEventInterprocess`*                                    |
| enum         |***`CUfilter_mode`***                                               |***`hipTextureFilterMode`***                                |
| typedef      |***`CUfilter_mode_enum`***                                          |***`hipTextureFilterMode`***                                |
|            0 |*`CU_TR_FILTER_MODE_POINT`*                                         |*`hipFilterModePoint`*                                      |
|            1 |*`CU_TR_FILTER_MODE_LINEAR`*                                        |*`hipFilterModeLinear`*                                     |
| enum         |***`CUfunc_cache`***                                                |***`hipFuncCache_t`***                                      |
| typedef      |***`CUfunc_cache_enum`***                                           |***`hipFuncCache_t`***                                      |
|         0x00 |*`CU_FUNC_CACHE_PREFER_NONE`*                                       |*`hipFuncCachePreferNone`*                                  |
|         0x01 |*`CU_FUNC_CACHE_PREFER_SHARED`*                                     |*`hipFuncCachePreferShared`*                                |
|         0x02 |*`CU_FUNC_CACHE_PREFER_L1`*                                         |*`hipFuncCachePreferL1`*                                    |
|         0x03 |*`CU_FUNC_CACHE_PREFER_EQUAL`*                                      |*`hipFuncCachePreferEqual`*                                 |
| enum         |***`CUfunction_attribute`***                                        |***`hipFunction_attribute`***                               |
| typedef      |***`CUfunction_attribute_enum`***                                   |***`hipFunction_attribute`***                               |
|            0 |*`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`*                         |*`HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`*                |
|            1 |*`CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`*                             |*`HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`*                    |
|            2 |*`CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES`*                              |*`HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES`*                     |
|            3 |*`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`*                              |*`HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`*                     |
|            4 |*`CU_FUNC_ATTRIBUTE_NUM_REGS`*                                      |*`HIP_FUNC_ATTRIBUTE_NUM_REGS`*                             |
|            5 |*`CU_FUNC_ATTRIBUTE_PTX_VERSION`*                                   |*`HIP_FUNC_ATTRIBUTE_PTX_VERSION`*                          |
|            6 |*`CU_FUNC_ATTRIBUTE_BINARY_VERSION`*                                |*`HIP_FUNC_ATTRIBUTE_BINARY_VERSION`*                       |
|            7 |*`CU_FUNC_ATTRIBUTE_CACHE_MODE_CA`*                                 |*`HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA`*                        |
|            8 |*`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`*                 |*`HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`*        | 9.0              |
|            9 |*`CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`*              |*`HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`*     | 9.0              |
|           10 |*`CU_FUNC_ATTRIBUTE_MAX`*                                           |*`HIP_FUNC_ATTRIBUTE_MAX`*                                  |
| enum         |***`CUgraphicsMapResourceFlags`***                                  |                                                            |
| typedef      |***`CUgraphicsMapResourceFlags_enum`***                             |                                                            |
|         0x00 |*`CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE`*                             |                                                            |
|         0x01 |*`CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY`*                        |                                                            |
|         0x02 |*`CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD`*                    |                                                            |
| enum         |***`CUgraphicsRegisterFlags`***                                     |                                                            |
| typedef      |***`CUgraphicsRegisterFlags_enum`***                                |                                                            |
|         0x00 |*`CU_GRAPHICS_REGISTER_FLAGS_NONE`*                                 |                                                            |
|         0x01 |*`CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY`*                            |                                                            |
|         0x02 |*`CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD`*                        |                                                            |
|         0x04 |*`CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST`*                         |                                                            |
|         0x08 |*`CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER`*                       |                                                            |
| enum         |***`CUipcMem_flags`***                                              |                                                            |
| typedef      |***`CUipcMem_flags_enum`***                                         |                                                            |
|          0x1 |*`CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS`*                              |*`hipIpcMemLazyEnablePeerAccess`*                           |
| enum         |***`CUjit_cacheMode`***                                             |                                                            |
| typedef      |***`CUjit_cacheMode_enum`***                                        |                                                            |
|            0 |*`CU_JIT_CACHE_OPTION_NONE`*                                        |                                                            |
|              |*`CU_JIT_CACHE_OPTION_CG`*                                          |                                                            |
|              |*`CU_JIT_CACHE_OPTION_CA`*                                          |                                                            |
| enum         |***`CUjit_fallback`***                                              |                                                            |
| typedef      |***`CUjit_fallback_enum`***                                         |                                                            |
|            0 |*`CU_PREFER_PTX`*                                                   |                                                            |
|              |*`CU_PREFER_BINARY`*                                                |                                                            |
| enum         |***`CUjit_option`***                                                |                                                            |
| typedef      |***`CUjit_option_enum`***                                           |                                                            |
|            0 |*`CU_JIT_MAX_REGISTERS`*                                            |                                                            |
|              |*`CU_JIT_THREADS_PER_BLOCK`*                                        |                                                            |
|              |*`CU_JIT_WALL_TIME`*                                                |                                                            |
|              |*`CU_JIT_INFO_LOG_BUFFER`*                                          |                                                            |
|              |*`CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES`*                               |                                                            |
|              |*`CU_JIT_ERROR_LOG_BUFFER`*                                         |                                                            |
|              |*`CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`*                              |                                                            |
|              |*`CU_JIT_OPTIMIZATION_LEVEL`*                                       |                                                            |
|              |*`CU_JIT_TARGET_FROM_CUCONTEXT`*                                    |                                                            |
|              |*`CU_JIT_TARGET`*                                                   |                                                            |
|              |*`CU_JIT_FALLBACK_STRATEGY`*                                        |                                                            |
|              |*`CU_JIT_GENERATE_DEBUG_INFO`*                                      |                                                            |
|              |*`CU_JIT_LOG_VERBOSE`*                                              |                                                            |
|              |*`CU_JIT_GENERATE_LINE_INFO`*                                       |                                                            |
|              |*`CU_JIT_CACHE_MODE`*                                               |                                                            |
|              |*`CU_JIT_NEW_SM3X_OPT`*                                             |                                                            | 8.0              |
|              |*`CU_JIT_FAST_COMPILE`*                                             |                                                            | 8.0              |
|              |*`CU_JIT_GLOBAL_SYMBOL_NAMES`*                                      |                                                            | 10.0             |
|              |*`CU_JIT_GLOBAL_SYMBOL_ADDRESSES`*                                  |                                                            | 10.0             |
|              |*`CU_JIT_GLOBAL_SYMBOL_COUNT`*                                      |                                                            | 10.0             |
|              |*`CU_JIT_NUM_OPTIONS`*                                              |                                                            |
| enum         |***`CUjit_target`***                                                |                                                            |
| typedef      |***`CUjit_target_enum`***                                           |                                                            |
|           10 |*`CU_TARGET_COMPUTE_10`*                                            |                                                            |
|           11 |*`CU_TARGET_COMPUTE_11`*                                            |                                                            |
|           12 |*`CU_TARGET_COMPUTE_12`*                                            |                                                            |
|           13 |*`CU_TARGET_COMPUTE_13`*                                            |                                                            |
|           20 |*`CU_TARGET_COMPUTE_20`*                                            |                                                            |
|           21 |*`CU_TARGET_COMPUTE_21`*                                            |                                                            |
|           30 |*`CU_TARGET_COMPUTE_30`*                                            |                                                            |
|           32 |*`CU_TARGET_COMPUTE_32`*                                            |                                                            |
|           35 |*`CU_TARGET_COMPUTE_35`*                                            |                                                            |
|           37 |*`CU_TARGET_COMPUTE_37`*                                            |                                                            |
|           50 |*`CU_TARGET_COMPUTE_50`*                                            |                                                            |
|           52 |*`CU_TARGET_COMPUTE_52`*                                            |                                                            |
|           53 |*`CU_TARGET_COMPUTE_53`*                                            |                                                            | 8.0              |
|           60 |*`CU_TARGET_COMPUTE_60`*                                            |                                                            | 8.0              |
|           61 |*`CU_TARGET_COMPUTE_61`*                                            |                                                            | 8.0              |
|           62 |*`CU_TARGET_COMPUTE_62`*                                            |                                                            | 8.0              |
|           70 |*`CU_TARGET_COMPUTE_70`*                                            |                                                            | 9.0              |
|           72 |*`CU_TARGET_COMPUTE_72`*                                            |                                                            | 10.1             |
|           73 |*`CU_TARGET_COMPUTE_73`*                                            |                                                            | 9.1 - 9.2        |
|           75 |*`CU_TARGET_COMPUTE_75`*                                            |                                                            | 9.1              |
|           80 |*`CU_TARGET_COMPUTE_80`*                                            |                                                            | 11.0             |
| enum         |***`CUjitInputType`***                                              |                                                            |
| typedef      |***`CUjitInputType_enum`***                                         |                                                            |
|            0 |*`CU_JIT_INPUT_CUBIN`*                                              |                                                            |
|              |*`CU_JIT_INPUT_PTX`*                                                |                                                            |
|              |*`CU_JIT_INPUT_FATBINARY`*                                          |                                                            |
|              |*`CU_JIT_INPUT_OBJECT`*                                             |                                                            |
|              |*`CU_JIT_INPUT_LIBRARY`*                                            |                                                            |
|              |*`CU_JIT_NUM_INPUT_TYPES`*                                          |                                                            |
| enum         |***`CUlimit`***                                                     |***`hipLimit_t`***                                          |
| typedef      |***`CUlimit_enum`***                                                |***`hipLimit_t`***                                          |
|         0x00 |*`CU_LIMIT_STACK_SIZE`*                                             |                                                            |
|         0x01 |*`CU_LIMIT_PRINTF_FIFO_SIZE`*                                       |                                                            |
|         0x02 |*`CU_LIMIT_MALLOC_HEAP_SIZE`*                                       |*`hipLimitMallocHeapSize`*                                  |
|         0x03 |*`CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH`*                                 |                                                            |
|         0x04 |*`CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT`*                       |                                                            |
|         0x05 |*`CU_LIMIT_MAX_L2_FETCH_GRANULARITY`*                               |                                                            | 10.0             |
|         0x06 |*`CU_LIMIT_PERSISTING_L2_CACHE_SIZE`*                               |                                                            | 11.0             |
|              |*`CU_LIMIT_MAX`*                                                    |                                                            |
| enum         |***`CUmem_advise`***                                                |                                                            | 8.0              |
| typedef      |***`CUmem_advise_enum`***                                           |                                                            | 8.0              |
|            1 |*`CU_MEM_ADVISE_SET_READ_MOSTLY`*                                   |                                                            | 8.0              |
|            2 |*`CU_MEM_ADVISE_UNSET_READ_MOSTLY`*                                 |                                                            | 8.0              |
|            3 |*`CU_MEM_ADVISE_SET_PREFERRED_LOCATION`*                            |                                                            | 8.0              |
|            4 |*`CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION`*                          |                                                            | 8.0              |
|            5 |*`CU_MEM_ADVISE_SET_ACCESSED_BY`*                                   |                                                            | 8.0              |
|            6 |*`CU_MEM_ADVISE_UNSET_ACCESSED_BY`*                                 |                                                            | 8.0              |
| enum         |***`CUmemAttach_flags`***                                           |                                                            |
| typedef      |***`CUmemAttach_flags_enum`***                                      |                                                            |
|          0x1 |*`CU_MEM_ATTACH_GLOBAL`*                                            |*`hipMemAttachGlobal`*                                      |
|          0x2 |*`CU_MEM_ATTACH_HOST`*                                              |*`hipMemAttachHost`*                                        |
|          0x4 |*`CU_MEM_ATTACH_SINGLE`*                                            |                                                            |
| enum         |***`CUmemorytype`***                                                |*`hipMemoryType`*                                           |
| typedef      |***`CUmemorytype_enum`***                                           |*`hipMemoryType`*                                           |
|         0x01 |*`CU_MEMORYTYPE_HOST`*                                              |*`hipMemoryTypeHost`*                                       |
|         0x02 |*`CU_MEMORYTYPE_DEVICE`*                                            |*`hipMemoryTypeDevice`*                                     |
|         0x03 |*`CU_MEMORYTYPE_ARRAY`*                                             |*`hipMemoryTypeArray`*                                      |
|         0x04 |*`CU_MEMORYTYPE_UNIFIED`*                                           |*`hipMemoryTypeUnified`*                                    |
| enum         |***`CUmem_range_attribute`***                                       |                                                            | 8.0              |
| typedef      |***`CUmem_range_attribute_enum`***                                  |                                                            | 8.0              |
|            1 |*`CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY`*                              |                                                            | 8.0              |
|            2 |*`CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION`*                       |                                                            | 8.0              |
|            3 |*`CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY`*                              |                                                            | 8.0              |
|            4 |*`CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION`*                   |                                                            | 8.0              |
| enum         |***`CUcomputemode`***                                               |***`hipComputeMode`***                                      |
| typedef      |***`CUcomputemode_enum`***                                          |***`hipComputeMode`***                                      |
|            0 |*`CU_COMPUTEMODE_DEFAULT`*                                          |*`hipComputeModeDefault`*                                   |
|            1 |*`CU_COMPUTEMODE_EXCLUSIVE`*                                        |*`hipComputeModeExclusive`*                                 |
|            2 |*`CU_COMPUTEMODE_PROHIBITED`*                                       |*`hipComputeModeProhibited`*                                |
|            3 |*`CU_COMPUTEMODE_EXCLUSIVE_PROCESS`*                                |*`hipComputeModeExclusiveProcess`*                          |
| enum         |***`CUoccupancy_flags`***                                           |                                                            |
| typedef      |***`CUoccupancy_flags_enum`***                                      |                                                            |
|         0x00 |*`CU_OCCUPANCY_DEFAULT`*                                            |                                                            |
|         0x01 |*`CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE`*                           |                                                            |
| enum         |***`CUpointer_attribute`***                                         |                                                            |
| typedef      |***`CUpointer_attribute_enum`***                                    |                                                            |
|            1 |*`CU_POINTER_ATTRIBUTE_CONTEXT`*                                    |                                                            |
|            2 |*`CU_POINTER_ATTRIBUTE_MEMORY_TYPE`*                                |                                                            |
|            3 |*`CU_POINTER_ATTRIBUTE_DEVICE_POINTER`*                             |                                                            |
|            4 |*`CU_POINTER_ATTRIBUTE_HOST_POINTER`*                               |                                                            |
|            5 |*`CU_POINTER_ATTRIBUTE_P2P_TOKENS`*                                 |                                                            |
|            6 |*`CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`*                                |                                                            |
|            7 |*`CU_POINTER_ATTRIBUTE_BUFFER_ID`*                                  |                                                            |
|            8 |*`CU_POINTER_ATTRIBUTE_IS_MANAGED`*                                 |                                                            |
|            9 |*`CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL`*                             |                                                            | 9.2              |
|           10 |*`CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE`*                 |                                                            | 10.2             |
|           11 |*`CU_POINTER_ATTRIBUTE_RANGE_START_ADDR`*                           |                                                            | 10.2             |
|           12 |*`CU_POINTER_ATTRIBUTE_RANGE_SIZE`*                                 |                                                            | 10.2             |
|           13 |*`CU_POINTER_ATTRIBUTE_MAPPED`*                                     |                                                            | 10.2             |
|           14 |*`CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES`*                       |                                                            | 10.2             |
|           15 |*`CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE`*                 |                                                            | 11.0             |
| enum         |***`CUresourcetype`***                                              |                                                            |
| typedef      |***`CUresourcetype_enum`***                                         |                                                            |
|         0x00 |*`CU_RESOURCE_TYPE_ARRAY`*                                          |                                                            |
|         0x01 |*`CU_RESOURCE_TYPE_MIPMAPPED_ARRAY`*                                |                                                            |
|         0x02 |*`CU_RESOURCE_TYPE_LINEAR`*                                         |                                                            |
|         0x03 |*`CU_RESOURCE_TYPE_PITCH2D`*                                        |                                                            |
| enum         |***`CUresourceViewFormat`***                                        |***`hipResourceViewFormat`***                               |
| typedef      |***`CUresourceViewFormat_enum`***                                   |***`hipResourceViewFormat`***                               |
|         0x00 |*`CU_RES_VIEW_FORMAT_NONE`*                                         |*`hipResViewFormatNone`*                                    |
|         0x01 |*`CU_RES_VIEW_FORMAT_UINT_1X8`*                                     |*`hipResViewFormatUnsignedChar1`*                           |
|         0x02 |*`CU_RES_VIEW_FORMAT_UINT_2X8`*                                     |*`hipResViewFormatUnsignedChar2`*                           |
|         0x03 |*`CU_RES_VIEW_FORMAT_UINT_4X8`*                                     |*`hipResViewFormatUnsignedChar4`*                           |
|         0x04 |*`CU_RES_VIEW_FORMAT_SINT_1X8`*                                     |*`hipResViewFormatSignedChar1`*                             |
|         0x05 |*`CU_RES_VIEW_FORMAT_SINT_2X8`*                                     |*`hipResViewFormatSignedChar2`*                             |
|         0x06 |*`CU_RES_VIEW_FORMAT_SINT_4X8`*                                     |*`hipResViewFormatSignedChar4`*                             |
|         0x07 |*`CU_RES_VIEW_FORMAT_UINT_1X16`*                                    |*`hipResViewFormatUnsignedShort1`*                          |
|         0x08 |*`CU_RES_VIEW_FORMAT_UINT_2X16`*                                    |*`hipResViewFormatUnsignedShort2`*                          |
|         0x09 |*`CU_RES_VIEW_FORMAT_UINT_4X16`*                                    |*`hipResViewFormatUnsignedShort4`*                          |
|         0x0a |*`CU_RES_VIEW_FORMAT_SINT_1X16`*                                    |*`hipResViewFormatSignedShort1`*                            |
|         0x0b |*`CU_RES_VIEW_FORMAT_SINT_2X16`*                                    |*`hipResViewFormatSignedShort2`*                            |
|         0x0c |*`CU_RES_VIEW_FORMAT_SINT_4X16`*                                    |*`hipResViewFormatSignedShort4`*                            |
|         0x0d |*`CU_RES_VIEW_FORMAT_UINT_1X32`*                                    |*`hipResViewFormatUnsignedInt1`*                            |
|         0x0e |*`CU_RES_VIEW_FORMAT_UINT_2X32`*                                    |*`hipResViewFormatUnsignedInt2`*                            |
|         0x0f |*`CU_RES_VIEW_FORMAT_UINT_4X32`*                                    |*`hipResViewFormatUnsignedInt4`*                            |
|         0x10 |*`CU_RES_VIEW_FORMAT_SINT_1X32`*                                    |*`hipResViewFormatSignedInt1`*                              |
|         0x11 |*`CU_RES_VIEW_FORMAT_SINT_2X32`*                                    |*`hipResViewFormatSignedInt2`*                              |
|         0x12 |*`CU_RES_VIEW_FORMAT_SINT_4X32`*                                    |*`hipResViewFormatSignedInt4`*                              |
|         0x13 |*`CU_RES_VIEW_FORMAT_FLOAT_1X16`*                                   |*`hipResViewFormatHalf1`*                                   |
|         0x14 |*`CU_RES_VIEW_FORMAT_FLOAT_2X16`*                                   |*`hipResViewFormatHalf2`*                                   |
|         0x15 |*`CU_RES_VIEW_FORMAT_FLOAT_4X16`*                                   |*`hipResViewFormatHalf4`*                                   |
|         0x16 |*`CU_RES_VIEW_FORMAT_FLOAT_1X32`*                                   |*`hipResViewFormatFloat1`*                                  |
|         0x17 |*`CU_RES_VIEW_FORMAT_FLOAT_2X32`*                                   |*`hipResViewFormatFloat2`*                                  |
|         0x18 |*`CU_RES_VIEW_FORMAT_FLOAT_4X32`*                                   |*`hipResViewFormatFloat4`*                                  |
|         0x19 |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC1`*                                 |*`hipResViewFormatUnsignedBlockCompressed1`*                |
|         0x1a |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC2`*                                 |*`hipResViewFormatUnsignedBlockCompressed2`*                |
|         0x1b |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC3`*                                 |*`hipResViewFormatUnsignedBlockCompressed3`*                |
|         0x1c |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC4`*                                 |*`hipResViewFormatUnsignedBlockCompressed4`*                |
|         0x1d |*`CU_RES_VIEW_FORMAT_SIGNED_BC4`*                                   |*`hipResViewFormatSignedBlockCompressed4`*                  |
|         0x1e |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC5`*                                 |*`hipResViewFormatUnsignedBlockCompressed5`*                |
|         0x1f |*`CU_RES_VIEW_FORMAT_SIGNED_BC5`*                                   |*`hipResViewFormatSignedBlockCompressed5`*                  |
|         0x20 |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC6H`*                                |*`hipResViewFormatUnsignedBlockCompressed6H`*               |
|         0x21 |*`CU_RES_VIEW_FORMAT_SIGNED_BC6H`*                                  |*`hipResViewFormatSignedBlockCompressed6H`*                 |
|         0x22 |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC7`*                                 |*`hipResViewFormatUnsignedBlockCompressed7`*                |
| enum         |***`CUresult`***                                                    |***`hipError_t`***                                          |
| typedef      |`cudaError_enum`                                                    |***`hipError_t`***                                          |
|            0 |*`CUDA_SUCCESS`*                                                    |*`hipSuccess`*                                              |
|            1 |*`CUDA_ERROR_INVALID_VALUE`*                                        |*`hipErrorInvalidValue`*                                    |
|            2 |*`CUDA_ERROR_OUT_OF_MEMORY`*                                        |*`hipErrorOutOfMemory`*                                     |
|            3 |*`CUDA_ERROR_NOT_INITIALIZED`*                                      |*`hipErrorNotInitialized`*                                  |
|            4 |*`CUDA_ERROR_DEINITIALIZED`*                                        |*`hipErrorDeinitialized`*                                   |
|            5 |*`CUDA_ERROR_PROFILER_DISABLED`*                                    |*`hipErrorProfilerDisabled`*                                |
|            6 |*`CUDA_ERROR_PROFILER_NOT_INITIALIZED`*                             |*`hipErrorProfilerNotInitialized`*                          |
|            7 |*`CUDA_ERROR_PROFILER_ALREADY_STARTED`*                             |*`hipErrorProfilerAlreadyStarted`*                          |
|            8 |*`CUDA_ERROR_PROFILER_ALREADY_STOPPED`*                             |*`hipErrorProfilerAlreadyStopped`*                          |
|          100 |*`CUDA_ERROR_NO_DEVICE`*                                            |*`hipErrorNoDevice`*                                        |
|          101 |*`CUDA_ERROR_INVALID_DEVICE`*                                       |*`hipErrorInvalidDevice`*                                   |
|          200 |*`CUDA_ERROR_INVALID_IMAGE`*                                        |*`hipErrorInvalidImage`*                                    |
|          201 |*`CUDA_ERROR_INVALID_CONTEXT`*                                      |*`hipErrorInvalidContext`*                                  |
|          202 |*`CUDA_ERROR_CONTEXT_ALREADY_CURRENT`*                              |*`hipErrorContextAlreadyCurrent`*                           |
|          205 |*`CUDA_ERROR_MAP_FAILED`*                                           |*`hipErrorMapFailed`*                                       |
|          206 |*`CUDA_ERROR_UNMAP_FAILED`*                                         |*`hipErrorUnmapFailed`*                                     |
|          207 |*`CUDA_ERROR_ARRAY_IS_MAPPED`*                                      |*`hipErrorArrayIsMapped`*                                   |
|          208 |*`CUDA_ERROR_ALREADY_MAPPED`*                                       |*`hipErrorAlreadyMapped`*                                   |
|          209 |*`CUDA_ERROR_NO_BINARY_FOR_GPU`*                                    |*`hipErrorNoBinaryForGpu`*                                  |
|          210 |*`CUDA_ERROR_ALREADY_ACQUIRED`*                                     |*`hipErrorAlreadyAcquired`*                                 |
|          211 |*`CUDA_ERROR_NOT_MAPPED`*                                           |*`hipErrorNotMapped`*                                       |
|          212 |*`CUDA_ERROR_NOT_MAPPED_AS_ARRAY`*                                  |*`hipErrorNotMappedAsArray`*                                |
|          213 |*`CUDA_ERROR_NOT_MAPPED_AS_POINTER`*                                |*`hipErrorNotMappedAsPointer`*                              |
|          214 |*`CUDA_ERROR_ECC_UNCORRECTABLE`*                                    |*`hipErrorECCNotCorrectable`*                               |
|          215 |*`CUDA_ERROR_UNSUPPORTED_LIMIT`*                                    |*`hipErrorUnsupportedLimit`*                                |
|          216 |*`CUDA_ERROR_CONTEXT_ALREADY_IN_USE`*                               |*`hipErrorContextAlreadyInUse`*                             |
|          217 |*`CUDA_ERROR_PEER_ACCESS_UNSUPPORTED`*                              |*`hipErrorPeerAccessUnsupported`*                           |
|          218 |*`CUDA_ERROR_INVALID_PTX`*                                          |*`hipErrorInvalidKernelFile`*                               |
|          219 |*`CUDA_ERROR_INVALID_GRAPHICS_CONTEXT`*                             |*`hipErrorInvalidGraphicsContext`*                          |
|          220 |*`CUDA_ERROR_NVLINK_UNCORRECTABLE`*                                 |                                                            | 8.0              |
|          221 |*`CUDA_ERROR_JIT_COMPILER_NOT_FOUND`*                               |                                                            | 9.0              |
|          300 |*`CUDA_ERROR_INVALID_SOURCE`*                                       |*`hipErrorInvalidSource`*                                   |
|          301 |*`CUDA_ERROR_FILE_NOT_FOUND`*                                       |*`hipErrorFileNotFound`*                                    |
|          302 |*`CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND`*                       |*`hipErrorSharedObjectSymbolNotFound`*                      |
|          303 |*`CUDA_ERROR_SHARED_OBJECT_INIT_FAILED`*                            |*`hipErrorSharedObjectInitFailed`*                          |
|          304 |*`CUDA_ERROR_OPERATING_SYSTEM`*                                     |*`hipErrorOperatingSystem`*                                 |
|          400 |*`CUDA_ERROR_INVALID_HANDLE`*                                       |*`hipErrorInvalidHandle`*                                   |
|          401 |*`CUDA_ERROR_ILLEGAL_STATE`*                                        |                                                            | 10.0             |
|          500 |*`CUDA_ERROR_NOT_FOUND`*                                            |*`hipErrorNotFound`*                                        |
|          600 |*`CUDA_ERROR_NOT_READY`*                                            |*`hipErrorNotReady`*                                        |
|          700 |*`CUDA_ERROR_ILLEGAL_ADDRESS`*                                      |*`hipErrorIllegalAddress`*                                  |
|          701 |*`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`*                              |*`hipErrorLaunchOutOfResources`*                            |
|          702 |*`CUDA_ERROR_LAUNCH_TIMEOUT`*                                       |*`hipErrorLaunchTimeOut`*                                   |
|          703 |*`CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING`*                        |                                                            |
|          704 |*`CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED`*                          |*`hipErrorPeerAccessAlreadyEnabled`*                        |
|          705 |*`CUDA_ERROR_PEER_ACCESS_NOT_ENABLED`*                              |*`hipErrorPeerAccessNotEnabled`*                            |
|          708 |*`CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE`*                               |*`hipErrorSetOnActiveProcess`*                              |
|          709 |*`CUDA_ERROR_CONTEXT_IS_DESTROYED`*                                 |                                                            |
|          710 |*`CUDA_ERROR_ASSERT`*                                               |*`hipErrorAssert`*                                          |
|          711 |*`CUDA_ERROR_TOO_MANY_PEERS`*                                       |                                                            |
|          712 |*`CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`*                       |*`hipErrorHostMemoryAlreadyRegistered`*                     |
|          713 |*`CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED`*                           |*`hipErrorHostMemoryNotRegistered`*                         |
|          714 |*`CUDA_ERROR_HARDWARE_STACK_ERROR`*                                 |                                                            |
|          715 |*`CUDA_ERROR_ILLEGAL_INSTRUCTION`*                                  |                                                            |
|          716 |*`CUDA_ERROR_MISALIGNED_ADDRESS`*                                   |                                                            |
|          717 |*`CUDA_ERROR_INVALID_ADDRESS_SPACE`*                                |                                                            |
|          718 |*`CUDA_ERROR_INVALID_PC`*                                           |                                                            |
|          719 |*`CUDA_ERROR_LAUNCH_FAILED`*                                        |*`hipErrorLaunchFailure`*                                   |
|          720 |*`CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`*                         |*`hipErrorCooperativeLaunchTooLarge`*                       |
|          800 |*`CUDA_ERROR_NOT_PERMITTED`*                                        |                                                            |
|          801 |*`CUDA_ERROR_NOT_SUPPORTED`*                                        |*`hipErrorNotSupported`*                                    |
|          802 |*`CUDA_ERROR_SYSTEM_NOT_READY`*                                     |                                                            | 10.0             |
|          803 |*`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`*                               |                                                            | 10.1             |
|          804 |*`CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`*                       |                                                            | 10.1             |
|          900 |*`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`*                           |                                                            | 10.0             |
|          901 |*`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`*                           |                                                            | 10.0             |
|          902 |*`CUDA_ERROR_STREAM_CAPTURE_MERGE`*                                 |                                                            | 10.0             |
|          903 |*`CUDA_ERROR_STREAM_CAPTURE_UNMATCHED`*                             |                                                            | 10.0             |
|          904 |*`CUDA_ERROR_STREAM_CAPTURE_UNJOINED`*                              |                                                            | 10.0             |
|          905 |*`CUDA_ERROR_STREAM_CAPTURE_ISOLATION`*                             |                                                            | 10.0             |
|          906 |*`CUDA_ERROR_STREAM_CAPTURE_IMPLICIT`*                              |                                                            | 10.0             |
|          907 |*`CUDA_ERROR_CAPTURED_EVENT`*                                       |                                                            | 10.0             |
|          908 |*`CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD`*                          |                                                            | 10.1             |
|          909 |*`CUDA_ERROR_TIMEOUT`*                                              |                                                            | 10.2             |
|          910 |*`CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE`*                            |                                                            | 10.2             |
|          999 |*`CUDA_ERROR_UNKNOWN`*                                              |*`hipErrorUnknown`*                                         |
| enum         |***`CUsharedconfig`***                                              |***`hipSharedMemConfig`***                                  |
| typedef      |***`CUsharedconfig_enum`***                                         |***`hipSharedMemConfig`***                                  |
|         0x00 |*`CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE`*                          |*`hipSharedMemBankSizeDefault`*                             |
|         0x01 |*`CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE`*                        |*`hipSharedMemBankSizeFourByte`*                            |
|         0x02 |*`CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE`*                       |*`hipSharedMemBankSizeEightByte`*                           |
| enum         |***`CUshared_carveout`***                                           |                                                            | 9.0              |
| typedef      |***`CUshared_carveout_enum`***                                      |                                                            | 9.0              |
|           -1 |*`CU_SHAREDMEM_CARVEOUT_DEFAULT`*                                   |                                                            | 9.0              |
|          100 |*`CU_SHAREDMEM_CARVEOUT_MAX_SHARED`*                                |                                                            | 9.0              |
|            0 |*`CU_SHAREDMEM_CARVEOUT_MAX_L1`*                                    |                                                            | 9.0              |
| enum         |***`CUstream_flags`***                                              |                                                            |
| typedef      |***`CUstream_flags_enum`***                                         |                                                            |
|          0x0 |*`CU_STREAM_DEFAULT`*                                               |*`hipStreamDefault`*                                        |
|          0x1 |*`CU_STREAM_NON_BLOCKING`*                                          |*`hipStreamNonBlocking`*                                    |
| enum         |***`CUstreamBatchMemOpType`***                                      |                                                            | 8.0              |
| typedef      |***`CUstreamBatchMemOpType_enum`***                                 |                                                            | 8.0              |
|            1 |*`CU_STREAM_MEM_OP_WAIT_VALUE_32`*                                  |                                                            | 8.0              |
|            2 |*`CU_STREAM_MEM_OP_WRITE_VALUE_32`*                                 |                                                            | 8.0              |
|            3 |*`CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES`*                            |                                                            | 8.0              |
|            4 |*`CU_STREAM_MEM_OP_WAIT_VALUE_64`*                                  |                                                            | 9.0              |
|            5 |*`CU_STREAM_MEM_OP_WRITE_VALUE_64`*                                 |                                                            | 9.0              |
| enum         |***`CUGLDeviceList`***                                              |                                                            |
| typedef      |***`CUGLDeviceList_enum`***                                         |                                                            |
|         0x01 |*`CU_GL_DEVICE_LIST_ALL`*                                           |                                                            |
|         0x02 |*`CU_GL_DEVICE_LIST_CURRENT_FRAME`*                                 |                                                            |
|         0x03 |*`CU_GL_DEVICE_LIST_NEXT_FRAME`*                                    |                                                            |
| enum         |***`CUGLmap_flags`***                                               |                                                            |
| typedef      |***`CUGLmap_flags_enum`***                                          |                                                            |
|         0x00 |*`CU_GL_MAP_RESOURCE_FLAGS_NONE`*                                   |                                                            |
|         0x01 |*`CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY`*                              |                                                            |
|         0x02 |*`CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD`*                          |                                                            |
| enum         |***`CUd3d9DeviceList`***                                            |                                                            |
| typedef      |***`CUd3d9DeviceList_enum`***                                       |                                                            |
|         0x01 |*`CU_D3D9_DEVICE_LIST_ALL`*                                         |                                                            |
|         0x02 |*`CU_D3D9_DEVICE_LIST_CURRENT_FRAME`*                               |                                                            |
|         0x03 |*`CU_D3D9_DEVICE_LIST_NEXT_FRAME`*                                  |                                                            |
| enum         |***`CUd3d9map_flags`***                                             |                                                            |
| typedef      |***`CUd3d9map_flags_enum`***                                        |                                                            |
|         0x00 |*`CU_D3D9_MAPRESOURCE_FLAGS_NONE`*                                  |                                                            |
|         0x01 |*`CU_D3D9_MAPRESOURCE_FLAGS_READONLY`*                              |                                                            |
|         0x02 |*`CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD`*                          |                                                            |
| enum         |***`CUd3d9register_flags`***                                        |                                                            |
| typedef      |***`CUd3d9register_flags_enum`***                                   |                                                            |
|         0x00 |*`CU_D3D9_REGISTER_FLAGS_NONE`*                                     |                                                            |
|         0x01 |*`CU_D3D9_REGISTER_FLAGS_ARRAY`*                                    |                                                            |
| enum         |***`CUd3d10DeviceList`***                                           |                                                            |
| typedef      |***`CUd3d10DeviceList_enum`***                                      |                                                            |
|         0x01 |*`CU_D3D10_DEVICE_LIST_ALL`*                                        |                                                            |
|         0x02 |*`CU_D3D10_DEVICE_LIST_CURRENT_FRAME`*                              |                                                            |
|         0x03 |*`CU_D3D10_DEVICE_LIST_NEXT_FRAME`*                                 |                                                            |
| enum         |***`CUd3d10map_flags`***                                            |                                                            |
| typedef      |***`CUd3d10map_flags_enum`***                                       |                                                            |
|         0x00 |*`CU_D3D10_MAPRESOURCE_FLAGS_NONE`*                                 |                                                            |
|         0x01 |*`CU_D3D10_MAPRESOURCE_FLAGS_READONLY`*                             |                                                            |
|         0x02 |*`CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD`*                         |                                                            |
| enum         |***`CUd3d10register_flags`***                                       |                                                            |
| typedef      |***`CUd3d10register_flags_enum`***                                  |                                                            |
|         0x00 |*`CU_D3D10_REGISTER_FLAGS_NONE`*                                    |                                                            |
|         0x01 |*`CU_D3D10_REGISTER_FLAGS_ARRAY`*                                   |                                                            |
| enum         |***`CUd3d11DeviceList`***                                           |                                                            |
| typedef      |***`CUd3d11DeviceList_enum`***                                      |                                                            |
|         0x01 |*`CU_D3D11_DEVICE_LIST_ALL`*                                        |                                                            |
|         0x02 |*`CU_D3D11_DEVICE_LIST_CURRENT_FRAME`*                              |                                                            |
|         0x03 |*`CU_D3D11_DEVICE_LIST_NEXT_FRAME`*                                 |                                                            |
| struct       |`CUarray_st`                                                        |`hipArray`                                                  |
| typedef      |`CUarray`                                                           |`hipArray *`                                                |
| struct       |`CUctx`                                                             |`ihipCtx_t`                                                 |
| typedef      |`CUcontext_st`                                                      |`hipCtx_t`                                                  |
| typedef      |`CUdevice`                                                          |`hipDevice_t`                                               |
| typedef      |`CUdeviceptr`                                                       |`hipDeviceptr_t`                                            |
| typedef      |`CUdeviceptr_v1`                                                    |`hipDeviceptr_t`                                            | 11.0             |
| struct       |`CUeglStreamConnection_st`                                          |                                                            | 9.1              |
| typedef      |`CUeglStreamConnection`                                             |                                                            | 9.1              |
| typedef      |`CUevent`                                                           |`hipEvent_t`                                                |
| struct       |`CUevent_st`                                                        |`ihipEvent_t`                                               |
| typedef      |`CUfunction`                                                        |`hipFunction_t`                                             |
| struct       |`CUfunc_st`                                                         |`ihipModuleSymbol_t`                                        |
| typedef      |`CUgraphicsResource`                                                |                                                            |
| struct       |`CUgraphicsResource_st`                                             |                                                            |
| typedef      |`CUmipmappedArray`                                                  |`hipMipmappedArray_t`                                       |
| struct       |`CUmipmappedArray_st`                                               |`hipMipmappedArray`                                         |
| typedef      |`CUmodule`                                                          |`hipModule_t`                                               |
| struct       |`CUmod_st`                                                          |`ihipModule_t`                                              |
| typedef      |`CUstream`                                                          |`hipStream_t`                                               |
| struct       |`CUstream_st`                                                       |`ihipStream_t`                                              |
| typedef      |`CUstreamCallback`                                                  |`hipStreamCallback_t`                                       |
| typedef      |`CUsurfObject`                                                      |`hipSurfaceObject_t`                                        |
| typedef      |`CUsurfref`                                                         |                                                            |
| struct       |`CUsurfref_st`                                                      |                                                            |
| typedef      |`CUtexObject`                                                       |`hipTextureObject_t`                                        |
| typedef      |`CUtexref`                                                          |                                                            |
| struct       |`CUtexref_st`                                                       |`textureReference`                                          |
| define       |`CU_IPC_HANDLE_SIZE`                                                |                                                            |
| define       |`CU_LAUNCH_PARAM_BUFFER_POINTER`                                    |`HIP_LAUNCH_PARAM_BUFFER_POINTER`                           |
| define       |`CU_LAUNCH_PARAM_BUFFER_SIZE`                                       |`HIP_LAUNCH_PARAM_BUFFER_SIZE`                              |
| define       |`CU_LAUNCH_PARAM_END`                                               |`HIP_LAUNCH_PARAM_END`                                      |
| define       |`CU_MEMHOSTALLOC_DEVICEMAP`                                         |`hipHostMallocMapped`                                       |
| define       |`CU_MEMHOSTALLOC_PORTABLE`                                          |`hipHostMallocPortable`                                     |
| define       |`CU_MEMHOSTALLOC_WRITECOMBINED`                                     |`hipHostMallocWriteCombined`                                |
| define       |`CU_MEMHOSTREGISTER_DEVICEMAP`                                      |`hipHostRegisterMapped`                                     |
| define       |`CU_MEMHOSTREGISTER_IOMEMORY`                                       |`hipHostRegisterIoMemory`                                   | 7.5              |
| define       |`CU_MEMHOSTREGISTER_PORTABLE`                                       |`hipHostRegisterPortable`                                   |
| define       |`CU_PARAM_TR_DEFAULT`                                               |                                                            |
| define       |`CU_STREAM_LEGACY`                                                  |                                                            |
| define       |`CU_STREAM_PER_THREAD`                                              |                                                            |
| define       |`CU_TRSA_OVERRIDE_FORMAT`                                           |`HIP_TRSA_OVERRIDE_FORMAT`                                  |
| define       |`CU_TRSF_NORMALIZED_COORDINATES`                                    |`HIP_TRSF_NORMALIZED_COORDINATES`                           |
| define       |`CU_TRSF_READ_AS_INTEGER`                                           |`HIP_TRSF_READ_AS_INTEGER`                                  |
| define       |`CU_TRSF_SRGB`                                                      |                                                            |
| define       |`CUDA_ARRAY3D_2DARRAY`                                              |                                                            |
| define       |`CUDA_ARRAY3D_CUBEMAP`                                              |`hipArrayCubemap`                                           |
| define       |`CUDA_ARRAY3D_DEPTH_TEXTURE`                                        |                                                            |
| define       |`CUDA_ARRAY3D_LAYERED`                                              |`hipArrayLayered`                                           |
| define       |`CUDA_ARRAY3D_SURFACE_LDST`                                         |`hipArraySurfaceLoadStore`                                  |
| define       |`CUDA_ARRAY3D_TEXTURE_GATHER`                                       |`hipArrayTextureGather`                                     |
| define       |`CUDA_ARRAY3D_COLOR_ATTACHMENT`                                     |                                                            | 10.0             |
| define       |`CUDA_VERSION`                                                      |                                                            |
| typedef      |`CUexternalMemory`                                                  |                                                            | 10.0             |
| struct       |`CUextMemory_st`                                                    |                                                            | 10.0             |
| typedef      |`CUexternalSemaphore`                                               |                                                            | 10.0             |
| struct       |`CUextSemaphore_st`                                                 |                                                            | 10.0             |
| typedef      |`CUgraph`                                                           |                                                            | 10.0             |
| struct       |`CUgraph_st`                                                        |                                                            | 10.0             |
| typedef      |`CUgraphNode`                                                       |                                                            | 10.0             |
| struct       |`CUgraphNode_st`                                                    |                                                            | 10.0             |
| typedef      |`CUgraphExec`                                                       |                                                            | 10.0             |
| struct       |`CUgraphExec_st`                                                    |                                                            | 10.0             |
| typedef      |`CUhostFn`                                                          |                                                            | 10.0             |
| typedef      |`CUoccupancyB2DSize`                                                |                                                            |
| struct       |`CUDA_KERNEL_NODE_PARAMS`                                           |                                                            | 10.0             |
| typedef      |`CUDA_KERNEL_NODE_PARAMS_st`                                        |                                                            | 10.0             |
| struct       |`CUDA_LAUNCH_PARAMS`                                                |                                                            | 9.0              |
| typedef      |`CUDA_LAUNCH_PARAMS_st`                                             |                                                            | 9.0              |
| struct       |`CUDA_MEMSET_NODE_PARAMS`                                           |                                                            | 10.0             |
| typedef      |`CUDA_MEMSET_NODE_PARAMS_st`                                        |                                                            | 10.0             |
| struct       |`CUDA_HOST_NODE_PARAMS`                                             |                                                            | 10.0             |
| typedef      |`CUDA_HOST_NODE_PARAMS_st`                                          |                                                            | 10.0             |
| enum         |***`CUgraphNodeType`***                                             |                                                            | 10.0             |
| typedef      |***`CUgraphNodeType_enum`***                                        |                                                            | 10.0             |
|            0 |*`CU_GRAPH_NODE_TYPE_KERNEL`*                                       |                                                            | 10.0             |
|            1 |*`CU_GRAPH_NODE_TYPE_MEMCPY`*                                       |                                                            | 10.0             |
|            2 |*`CU_GRAPH_NODE_TYPE_MEMSET`*                                       |                                                            | 10.0             |
|            3 |*`CU_GRAPH_NODE_TYPE_HOST`*                                         |                                                            | 10.0             |
|            4 |*`CU_GRAPH_NODE_TYPE_GRAPH`*                                        |                                                            | 10.0             |
|            5 |*`CU_GRAPH_NODE_TYPE_EMPTY`*                                        |                                                            | 10.0             |
|            6 |*`CU_GRAPH_NODE_TYPE_COUNT`*                                        |                                                            | 10.0             |
| enum         |***`CUstreamCaptureStatus`***                                       |                                                            | 10.0             |
| typedef      |***`CUstreamCaptureStatus_enum`***                                  |                                                            | 10.0             |
|            0 |*`CU_STREAM_CAPTURE_STATUS_NONE`*                                   |                                                            | 10.0             |
|            1 |*`CU_STREAM_CAPTURE_STATUS_ACTIVE`*                                 |                                                            | 10.0             |
|            2 |*`CU_STREAM_CAPTURE_STATUS_INVALIDATED`*                            |                                                            | 10.0             |
| enum         |***`CUstreamCaptureMode`***                                         |                                                            | 10.1             |
| typedef      |***`CUstreamCaptureMode_enum`***                                    |                                                            | 10.1             |
|            0 |*`CU_STREAM_CAPTURE_MODE_GLOBAL`*                                   |                                                            | 10.1             |
|            1 |*`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`*                             |                                                            | 10.1             |
|            2 |*`CU_STREAM_CAPTURE_MODE_RELAXED`*                                  |                                                            | 10.1             |
| enum         |***`CUstreamWaitValue_flags`***                                     |                                                            | 8.0              |
| typedef      |***`CUstreamWaitValue_flags_enum`***                                |                                                            | 8.0              |
|          0x0 |*`CU_STREAM_WAIT_VALUE_GEQ`*                                        |                                                            | 8.0              |
|          0x1 |*`CU_STREAM_WAIT_VALUE_EQ`*                                         |                                                            | 8.0              |
|          0x2 |*`CU_STREAM_WAIT_VALUE_AND`*                                        |                                                            | 8.0              |
|        1<<30 |*`CU_STREAM_WAIT_VALUE_FLUSH`*                                      |                                                            | 8.0              |
| enum         |***`CUstreamWriteValue_flags`***                                    |                                                            | 8.0              |
| typedef      |***`CUstreamWriteValue_flags_enum`***                               |                                                            | 8.0              |
|          0x0 |*`CU_STREAM_WRITE_VALUE_DEFAULT`*                                   |                                                            | 8.0              |
|          0x1 |*`CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER`*                         |                                                            | 8.0              |
| enum         |***`CUdevice_P2PAttribute`***                                       |                                                            | 8.0              |
| typedef      |***`CUdevice_P2PAttribute_enum`***                                  |                                                            | 8.0              |
|         0x01 |*`CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK`*                        |                                                            | 8.0              |
|         0x02 |*`CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED`*                        |                                                            | 8.0              |
|         0x03 |*`CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED`*                 |                                                            | 8.0              |
|         0x04 |*`CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED`*                 |                                                            | 10.1             |
|         0x04 |*`CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED`*           |                                                            | 9.2 - 10.0       |
|         0x04 |*`CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED`*             |                                                            | 9.2              |
| enum         |***`CUeglColorFormat`***                                            |                                                            | 8.0              |
| typedef      |***`CUeglColorFormate_enum`***                                      |                                                            | 8.0              |
|         0x00 |*`CU_EGL_COLOR_FORMAT_YUV420_PLANAR`*                               |                                                            | 8.0              |
|         0x01 |*`CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR`*                           |                                                            | 8.0              |
|         0x02 |*`CU_EGL_COLOR_FORMAT_YUV422_PLANAR`*                               |                                                            | 8.0              |
|         0x03 |*`CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR`*                           |                                                            | 8.0              |
|         0x04 |*`CU_EGL_COLOR_FORMAT_RGB`*                                         |                                                            | 8.0              |
|         0x05 |*`CU_EGL_COLOR_FORMAT_BGR`*                                         |                                                            | 8.0              |
|         0x06 |*`CU_EGL_COLOR_FORMAT_ARGB`*                                        |                                                            | 8.0              |
|         0x07 |*`CU_EGL_COLOR_FORMAT_RGBA`*                                        |                                                            | 8.0              |
|         0x08 |*`CU_EGL_COLOR_FORMAT_L`*                                           |                                                            | 8.0              |
|         0x09 |*`CU_EGL_COLOR_FORMAT_R`*                                           |                                                            | 8.0              |
|         0x0A |*`CU_EGL_COLOR_FORMAT_YUV444_PLANAR`*                               |                                                            | 9.0              |
|         0x0B |*`CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR`*                           |                                                            | 9.0              |
|         0x0C |*`CU_EGL_COLOR_FORMAT_YUYV_422`*                                    |                                                            | 9.0              |
|         0x0D |*`CU_EGL_COLOR_FORMAT_UYVY_422`*                                    |                                                            | 9.0              |
|         0x0E |*`CU_EGL_COLOR_FORMAT_ABGR`*                                        |                                                            | 9.1              |
|         0x0F |*`CU_EGL_COLOR_FORMAT_BGRA`*                                        |                                                            | 9.1              |
|         0x10 |*`CU_EGL_COLOR_FORMAT_A`*                                           |                                                            | 9.1              |
|         0x11 |*`CU_EGL_COLOR_FORMAT_RG`*                                          |                                                            | 9.1              |
|         0x12 |*`CU_EGL_COLOR_FORMAT_AYUV`*                                        |                                                            | 9.1              |
|         0x13 |*`CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR`*                           |                                                            | 9.1              |
|         0x14 |*`CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR`*                           |                                                            | 9.1              |
|         0x15 |*`CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR`*                           |                                                            | 9.1              |
|         0x16 |*`CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR`*                    |                                                            | 9.1              |
|         0x17 |*`CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR`*                    |                                                            | 9.1              |
|         0x18 |*`CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR`*                    |                                                            | 9.1              |
|         0x19 |*`CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR`*                    |                                                            | 9.1              |
|         0x1A |*`CU_EGL_COLOR_FORMAT_VYUY_ER`*                                     |                                                            | 9.1              |
|         0x1B |*`CU_EGL_COLOR_FORMAT_UYVY_ER`*                                     |                                                            | 9.1              |
|         0x1C |*`CU_EGL_COLOR_FORMAT_YUYV_ER`*                                     |                                                            | 9.1              |
|         0x1D |*`CU_EGL_COLOR_FORMAT_YVYU_ER`*                                     |                                                            | 9.1              |
|         0x1E |*`CU_EGL_COLOR_FORMAT_YUV_ER`*                                      |                                                            | 9.1              |
|         0x1F |*`CU_EGL_COLOR_FORMAT_YUVA_ER`*                                     |                                                            | 9.1              |
|         0x20 |*`CU_EGL_COLOR_FORMAT_AYUV_ER`*                                     |                                                            | 9.1              |
|         0x21 |*`CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER`*                            |                                                            | 9.1              |
|         0x22 |*`CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER`*                            |                                                            | 9.1              |
|         0x23 |*`CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER`*                            |                                                            | 9.1              |
|         0x24 |*`CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER`*                        |                                                            | 9.1              |
|         0x25 |*`CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER`*                        |                                                            | 9.1              |
|         0x26 |*`CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER`*                        |                                                            | 9.1              |
|         0x27 |*`CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER`*                            |                                                            | 9.1              |
|         0x28 |*`CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER`*                            |                                                            | 9.1              |
|         0x29 |*`CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER`*                            |                                                            | 9.1              |
|         0x2A |*`CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER`*                        |                                                            | 9.1              |
|         0x2B |*`CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER`*                        |                                                            | 9.1              |
|         0x2C |*`CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER`*                        |                                                            | 9.1              |
|         0x2D |*`CU_EGL_COLOR_FORMAT_BAYER_RGGB`*                                  |                                                            | 9.1              |
|         0x2E |*`CU_EGL_COLOR_FORMAT_BAYER_BGGR`*                                  |                                                            | 9.1              |
|         0x2F |*`CU_EGL_COLOR_FORMAT_BAYER_GRBG`*                                  |                                                            | 9.1              |
|         0x30 |*`CU_EGL_COLOR_FORMAT_BAYER_GBRG`*                                  |                                                            | 9.1              |
|         0x31 |*`CU_EGL_COLOR_FORMAT_BAYER10_RGGB`*                                |                                                            | 9.1              |
|         0x32 |*`CU_EGL_COLOR_FORMAT_BAYER10_BGGR`*                                |                                                            | 9.1              |
|         0x33 |*`CU_EGL_COLOR_FORMAT_BAYER10_GRBG`*                                |                                                            | 9.1              |
|         0x34 |*`CU_EGL_COLOR_FORMAT_BAYER10_GBRG`*                                |                                                            | 9.1              |
|         0x35 |*`CU_EGL_COLOR_FORMAT_BAYER12_RGGB`*                                |                                                            | 9.1              |
|         0x36 |*`CU_EGL_COLOR_FORMAT_BAYER12_BGGR`*                                |                                                            | 9.1              |
|         0x37 |*`CU_EGL_COLOR_FORMAT_BAYER12_GRBG`*                                |                                                            | 9.1              |
|         0x38 |*`CU_EGL_COLOR_FORMAT_BAYER12_GBRG`*                                |                                                            | 9.1              |
|         0x39 |*`CU_EGL_COLOR_FORMAT_BAYER14_RGGB`*                                |                                                            | 9.1              |
|         0x3A |*`CU_EGL_COLOR_FORMAT_BAYER14_BGGR`*                                |                                                            | 9.1              |
|         0x3B |*`CU_EGL_COLOR_FORMAT_BAYER14_GRBG`*                                |                                                            | 9.1              |
|         0x3C |*`CU_EGL_COLOR_FORMAT_BAYER14_GBRG`*                                |                                                            | 9.1              |
|         0x3D |*`CU_EGL_COLOR_FORMAT_BAYER20_RGGB`*                                |                                                            | 9.1              |
|         0x3E |*`CU_EGL_COLOR_FORMAT_BAYER20_BGGR`*                                |                                                            | 9.1              |
|         0x3F |*`CU_EGL_COLOR_FORMAT_BAYER20_GRBG`*                                |                                                            | 9.1              |
|         0x40 |*`CU_EGL_COLOR_FORMAT_BAYER20_GBRG`*                                |                                                            | 9.1              |
|         0x41 |*`CU_EGL_COLOR_FORMAT_YVU444_PLANAR`*                               |                                                            | 9.1              |
|         0x42 |*`CU_EGL_COLOR_FORMAT_YVU422_PLANAR`*                               |                                                            | 9.1              |
|         0x43 |*`CU_EGL_COLOR_FORMAT_YVU420_PLANAR`*                               |                                                            | 9.1              |
|         0x44 |*`CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB`*                              |                                                            | 9.2              |
|         0x45 |*`CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR`*                              |                                                            | 9.2              |
|         0x46 |*`CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG`*                              |                                                            | 9.2              |
|         0x47 |*`CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG`*                              |                                                            | 9.2              |
|         0x48 |*`CU_EGL_COLOR_FORMAT_MAX`*                                         |                                                            | 9.0              |
| enum         |***`CUeglFrameType`***                                              |                                                            | 8.0              |
| typedef      |***`CUeglFrameType_enum`***                                         |                                                            | 8.0              |
|            0 |*`CU_EGL_FRAME_TYPE_ARRAY`*                                         |                                                            | 8.0              |
|            1 |*`CU_EGL_FRAME_TYPE_PITCH`*                                         |                                                            | 8.0              |
| enum         |***`CUeglResourceLocationFlags`***                                  |                                                            | 8.0              |
| typedef      |***`CUeglResourceLocationFlags_enum`***                             |                                                            | 8.0              |
|         0x00 |*`CU_EGL_RESOURCE_LOCATION_SYSMEM`*                                 |                                                            | 8.0              |
|         0x01 |*`CU_EGL_RESOURCE_LOCATION_VIDMEM`*                                 |                                                            | 8.0              |
| enum         |***`CUexternalMemoryHandleType`***                                  |                                                            | 10.0             |
| typedef      |***`CUexternalMemoryHandleType_enum`***                             |                                                            | 10.0             |
|            1 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD`*                        |                                                            | 10.0             |
|            2 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32`*                     |                                                            | 10.0             |
|            3 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT`*                 |                                                            | 10.0             |
|            4 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP`*                       |                                                            | 10.0             |
|            5 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE`*                   |                                                            | 10.0             |
|            6 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE`*                   |                                                            | 10.2             |
|            7 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT`*               |                                                            | 10.2             |
|            8 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF`*                         |                                                            | 10.2             |
| define       |`CUDA_EXTERNAL_MEMORY_DEDICATED`                                    |                                                            | 10.0             |
| define       |`CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC`              |                                                            | 10.2             |
| define       |`CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC`                |                                                            | 10.2             |
| define       |`CUDA_NVSCISYNC_ATTR_SIGNAL`                                        |                                                            | 10.2             |
| define       |`CUDA_NVSCISYNC_ATTR_WAIT`                                          |                                                            | 10.2             |
| struct       |`CUDA_EXTERNAL_MEMORY_HANDLE_DESC`                                  |                                                            | 10.0             |
| typedef      |`CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st`                               |                                                            | 10.0             |
| struct       |`CUDA_EXTERNAL_MEMORY_BUFFER_DESC`                                  |                                                            | 10.0             |
| typedef      |`CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st`                               |                                                            | 10.0             |
| struct       |`CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC`                         |                                                            | 10.0             |
| typedef      |`CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st`                      |                                                            | 10.0             |
| enum         |***`CUexternalSemaphoreHandleType`***                               |                                                            | 10.0             |
| typedef      |***`CUexternalSemaphoreHandleType_enum`***                          |                                                            | 10.0             |
|            1 |*`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD`*                     |                                                            | 10.0             |
|            2 |*`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32`*                  |                                                            | 10.0             |
|            3 |*`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT`*              |                                                            | 10.0             |
|            4 |*`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE`*                   |                                                            | 10.0             |
|            5 |*`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE`*                   |                                                            | 10.2             |
|            6 |*`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC`*                     |                                                            | 10.2             |
|            7 |*`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX`*             |                                                            | 10.2             |
|            8 |*`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT`*         |                                                            | 10.2             |
| struct       |`CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC`                               |                                                            | 10.0             |
| typedef      |`CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st`                            |                                                            | 10.0             |
| struct       |`CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS`                             |                                                            | 10.0             |
| typedef      |`CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st`                          |                                                            | 10.0             |
| struct       |`CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS`                               |                                                            | 10.0             |
| typedef      |`CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st`                            |                                                            | 10.0             |
| define       |`CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC`           |                                                            | 9.0              |
| define       |`CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC`          |                                                            | 9.0              |
| define       |`__CUDACC__`                                                        |`__HIPCC__`                                                 |
| define       |`CUDA_CB`                                                           |                                                            |
| define       |`CU_DEVICE_CPU`                                                     |                                                            | 8.0              |
| define       |`CU_DEVICE_INVALID`                                                 |                                                            | 8.0              |
| struct       |`CUuuid`                                                            |                                                            |
| typedef      |`CUuuid_st`                                                         |                                                            |
| enum         |***`CUmemAllocationHandleType`***                                   |                                                            | 10.2             |
| typedef      |***`CUmemAllocationHandleType_enum`***                              |                                                            | 10.2             |
|          0x1 |*`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`*                        |                                                            | 10.2             |
|          0x2 |*`CU_MEM_HANDLE_TYPE_WIN32`*                                        |                                                            | 10.2             |
|          0x4 |*`CU_MEM_HANDLE_TYPE_WIN32_KMT`*                                    |                                                            | 10.2             |
|   0xFFFFFFFF |*`CU_MEM_HANDLE_TYPE_MAX`*                                          |                                                            | 10.2             |
| enum         |***`CUmemAccess_flags`***                                           |                                                            | 10.2             |
| typedef      |***`CUmemAccess_flags_enum`***                                      |                                                            | 10.2             |
|          0x1 |*`CU_MEM_ACCESS_FLAGS_PROT_NONE`*                                   |                                                            | 10.2             |
|          0x2 |*`CU_MEM_ACCESS_FLAGS_PROT_READ`*                                   |                                                            | 10.2             |
|          0x3 |*`CU_MEM_ACCESS_FLAGS_PROT_READWRITE`*                              |                                                            | 10.2             |
|   0xFFFFFFFF |*`CU_MEM_ACCESS_FLAGS_PROT_MAX`*                                    |                                                            | 10.2             |
| enum         |***`CUmemLocationType`***                                           |                                                            | 10.2             |
| typedef      |***`CUmemLocationType_enum`***                                      |                                                            | 10.2             |
|          0x0 |*`CU_MEM_LOCATION_TYPE_INVALID`*                                    |                                                            | 10.2             |
|          0x1 |*`CU_MEM_LOCATION_TYPE_DEVICE`*                                     |                                                            | 10.2             |
|   0xFFFFFFFF |*`CU_MEM_LOCATION_TYPE_MAX`*                                        |                                                            | 10.2             |
| enum         |***`CUmemAllocationGranularity_flags`***                            |                                                            | 10.2             |
| typedef      |***`CUmemAllocationGranularity_flags_enum`***                       |                                                            | 10.2             |
|          0x0 |*`CU_MEM_ALLOC_GRANULARITY_MINIMUM`*                                |                                                            | 10.2             |
|          0x1 |*`CU_MEM_ALLOC_GRANULARITY_RECOMMENDED`*                            |                                                            | 10.2             |
| struct       |`CUmemLocation`                                                     |                                                            | 10.2             |
| typedef      |`CUmemLocation_st`                                                  |                                                            | 10.2             |
| struct       |`CUmemAllocationProp`                                               |                                                            | 10.2             |
| typedef      |`CUmemAllocationProp_st`                                            |                                                            | 10.2             |
| struct       |`CUmemAccessDesc`                                                   |                                                            | 10.2             |
| typedef      |`CUmemAccessDesc_st`                                                |                                                            | 10.2             |
| enum         |***`CUgraphExecUpdateResult`***                                     |                                                            | 10.2             |
| typedef      |***`CUgraphExecUpdateResult_enum`***                                |                                                            | 10.2             |
|          0x0 |*`CU_GRAPH_EXEC_UPDATE_SUCCESS`*                                    |                                                            | 10.2             |
|          0x1 |*`CU_GRAPH_EXEC_UPDATE_ERROR`*                                      |                                                            | 10.2             |
|          0x2 |*`CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED`*                     |                                                            | 10.2             |
|          0x3 |*`CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED`*                    |                                                            | 10.2             |
|          0x4 |*`CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED`*                     |                                                            | 10.2             |
|          0x5 |*`CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED`*                   |                                                            | 10.2             |
|          0x6 |*`CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED`*                        |                                                            | 10.2             |
| typedef      |`CUmemGenericAllocationHandle`                                      |                                                            | 10.2             |
| enum         |***`CUaccessProperty`***                                            |                                                            | 11.0             |
| typedef      |***`CUaccessProperty_enum`***                                       |                                                            | 11.0             |
|            0 |*`CU_ACCESS_PROPERTY_NORMAL`*                                       |                                                            | 11.0             |
|            1 |*`CU_ACCESS_PROPERTY_STREAMING`*                                    |                                                            | 11.0             |
|            2 |*`CU_ACCESS_PROPERTY_PERSISTING`*                                   |                                                            | 11.0             |
| struct       |`CUaccessPolicyWindow`                                              |                                                            | 11.0             |
| typedef      |`CUaccessPolicyWindow_st`                                           |                                                            | 11.0             |
| enum         |***`CUsynchronizationPolicy`***                                     |                                                            | 11.0             |
| typedef      |***`CUsynchronizationPolicy_enum`***                                |                                                            | 11.0             |
|            1 |*`CU_SYNC_POLICY_AUTO`*                                             |                                                            | 11.0             |
|            2 |*`CU_SYNC_POLICY_SPIN`*                                             |                                                            | 11.0             |
|            3 |*`CU_SYNC_POLICY_YIELD`*                                            |                                                            | 11.0             |
|            4 |*`CU_SYNC_POLICY_BLOCKING_SYNC`*                                    |                                                            | 11.0             |
| enum         |***`CUkernelNodeAttrID`***                                          |                                                            | 11.0             |
| typedef      |***`CUkernelNodeAttrID_enum`***                                     |                                                            | 11.0             |
|            1 |*`CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW`*                   |                                                            | 11.0             |
|            2 |*`CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE`*                            |                                                            | 11.0             |
| union        |`CUkernelNodeAttrValue`                                             |                                                            | 11.0             |
| typedef      |`CUkernelNodeAttrValue_union`                                       |                                                            | 11.0             |
| enum         |***`CUstreamAttrID`***                                              |                                                            | 11.0             |
| typedef      |***`CUstreamAttrID_enum`***                                         |                                                            | 11.0             |
|            1 |*`CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW`*                        |                                                            | 11.0             |
|            3 |*`CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY`*                      |                                                            | 11.0             |
| union        |`CUstreamAttrValue`                                                 |                                                            | 11.0             |
| typedef      |`CUstreamAttrValue_union`                                           |                                                            | 11.0             |
| enum         |***`CUmemAllocationCompType`***                                     |                                                            | 11.0             |
| typedef      |***`CUmemAllocationCompType_enum`***                                |                                                            | 11.0             |
|          0x0 |*`CU_MEM_ALLOCATION_COMP_NONE`*                                     |                                                            | 11.0             |
|          0x1 |*`CU_MEM_ALLOCATION_COMP_GENERIC`*                                  |                                                            | 11.0             |
| define       |`CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION`                            |                                                            | 11.0             |

## **2. Error Handling**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuGetErrorName`                                          |                               |
| `cuGetErrorString`                                        |                               |

## **3. Initialization**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuInit`                                                  | `hipInit`                     |

## **4. Version Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuDriverGetVersion`                                      | `hipDriverGetVersion`         |

## **5. Device Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cuDriverGetVersion`                                      | `hipGetDevice`                |
| `cuDeviceGetAttribute`                                    | `hipDeviceGetAttribute`       |
| `cuDeviceGetCount`                                        | `hipGetDeviceCount`           |
| `cuDeviceGetName`                                         | `hipDeviceGetName`            |
| `cuDeviceGetNvSciSyncAttributes`                          |                               | 10.2             |
| `cuDeviceTotalMem`                                        | `hipDeviceTotalMem`           |
| `cuDeviceGetLuid`                                         |                               | 10.0             |
| `cuDeviceGetUuid`                                         |                               | 9.2              |

## **6. Device Management [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuDeviceComputeCapability`                               | `hipDeviceComputeCapability`  |
| `cuDeviceGetProperties`                                   |                               |

## **7. Primary Context Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuDevicePrimaryCtxGetState`                              | `hipDevicePrimaryCtxGetState` |
| `cuDevicePrimaryCtxRelease`                               | `hipDevicePrimaryCtxRelease`  |
| `cuDevicePrimaryCtxReset`                                 | `hipDevicePrimaryCtxReset`    |
| `cuDevicePrimaryCtxRetain`                                | `hipDevicePrimaryCtxRetain`   |
| `cuDevicePrimaryCtxSetFlags`                              | `hipDevicePrimaryCtxSetFlags` |

## **8. Context Management**

|   **CUDA**                                                |   **HIP**                        |**CUDA version\***|
|-----------------------------------------------------------|----------------------------------|------------------|
| `cuCtxCreate`                                             | `hipCtxCreate`                   |
| `cuCtxDestroy`                                            | `hipCtxDestroy`                  |
| `cuCtxGetApiVersion`                                      | `hipCtxGetApiVersion`            |
| `cuCtxGetCacheConfig`                                     | `hipCtxGetCacheConfig`           |
| `cuCtxGetCurrent`                                         | `hipCtxGetCurrent`               |
| `cuCtxGetDevice`                                          | `hipCtxGetDevice`                |
| `cuCtxGetFlags`                                           | `hipCtxGetFlags`                 |
| `cuCtxGetLimit`                                           | `hipDeviceGetLimit`              |
| `cuCtxGetSharedMemConfig`                                 | `hipCtxGetSharedMemConfig`       |
| `cuCtxGetStreamPriorityRange`                             | `hipDeviceGetStreamPriorityRange`|
| `cuCtxPopCurrent`                                         | `hipCtxPopCurrent`               |
| `cuCtxPushCurrent`                                        | `hipCtxPushCurrent`              |
| `cuCtxResetPersistingL2Cache`                             |                                  | 11.0             |
| `cuCtxSetCacheConfig`                                     | `hipCtxSetCacheConfig`           |
| `cuCtxSetCurrent`                                         | `hipCtxSetCurrent`               |
| `cuCtxSetLimit`                                           | `hipDeviceSetLimit`              |
| `cuCtxSetSharedMemConfig`                                 | `hipCtxSetSharedMemConfig`       |
| `cuCtxSynchronize`                                        | `hipCtxSynchronize`              |

## **9. Context Management [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuCtxAttach`                                             |                               |
| `cuCtxDetach`                                             |                               |

## **10. Module Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuLinkAddData`                                           |                               |
| `cuLinkAddFile`                                           |                               |
| `cuLinkComplete`                                          |                               |
| `cuLinkCreate`                                            |                               |
| `cuLinkDestroy`                                           |                               |
| `cuModuleGetFunction`                                     | `hipModuleGetFunction`        |
| `cuModuleGetGlobal`                                       | `hipModuleGetGlobal`          |
| `cuModuleGetSurfRef`                                      |                               |
| `cuModuleGetTexRef`                                       | `hipModuleGetTexRef`          |
| `cuModuleLoad`                                            | `hipModuleLoad`               |
| `cuModuleLoadData`                                        | `hipModuleLoadData`           |
| `cuModuleLoadDataEx`                                      | `hipModuleLoadDataEx`         |
| `cuModuleLoadFatBinary`                                   |                               |
| `cuModuleUnload`                                          | `hipModuleUnload`             |

## **11. Memory Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuArray3DCreate`                                         | `hipArray3DCreate`            |
| `cuArray3DGetDescriptor`                                  |                               |
| `cuArrayCreate`                                           | `hipArrayCreate`              |
| `cuArrayDestroy`                                          |                               |
| `cuArrayGetDescriptor`                                    |                               |
| `cuDeviceGetByPCIBusId`                                   | `hipDeviceGetByPCIBusId`      |
| `cuDeviceGetPCIBusId`                                     | `hipDeviceGetPCIBusId`        |
| `cuIpcCloseMemHandle`                                     | `hipIpcCloseMemHandle`        |
| `cuIpcGetEventHandle`                                     |                               |
| `cuIpcGetMemHandle`                                       | `hipIpcGetMemHandle`          |
| `cuIpcOpenEventHandle`                                    |                               |
| `cuIpcOpenMemHandle`                                      | `hipIpcOpenMemHandle`         |
| `cuMemAlloc`                                              | `hipMalloc`                   |
| `cuMemAllocHost`                                          | `hipHostMalloc`               |
| `cuMemAllocManaged`                                       | `hipMallocManaged`            |
| `cuMemAllocPitch`                                         | `hipMemAllocPitch`            |
| `cuMemcpy`                                                |                               |
| `cuMemcpy2D`                                              | `hipMemcpyParam2D`            |
| `cuMemcpy2DAsync`                                         | `hipMemcpyParam2DAsync`       |
| `cuMemcpy2DUnaligned`                                     |                               |
| `cuMemcpy3D`                                              | `hipDrvMemcpy3D`              |
| `cuMemcpy3DAsync`                                         | `hipDrvMemcpy3DAsync`         |
| `cuMemcpy3DPeer`                                          |                               |
| `cuMemcpy3DPeerAsync`                                     |                               |
| `cuMemcpyAsync`                                           |                               |
| `cuMemcpyAtoA`                                            |                               |
| `cuMemcpyAtoD`                                            |                               |
| `cuMemcpyAtoH`                                            | `hipMemcpyAtoH`               |
| `cuMemcpyAtoHAsync`                                       |                               |
| `cuMemcpyDtoA`                                            |                               |
| `cuMemcpyDtoD`                                            | `hipMemcpyDtoD`               |
| `cuMemcpyDtoDAsync`                                       | `hipMemcpyDtoDAsync`          |
| `cuMemcpyDtoH`                                            | `hipMemcpyDtoH`               |
| `cuMemcpyDtoHAsync`                                       | `hipMemcpyDtoHAsync`          |
| `cuMemcpyHtoA`                                            | `hipMemcpyHtoA`               |
| `cuMemcpyHtoAAsync`                                       |                               |
| `cuMemcpyHtoD`                                            | `hipMemcpyHtoD`               |
| `cuMemcpyHtoDAsync`                                       | `hipMemcpyHtoDAsync`          |
| `cuMemcpyPeer`                                            |                               |
| `cuMemcpyPeerAsync`                                       |                               |
| `cuMemFree`                                               | `hipFree`                     |
| `cuMemFreeHost`                                           | `hipHostFree`                 |
| `cuMemGetAddressRange`                                    | `hipMemGetAddressRange`       |
| `cuMemGetInfo`                                            | `hipMemGetInfo`               |
| `cuMemHostAlloc`                                          | `hipHostMalloc`               |
| `cuMemHostGetDevicePointer`                               | `hipHostGetDevicePointer`     |
| `cuMemHostGetFlags`                                       | `hipHostGetFlags`             |
| `cuMemHostRegister`                                       | `hipHostRegister`             |
| `cuMemHostUnregister`                                     | `hipHostUnregister`           |
| `cuMemsetD16`                                             | `hipMemsetD16`                |
| `cuMemsetD16Async`                                        | `hipMemsetD16Async`           |
| `cuMemsetD2D16`                                           |                               |
| `cuMemsetD2D16Async`                                      |                               |
| `cuMemsetD2D32`                                           |                               |
| `cuMemsetD2D32Async`                                      |                               |
| `cuMemsetD2D8`                                            |                               |
| `cuMemsetD2D8Async`                                       |                               |
| `cuMemsetD32`                                             | `hipMemsetD32`                |
| `cuMemsetD32Async`                                        | `hipMemsetD32Async`           |
| `cuMemsetD8`                                              | `hipMemsetD8`                 |
| `cuMemsetD8Async`                                         | `hipMemsetD8Async`            |
| `cuMipmappedArrayCreate`                                  |                               |
| `cuMipmappedArrayDestroy`                                 |                               |
| `cuMipmappedArrayGetLevel`                                |                               |

## **12. Virtual Memory Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cuMemAddressFree`                                        |                               | 10.2             |
| `cuMemAddressReserve`                                     |                               | 10.2             |
| `cuMemCreate`                                             |                               | 10.2             |
| `cuMemExportToShareableHandle`                            |                               | 10.2             |
| `cuMemGetAccess`                                          |                               | 10.2             |
| `cuMemGetAllocationGranularity`                           |                               | 10.2             |
| `cuMemGetAllocationPropertiesFromHandle`                  |                               | 10.2             |
| `cuMemImportFromShareableHandle`                          |                               | 10.2             |
| `cuMemMap`                                                |                               | 10.2             |
| `cuMemRelease`                                            |                               | 10.2             |
| `cuMemRetainAllocationHandle`                             |                               | 11.0             |
| `cuMemSetAccess`                                          |                               | 10.2             |
| `cuMemUnmap`                                              |                               | 10.2             |

## **13. Unified Addressing**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cuMemAdvise`                                             |                               | 8.0              |
| `cuMemPrefetchAsync`                                      |                               | 8.0              |
| `cuMemRangeGetAttribute`                                  |                               | 8.0              |
| `cuMemRangeGetAttributes`                                 |                               | 8.0              |
| `cuPointerGetAttribute`                                   |                               |
| `cuPointerGetAttributes`                                  |                               |
| `cuPointerSetAttribute`                                   |                               |

## **14. Stream Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cuStreamAddCallback`                                     | `hipStreamAddCallback`        |
| `cuStreamAttachMemAsync`                                  |                               |
| `cuStreamCopyAttributes`                                  |                               | 11.0             |
| `cuStreamCreate`                                          | `hipStreamCreateWithFlags`    |
| `cuStreamCreateWithPriority`                              | `hipStreamCreateWithPriority` |
| `cuStreamDestroy`                                         | `hipStreamDestroy`            |
| `cuStreamGetFlags`                                        | `hipStreamGetFlags`           |
| `cuStreamGetPriority`                                     | `hipStreamGetPriority`        |
| `cuStreamQuery`                                           | `hipStreamQuery`              |
| `cuStreamSetAttribute`                                    |                               | 11.0             |
| `cuStreamSynchronize`                                     | `hipStreamSynchronize`        |
| `cuStreamWaitEvent`                                       | `hipStreamWaitEvent`          |
| `cuStreamBeginCapture`                                    |                               | 10.0             |
| `cuStreamBeginCapture_ptsz`                               |                               | 10.1             |
| `cuStreamEndCapture`                                      |                               | 10.0             |
| `cuStreamGetAttribute`                                    |                               | 11.0             |
| `cuStreamGetCaptureInfo`                                  |                               | 10.1             |
| `cuStreamIsCapturing`                                     |                               | 10.0             |
| `cuThreadExchangeStreamCaptureMode`                       |                               | 10.1             |

## **15. Event Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuEventCreate`                                           | `hipEventCreateWithFlags`     |
| `cuEventDestroy`                                          | `hipEventDestroy`             |
| `cuEventElapsedTime`                                      | `hipEventElapsedTime`         |
| `cuEventQuery`                                            | `hipEventQuery`               |
| `cuEventRecord`                                           | `hipEventRecord`              |
| `cuEventSynchronize`                                      | `hipEventSynchronize`         |

## **16. External Resource Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cuSignalExternalSemaphoresAsync`                         |                               | 10.0             |
| `cuWaitExternalSemaphoresAsync`                           |                               | 10.0             |
| `cuImportExternalMemory`                                  |                               | 10.0             |
| `cuExternalMemoryGetMappedBuffer`                         |                               | 10.0             |
| `cuExternalMemoryGetMappedMipmappedArray`                 |                               | 10.0             |
| `cuDestroyExternalMemory`                                 |                               | 10.0             |
| `cuImportExternalSemaphore`                               |                               | 10.0             |
| `cuDestroyExternalSemaphore`                              |                               | 10.0             |

## **17. Stream Memory Operations**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cuStreamBatchMemOp`                                      |                               | 8.0              |
| `cuStreamWaitValue32`                                     |                               | 8.0              |
| `cuStreamWaitValue64`                                     |                               | 9.0              |
| `cuStreamWriteValue32`                                    |                               | 8.0              |
| `cuStreamWriteValue64`                                    |                               | 9.0              |

## **18. Execution Control**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cuFuncGetAttribute`                                      | `hipFuncGetAttribute`         |
| `cuFuncGetModule`                                         |                               | 11.0             |
| `cuFuncSetAttribute`                                      |                               | 9.0              |
| `cuFuncSetCacheConfig`                                    | `hipFuncSetCacheConfig`       |
| `cuFuncSetSharedMemConfig`                                |                               |
| `cuLaunchKernel`                                          | `hipModuleLaunchKernel`       |
| `cuLaunchHostFunc`                                        |                               | 10.0             |
| `cuLaunchCooperativeKernel`                               |                               | 9.0              |
| `cuLaunchCooperativeKernelMultiDevice`                    |                               | 9.0              |

## **19. Execution Control [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuFuncSetBlockShape`                                     |                               |
| `cuFuncSetSharedSize`                                     |                               |
| `cuLaunch`                                                |                               |
| `cuLaunchGrid`                                            |                               |
| `cuLaunchGridAsync`                                       |                               |
| `cuParamSetf`                                             |                               |
| `cuParamSeti`                                             |                               |
| `cuParamSetTexRef`                                        |                               |
| `cuParamSetv`                                             |                               |

## **20. Graph Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cuGraphCreate`                                           |                               | 10.0             |
| `cuGraphLaunch`                                           |                               | 10.0             |
| `cuGraphAddKernelNode`                                    |                               | 10.0             |
| `cuGraphKernelNodeGetParams`                              |                               | 10.0             |
| `cuGraphKernelNodeSetAttribute`                           |                               | 11.0             |
| `cuGraphKernelNodeSetParams`                              |                               | 10.0             |
| `cuGraphAddMemcpyNode`                                    |                               | 10.0             |
| `cuGraphMemcpyNodeGetParams`                              |                               | 10.0             |
| `cuGraphMemcpyNodeSetParams`                              |                               | 10.0             |
| `cuGraphAddMemsetNode`                                    |                               | 10.0             |
| `cuGraphMemsetNodeGetParams`                              |                               | 10.0             |
| `cuGraphMemsetNodeSetParams`                              |                               | 10.0             |
| `cuGraphAddHostNode`                                      |                               | 10.0             |
| `cuGraphHostNodeGetParams`                                |                               | 10.0             |
| `cuGraphHostNodeSetParams`                                |                               | 10.0             |
| `cuGraphAddChildGraphNode`                                |                               | 10.0             |
| `cuGraphChildGraphNodeGetGraph`                           |                               | 10.0             |
| `cuGraphAddEmptyNode`                                     |                               | 10.0             |
| `cuGraphClone`                                            |                               | 10.0             |
| `cuGraphNodeFindInClone`                                  |                               | 10.0             |
| `cuGraphNodeGetType`                                      |                               | 10.0             |
| `cuGraphGetNodes`                                         |                               | 10.0             |
| `cuGraphGetRootNodes`                                     |                               | 10.0             |
| `cuGraphGetEdges`                                         |                               | 10.0             |
| `cuGraphNodeGetDependencies`                              |                               | 10.0             |
| `cuGraphNodeGetDependentNodes`                            |                               | 10.0             |
| `cuGraphAddDependencies`                                  |                               | 10.0             |
| `cuGraphRemoveDependencies`                               |                               | 10.0             |
| `cuGraphDestroyNode`                                      |                               | 10.0             |
| `cuGraphInstantiate`                                      |                               | 10.0             |
| `cuGraphKernelNodeCopyAttributes`                         |                               | 11.0             |
| `cuGraphKernelNodeGetAttribute`                           |                               | 11.0             |
| `cuGraphExecDestroy`                                      |                               | 10.0             |
| `cuGraphExecKernelNodeSetParams`                          |                               | 10.1             |
| `cuGraphExecMemcpyNodeSetParams`                          |                               | 10.2             |
| `cuGraphExecMemsetNodeSetParams`                          |                               | 10.2             |
| `cuGraphExecHostNodeSetParams`                            |                               | 10.2             |
| `cuGraphExecUpdate`                                       |                               | 10.2             |
| `cuGraphDestroy`                                          |                               | 10.0             |

## **21. Occupancy**

|   **CUDA**                                                |   **HIP**                                                  |**CUDA version\***|
|-----------------------------------------------------------|------------------------------------------------------------|------------------|
| `cuOccupancyAvailableDynamicSMemPerBlock`                 |                                                            | 11.0             |
| `cuOccupancyMaxActiveBlocksPerMultiprocessor`             |`hipDrvOccupancyMaxActiveBlocksPerMultiprocessor`           |
| `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`    |`hipDrvOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`  |
| `cuOccupancyMaxPotentialBlockSize`                        |`hipOccupancyMaxPotentialBlockSize`                         |
| `cuOccupancyMaxPotentialBlockSizeWithFlags`               |                                                            |

## **22. Texture Reference Management [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cuTexRefGetAddress`                                      |`hipTexRefGetAddress`          |
| `cuTexRefGetAddressMode`                                  |`hipTexRefGetAddressMode`      |
| `cuTexRefGetArray`                                        |`hipTexRefGetArray`            |
| `cuTexRefGetBorderColor`                                  |                               | 8.0              |
| `cuTexRefGetFilterMode`                                   |                               |
| `cuTexRefGetFlags`                                        |                               |
| `cuTexRefGetFormat`                                       |                               |
| `cuTexRefGetMaxAnisotropy`                                |                               |
| `cuTexRefGetMipmapFilterMode`                             |                               |
| `cuTexRefGetMipmapLevelBias`                              |                               |
| `cuTexRefGetMipmapLevelClamp`                             |                               |
| `cuTexRefGetMipmappedArray`                               |                               |
| `cuTexRefSetAddress`                                      | `hipTexRefSetAddress`         |
| `cuTexRefSetAddress2D`                                    | `hipTexRefSetAddress2D`       |
| `cuTexRefSetAddressMode`                                  | `hipTexRefSetAddressMode`     |
| `cuTexRefSetArray`                                        | `hipTexRefSetArray`           |
| `cuTexRefSetBorderColor`                                  |                               | 8.0              |
| `cuTexRefSetFilterMode`                                   | `hipTexRefSetFilterMode`      |
| `cuTexRefSetFlags`                                        | `hipTexRefSetFlags`           |
| `cuTexRefSetFormat`                                       | `hipTexRefSetFormat`          |
| `cuTexRefSetMaxAnisotropy`                                |                               |
| `cuTexRefSetMipmapFilterMode`                             |                               |
| `cuTexRefSetMipmapLevelBias`                              |                               |
| `cuTexRefSetMipmapLevelClamp`                             |                               |
| `cuTexRefSetMipmappedArray`                               |                               |
| `cuTexRefCreate`                                          |                               |
| `cuTexRefDestroy`                                         |                               |

## **23. Surface Reference Management [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuSurfRefGetArray`                                       |                               |
| `cuSurfRefSetArray`                                       |                               |

## **24. Texture Object Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuTexObjectCreate`                                       |                               |
| `cuTexObjectDestroy`                                      |                               |
| `cuTexObjectGetResourceDesc`                              |                               |
| `cuTexObjectGetResourceViewDesc`                          |                               |
| `cuTexObjectGetTextureDesc`                               |                               |

## **25. Surface Object Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuSurfObjectCreate`                                      |                               |
| `cuSurfObjectDestroy`                                     |                               |
| `cuSurfObjectGetResourceDesc`                             |                               |

## **26. Peer Context Memory Access**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cuCtxEnablePeerAccess`                                   | `hipCtxEnablePeerAccess`      |
| `cuCtxDisablePeerAccess`                                  | `hipCtxDisablePeerAccess`     |
| `cuDeviceCanAccessPeer`                                   | `hipDeviceCanAccessPeer`      |
| `cuDeviceGetP2PAttribute`                                 |                               | 8.0              |

## **27. Graphics Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuGraphicsMapResources`                                  |                               |
| `cuGraphicsResourceGetMappedMipmappedArray`               |                               |
| `cuGraphicsResourceGetMappedPointer`                      |                               |
| `cuGraphicsResourceSetMapFlags`                           |                               |
| `cuGraphicsSubResourceGetMappedArray`                     |                               |
| `cuGraphicsUnmapResources`                                |                               |
| `cuGraphicsUnregisterResource`                            |                               |

## **28. Profiler Control**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuProfilerInitialize`                                    |                               |
| `cuProfilerStart`                                         | `hipProfilerStart`            |
| `cuProfilerStop`                                          | `hipProfilerStop`             |

## **29. OpenGL Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuGLGetDevices`                                          |                               |
| `cuGraphicsGLRegisterBuffer`                              |                               |
| `cuGraphicsGLRegisterImage`                               |                               |
| `cuWGLGetDevice`                                          |                               |

## **29.1. OpenGL Interoperability [DEPRECATED]**
|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuGLCtxCreate`                                           |                               |
| `cuGLInit`                                                |                               |
| `cuGLMapBufferObject`                                     |                               |
| `cuGLMapBufferObjectAsync`                                |                               |
| `cuGLRegisterBufferObject`                                |                               |
| `cuGLSetBufferObjectMapFlags`                             |                               |
| `cuGLUnmapBufferObject`                                   |                               |
| `cuGLUnmapBufferObjectAsync`                              |                               |
| `cuGLUnregisterBufferObject`                              |                               |

## **30. Direct3D 9 Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuD3D9CtxCreate`                                         |                               |
| `cuD3D9CtxCreateOnDevice`                                 |                               |
| `cuD3D9GetDevice`                                         |                               |
| `cuD3D9GetDevices`                                        |                               |
| `cuD3D9GetDirect3DDevice`                                 |                               |
| `cuGraphicsD3D9RegisterResource`                          |                               |

## **30.1. Direct3D 9 Interoperability [DEPRECATED]**
|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuD3D9MapResources`                                      |                               |
| `cuD3D9RegisterResource`                                  |                               |
| `cuD3D9ResourceGetMappedArray`                            |                               |
| `cuD3D9ResourceGetMappedPitch`                            |                               |
| `cuD3D9ResourceGetMappedPointer`                          |                               |
| `cuD3D9ResourceGetMappedSize`                             |                               |
| `cuD3D9ResourceGetSurfaceDimensions`                      |                               |
| `cuD3D9ResourceSetMapFlags`                               |                               |
| `cuD3D9UnmapResources`                                    |                               |
| `cuD3D9UnregisterResource`                                |                               |

## **31. Direct3D 10 Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuD3D10GetDevice`                                        |                               |
| `cuD3D10GetDevices`                                       |                               |
| `cuGraphicsD3D10RegisterResource`                         |                               |

## **31.1. Direct3D 10 Interoperability [DEPRECATED]**
|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuD3D10CtxCreate`                                        |                               |
| `cuD3D10CtxCreateOnDevice`                                |                               |
| `cuD3D10GetDirect3DDevice`                                |                               |
| `cuD3D10MapResources`                                     |                               |
| `cuD3D10RegisterResource`                                 |                               |
| `cuD3D10ResourceGetMappedArray`                           |                               |
| `cuD3D10ResourceGetMappedPitch`                           |                               |
| `cuD3D10ResourceGetMappedPointer`                         |                               |
| `cuD3D10ResourceGetMappedSize`                            |                               |
| `cuD3D10ResourceGetSurfaceDimensions`                     |                               |
| `cuD3D10ResourceSetMapFlags`                              |                               |
| `cuD3D10UnmapResources`                                   |                               |
| `cuD3D10UnregisterResource`                               |                               |

## **32. Direct3D 11 Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuD3D11GetDevice`                                        |                               |
| `cuD3D11GetDevices`                                       |                               |
| `cuGraphicsD3D11RegisterResource`                         |                               |

## **32.1. Direct3D 11 Interoperability [DEPRECATED]**
|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuD3D11CtxCreate`                                        |                               |
| `cuD3D11CtxCreateOnDevice`                                |                               |
| `cuD3D11GetDirect3DDevice`                                |                               |

## **33. VDPAU Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|------------------|
| `cuGraphicsVDPAURegisterOutputSurface`                    |                               |
| `cuGraphicsVDPAURegisterVideoSurface`                     |                               |
| `cuVDPAUCtxCreate`                                        |                               |
| `cuVDPAUGetDevice`                                        |                               |

## **34. EGL Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cuEGLStreamConsumerAcquireFrame`                         |                               | 8.0              |
| `cuEGLStreamConsumerConnect`                              |                               | 8.0              |
| `cuEGLStreamConsumerConnectWithFlags`                     |                               | 9.1              |
| `cuEGLStreamConsumerDisconnect`                           |                               | 9.1              |
| `cuEGLStreamConsumerReleaseFrame`                         |                               | 9.1              |
| `cuEGLStreamProducerConnect`                              |                               | 9.1              |
| `cuEGLStreamProducerDisconnect`                           |                               | 9.1              |
| `cuEGLStreamProducerPresentFrame`                         |                               | 9.1              |
| `cuEGLStreamProducerReturnFrame`                          |                               | 9.1              |
| `cuGraphicsEGLRegisterImage`                              |                               | 9.1              |
| `cuGraphicsResourceGetMappedEglFrame`                     |                               | 9.1              |
| `cuEventCreateFromEGLSync`                                |                               | 9.0              |

\* CUDA version, in which API has appeared and (optional) last version before abandoning it; no value in case of earlier versions < 7.5.
