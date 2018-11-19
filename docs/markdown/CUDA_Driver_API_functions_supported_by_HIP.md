# CUDA Driver API functions supported by HIP

## **1. Data types used by CUDA driver**

| **type**     |   **CUDA**                                                         |   **HIP**                                                  |
|-------------:|--------------------------------------------------------------------|------------------------------------------------------------|
| struct       |`CUDA_ARRAY3D_DESCRIPTOR`                                           |                                                            |
| typedef      |`CUDA_ARRAY3D_DESCRIPTOR_st`                                        |                                                            |
| struct       |`CUDA_ARRAY_DESCRIPTOR`                                             |`HIP_ARRAY_DESCRIPTOR`                                      |
| typedef      |`CUDA_ARRAY_DESCRIPTOR_st`                                          |`HIP_ARRAY_DESCRIPTOR`                                      |
| struct       |`CUDA_MEMCPY2D`                                                     |`hip_Memcpy2D`                                              |
| typedef      |`CUDA_MEMCPY2D_st`                                                  |`hip_Memcpy2D`                                              |
| struct       |`CUDA_MEMCPY3D`                                                     |                                                            |
| typedef      |`CUDA_MEMCPY3D_st`                                                  |                                                            |
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
| union        |`CUstreamBatchMemOpParams`                                          |                                                            |
| typedef      |`CUstreamBatchMemOpParams_union`                                    |                                                            |
| enum         |***`CUaddress_mode`***                                              |                                                            |
| typedef      |***`CUaddress_mode_enum`***                                         |                                                            |
|            0 |*`CU_TR_ADDRESS_MODE_WRAP`*                                         |                                                            |
|            1 |*`CU_TR_ADDRESS_MODE_CLAMP`*                                        |                                                            |
|            2 |*`CU_TR_ADDRESS_MODE_MIRROR`*                                       |                                                            |
|            3 |*`CU_TR_ADDRESS_MODE_BORDER`*                                       |                                                            |
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
|           11 |*`CU_DEVICE_ATTRIBUTE_MAX_PITCH`*                                   |                                                            |
|           12 |*`CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK`*                     |*`hipDeviceAttributeMaxRegistersPerBlock`*                  |
|           12 |*`CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK`*                         |*`hipDeviceAttributeMaxRegistersPerBlock`*                  |
|           13 |*`CU_DEVICE_ATTRIBUTE_CLOCK_RATE`*                                  |*`hipDeviceAttributeClockRate`*                             |
|           14 |*`CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT`*                           |                                                            |
|           15 |*`CU_DEVICE_ATTRIBUTE_GPU_OVERLAP`*                                 |                                                            |
|           16 |*`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`*                        |*`hipDeviceAttributeMultiprocessorCount`*                   |
|           17 |*`CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT`*                         |                                                            |
|           18 |*`CU_DEVICE_ATTRIBUTE_INTEGRATED`*                                  |*`hipDeviceAttributeIntegrated`*                            |
|           19 |*`CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY`*                         |                                                            |
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
|           32 |*`CU_DEVICE_ATTRIBUTE_ECC_ENABLED`*                                 |                                                            |
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
|           86 |*`CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED`*                |                                                            |
|           87 |*`CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO`*       |                                                            |
|           88 |*`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`*                      |                                                            |
|           89 |*`CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`*                   |                                                            |
|           90 |*`CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED`*                |                                                            |
|           91 |*`CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM`*     |                                                            |
|           92 |*`CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS`*                      |                                                            |
|           93 |*`CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS`*               |                                                            |
|           94 |*`CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR`*               |                                                            |
|           95 |*`CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`*                          |                                                            |
|           96 |*`CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH`*             |                                                            |
|           97 |*`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`*           |                                                            |
|           98 |*`CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES`*                     |                                                            |
|           99 |*`CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED`*                     |                                                            |
|          100 |*`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`*|                                                            |
|          101 |*`CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST`*         |                                                            |
|          102 |*`CU_DEVICE_ATTRIBUTE_MAX`*                                         |                                                            |
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
| enum         |***`CUfunction_attribute`***                                        |                                                            |
| typedef      |***`CUfunction_attribute_enum`***                                   |                                                            |
|            0 |*`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`*                         |                                                            |
|            1 |*`CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`*                             |                                                            |
|            2 |*`CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES`*                              |                                                            |
|            3 |*`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`*                              |                                                            |
|            4 |*`CU_FUNC_ATTRIBUTE_NUM_REGS`*                                      |                                                            |
|            5 |*`CU_FUNC_ATTRIBUTE_PTX_VERSION`*                                   |                                                            |
|            6 |*`CU_FUNC_ATTRIBUTE_BINARY_VERSION`*                                |                                                            |
|            7 |*`CU_FUNC_ATTRIBUTE_CACHE_MODE_CA`*                                 |                                                            |
|            8 |*`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`*                 |                                                            |
|            9 |*`CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`*              |                                                            |
|           10 |*`CU_FUNC_ATTRIBUTE_MAX`*                                           |                                                            |
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
|              |*`CU_JIT_NEW_SM3X_OPT`*                                             |                                                            |
|              |*`CU_JIT_FAST_COMPILE`*                                             |                                                            |
|              |*`CU_JIT_GLOBAL_SYMBOL_NAMES`*                                      |                                                            |
|              |*`CU_JIT_GLOBAL_SYMBOL_ADDRESSES`*                                  |                                                            |
|              |*`CU_JIT_GLOBAL_SYMBOL_COUNT`*                                      |                                                            |
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
|           53 |*`CU_TARGET_COMPUTE_53`*                                            |                                                            |
|           60 |*`CU_TARGET_COMPUTE_60`*                                            |                                                            |
|           61 |*`CU_TARGET_COMPUTE_61`*                                            |                                                            |
|           62 |*`CU_TARGET_COMPUTE_62`*                                            |                                                            |
|           70 |*`CU_TARGET_COMPUTE_70`*                                            |                                                            |
|           73 |*`CU_TARGET_COMPUTE_73`*                                            |                                                            |
|           75 |*`CU_TARGET_COMPUTE_75`*                                            |                                                            |
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
|         0x05 |*`CU_LIMIT_MAX_L2_FETCH_GRANULARITY`*                               |                                                            |
|              |*`CU_LIMIT_MAX`*                                                    |                                                            |
| enum         |***`CUmem_advise`***                                                |                                                            |
| typedef      |***`CUmem_advise_enum`***                                           |                                                            |
|            1 |*`CU_MEM_ADVISE_SET_READ_MOSTLY`*                                   |                                                            |
|            2 |*`CU_MEM_ADVISE_UNSET_READ_MOSTLY`*                                 |                                                            |
|            3 |*`CU_MEM_ADVISE_SET_PREFERRED_LOCATION`*                            |                                                            |
|            4 |*`CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION`*                          |                                                            |
|            5 |*`CU_MEM_ADVISE_SET_ACCESSED_BY`*                                   |                                                            |
|            6 |*`CU_MEM_ADVISE_UNSET_ACCESSED_BY`*                                 |                                                            |
| enum         |***`CUmemAttach_flags`***                                           |                                                            |
| typedef      |***`CUmemAttach_flags_enum`***                                      |                                                            |
|          0x1 |*`CU_MEM_ATTACH_GLOBAL`*                                            |                                                            |
|          0x2 |*`CU_MEM_ATTACH_HOST`*                                              |                                                            |
|          0x4 |*`CU_MEM_ATTACH_SINGLE`*                                            |                                                            |
| enum         |***`CUmemorytype`***                                                |*`hipMemoryType`*                                           |
| typedef      |***`CUmemorytype_enum`***                                           |*`hipMemoryType`*                                           |
|         0x01 |*`CU_MEMORYTYPE_HOST`*                                              |*`hipMemoryTypeHost`*                                       |
|         0x02 |*`CU_MEMORYTYPE_DEVICE`*                                            |*`hipMemoryTypeDevice`*                                     |
|         0x03 |*`CU_MEMORYTYPE_ARRAY`*                                             |*`hipMemoryTypeArray`*                                      |
|         0x04 |*`CU_MEMORYTYPE_UNIFIED`*                                           |*`hipMemoryTypeUnified`*                                    |
| enum         |***`CUmem_range_attribute`***                                       |                                                            |
| typedef      |***`CUmem_range_attribute_enum`***                                  |                                                            |
|            1 |*`CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY`*                              |                                                            |
|            2 |*`CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION`*                       |                                                            |
|            3 |*`CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY`*                              |                                                            |
|            4 |*`CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION`*                   |                                                            |
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
|            9 |*`CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL`*                             |                                                            |
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
|            2 |*`CUDA_ERROR_OUT_OF_MEMORY`*                                        |*`hipErrorMemoryAllocation`*                                |
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
|          220 |*`CUDA_ERROR_NVLINK_UNCORRECTABLE`*                                 |                                                            |
|          221 |*`CUDA_ERROR_JIT_COMPILER_NOT_FOUND`*                               |                                                            |
|          300 |*`CUDA_ERROR_INVALID_SOURCE`*                                       |*`hipErrorInvalidSource`*                                   |
|          301 |*`CUDA_ERROR_FILE_NOT_FOUND`*                                       |*`hipErrorFileNotFound`*                                    |
|          302 |*`CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND`*                       |*`hipErrorSharedObjectSymbolNotFound`*                      |
|          303 |*`CUDA_ERROR_SHARED_OBJECT_INIT_FAILED`*                            |*`hipErrorSharedObjectInitFailed`*                          |
|          304 |*`CUDA_ERROR_OPERATING_SYSTEM`*                                     |*`hipErrorOperatingSystem`*                                 |
|          400 |*`CUDA_ERROR_INVALID_HANDLE`*                                       |*`hipErrorInvalidResourceHandle`*                           |
|          401 |*`CUDA_ERROR_ILLEGAL_STATE`*                                        |                                                            |
|          500 |*`CUDA_ERROR_NOT_FOUND`*                                            |*`hipErrorNotFound`*                                        |
|          600 |*`CUDA_ERROR_NOT_READY`*                                            |*`hipErrorNotReady`*                                        |
|          700 |*`CUDA_ERROR_ILLEGAL_ADDRESS`*                                      |*`hipErrorIllegalAddress`*                                  |
|          701 |*`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`*                              |*`hipErrorLaunchOutOfResources`*                            |
|          702 |*`CUDA_ERROR_LAUNCH_TIMEOUT`*                                       |*`hipErrorLaunchTimeOut`*                                   |
|          703 |*`CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING`*                        |                                                            |
|          704 |*`CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED`*                          |*`hipErrorPeerAccessAlreadyEnabled`*                        |
|          705 |*`CUDA_ERROR_PEER_ACCESS_NOT_ENABLED`*                              |*`hipErrorPeerAccessNotEnabled`*                            |
|          708 |*`CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE`*                               |                                                            |
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
|          800 |*`CUDA_ERROR_NOT_PERMITTED`*                                        |                                                            |
|          801 |*`CUDA_ERROR_NOT_SUPPORTED`*                                        |                                                            |
|          802 |*`CUDA_ERROR_SYSTEM_NOT_READY`*                                     |                                                            |
|          900 |*`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`*                           |                                                            |
|          901 |*`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`*                           |                                                            |
|          902 |*`CUDA_ERROR_STREAM_CAPTURE_MERGE`*                                 |                                                            |
|          903 |*`CUDA_ERROR_STREAM_CAPTURE_UNMATCHED`*                             |                                                            |
|          904 |*`CUDA_ERROR_STREAM_CAPTURE_UNJOINED`*                              |                                                            |
|          905 |*`CUDA_ERROR_STREAM_CAPTURE_ISOLATION`*                             |                                                            |
|          906 |*`CUDA_ERROR_STREAM_CAPTURE_IMPLICIT`*                              |                                                            |
|          907 |*`CUDA_ERROR_CAPTURED_EVENT`*                                       |                                                            |
|          999 |*`CUDA_ERROR_UNKNOWN`*                                              |                                                            |
| enum         |***`CUsharedconfig`***                                              |***`hipSharedMemConfig`***                                  |
| typedef      |***`CUsharedconfig_enum`***                                         |***`hipSharedMemConfig`***                                  |
|         0x00 |*`CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE`*                          |*`hipSharedMemBankSizeDefault`*                             |
|         0x01 |*`CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE`*                        |*`hipSharedMemBankSizeFourByte`*                            |
|         0x02 |*`CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE`*                       |*`hipSharedMemBankSizeEightByte`*                           |
| enum         |***`CUshared_carveout`***                                           |                                                            |
| typedef      |`CUshared_carveout_enum`                                            |                                                            |
|           -1 |*`CU_SHAREDMEM_CARVEOUT_DEFAULT`*                                   |                                                            |
|          100 |*`CU_SHAREDMEM_CARVEOUT_MAX_SHARED`*                                |                                                            |
|            0 |*`CU_SHAREDMEM_CARVEOUT_MAX_L1`*                                    |                                                            |
| enum         |***`CUstream_flags`***                                              |                                                            |
| typedef      |***`CUstream_flags_enum`***                                         |                                                            |
|          0x0 |*`CU_STREAM_DEFAULT`*                                               |*`hipStreamDefault`*                                        |
|          0x1 |*`CU_STREAM_NON_BLOCKING`*                                          |*`hipStreamNonBlocking`*                                    |
| enum         |***`CUstreamBatchMemOpType`***                                      |                                                            |
| typedef      |***`CUstreamBatchMemOpType_enum`***                                 |                                                            |
|            1 |*`CU_STREAM_MEM_OP_WAIT_VALUE_32`*                                  |                                                            |
|            2 |*`CU_STREAM_MEM_OP_WRITE_VALUE_32`*                                 |                                                            |
|            3 |*`CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES`*                            |                                                            |
|            4 |*`CU_STREAM_MEM_OP_WAIT_VALUE_64`*                                  |                                                            |
|            5 |*`CU_STREAM_MEM_OP_WRITE_VALUE_64`*                                 |                                                            |
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
| struct       |`CUeglStreamConnection_st`                                          |                                                            |
| typedef      |`CUeglStreamConnection`                                             |                                                            |
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
| define       |`CU_MEMHOSTALLOC_WRITECOMBINED`                                     |`hipHostAllocWriteCombined`                                 |
| define       |`CU_MEMHOSTREGISTER_DEVICEMAP`                                      |`hipHostRegisterMapped`                                     |
| define       |`CU_MEMHOSTREGISTER_IOMEMORY`                                       |`hipHostRegisterIoMemory`                                   |
| define       |`CU_MEMHOSTREGISTER_PORTABLE`                                       |`hipHostRegisterPortable`                                   |
| define       |`CU_PARAM_TR_DEFAULT`                                               |                                                            |
| define       |`CU_STREAM_LEGACY`                                                  |                                                            |
| define       |`CU_STREAM_PER_THREAD`                                              |                                                            |
| define       |`CU_TRSA_OVERRIDE_FORMAT`                                           |                                                            |
| define       |`CU_TRSF_NORMALIZED_COORDINATES`                                    |                                                            |
| define       |`CU_TRSF_READ_AS_INTEGER`                                           |                                                            |
| define       |`CU_TRSF_SRGB`                                                      |                                                            |
| define       |`CUDA_ARRAY3D_2DARRAY`                                              |                                                            |
| define       |`CUDA_ARRAY3D_CUBEMAP`                                              |`hipArrayCubemap`                                           |
| define       |`CUDA_ARRAY3D_DEPTH_TEXTURE`                                        |                                                            |
| define       |`CUDA_ARRAY3D_LAYERED`                                              |`hipArrayLayered`                                           |
| define       |`CUDA_ARRAY3D_SURFACE_LDST`                                         |`hipArraySurfaceLoadStore`                                  |
| define       |`CUDA_ARRAY3D_TEXTURE_GATHER`                                       |`hipArrayTextureGather`                                     |
| define       |`CUDA_ARRAY3D_COLOR_ATTACHMENT`                                     |                                                            |
| define       |`CUDA_VERSION`                                                      |                                                            |
| typedef      |`CUexternalMemory`                                                  |                                                            |
| struct       |`CUextMemory_st`                                                    |                                                            |
| typedef      |`CUexternalSemaphore`                                               |                                                            |
| struct       |`CUextSemaphore_st`                                                 |                                                            |
| typedef      |`CUgraph`                                                           |                                                            |
| struct       |`CUgraph_st`                                                        |                                                            |
| typedef      |`CUgraphNode`                                                       |                                                            |
| struct       |`CUgraphNode_st`                                                    |                                                            |
| typedef      |`CUgraphExec`                                                       |                                                            |
| struct       |`CUgraphExec_st`                                                    |                                                            |
| typedef      |`CUhostFn`                                                          |                                                            |
| typedef      |`CUoccupancyB2DSize`                                                |                                                            |
| struct       |`CUDA_KERNEL_NODE_PARAMS`                                           |                                                            |
| typedef      |`CUDA_KERNEL_NODE_PARAMS_st`                                        |                                                            |
| struct       |`CUDA_LAUNCH_PARAMS`                                                |                                                            |
| typedef      |`CUDA_LAUNCH_PARAMS_st`                                             |                                                            |
| struct       |`CUDA_MEMSET_NODE_PARAMS`                                           |                                                            |
| typedef      |`CUDA_MEMSET_NODE_PARAMS_st`                                        |                                                            |
| struct       |`CUDA_HOST_NODE_PARAMS`                                             |                                                            |
| typedef      |`CUDA_HOST_NODE_PARAMS_st`                                          |                                                            |
| enum         |***`CUgraphNodeType`***                                             |                                                            |
| typedef      |***`CUgraphNodeType_enum`***                                        |                                                            |
|            0 |*`CU_GRAPH_NODE_TYPE_KERNEL`*                                       |                                                            |
|            1 |*`CU_GRAPH_NODE_TYPE_MEMCPY`*                                       |                                                            |
|            2 |*`CU_GRAPH_NODE_TYPE_MEMSET`*                                       |                                                            |
|            3 |*`CU_GRAPH_NODE_TYPE_HOST`*                                         |                                                            |
|            4 |*`CU_GRAPH_NODE_TYPE_GRAPH`*                                        |                                                            |
|            5 |*`CU_GRAPH_NODE_TYPE_EMPTY`*                                        |                                                            |
|            6 |*`CU_GRAPH_NODE_TYPE_COUNT`*                                        |                                                            |
| enum         |***`CUstreamCaptureStatus`***                                       |                                                            |
| typedef      |***`CUstreamCaptureStatus_enum`***                                  |                                                            |
|            0 |*`CU_STREAM_CAPTURE_STATUS_NONE`*                                   |                                                            |
|            1 |*`CU_STREAM_CAPTURE_STATUS_ACTIVE`*                                 |                                                            |
|            2 |*`CU_STREAM_CAPTURE_STATUS_INVALIDATED`*                            |                                                            |
| enum         |***`CUstreamWaitValue_flags`***                                     |                                                            |
| typedef      |***`CUstreamWaitValue_flags_enum`***                                |                                                            |
|          0x0 |*`CU_STREAM_WAIT_VALUE_GEQ`*                                        |                                                            |
|          0x1 |*`CU_STREAM_WAIT_VALUE_EQ`*                                         |                                                            |
|          0x2 |*`CU_STREAM_WAIT_VALUE_AND`*                                        |                                                            |
|        1<<30 |*`CU_STREAM_WAIT_VALUE_FLUSH`*                                      |                                                            |
| enum         |***`CUstreamWriteValue_flags`***                                    |                                                            |
| typedef      |***`CUstreamWriteValue_flags_enum`***                               |                                                            |
|          0x0 |*`CU_STREAM_WRITE_VALUE_DEFAULT`*                                   |                                                            |
|          0x1 |*`CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER`*                         |                                                            |
| enum         |***`CUdevice_P2PAttribute`***                                       |                                                            |
| typedef      |***`CUdevice_P2PAttribute_enum`***                                  |                                                            |
|         0x01 |*`CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK`*                        |                                                            |
|         0x02 |*`CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED`*                        |                                                            |
|         0x03 |*`CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED`*                 |                                                            |
|         0x04 |*`CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED`*           |                                                            |
|         0x04 |*`CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED`*             |                                                            |
| enum         |***`CUeglColorFormat`***                                            |                                                            |
| typedef      |***`CUeglColorFormate_enum`***                                      |                                                            |
|         0x00 |*`CU_EGL_COLOR_FORMAT_YUV420_PLANAR`*                               |                                                            |
|         0x01 |*`CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR`*                           |                                                            |
|         0x02 |*`CU_EGL_COLOR_FORMAT_YUV422_PLANAR`*                               |                                                            |
|         0x03 |*`CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR`*                           |                                                            |
|         0x04 |*`CU_EGL_COLOR_FORMAT_RGB`*                                         |                                                            |
|         0x05 |*`CU_EGL_COLOR_FORMAT_BGR`*                                         |                                                            |
|         0x06 |*`CU_EGL_COLOR_FORMAT_ARGB`*                                        |                                                            |
|         0x07 |*`CU_EGL_COLOR_FORMAT_RGBA`*                                        |                                                            |
|         0x08 |*`CU_EGL_COLOR_FORMAT_L`*                                           |                                                            |
|         0x09 |*`CU_EGL_COLOR_FORMAT_R`*                                           |                                                            |
|         0x0A |*`CU_EGL_COLOR_FORMAT_YUV444_PLANAR`*                               |                                                            |
|         0x0B |*`CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR`*                           |                                                            |
|         0x0C |*`CU_EGL_COLOR_FORMAT_YUYV_422`*                                    |                                                            |
|         0x0D |*`CU_EGL_COLOR_FORMAT_UYVY_422`*                                    |                                                            |
|         0x0E |*`CU_EGL_COLOR_FORMAT_ABGR`*                                        |                                                            |
|         0x0F |*`CU_EGL_COLOR_FORMAT_BGRA`*                                        |                                                            |
|         0x10 |*`CU_EGL_COLOR_FORMAT_A`*                                           |                                                            |
|         0x11 |*`CU_EGL_COLOR_FORMAT_RG`*                                          |                                                            |
|         0x12 |*`CU_EGL_COLOR_FORMAT_AYUV`*                                        |                                                            |
|         0x13 |*`CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR`*                           |                                                            |
|         0x14 |*`CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR`*                           |                                                            |
|         0x15 |*`CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR`*                           |                                                            |
|         0x16 |*`CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR`*                    |                                                            |
|         0x17 |*`CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR`*                    |                                                            |
|         0x18 |*`CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR`*                    |                                                            |
|         0x19 |*`CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR`*                    |                                                            |
|         0x1A |*`CU_EGL_COLOR_FORMAT_VYUY_ER`*                                     |                                                            |
|         0x1B |*`CU_EGL_COLOR_FORMAT_UYVY_ER`*                                     |                                                            |
|         0x1C |*`CU_EGL_COLOR_FORMAT_YUYV_ER`*                                     |                                                            |
|         0x1D |*`CU_EGL_COLOR_FORMAT_YVYU_ER`*                                     |                                                            |
|         0x1E |*`CU_EGL_COLOR_FORMAT_YUV_ER`*                                      |                                                            |
|         0x1F |*`CU_EGL_COLOR_FORMAT_YUVA_ER`*                                     |                                                            |
|         0x20 |*`CU_EGL_COLOR_FORMAT_AYUV_ER`*                                     |                                                            |
|         0x21 |*`CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER`*                            |                                                            |
|         0x22 |*`CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER`*                            |                                                            |
|         0x23 |*`CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER`*                            |                                                            |
|         0x24 |*`CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER`*                        |                                                            |
|         0x25 |*`CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER`*                        |                                                            |
|         0x26 |*`CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER`*                        |                                                            |
|         0x27 |*`CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER`*                            |                                                            |
|         0x28 |*`CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER`*                            |                                                            |
|         0x29 |*`CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER`*                            |                                                            |
|         0x2A |*`CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER`*                        |                                                            |
|         0x2B |*`CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER`*                        |                                                            |
|         0x2C |*`CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER`*                        |                                                            |
|         0x2D |*`CU_EGL_COLOR_FORMAT_BAYER_RGGB`*                                  |                                                            |
|         0x2E |*`CU_EGL_COLOR_FORMAT_BAYER_BGGR`*                                  |                                                            |
|         0x2F |*`CU_EGL_COLOR_FORMAT_BAYER_GRBG`*                                  |                                                            |
|         0x30 |*`CU_EGL_COLOR_FORMAT_BAYER_GBRG`*                                  |                                                            |
|         0x31 |*`CU_EGL_COLOR_FORMAT_BAYER10_RGGB`*                                |                                                            |
|         0x32 |*`CU_EGL_COLOR_FORMAT_BAYER10_BGGR`*                                |                                                            |
|         0x33 |*`CU_EGL_COLOR_FORMAT_BAYER10_GRBG`*                                |                                                            |
|         0x34 |*`CU_EGL_COLOR_FORMAT_BAYER10_GBRG`*                                |                                                            |
|         0x35 |*`CU_EGL_COLOR_FORMAT_BAYER12_RGGB`*                                |                                                            |
|         0x36 |*`CU_EGL_COLOR_FORMAT_BAYER12_BGGR`*                                |                                                            |
|         0x37 |*`CU_EGL_COLOR_FORMAT_BAYER12_GRBG`*                                |                                                            |
|         0x38 |*`CU_EGL_COLOR_FORMAT_BAYER12_GBRG`*                                |                                                            |
|         0x39 |*`CU_EGL_COLOR_FORMAT_BAYER14_RGGB`*                                |                                                            |
|         0x3A |*`CU_EGL_COLOR_FORMAT_BAYER14_BGGR`*                                |                                                            |
|         0x3B |*`CU_EGL_COLOR_FORMAT_BAYER14_GRBG`*                                |                                                            |
|         0x3C |*`CU_EGL_COLOR_FORMAT_BAYER14_GBRG`*                                |                                                            |
|         0x3D |*`CU_EGL_COLOR_FORMAT_BAYER20_RGGB`*                                |                                                            |
|         0x3E |*`CU_EGL_COLOR_FORMAT_BAYER20_BGGR`*                                |                                                            |
|         0x3F |*`CU_EGL_COLOR_FORMAT_BAYER20_GRBG`*                                |                                                            |
|         0x40 |*`CU_EGL_COLOR_FORMAT_BAYER20_GBRG`*                                |                                                            |
|         0x41 |*`CU_EGL_COLOR_FORMAT_YVU444_PLANAR`*                               |                                                            |
|         0x42 |*`CU_EGL_COLOR_FORMAT_YVU422_PLANAR`*                               |                                                            |
|         0x43 |*`CU_EGL_COLOR_FORMAT_YVU420_PLANAR`*                               |                                                            |
|         0x44 |*`CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB`*                              |                                                            |
|         0x45 |*`CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR`*                              |                                                            |
|         0x46 |*`CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG`*                              |                                                            |
|         0x47 |*`CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG`*                              |                                                            |
|         0x48 |*`CU_EGL_COLOR_FORMAT_MAX`*                                         |                                                            |
| enum         |***`CUeglFrameType`***                                              |                                                            |
| typedef      |***`CUeglFrameType_enum`***                                         |                                                            |
|            0 |*`CU_EGL_FRAME_TYPE_ARRAY`*                                         |                                                            |
|            1 |*`CU_EGL_FRAME_TYPE_PITCH`*                                         |                                                            |
| enum         |***`CUeglResourceLocationFlags`***                                  |                                                            |
| typedef      |***`CUeglResourceLocationFlags_enum`***                             |                                                            |
|         0x00 |*`CU_EGL_RESOURCE_LOCATION_SYSMEM`*                                 |                                                            |
|         0x01 |*`CU_EGL_RESOURCE_LOCATION_VIDMEM`*                                 |                                                            |
| enum         |***`CUexternalMemoryHandleType`***                                  |                                                            |
| typedef      |***`CUexternalMemoryHandleType_enum`***                             |                                                            |
|            1 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD`*                        |                                                            |
|            2 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32`*                     |                                                            |
|            3 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT`*                 |                                                            |
|            4 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP`*                       |                                                            |
|            5 |*`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE`*                   |                                                            |
| define       |`CUDA_EXTERNAL_MEMORY_DEDICATED`                                    |                                                            |
| struct       |`CUDA_EXTERNAL_MEMORY_HANDLE_DESC`                                  |                                                            |
| typedef      |`CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st`                               |                                                            |
| struct       |`CUDA_EXTERNAL_MEMORY_BUFFER_DESC`                                  |                                                            |
| typedef      |`CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st`                               |                                                            |
| struct       |`CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC`                         |                                                            |
| typedef      |`CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st`                      |                                                            |
| enum         |***`CUexternalSemaphoreHandleType`***                               |                                                            |
| typedef      |***`CUexternalSemaphoreHandleType_enum`***                          |                                                            |
|            1 |*`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD`*                     |                                                            |
|            2 |*`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32`*                  |                                                            |
|            3 |*`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT`*              |                                                            |
|            4 |*`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE`*                   |                                                            |
| struct       |`CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC`                               |                                                            |
| typedef      |`CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st`                            |                                                            |
| struct       |`CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS`                             |                                                            |
| typedef      |`CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st`                          |                                                            |
| struct       |`CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS`                               |                                                            |
| typedef      |`CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st`                            |                                                            |
| define       |`CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC`           |                                                            |
| define       |`CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC`          |                                                            |
| define       |`__CUDACC__`                                                        |`__HIPCC__`                                                 |
| define       |`CUDA_CB`                                                           |                                                            |
| define       |`CU_DEVICE_CPU`                                                     |                                                            |
| define       |`CU_DEVICE_INVALID`                                                 |                                                            |
| struct       |`CUuuid`                                                            |                                                            |
| typedef      |`CUuuid_st`                                                         |                                                            |

## **2. Error Handling**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuGetErrorName`                                          |                               |
| `cuGetErrorString`                                        |                               |

## **3. Initialization**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuInit`                                                  | `hipInit`                     |

## **4. Version Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuDriverGetVersion`                                      | `hipDriverGetVersion`         |

## **5. Device Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuDriverGetVersion`                                      | `hipGetDevice`                |
| `cuDeviceGetAttribute`                                    | `hipDeviceGetAttribute`       |
| `cuDeviceGetCount`                                        | `hipGetDeviceCount`           |
| `cuDeviceGetName`                                         | `hipDeviceGetName`            |
| `cuDeviceTotalMem`                                        | `hipDeviceTotalMem`           |
| `cuDeviceGetLuid`                                         |                               |
| `cuDeviceGetUuid`                                         |                               |

## **6. Device Management [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuDeviceComputeCapability`                               | `hipDeviceComputeCapability`  |
| `cuDeviceGetProperties`                                   | `hipGetDeviceProperties`      |

## **7. Primary Context Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuDevicePrimaryCtxGetState`                              | `hipDevicePrimaryCtxGetState` |
| `cuDevicePrimaryCtxRelease`                               | `hipDevicePrimaryCtxRelease`  |
| `cuDevicePrimaryCtxReset`                                 | `hipDevicePrimaryCtxReset`    |
| `cuDevicePrimaryCtxRetain`                                | `hipDevicePrimaryCtxRetain`   |
| `cuDevicePrimaryCtxSetFlags`                              | `hipDevicePrimaryCtxSetFlags` |

## **8. Context Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuCtxCreate`                                             | `hipCtxCreate`                |
| `cuCtxDestroy`                                            | `hipCtxDestroy`               |
| `cuCtxGetApiVersion`                                      | `hipCtxGetApiVersion`         |
| `cuCtxGetCacheConfig`                                     | `hipCtxGetCacheConfig`        |
| `cuCtxGetCurrent`                                         | `hipCtxGetCurrent`            |
| `cuCtxGetDevice`                                          | `hipCtxGetDevice`             |
| `cuCtxGetFlags`                                           | `hipCtxGetFlags`              |
| `cuCtxGetLimit`                                           | `hipDeviceGetLimit`           |
| `cuCtxGetSharedMemConfig`                                 | `hipCtxGetSharedMemConfig`    |
| `cuCtxGetStreamPriorityRange`                             | `hipDeviceGetStreamPriorityRange`|
| `cuCtxPopCurrent`                                         | `hipCtxPopCurrent`            |
| `cuCtxPushCurrent`                                        | `hipCtxPushCurrent`           |
| `cuCtxSetCacheConfig`                                     | `hipCtxSetCacheConfig`        |
| `cuCtxSetCurrent`                                         | `hipCtxSetCurrent`            |
| `cuCtxSetLimit`                                           |                               |
| `cuCtxSetSharedMemConfig`                                 | `hipCtxSetSharedMemConfig`    |
| `cuCtxSynchronize`                                        | `hipCtxSynchronize`           |

## **9. Context Management [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuCtxAttach`                                             |                               |
| `cuCtxDetach`                                             |                               |

## **10. Module Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
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

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
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
| `cuMemAllocHost`                                          |                               |
| `cuMemAllocManaged`                                       |                               |
| `cuMemAllocPitch`                                         |                               |
| `cuMemcpy`                                                |                               |
| `cuMemcpy2D`                                              |                               |
| `cuMemcpy2DAsync`                                         |                               |
| `cuMemcpy2DUnaligned`                                     |                               |
| `cuMemcpy3D`                                              |                               |
| `cuMemcpy3DAsync`                                         |                               |
| `cuMemcpy3DPeer`                                          |                               |
| `cuMemcpy3DPeerAsync`                                     |                               |
| `cuMemcpyAsync`                                           |                               |
| `cuMemcpyAtoA`                                            |                               |
| `cuMemcpyAtoD`                                            |                               |
| `cuMemcpyAtoH`                                            |                               |
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
| `cuMemFreeHost`                                           | `hipFreeHost`                 |
| `cuMemGetAddressRange`                                    | `hipMemGetAddressRange`       |
| `cuMemGetInfo`                                            | `hipMemGetInfo`               |
| `cuMemHostAlloc`                                          | `hipHostMalloc`               |
| `cuMemHostGetDevicePointer`                               | `hipHostGetDevicePointer`     |
| `cuMemHostGetFlags`                                       | `hipHostGetFlags`             |
| `cuMemHostRegister`                                       | `hipHostRegister`             |
| `cuMemHostUnregister`                                     | `hipHostUnregister`           |
| `cuMemsetD16`                                             |                               |
| `cuMemsetD16Async`                                        |                               |
| `cuMemsetD2D16`                                           |                               |
| `cuMemsetD2D16Async`                                      |                               |
| `cuMemsetD2D32`                                           |                               |
| `cuMemsetD2D32Async`                                      |                               |
| `cuMemsetD2D8`                                            |                               |
| `cuMemsetD2D8Async`                                       |                               |
| `cuMemsetD32`                                             | `hipMemset`                   |
| `cuMemsetD32Async`                                        | `hipMemsetAsync`              |
| `cuMemsetD8`                                              | `hipMemsetD8`                 |
| `cuMemsetD8Async`                                         |                               |
| `cuMipmappedArrayCreate`                                  |                               |
| `cuMipmappedArrayDestroy`                                 |                               |
| `cuMipmappedArrayGetLevel`                                |                               |

## **12. Unified Addressing**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuMemAdvise`                                             |                               |
| `cuMemPrefetchAsync`                                      |                               |
| `cuMemRangeGetAttribute`                                  |                               |
| `cuMemRangeGetAttributes`                                 |                               |
| `cuPointerGetAttribute`                                   |                               |
| `cuPointerGetAttributes`                                  |                               |
| `cuPointerSetAttribute`                                   |                               |

## **13. Stream Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuStreamAddCallback`                                     | `hipStreamAddCallback`        |
| `cuStreamAttachMemAsync`                                  |                               |
| `cuStreamCreate`                                          | `hipStreamCreateWithFlags`    |
| `cuStreamCreateWithPriority`                              | `hipStreamCreateWithPriority` |
| `cuStreamDestroy`                                         | `hipStreamDestroy`            |
| `cuStreamGetFlags`                                        | `hipStreamGetFlags`           |
| `cuStreamGetPriority`                                     | `hipStreamGetPriority`        |
| `cuStreamQuery`                                           | `hipStreamQuery`              |
| `cuStreamSynchronize`                                     | `hipStreamSynchronize`        |
| `cuStreamWaitEvent`                                       | `hipStreamWaitEvent`          |
| `cuStreamBeginCapture`                                    |                               |
| `cuStreamEndCapture`                                      |                               |
| `cuStreamIsCapturing`                                     |                               |

## **14. Event Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuEventCreate`                                           | `hipEventCreateWithFlags`     |
| `cuEventDestroy`                                          | `hipEventDestroy`             |
| `cuEventElapsedTime`                                      | `hipEventElapsedTime`         |
| `cuEventQuery`                                            | `hipEventQuery`               |
| `cuEventRecord`                                           | `hipEventRecord`              |
| `cuEventSynchronize`                                      | `hipEventSynchronize`         |

## **15. External Resource Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuSignalExternalSemaphoresAsync`                         |                               |
| `cuWaitExternalSemaphoresAsync`                           |                               |
| `cuImportExternalMemory`                                  |                               |
| `cuExternalMemoryGetMappedBuffer`                         |                               |
| `cuExternalMemoryGetMappedMipmappedArray`                 |                               |
| `cuDestroyExternalMemory`                                 |                               |
| `cuImportExternalSemaphore`                               |                               |
| `cuDestroyExternalSemaphore`                              |                               |

## **16. Stream Memory Operations**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuStreamBatchMemOp`                                      |                               |
| `cuStreamWaitValue32`                                     |                               |
| `cuStreamWaitValue64`                                     |                               |
| `cuStreamWriteValue32`                                    |                               |
| `cuStreamWriteValue64`                                    |                               |

## **17. Execution Control**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuFuncGetAttribute`                                      |                               |
| `cuFuncSetAttribute`                                      |                               |
| `cuFuncSetCacheConfig`                                    | `hipFuncSetCacheConfig`       |
| `cuFuncSetSharedMemConfig`                                |                               |
| `cuLaunchKernel`                                          | `hipModuleLaunchKernel`       |
| `cuLaunchHostFunc`                                        |                               |
| `cuLaunchCooperativeKernel`                               |                               |
| `cuLaunchCooperativeKernelMultiDevice`                    |                               |

## **18. Execution Control [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuFuncSetBlockShape`                                     |                               |
| `cuFuncSetSharedSize`                                     |                               |
| `cuLaunch`                                                |                               |
| `cuLaunchGrid`                                            |                               |
| `cuLaunchGridAsync`                                       |                               |
| `cuParamSetf`                                             |                               |
| `cuParamSeti`                                             |                               |
| `cuParamSetTexRef`                                        |                               |
| `cuParamSetv`                                             |                               |

## **19. Graph Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuGraphCreate`                                           |                               |
| `cuGraphLaunch`                                           |                               |
| `cuGraphAddKernelNode`                                    |                               |
| `cuGraphKernelNodeGetParams`                              |                               |
| `cuGraphKernelNodeSetParams`                              |                               |
| `cuGraphAddMemcpyNode`                                    |                               |
| `cuGraphMemcpyNodeGetParams`                              |                               |
| `cuGraphMemcpyNodeSetParams`                              |                               |
| `cuGraphAddMemsetNode`                                    |                               |
| `cuGraphMemsetNodeGetParams`                              |                               |
| `cuGraphMemsetNodeSetParams`                              |                               |
| `cuGraphAddHostNode`                                      |                               |
| `cuGraphHostNodeGetParams`                                |                               |
| `cuGraphHostNodeSetParams`                                |                               |
| `cuGraphAddChildGraphNode`                                |                               |
| `cuGraphChildGraphNodeGetGraph`                           |                               |
| `cuGraphAddEmptyNode`                                     |                               |
| `cuGraphClone`                                            |                               |
| `cuGraphNodeFindInClone`                                  |                               |
| `cuGraphNodeGetType`                                      |                               |
| `cuGraphGetNodes`                                         |                               |
| `cuGraphGetRootNodes`                                     |                               |
| `cuGraphGetEdges`                                         |                               |
| `cuGraphNodeGetDependencies`                              |                               |
| `cuGraphNodeGetDependentNodes`                            |                               |
| `cuGraphAddDependencies`                                  |                               |
| `cuGraphRemoveDependencies`                               |                               |
| `cuGraphDestroyNode`                                      |                               |
| `cuGraphInstantiate`                                      |                               |
| `cuGraphExecDestroy`                                      |                               |
| `cuGraphDestroy`                                          |                               |

## **20. Occupancy**

|   **CUDA**                                                |   **HIP**                                               |
|-----------------------------------------------------------|---------------------------------------------------------|
| `cuOccupancyMaxActiveBlocksPerMultiprocessor`             | `hipOccupancyMaxActiveBlocksPerMultiprocessor`          |
| `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`    |                                                         |
| `cuOccupancyMaxPotentialBlockSize`                        | `hipOccupancyMaxPotentialBlockSize`                     |
| `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`    |                                                         |

## **21. Texture Reference Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuTexRefGetAddress`                                      |                               |
| `cuTexRefGetAddressMode`                                  |                               |
| `cuTexRefGetArray`                                        |                               |
| `cuTexRefGetBorderColor`                                  |                               |
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
| `cuTexRefSetBorderColor`                                  |                               |
| `cuTexRefSetFilterMode`                                   | `hipTexRefSetFilterMode`      |
| `cuTexRefSetFlags`                                        | `hipTexRefSetFlags`           |
| `cuTexRefSetFormat`                                       | `hipTexRefSetFormat`          |
| `cuTexRefSetMaxAnisotropy`                                |                               |
| `cuTexRefSetMipmapFilterMode`                             |                               |
| `cuTexRefSetMipmapLevelBias`                              |                               |
| `cuTexRefSetMipmapLevelClamp`                             |                               |
| `cuTexRefSetMipmappedArray`                               |                               |

## **22. Texture Reference Management [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuTexRefCreate`                                          |                               |
| `cuTexRefDestroy`                                         |                               |

## **23. Surface Reference Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuSurfRefGetArray`                                       |                               |
| `cuSurfRefSetArray`                                       |                               |

## **24. Texture Object Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuTexObjectCreate`                                       |                               |
| `cuTexObjectDestroy`                                      |                               |
| `cuTexObjectGetResourceDesc`                              |                               |
| `cuTexObjectGetResourceViewDesc`                          |                               |
| `cuTexObjectGetTextureDesc`                               |                               |

## **25. Surface Object Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuSurfObjectCreate`                                      |                               |
| `cuSurfObjectDestroy`                                     |                               |
| `cuSurfObjectGetResourceDesc`                             |                               |

## **26. Peer Context Memory Access**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuCtxEnablePeerAccess`                                   | `hipCtxEnablePeerAccess`      |
| `cuCtxDisablePeerAccess`                                  | `hipCtxDisablePeerAccess`     |
| `cuDeviceCanAccessPeer`                                   | `hipDeviceCanAccessPeer`      |
| `cuDeviceGetP2PAttribute`                                 |                               |

## **27. Graphics Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuGraphicsMapResources`                                  |                               |
| `cuGraphicsResourceGetMappedMipmappedArray`               |                               |
| `cuGraphicsResourceGetMappedPointer`                      |                               |
| `cuGraphicsResourceSetMapFlags`                           |                               |
| `cuGraphicsSubResourceGetMappedArray`                     |                               |
| `cuGraphicsUnmapResources`                                |                               |
| `cuGraphicsUnregisterResource`                            |                               |

## **28. Profiler Control**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuProfilerInitialize`                                    |                               |
| `cuProfilerStart`                                         | `hipProfilerStart`            |
| `cuProfilerStop`                                          | `hipProfilerStop`             |

## **29. OpenGL Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuGLGetDevices`                                          |                               |
| `cuGraphicsGLRegisterBuffer`                              |                               |
| `cuGraphicsGLRegisterImage`                               |                               |
| `cuWGLGetDevice`                                          |                               |

## **29.1. OpenGL Interoperability [DEPRECATED]**
|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
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

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuD3D9CtxCreate`                                         |                               |
| `cuD3D9CtxCreateOnDevice`                                 |                               |
| `cuD3D9GetDevice`                                         |                               |
| `cuD3D9GetDevices`                                        |                               |
| `cuD3D9GetDirect3DDevice`                                 |                               |
| `cuGraphicsD3D9RegisterResource`                          |                               |

## **30.1. Direct3D 9 Interoperability [DEPRECATED]**
|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
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

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuD3D10GetDevice`                                        |                               |
| `cuD3D10GetDevices`                                       |                               |
| `cuGraphicsD3D10RegisterResource`                         |                               |

## **31.1. Direct3D 10 Interoperability [DEPRECATED]**
|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
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

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuD3D11GetDevice`                                        |                               |
| `cuD3D11GetDevices`                                       |                               |
| `cuGraphicsD3D11RegisterResource`                         |                               |

## **32.1. Direct3D 11 Interoperability [DEPRECATED]**
|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuD3D11CtxCreate`                                        |                               |
| `cuD3D11CtxCreateOnDevice`                                |                               |
| `cuD3D11GetDirect3DDevice`                                |                               |

## **33. VDPAU Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuGraphicsVDPAURegisterOutputSurface`                    |                               |
| `cuGraphicsVDPAURegisterVideoSurface`                     |                               |
| `cuVDPAUCtxCreate`                                        |                               |
| `cuVDPAUGetDevice`                                        |                               |

## **34. EGL Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuEGLStreamConsumerAcquireFrame`                         |                               |
| `cuEGLStreamConsumerConnect`                              |                               |
| `cuEGLStreamConsumerConnectWithFlags`                     |                               |
| `cuEGLStreamConsumerDisconnect`                           |                               |
| `cuEGLStreamConsumerReleaseFrame`                         |                               |
| `cuEGLStreamProducerConnect`                              |                               |
| `cuEGLStreamProducerDisconnect`                           |                               |
| `cuEGLStreamProducerPresentFrame`                         |                               |
| `cuEGLStreamProducerReturnFrame`                          |                               |
| `cuGraphicsEGLRegisterImage`                              |                               |
| `cuGraphicsResourceGetMappedEglFrame`                     |                               |
| `cuEventCreateFromEGLSync`                                |                               |
