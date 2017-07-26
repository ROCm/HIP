# CUDA Driver API functions supported by HIP

## **1. Data types used by CUDA driver**

| **type**     |   **CUDA**                                                    |   **HIP**                                                  |
|-------------:|---------------------------------------------------------------|------------------------------------------------------------|
| struct       | `CUDA_ARRAY3D_DESCRIPTOR`                                     |                                                            |
| struct       | `CUDA_ARRAY_DESCRIPTOR`                                       |                                                            |
| struct       | `CUDA_MEMCPY2D`                                               |                                                            |
| struct       | `CUDA_MEMCPY3D`                                               |                                                            |
| struct       | `CUDA_MEMCPY3D_PEER`                                          |                                                            |
| struct       | `CUDA_POINTER_ATTRIBUTE_P2P_TOKENS`                           |                                                            |
| struct       | `CUDA_RESOURCE_DESC`                                          |                                                            |
| struct       | `CUDA_RESOURCE_VIEW_DESC`                                     |                                                            |
| struct       | `CUdevprop`                                                   | `hipDeviceProp_t`                                          |
| struct       | `CUipcEventHandle`                                            |                                                            |
| struct       | `CUipcMemHandle`                                              |                                                            |
| enum         |***`CUaddress_mode`***                                         |                                                            |
|            0 |*`CU_TR_ADDRESS_MODE_WRAP`*                                    |                                                            |
|            1 |*`CU_TR_ADDRESS_MODE_CLAMP`*                                   |                                                            |
|            2 |*`CU_TR_ADDRESS_MODE_MIRROR`*                                  |                                                            |
|            3 |*`CU_TR_ADDRESS_MODE_BORDER`*                                  |                                                            |
| enum         |***`CUarray_cubemap_face`***                                   |                                                            |
|         0x00 |*`CU_CUBEMAP_FACE_POSITIVE_X`*                                 |                                                            |
|         0x01 |*`CU_CUBEMAP_FACE_NEGATIVE_X`*                                 |                                                            |
|         0x02 |*`CU_CUBEMAP_FACE_POSITIVE_Y`*                                 |                                                            |
|         0x03 |*`CU_CUBEMAP_FACE_NEGATIVE_Y`*                                 |                                                            |
|         0x04 |*`CU_CUBEMAP_FACE_POSITIVE_Z`*                                 |                                                            |
|         0x05 |*`CU_CUBEMAP_FACE_NEGATIVE_Z`*                                 |                                                            |
| enum         |***`CUarray_format`***                                         |                                                            |
|         0x01 |*`CU_AD_FORMAT_UNSIGNED_INT8`*                                 |                                                            |
|         0x02 |*`CU_AD_FORMAT_UNSIGNED_INT16`*                                |                                                            |
|         0x03 |*`CU_AD_FORMAT_UNSIGNED_INT32`*                                |                                                            |
|         0x08 |*`CU_AD_FORMAT_SIGNED_INT8`*                                   |                                                            |
|         0x09 |*`CU_AD_FORMAT_SIGNED_INT16`*                                  |                                                            |
|         0x0a |*`CU_AD_FORMAT_SIGNED_INT32`*                                  |                                                            |
|         0x10 |*`CU_AD_FORMAT_HALF`*                                          |                                                            |
|         0x20 |*`CU_AD_FORMAT_FLOAT`*                                         |                                                            |
| enum         |***`CUctx_flags`***                                            |                                                            |
|         0x00 |*`CU_CTX_SCHED_AUTO`*                                          |                                                            |
|         0x01 |*`CU_CTX_SCHED_SPIN`*                                          |                                                            |
|         0x02 |*`CU_CTX_SCHED_YIELD`*                                         |                                                            |
|         0x04 |*`CU_CTX_SCHED_BLOCKING_SYNC`*                                 |                                                            |
|         0x04 |*`CU_CTX_BLOCKING_SYNC`*                                       |                                                            |
|         0x07 |*`CU_CTX_SCHED_MASK`*                                          |                                                            |
|         0x08 |*`CU_CTX_MAP_HOST`*                                            |                                                            |
|         0x10 |*`CU_CTX_LMEM_RESIZE_TO_MAX`*                                  |                                                            |
|         0x1f |*`CU_CTX_FLAGS_MASK`*                                          |                                                            |
| enum         |***`CUdevice_attribute`***                                     |                                                            |
|            1 |*`CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`*                  |*`hipDeviceAttributeMaxThreadsPerBlock`*                    |
|            2 |*`CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X`*                        |*`hipDeviceAttributeMaxBlockDimX`*                          |
|            3 |*`CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y`*                        |*`hipDeviceAttributeMaxBlockDimY`*                          |
|            4 |*`CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z`*                        |*`hipDeviceAttributeMaxBlockDimZ`*                          |
|            5 |*`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X`*                         |*`hipDeviceAttributeMaxGridDimX`*                           |
|            6 |*`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y`*                         |*`hipDeviceAttributeMaxGridDimY`*                           |
|            7 |*`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z`*                         |*`hipDeviceAttributeMaxGridDimZ`*                           |
|            8 |*`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`*            |*`hipDeviceAttributeMaxSharedMemoryPerBlock`*               |
|            8 |*`CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK`*                |*`hipDeviceAttributeMaxSharedMemoryPerBlock`*               |
|            9 |*`CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY`*                  |*`hipDeviceAttributeTotalConstantMemory`*                   |
|           10 |*`CU_DEVICE_ATTRIBUTE_WARP_SIZE`*                              |*`hipDeviceAttributeWarpSize`*                              |
|           11 |*`CU_DEVICE_ATTRIBUTE_MAX_PITCH`*                              |                                                            |
|           12 |*`CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK`*                |*`hipDeviceAttributeMaxRegistersPerBlock`*                  |
|           12 |*`CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK`*                    |*`hipDeviceAttributeMaxRegistersPerBlock`*                  |
|           13 |*`CU_DEVICE_ATTRIBUTE_CLOCK_RATE`*                             |*`hipDeviceAttributeClockRate`*                             |
|           14 |*`CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT`*                      |                                                            |
|           15 |*`CU_DEVICE_ATTRIBUTE_GPU_OVERLAP`*                            |                                                            |
|           16 |*`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`*                   |*`hipDeviceAttributeMultiprocessorCount`*                   |
|           17 |*`CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT`*                    |                                                            |
|           18 |*`CU_DEVICE_ATTRIBUTE_INTEGRATED`*                             |                                                            |
|           19 |*`CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY`*                    |                                                            |
|           20 |*`CU_DEVICE_ATTRIBUTE_COMPUTE_MODE`*                           |*`hipDeviceAttributeComputeMode`*                           |
|           21 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH`*                |                                                            |
|           22 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH`*                |                                                            |
|           23 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT`*               |                                                            |
|           24 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH`*                |                                                            |
|           25 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT`*               |                                                            |
|           26 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH`*                |                                                            |
|           27 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH`*        |                                                            |
|           28 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT`*       |                                                            |
|           29 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS`*       |                                                            |
|           27 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH`*          |                                                            |
|           28 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT`*         |                                                            |
|           29 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES`*      |                                                            |
|           30 |*`CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT`*                      |                                                            |
|           31 |*`CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS`*                     |*`hipDeviceAttributeConcurrentKernels`*                     |
|           32 |*`CU_DEVICE_ATTRIBUTE_ECC_ENABLED`*                            |                                                            |
|           33 |*`CU_DEVICE_ATTRIBUTE_PCI_BUS_ID`*                             |*`hipDeviceAttributePciBusId`*                              |
|           34 |*`CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID`*                          |*`hipDeviceAttributePciDeviceId`*                           |
|           35 |*`CU_DEVICE_ATTRIBUTE_TCC_DRIVER`*                             |                                                            |
|           36 |*`CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE`*                      |*`hipDeviceAttributeMemoryClockRate`*                       |
|           37 |*`CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH`*                |*`hipDeviceAttributeMemoryBusWidth`*                        |
|           38 |*`CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`*                          |*`hipDeviceAttributeL2CacheSize`*                           |
|           39 |*`CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR`*         |*`hipDeviceAttributeMaxThreadsPerMultiProcessor`*           |
|           40 |*`CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT`*                     |                                                            |
|           41 |*`CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING`*                     |                                                            |
|           42 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH`*        |                                                            |
|           43 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS`*       |                                                            |
|           44 |*`CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER`*                       |                                                            |
|           45 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH`*         |                                                            |
|           46 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT`*        |                                                            |
|           47 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE`*      |                                                            |
|           48 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE`*     |                                                            |
|           49 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE`*      |                                                            |
|           50 |*`CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID`*                          |                                                            |
|           51 |*`CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT`*                |                                                            |
|           52 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH`*           |                                                            |
|           53 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH`*   |                                                            |
|           54 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS`*  |                                                            |
|           55 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH`*                |                                                            |
|           56 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH`*                |                                                            |
|           57 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT`*               |                                                            |
|           58 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH`*                |                                                            |
|           59 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT`*               |                                                            |
|           60 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH`*                |                                                            |
|           61 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH`*        |                                                            |
|           62 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS`*       |                                                            |
|           63 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH`*        |                                                            |
|           64 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT`*       |                                                            |
|           65 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS`*       |                                                            |
|           66 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH`*           |                                                            |
|           67 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH`*   |                                                            |
|           68 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS`*  |                                                            |
|           69 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH`*         |                                                            |
|           70 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH`*         |                                                            |
|           71 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT`*        |                                                            |
|           72 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH`*         |                                                            |
|           73 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH`*      |                                                            |
|           74 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT`*     |                                                            |
|           75 |*`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`*               |*`hipDeviceAttributeComputeCapabilityMajor`*                |
|           76 |*`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`*               |*`hipDeviceAttributeComputeCapabilityMinor`*                |
|           77 |*`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH`*      |                                                            |
|           78 |*`CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED`*            |                                                            |
|           79 |*`CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED`*              |                                                            |
|           80 |*`CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED`*               |                                                            |
|           81 |*`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`*   |*`hipDeviceAttributeMaxSharedMemoryPerMultiprocessor`*      |
|           82 |*`CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR`*       |                                                            |
|           83 |*`CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY`*                         |*`hipDeviceAttributeManagedMemory`*                         |
|           84 |*`CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD`*                        |                                                            |
|           85 |*`CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID`*               |                                                            |
|           86 |*`CU_DEVICE_ATTRIBUTE_MAX`*                                    |                                                            |
| enum         |***`CUevent_flags`***                                          |                                                            |
|         0x00 |*`CU_EVENT_DEFAULT`*                                           |*`hipEventDefault`*                                         |
|         0x01 |*`CU_EVENT_BLOCKING_SYNC`*                                     |*`hipEventBlockingSync`*                                    |
|         0x02 |*`CU_EVENT_DISABLE_TIMING`*                                    |*`hipEventDisableTiming`*                                   |
|         0x04 |*`CU_EVENT_INTERPROCESS`*                                      |*`hipEventInterprocess`*                                    |
| enum         |***`CUfilter_mode`***                                          |***`hipTextureFilterMode`***                                |
|            0 |*`CU_TR_FILTER_MODE_POINT`*                                    |*`hipFilterModePoint`*                                      |
|            1 |*`CU_TR_FILTER_MODE_LINEAR`*                                   |*`hipFilterModeLinear`*                                     |
| enum         |***`CUfunc_cache`***                                           |***`hipFuncCache`***                                        |
|         0x00 |*`CU_FUNC_CACHE_PREFER_NONE`*                                  |*`hipFuncCachePreferNone`*                                  |
|         0x01 |*`CU_FUNC_CACHE_PREFER_SHARED`*                                |*`hipFuncCachePreferShared`*                                |
|         0x02 |*`CU_FUNC_CACHE_PREFER_L1`*                                    |*`hipFuncCachePreferL1`*                                    |
|         0x03 |*`CU_FUNC_CACHE_PREFER_EQUAL`*                                 |*`hipFuncCachePreferEqual`*                                 |
| enum         |***`CUfunction_attribute`***                                   |                                                            |
|            0 |*`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`*                    |                                                            |
|            1 |*`CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`*                        |                                                            |
|            2 |*`CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES`*                         |                                                            |
|            3 |*`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`*                         |                                                            |
|            4 |*`CU_FUNC_ATTRIBUTE_NUM_REGS`*                                 |                                                            |
|            5 |*`CU_FUNC_ATTRIBUTE_PTX_VERSION`*                              |                                                            |
|            6 |*`CU_FUNC_ATTRIBUTE_BINARY_VERSION`*                           |                                                            |
|            7 |*`CU_FUNC_ATTRIBUTE_CACHE_MODE_CA`*                            |                                                            |
|            8 |*`CU_FUNC_ATTRIBUTE_MAX`*                                      |                                                            |
| enum         |***`CUgraphicsMapResourceFlags`***                             |                                                            |
|         0x00 |*`CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE`*                        |                                                            |
|         0x01 |*`CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY`*                   |                                                            |
|         0x02 |*`CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD`*               |                                                            |
| enum         |***`CUgraphicsRegisterFlags`***                                |                                                            |
|         0x00 |*`CU_GRAPHICS_REGISTER_FLAGS_NONE`*                            |                                                            |
|         0x01 |*`CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY`*                       |                                                            |
|         0x02 |*`CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD`*                   |                                                            |
|         0x04 |*`CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST`*                    |                                                            |
|         0x08 |*`CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER`*                  |                                                            |
| enum         |***`CUipcMem_flags`***                                         |                                                            |
|          0x1 |*`CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS`*                         |*`hipIpcMemLazyEnablePeerAccess`*                           |
| enum         |***`CUjit_cacheMode`***                                        |                                                            |
|            0 |*`CU_JIT_CACHE_OPTION_NONE`*                                   |                                                            |
|              |*`CU_JIT_CACHE_OPTION_CG`*                                     |                                                            |
|              |*`CU_JIT_CACHE_OPTION_CA`*                                     |                                                            |
| enum         |***`CUjit_fallback`***                                         |                                                            |
|            0 |*`CU_PREFER_PTX`*                                              |                                                            |
|              |*`CU_PREFER_BINARY`*                                           |                                                            |
| enum         |***`CUjit_option`***                                           |                                                            |
|            0 |*`CU_JIT_MAX_REGISTERS`*                                       |                                                            |
|              |*`CU_JIT_THREADS_PER_BLOCK`*                                   |                                                            |
|              |*`CU_JIT_WALL_TIME`*                                           |                                                            |
|              |*`CU_JIT_INFO_LOG_BUFFER`*                                     |                                                            |
|              |*`CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES`*                          |                                                            |
|              |*`CU_JIT_OPTIMIZATION_LEVEL`*                                  |                                                            |
|              |*`CU_JIT_TARGET_FROM_CUCONTEXT`*                               |                                                            |
|              |*`CU_JIT_TARGET`*                                              |                                                            |
|              |*`CU_JIT_FALLBACK_STRATEGY`*                                   |                                                            |
|              |*`CU_JIT_GENERATE_DEBUG_INFO`*                                 |                                                            |
|              |*`CU_JIT_LOG_VERBOSE`*                                         |                                                            |
|              |*`CU_JIT_GENERATE_LINE_INFO`*                                  |                                                            |
|              |*`CU_JIT_CACHE_MODE`*                                          |                                                            |
|              |*`CU_JIT_NUM_OPTIONS`*                                         |                                                            |
| enum         |***`CUjit_target`***                                           |                                                            |
|           10 |*`CU_TARGET_COMPUTE_10`*                                       |                                                            |
|           11 |*`CU_TARGET_COMPUTE_11`*                                       |                                                            |
|           12 |*`CU_TARGET_COMPUTE_12`*                                       |                                                            |
|           13 |*`CU_TARGET_COMPUTE_13`*                                       |                                                            |
|           20 |*`CU_TARGET_COMPUTE_20`*                                       |                                                            |
|           21 |*`CU_TARGET_COMPUTE_21`*                                       |                                                            |
|           30 |*`CU_TARGET_COMPUTE_30`*                                       |                                                            |
|           32 |*`CU_TARGET_COMPUTE_32`*                                       |                                                            |
|           35 |*`CU_TARGET_COMPUTE_35`*                                       |                                                            |
|           37 |*`CU_TARGET_COMPUTE_37`*                                       |                                                            |
|           50 |*`CU_TARGET_COMPUTE_50`*                                       |                                                            |
|           52 |*`CU_TARGET_COMPUTE_52`*                                       |                                                            |
| enum         |***`CUjitInputType`***                                         |                                                            |
|            0 |*`CU_JIT_INPUT_CUBIN`*                                         |                                                            |
|              |*`CU_JIT_INPUT_PTX`*                                           |                                                            |
|              |*`CU_JIT_INPUT_FATBINARY`*                                     |                                                            |
|              |*`CU_JIT_INPUT_OBJECT`*                                        |                                                            |
|              |*`CU_JIT_INPUT_LIBRARY`*                                       |                                                            |
|              |*`CU_JIT_NUM_INPUT_TYPES`*                                     |                                                            |
| enum         |***`CUlimit`***                                                |***`hipLimit_t`***                                          |
|         0x00 |*`CU_LIMIT_STACK_SIZE`*                                        |                                                            |
|         0x01 |*`CU_LIMIT_PRINTF_FIFO_SIZE`*                                  |                                                            |
|         0x02 |*`CU_LIMIT_MALLOC_HEAP_SIZE`*                                  |*`hipLimitMallocHeapSize`*                                  |
|         0x03 |*`CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH`*                            |                                                            |
|         0x04 |*`CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT`*                  |                                                            |
|              |*`CU_LIMIT_MAX`*                                               |                                                            |
| enum         |***`CUmemAttach_flags`***                                      |                                                            |
|          0x1 |*`CU_MEM_ATTACH_GLOBAL`*                                       |                                                            |
|          0x2 |*`CU_MEM_ATTACH_HOST`*                                         |                                                            |
|          0x4 |*`CU_MEM_ATTACH_SINGLE`*                                       |                                                            |
| enum         |***`CUmemorytype`***                                           |                                                            |
|         0x01 |*`CU_MEMORYTYPE_HOST`*                                         |                                                            |
|         0x02 |*`CU_MEMORYTYPE_DEVICE`*                                       |                                                            |
|         0x03 |*`CU_MEMORYTYPE_ARRAY`*                                        |                                                            |
|         0x04 |*`CU_MEMORYTYPE_UNIFIED`*                                      |                                                            |
| enum         |***`CUoccupancy_flags`***                                      |                                                            |
|         0x00 |*`CU_OCCUPANCY_DEFAULT`*                                       |                                                            |
|         0x01 |*`CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE`*                      |                                                            |
| enum         |***`CUpointer_attribute`***                                    |                                                            |
|            1 |*`CU_POINTER_ATTRIBUTE_CONTEXT`*                               |                                                            |
|            2 |*`CU_POINTER_ATTRIBUTE_MEMORY_TYPE`*                           |                                                            |
|            3 |*`CU_POINTER_ATTRIBUTE_DEVICE_POINTER`*                        |                                                            |
|            4 |*`CU_POINTER_ATTRIBUTE_HOST_POINTER`*                          |                                                            |
|            5 |*`CU_POINTER_ATTRIBUTE_P2P_TOKENS`*                            |                                                            |
|            6 |*`CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`*                           |                                                            |
|            7 |*`CU_POINTER_ATTRIBUTE_BUFFER_ID`*                             |                                                            |
|            8 |*`CU_POINTER_ATTRIBUTE_IS_MANAGED`*                            |                                                            |
| enum         |***`CUmemorytype`***                                           |                                                            |
|         0x00 |*`CU_RESOURCE_TYPE_ARRAY`*                                     |                                                            |
|         0x01 |*`CU_RESOURCE_TYPE_MIPMAPPED_ARRAY`*                           |                                                            |
|         0x02 |*`CU_RESOURCE_TYPE_LINEAR`*                                    |                                                            |
|         0x03 |*`CU_RESOURCE_TYPE_PITCH2D`*                                   |                                                            |
| enum         |***`CUresourceViewFormat`***                                   |                                                            |
|         0x00 |*`CU_RES_VIEW_FORMAT_NONE`*                                    |                                                            |
|         0x01 |*`CU_RES_VIEW_FORMAT_UINT_1X8`*                                |                                                            |
|         0x02 |*`CU_RES_VIEW_FORMAT_UINT_2X8`*                                |                                                            |
|         0x03 |*`CU_RES_VIEW_FORMAT_UINT_4X8`*                                |                                                            |
|         0x04 |*`CU_RES_VIEW_FORMAT_SINT_1X8`*                                |                                                            |
|         0x05 |*`CU_RES_VIEW_FORMAT_SINT_2X8`*                                |                                                            |
|         0x06 |*`CU_RES_VIEW_FORMAT_SINT_4X8`*                                |                                                            |
|         0x07 |*`CU_RES_VIEW_FORMAT_UINT_1X16`*                               |                                                            |
|         0x08 |*`CU_RES_VIEW_FORMAT_UINT_2X16`*                               |                                                            |
|         0x09 |*`CU_RES_VIEW_FORMAT_UINT_4X16`*                               |                                                            |
|         0x0a |*`CU_RES_VIEW_FORMAT_SINT_1X16`*                               |                                                            |
|         0x0b |*`CU_RES_VIEW_FORMAT_SINT_2X16`*                               |                                                            |
|         0x0c |*`CU_RES_VIEW_FORMAT_SINT_4X16`*                               |                                                            |
|         0x0d |*`CU_RES_VIEW_FORMAT_UINT_1X32`*                               |                                                            |
|         0x0e |*`CU_RES_VIEW_FORMAT_UINT_2X32`*                               |                                                            |
|         0x0f |*`CU_RES_VIEW_FORMAT_UINT_4X32`*                               |                                                            |
|         0x10 |*`CU_RES_VIEW_FORMAT_SINT_1X32`*                               |                                                            |
|         0x11 |*`CU_RES_VIEW_FORMAT_SINT_2X32`*                               |                                                            |
|         0x12 |*`CU_RES_VIEW_FORMAT_SINT_4X32`*                               |                                                            |
|         0x13 |*`CU_RES_VIEW_FORMAT_FLOAT_1X16`*                              |                                                            |
|         0x14 |*`CU_RES_VIEW_FORMAT_FLOAT_2X16`*                              |                                                            |
|         0x15 |*`CU_RES_VIEW_FORMAT_FLOAT_4X16`*                              |                                                            |
|         0x16 |*`CU_RES_VIEW_FORMAT_FLOAT_1X32`*                              |                                                            |
|         0x17 |*`CU_RES_VIEW_FORMAT_FLOAT_2X32`*                              |                                                            |
|         0x18 |*`CU_RES_VIEW_FORMAT_FLOAT_4X32`*                              |                                                            |
|         0x19 |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC1`*                            |                                                            |
|         0x1a |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC3`*                            |                                                            |
|         0x1b |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC3`*                            |                                                            |
|         0x1c |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC4`*                            |                                                            |
|         0x1d |*`CU_RES_VIEW_FORMAT_SIGNED_BC4`*                              |                                                            |
|         0x1e |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC5`*                            |                                                            |
|         0x1f |*`CU_RES_VIEW_FORMAT_SIGNED_BC5`*                              |                                                            |
|         0x20 |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC6H`*                           |                                                            |
|         0x21 |*`CU_RES_VIEW_FORMAT_SIGNED_BC6H`*                             |                                                            |
|         0x22 |*`CU_RES_VIEW_FORMAT_UNSIGNED_BC7`*                            |                                                            |
| enum         |***`CUresult`***                                               |***`hipError_t`***                                          |
|            0 |*`CUDA_SUCCESS`*                                               |*`hipSuccess`*                                              |
|            1 |*`CUDA_ERROR_INVALID_VALUE`*                                   |*`hipErrorInvalidValue`*                                    |
|            2 |*`CUDA_ERROR_OUT_OF_MEMORY`*                                   |*`hipErrorMemoryAllocation`*                                |
|            3 |*`CUDA_ERROR_NOT_INITIALIZED`*                                 |*`hipErrorNotInitialized`*                                  |
|            4 |*`CUDA_ERROR_DEINITIALIZED`*                                   |*`hipErrorDeinitialized`*                                   |
|            5 |*`CUDA_ERROR_PROFILER_DISABLED`*                               |*`hipErrorProfilerDisabled`*                                |
|            6 |*`CUDA_ERROR_PROFILER_NOT_INITIALIZED`*                        |*`hipErrorProfilerNotInitialized`*                          |
|            7 |*`CUDA_ERROR_PROFILER_ALREADY_STARTED`*                        |*`hipErrorProfilerAlreadyStarted`*                          |
|            8 |*`CUDA_ERROR_PROFILER_ALREADY_STOPPED`*                        |*`hipErrorProfilerAlreadyStopped`*                          |
|          100 |*`CUDA_ERROR_NO_DEVICE`*                                       |*`hipErrorNoDevice`*                                        |
|          101 |*`CUDA_ERROR_INVALID_DEVICE`*                                  |*`hipErrorInvalidDevice`*                                   |
|          200 |*`CUDA_ERROR_INVALID_IMAGE`*                                   |*`hipErrorInvalidImage`*                                    |
|          201 |*`CUDA_ERROR_INVALID_CONTEXT`*                                 |*`hipErrorInvalidContext`*                                  |
|          202 |*`CUDA_ERROR_CONTEXT_ALREADY_CURRENT`*                         |*`hipErrorContextAlreadyCurrent`*                           |
|          205 |*`CUDA_ERROR_MAP_FAILED`*                                      |*`hipErrorMapFailed`*                                       |
|          206 |*`CUDA_ERROR_UNMAP_FAILED`*                                    |*`hipErrorUnmapFailed`*                                     |
|          207 |*`CUDA_ERROR_ARRAY_IS_MAPPED`*                                 |*`hipErrorArrayIsMapped`*                                   |
|          208 |*`CUDA_ERROR_ALREADY_MAPPED`*                                  |*`hipErrorAlreadyMapped`*                                   |
|          209 |*`CUDA_ERROR_NO_BINARY_FOR_GPU`*                               |*`hipErrorNoBinaryForGpu*                                   |
|          210 |*`CUDA_ERROR_ALREADY_ACQUIRED`*                                |*`hipErrorAlreadyAcquired*                                  |
|          211 |*`CUDA_ERROR_NOT_MAPPED`*                                      |*`hipErrorNotMapped`*                                       |
|          212 |*`CUDA_ERROR_NOT_MAPPED_AS_ARRAY`*                             |*`hipErrorNotMappedAsArray`*                                |
|          213 |*`CUDA_ERROR_NOT_MAPPED_AS_POINTER`*                           |*`hipErrorNotMappedAsPointer`*                              |
|          214 |*`CUDA_ERROR_ECC_UNCORRECTABLE`*                               |*`hipErrorECCNotCorrectable`*                               |
|          215 |*`CUDA_ERROR_UNSUPPORTED_LIMIT`*                               |*`hipErrorUnsupportedLimit`*                                |
|          216 |*`CUDA_ERROR_CONTEXT_ALREADY_IN_USE`*                          |*`hipErrorContextAlreadyInUse`*                             |
|          217 |*`CUDA_ERROR_PEER_ACCESS_UNSUPPORTED`*                         |*`hipErrorPeerAccessUnsupported`*                           |
|          218 |*`CUDA_ERROR_INVALID_PTX`*                                     |*`hipErrorInvalidKernelFile`*                               |
|          219 |*`CUDA_ERROR_INVALID_GRAPHICS_CONTEXT`*                        |*`hipErrorInvalidGraphicsContext`*                          |
|          300 |*`CUDA_ERROR_INVALID_SOURCE`*                                  |*`hipErrorInvalidSource`*                                   |
|          301 |*`CUDA_ERROR_FILE_NOT_FOUND`*                                  |*`hipErrorFileNotFound`*                                    |
|          302 |*`CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND`*                  |*`hipErrorSharedObjectSymbolNotFound`*                      |
|          303 |*`CUDA_ERROR_SHARED_OBJECT_INIT_FAILED`*                       |*`hipErrorSharedObjectInitFailed`*                          |
|          304 |*`CUDA_ERROR_OPERATING_SYSTEM`*                                |*`hipErrorOperatingSystem`*                                 |
|          400 |*`CUDA_ERROR_INVALID_HANDLE`*                                  |*`hipErrorInvalidResourceHandle`*                           |
|          500 |*`CUDA_ERROR_NOT_FOUND`*                                       |*`hipErrorNotFound`*                                        |
|          600 |*`CUDA_ERROR_NOT_READY`*                                       |*`hipErrorNotReady`*                                        |
|          700 |*`CUDA_ERROR_ILLEGAL_ADDRESS`*                                 |*`hipErrorIllegalAddress`*                                  |
|          701 |*`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`*                         |*`hipErrorLaunchOutOfResources`*                            |
|          702 |*`CUDA_ERROR_LAUNCH_TIMEOUT`*                                  |*`hipErrorLaunchTimeOut`*                                   |
|          703 |*`CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING`*                   |                                                            |
|          704 |*`CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED`*                     |*`hipErrorPeerAccessAlreadyEnabled`*                        |
|          705 |*`CUDA_ERROR_PEER_ACCESS_NOT_ENABLED`*                         |*`hipErrorPeerAccessNotEnabled`*                            |
|          708 |*`CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE`*                          |                                                            |
|          709 |*`CUDA_ERROR_CONTEXT_IS_DESTROYED`*                            |                                                            |
|          710 |*`CUDA_ERROR_ASSERT`*                                          |                                                            |
|          711 |*`CUDA_ERROR_TOO_MANY_PEERS`*                                  |                                                            |
|          712 |*`CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`*                  |*`hipErrorHostMemoryAlreadyRegistered`*                     |
|          713 |*`CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED`*                      |*`hipErrorHostMemoryNotRegistered`*                         |
|          714 |*`CUDA_ERROR_HARDWARE_STACK_ERROR`*                            |                                                            |
|          715 |*`CUDA_ERROR_ILLEGAL_INSTRUCTION`*                             |                                                            |
|          716 |*`CUDA_ERROR_MISALIGNED_ADDRESS`*                              |                                                            |
|          717 |*`CUDA_ERROR_INVALID_ADDRESS_SPACE`*                           |                                                            |
|          718 |*`CUDA_ERROR_INVALID_PC`*                                      |                                                            |
|          719 |*`CUDA_ERROR_LAUNCH_FAILED`*                                   |                                                            |
|          800 |*`CUDA_ERROR_NOT_PERMITTED`*                                   |                                                            |
|          801 |*`CUDA_ERROR_NOT_SUPPORTED`*                                   |                                                            |
|          999 |*`CUDA_ERROR_UNKNOWN`*                                         |                                                            |
| enum         |***`CUstream_flags`***                                         |***`hipStreamFlags`***                                      |
|          0x0 |*`CU_STREAM_DEFAULT`*                                          |*`hipStreamDefault`*                                        |
|          0x1 |*`CU_STREAM_NON_BLOCKING`*                                     |*`hipStreamNonBlocking`*                                    |
| enum         |***`CUGLDeviceList`***                                         |                                                            |
|         0x01 |*`CU_GL_DEVICE_LIST_ALL`*                                      |                                                            |
|         0x02 |*`CU_GL_DEVICE_LIST_CURRENT_FRAME`*                            |                                                            |
|         0x03 |*`CU_GL_DEVICE_LIST_NEXT_FRAME`*                               |                                                            |
| enum         |***`CUGLmap_flags`***                                          |                                                            |
|         0x00 |*`CU_GL_MAP_RESOURCE_FLAGS_NONE`*                              |                                                            |
|         0x01 |*`CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY`*                         |                                                            |
|         0x02 |*`CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD`*                     |                                                            |
| enum         |***`CUd3d9DeviceList`***                                       |                                                            |
|         0x01 |*`CU_D3D9_DEVICE_LIST_ALL`*                                    |                                                            |
|         0x02 |*`CU_D3D9_DEVICE_LIST_CURRENT_FRAME`*                          |                                                            |
|         0x03 |*`CU_D3D9_DEVICE_LIST_NEXT_FRAME`*                             |                                                            |
| enum         |***`CUd3d9map_flags`***                                        |                                                            |
|         0x00 |*`CU_D3D9_MAPRESOURCE_FLAGS_NONE`*                             |                                                            |
|         0x01 |*`CU_D3D9_MAPRESOURCE_FLAGS_READONLY`*                         |                                                            |
|         0x02 |*`CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD`*                     |                                                            |
| enum         |***`CUd3d9register_flags`***                                   |                                                            |
|         0x00 |*`CU_D3D9_REGISTER_FLAGS_NONE`*                                |                                                            |
|         0x01 |*`CU_D3D9_REGISTER_FLAGS_ARRAY`*                               |                                                            |
| enum         |***`CUd3d10DeviceList`***                                      |                                                            |
|         0x01 |*`CU_D3D10_DEVICE_LIST_ALL`*                                   |                                                            |
|         0x02 |*`CU_D3D10_DEVICE_LIST_CURRENT_FRAME`*                         |                                                            |
|         0x03 |*`CU_D3D10_DEVICE_LIST_NEXT_FRAME`*                            |                                                            |
| enum         |***`CUd3d10map_flags`***                                       |                                                            |
|         0x00 |*`CU_D3D10_MAPRESOURCE_FLAGS_NONE`*                            |                                                            |
|         0x01 |*`CU_D3D10_MAPRESOURCE_FLAGS_READONLY`*                        |                                                            |
|         0x02 |*`CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD`*                    |                                                            |
| enum         |***`CUd3d10register_flags`***                                  |                                                            |
|         0x00 |*`CU_D3D10_REGISTER_FLAGS_NONE`*                               |                                                            |
|         0x01 |*`CU_D3D10_REGISTER_FLAGS_ARRAY`*                              |                                                            |
| enum         |***`CUd3d11DeviceList`***                                      |                                                            |
|         0x01 |*`CU_D3D11_DEVICE_LIST_ALL`*                                   |                                                            |
|         0x02 |*`CU_D3D11_DEVICE_LIST_CURRENT_FRAME`*                         |                                                            |
|         0x03 |*`CU_D3D11_DEVICE_LIST_NEXT_FRAME`*                            |                                                            |
| typedef      | `CUarray`                                                     | `hipArray *`                                               |
| struct       | `CUarray_st`                                                  | `hipArray`                                                 |
| typedef      | `CUcontext`                                                   | `hipCtx_t`                                                 |
| typedef      | `CUdevice`                                                    | `hipDevice_t`                                              |
| typedef      | `CUdeviceptr`                                                 | `hipDeviceptr_t`                                           |
| typedef      | `CUevent`                                                     | `hipEvent_t`                                               |
| typedef      | `CUfunction`                                                  | `hipFunction_t`                                            |
| typedef      | `CUgraphicsResource`                                          |                                                            |
| typedef      | `CUmipmappedArray`                                            |                                                            |
| typedef      | `CUmodule`                                                    | `hipModule_t`                                              |
| typedef      | `CUstream`                                                    | `hipStream_t`                                              |
| typedef      | `CUstreamCallback`                                            | `hipStreamCallback_t`                                      |
| typedef      | `CUsurfObject`                                                |                                                            |
| typedef      | `CUsurfref`                                                   |                                                            |
| typedef      | `CUtexObject`                                                 |                                                            |
| typedef      | `CUtexref`                                                    |                                                            |
| define       |`CU_IPC_HANDLE_SIZE`                                           |                                                            |
| define       |`CU_LAUNCH_PARAM_BUFFER_POINTER`                               | `HIP_LAUNCH_PARAM_BUFFER_POINTER`                          |
| define       |`CU_LAUNCH_PARAM_BUFFER_SIZE`                                  | `HIP_LAUNCH_PARAM_BUFFER_SIZE`                             |
| define       |`CU_LAUNCH_PARAM_END`                                          | `HIP_LAUNCH_PARAM_END`                                     |
| define       |`CU_MEMHOSTALLOC_DEVICEMAP`                                    |                                                            |
| define       |`CU_MEMHOSTALLOC_PORTABLE`                                     |                                                            |
| define       |`CU_MEMHOSTALLOC_WRITECOMBINED`                                |                                                            |
| define       |`CU_MEMHOSTREGISTER_DEVICEMAP`                                 |                                                            |
| define       |`CU_MEMHOSTREGISTER_IOMEMORY`                                  |                                                            |
| define       |`CU_MEMHOSTREGISTER_PORTABLE`                                  |                                                            |
| define       |`CU_PARAM_TR_DEFAULT`                                          |                                                            |
| define       |`CU_STREAM_LEGACY`                                             |                                                            |
| define       |`CU_STREAM_PER_THREAD`                                         |                                                            |
| define       |`CU_TRSA_OVERRIDE_FORMAT`                                      |                                                            |
| define       |`CU_TRSF_NORMALIZED_COORDINATES`                               |                                                            |
| define       |`CU_TRSF_SRGB`                                                 |                                                            |
| define       |`CUDA_ARRAY3D_2DARRAY`                                         |                                                            |
| define       |`CUDA_ARRAY3D_CUBEMAP`                                         |                                                            |
| define       |`CUDA_ARRAY3D_DEPTH_TEXTURE`                                   |                                                            |
| define       |`CUDA_ARRAY3D_LAYERED`                                         |                                                            |
| define       |`CUDA_ARRAY3D_SURFACE_LDST`                                    |                                                            |
| define       |`CUDA_ARRAY3D_TEXTURE_GATHER`                                  |                                                            |
| define       |`CUDA_VERSION`                                                 |                                                            |

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
| `cuCtxGetLimit`                                           |                               |
| `cuCtxGetSharedMemConfig`                                 | `hipCtxGetSharedMemConfig`    |
| `cuCtxGetStreamPriorityRange`                             |                               |
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
| `cuModuleGetTexRef`                                       |                               |
| `cuModuleLoad`                                            | `hipModuleLoad`               |
| `cuModuleLoadData`                                        | `hipModuleLoadData`           |
| `cuModuleLoadDataEx`                                      | `hipModuleLoadDataEx`         |
| `cuModuleLoadFatBinary`                                   |                               |
| `cuModuleUnload`                                          | `hipModuleUnload`             |

## **11. Memory Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuArray3DCreate`                                         |                               |
| `cuArray3DGetDescriptor`                                  |                               |
| `cuArrayCreate`                                           |                               |
| `cuArrayDestroy`                                          |                               |
| `cuArrayGetDescriptor`                                    |                               |
| `cuDeviceGetByPCIBusId`                                   | `hipDeviceGetByPCIBusId`      |
| `cuDeviceGetPCIBusId`                                     | `hipDeviceGetPCIBusId`        |
| `cuIpcCloseMemHandle`                                     |                               |
| `cuIpcGetEventHandle`                                     |                               |
| `cuIpcGetMemHandle`                                       |                               |
| `cuIpcOpenEventHandle`                                    |                               |
| `cuIpcOpenMemHandle`                                      |                               |
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
| `cuMemcpyHtoA`                                            |                               |
| `cuMemcpyHtoAAsync`                                       |                               |
| `cuMemcpyHtoD`                                            | `hipMemcpyHtoD`               |
| `cuMemcpyHtoDAsync`                                       | `hipMemcpyHtoDAsync`          |
| `cuMemcpyPeer`                                            |                               |
| `cuMemcpyPeerAsync`                                       |                               |
| `cuMemFree`                                               | `hipFree`                     |
| `cuMemFreeHost`                                           | `hipFreeHost`                 |
| `cuMemGetAddressRange`                                    |                               |
| `cuMemGetInfo`                                            | `hipMemGetInfo`               |
| `cuMemHostAlloc`                                          | `hipHostMalloc`               |
| `cuMemHostGetDevicePointer`                               |                               |
| `cuMemHostGetFlags`                                       |                               |
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
| `cuMemsetD2D8`                                            |                               |
| `cuMemsetD2D8Async`                                       |                               |
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
| `cuStreamCreate`                                          |                               |
| `cuStreamCreateWithPriority`                              |                               |
| `cuStreamDestroy`                                         | `hipStreamDestroy`            |
| `cuStreamGetFlags`                                        | `hipStreamGetFlags`           |
| `cuStreamGetPriority`                                     | `hipStreamGetPriority`        |
| `cuStreamQuery`                                           | `hipStreamQuery`              |
| `cuStreamSynchronize`                                     | `hipStreamSynchronize`        |
| `cuStreamWaitEvent`                                       | `hipStreamWaitEvent`          |
| `cuStreamBatchMemOp`                                      |                               |
| `cuStreamWaitValue32`                                     |                               |
| `cuStreamWriteValue32`                                    |                               |

## **14. Event Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuEventCreate`                                           | `hipEventCreate`              |
| `cuEventDestroy`                                          | `hipEventDestroy`             |
| `cuEventElapsedTime`                                      | `hipEventElapsedTime`         |
| `cuEventQuery`                                            | `hipEventQuery`               |
| `cuEventRecord`                                           | `hipEventRecord`              |
| `cuEventSynchronize`                                      | `hipEventSynchronize`         |

## **15. Execution Control**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuFuncGetAttribute`                                      |                               |
| `cuFuncSetCacheConfig`                                    | `hipFuncSetCacheConfig`       |
| `cuFuncSetSharedMemConfig`                                |                               |
| `cuLaunchKernel`                                          | `hipModuleLaunchKernel`       |

## **16. Execution Control [DEPRECATED]**

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

## **17. Occupancy**

|   **CUDA**                                                |   **HIP**                                               |
|-----------------------------------------------------------|---------------------------------------------------------|
| `cuOccupancyMaxActiveBlocksPerMultiprocessor`             | `hipOccupancyMaxActiveBlocksPerMultiprocessor`          |
| `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`    |                                                         |
| `cuOccupancyMaxPotentialBlockSize`                        | `hipOccupancyMaxPotentialBlockSize`                     |
| `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`    |                                                         |

## **18. Texture Reference Management**

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
| `cuTexRefSetAddress`                                      |                               |
| `cuTexRefSetAddress2D`                                    |                               |
| `cuTexRefSetAddressMode`                                  |                               |
| `cuTexRefSetArray`                                        |                               |
| `cuTexRefSetBorderColor`                                  |                               |
| `cuTexRefSetFilterMode`                                   |                               |
| `cuTexRefSetFlags`                                        |                               |
| `cuTexRefSetFormat`                                       |                               |
| `cuTexRefSetMaxAnisotropy`                                |                               |
| `cuTexRefSetMipmapFilterMode`                             |                               |
| `cuTexRefSetMipmapLevelBias`                              |                               |
| `cuTexRefSetMipmapLevelClamp`                             |                               |
| `cuTexRefSetMipmappedArray`                               |                               |

## **19. Texture Reference Management [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuTexRefCreate`                                          |                               |
| `cuTexRefDestroy`                                         |                               |

## **20. Surface Reference Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuSurfRefGetArray`                                       |                               |
| `cuSurfRefSetArray`                                       |                               |

## **21. Texture Object Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuTexObjectCreate`                                       |                               |
| `cuTexObjectDestroy`                                      |                               |
| `cuTexObjectGetResourceDesc`                              |                               |
| `cuTexObjectGetResourceViewDesc`                          |                               |
| `cuTexObjectGetTextureDesc`                               |                               |

## **22. Surface Object Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuSurfObjectCreate`                                      |                               |
| `cuSurfObjectDestroy`                                     |                               |
| `cuSurfObjectGetResourceDesc`                             |                               |

## **23. Peer Context Memory Access**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuCtxEnablePeerAccess`                                   | `hipCtxEnablePeerAccess`      |
| `cuCtxDisablePeerAccess`                                  | `hipCtxDisablePeerAccess`     |
| `cuDeviceCanAccessPeer`                                   | `hipDeviceCanAccessPeer`      |
| `cuDeviceGetP2PAttribute`                                 |                               |

## **24. Graphics Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuGraphicsMapResources`                                  |                               |
| `cuGraphicsResourceGetMappedMipmappedArray`               |                               |
| `cuGraphicsResourceGetMappedPointer`                      |                               |
| `cuGraphicsResourceSetMapFlags`                           |                               |
| `cuGraphicsSubResourceGetMappedArray`                     |                               |
| `cuGraphicsUnmapResources`                                |                               |
| `cuGraphicsUnregisterResource`                            |                               |

## **25. Profiler Control**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuProfilerInitialize`                                    |                               |
| `cuProfilerStart`                                         | `hipProfilerStart`            |
| `cuProfilerStop`                                          | `hipProfilerStop`             |

## **26. OpenGL Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuGLGetDevices`                                          |                               |
| `cuGraphicsGLRegisterBuffer`                              |                               |
| `cuGraphicsGLRegisterImage`                               |                               |
| `cuWGLGetDevice`                                          |                               |

## **26.1. OpenGL Interoperability [DEPRECATED]**
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

## **27. Direct3D 9 Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuD3D9CtxCreate`                                         |                               |
| `cuD3D9CtxCreateOnDevice`                                 |                               |
| `cuD3D9GetDevice`                                         |                               |
| `cuD3D9GetDevices`                                        |                               |
| `cuD3D9GetDirect3DDevice`                                 |                               |
| `cuGraphicsD3D9RegisterResource`                          |                               |

## **27.1. Direct3D 9 Interoperability [DEPRECATED]**
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

## **28. Direct3D 10 Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuD3D10GetDevice`                                        |                               |
| `cuD3D10GetDevices`                                       |                               |
| `cuGraphicsD3D10RegisterResource`                         |                               |

## **28.1. Direct3D 10 Interoperability [DEPRECATED]**
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

## **29. Direct3D 11 Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuD3D11GetDevice`                                        |                               |
| `cuD3D11GetDevices`                                       |                               |
| `cuGraphicsD3D11RegisterResource`                         |                               |

## **29.1. Direct3D 11 Interoperability [DEPRECATED]**
|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuD3D11CtxCreate`                                        |                               |
| `cuD3D11CtxCreateOnDevice`                                |                               |
| `cuD3D11GetDirect3DDevice`                                |                               |

## **30. VDPAU Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cuGraphicsVDPAURegisterOutputSurface`                    |                               |
| `cuGraphicsVDPAURegisterVideoSurface`                     |                               |
| `cuVDPAUCtxCreate`                                        |                               |
| `cuVDPAUGetDevice`                                        |                               |

## **31. EGL Interoperability**

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
