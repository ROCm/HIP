# CUDA Runtime API functions supported by HIP

## **1. Device Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaChooseDevice`                                        | `hipChooseDevice`             |
| `cudaDeviceGetAttribute`                                  | `hipDeviceGetAttribute`       |
| `cudaDeviceGetByPCIBusId`                                 | `hipDeviceGetByPCIBusId`      |
| `cudaDeviceGetCacheConfig`                                | `hipDeviceGetCacheConfig`     |
| `cudaDeviceGetLimit`                                      | `hipDeviceGetLimit`           |
| `cudaDeviceGetPCIBusId`                                   | `hipDeviceGetPCIBusId`        |
| `cudaDeviceGetSharedMemConfig`                            | `hipDeviceGetSharedMemConfig` |
| `cudaDeviceGetStreamPriorityRange`                        |                               |
| `cudaDeviceReset`                                         | `hipDeviceReset`              |
| `cudaDeviceSetCacheConfig`                                | `hipDeviceSetCacheConfig`     |
| `cudaDeviceSetLimit`                                      | `hipDeviceSetLimit`           |
| `cudaDeviceSetSharedMemConfig`                            | `hipDeviceSetSharedMemConfig` |
| `cudaDeviceSynchronize`                                   | `hipDeviceSynchronize`        |
| `cudaGetDevice`                                           | `hipGetDevice`                |
| `cudaGetDeviceCount`                                      | `hipGetDeviceCount`           |
| `cudaGetDeviceFlags`                                      |                               |
| `cudaGetDeviceProperties`                                 | `hipGetDeviceProperties`      |
| `cudaIpcCloseMemHandle`                                   | `hipIpcCloseMemHandle`        |
| `cudaIpcGetEventHandle`                                   | `hipIpcGetEventHandle`        |
| `cudaIpcGetMemHandle`                                     | `hipIpcGetMemHandle`          |
| `cudaIpcOpenEventHandle`                                  | `hipIpcOpenEventHandle`       |
| `cudaIpcOpenMemHandle`                                    | `hipIpcOpenMemHandle`         |
| `cudaSetDevice`                                           | `hipSetDevice`                |
| `cudaSetDeviceFlags`                                      | `hipSetDeviceFlags`           |
| `cudaSetValidDevices`                                     |                               |

## **2. Thread Management [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaThreadExit`                                          | `hipDeviceReset`              |
| `cudaThreadGetCacheConfig`                                | `hipDeviceGetCacheConfig`     |
| `cudaThreadGetLimit`                                      |                               |
| `cudaThreadSetCacheConfig`                                | `hipDeviceSetCacheConfig`     |
| `cudaThreadSetLimit`                                      |                               |
| `cudaThreadSynchronize`                                   | `hipDeviceSynchronize`        |

## **3. Error Handling**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaGetErrorName`                                        | `hipGetErrorName`             |
| `cudaGetErrorString`                                      | `hipGetErrorString`           |
| `cudaGetLastError`                                        | `hipGetLastError`             |
| `cudaPeekAtLastError`                                     | `hipPeekAtLastError`          |

## **4. Stream Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaStreamAddCallback`                                   | `hipStreamAddCallback`        |
| `cudaStreamAttachMemAsync`                                |                               |
| `cudaStreamCreate`                                        | `hipStreamCreate`             |
| `cudaStreamCreateWithFlags`                               | `hipStreamCreateWithFlags`    |
| `cudaStreamCreateWithPriority`                            |                               |
| `cudaStreamDestroy`                                       | `hipStreamDestroy`            |
| `cudaStreamGetFlags`                                      | `hipStreamGetFlags`           |
| `cudaStreamGetPriority`                                   |                               |
| `cudaStreamQuery`                                         | `hipStreamQuery`              |
| `cudaStreamSynchronize`                                   | `hipStreamSynchronize`        |
| `cudaStreamWaitEvent`                                     | `hipStreamWaitEvent`          |

## **5. Event Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaEventCreate`                                         | `hipEventCreate`              |
| `cudaEventCreateWithFlags`                                | `hipEventCreateWithFlags`     |
| `cudaEventDestroy`                                        | `hipEventDestroy`             |
| `cudaEventElapsedTime`                                    | `hipEventElapsedTime`         |
| `cudaEventQuery`                                          | `hipEventQuery`               |
| `cudaEventRecord`                                         | `hipEventRecord`              |
| `cudaEventSynchronize`                                    | `hipEventSynchronize`         |

## **6. Execution Control**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaFuncGetAttributes`                                   |                               |
| `cudaFuncSetCacheConfig`                                  | `hipFuncSetCacheConfig`       |
| `cudaFuncSetSharedMemConfig`                              |                               |
| `cudaGetParameterBuffer`                                  |                               |
| `cudaGetParameterBufferV2`                                |                               |
| `cudaLaunchKernel`                                        | `hipLaunchKernel`             |
| `cudaSetDoubleForDevice`                                  |                               |
| `cudaSetDoubleForHost`                                    |                               |

## **7. Occupancy**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor`           | `hipOccupancyMaxActiveBlocksPerMultiprocessor`|
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`  |                               |

## **8. Execution Control [deprecated since 7.0]**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaConfigureCall`                                       |                               |
| `cudaLaunch`                                              |                               |
| `cudaSetupArgument`                                       |                               |

## **9. Memory Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaArrayGetInfo`                                        |                               |
| `cudaFree`                                                | `hipFree`                     |
| `cudaFreeArray`                                           | `hipFreeArray`                |
| `cudaFreeHost`                                            | `hipHostFree`                 |
| `cudaFreeMipmappedArray`                                  |                               |
| `cudaGetMipmappedArrayLevel`                              |                               |
| `cudaGetSymbolAddress`                                    |                               |
| `cudaGetSymbolSize`                                       |                               |
| `cudaHostAlloc`                                           | `hipHostMalloc`               |
| `cudaHostGetDevicePointer`                                | `hipHostGetDevicePointer`     |
| `cudaHostGetFlags`                                        | `hipHostGetFlags`             |
| `cudaHostRegister`                                        | `hipHostRegister`             |
| `cudaHostUnregister`                                      | `hipHostUnregister`           |
| `cudaMalloc`                                              | `hipMalloc`                   |
| `cudaMalloc3D`                                            |                               |
| `cudaMalloc3DArray`                                       | `hipMalloc3DArray`            |
| `cudaMallocArray`                                         | `hipMallocArray`              |
| `cudaMallocHost`                                          | `hipHostMalloc`               |
| `cudaMallocManaged`                                       |                               |
| `cudaMallocMipmappedArray`                                |                               |
| `cudaMallocPitch`                                         |                               |
| `cudaMemGetInfo`                                          | `hipMemGetInfo`               |
| `cudaMemcpy`                                              | `hipMemcpy`                   |
| `cudaMemcpy2D`                                            | `hipMemcpy2D`                 |
| `cudaMemcpy2DArrayToArray`                                |                               |
| `cudaMemcpy2DAsync`                                       |                               |
| `cudaMemcpy2DFromArray`                                   |                               |
| `cudaMemcpy2DFromArrayAsync`                              |                               |
| `cudaMemcpy2DToArray`                                     | `hipMemcpy2DToArray`          |
| `cudaMemcpy2DToArrayAsync`                                |                               |
| `cudaMemcpy3D`                                            | `hipMemcpy3D`                 |
| `cudaMemcpy3DAsync`                                       |                               |
| `cudaMemcpy3DPeer`                                        |                               |
| `cudaMemcpy3DPeerAsync`                                   |                               |
| `cudaMemcpyArrayToArray`                                  |                               |
| `cudaMemcpyAsync`                                         | `hipMemcpyAsync`              |
| `cudaMemcpyFromArray`                                     | `MemcpyFromArray`             |
| `cudaMemcpyFromArrayAsync`                                |                               |
| `cudaMemcpyFromSymbol`                                    | `hipMemcpyFromSymbol`         |
| `cudaMemcpyFromSymbolAsync`                               |                               |
| `cudaMemcpyPeer`                                          | `hipMemcpyPeer`               |
| `cudaMemcpyPeerAsync`                                     | `hipMemcpyPeerAsync`          |
| `cudaMemcpyToArray`                                       | `hipMemcpyToArray`            |
| `cudaMemcpyToArrayAsync`                                  |                               |
| `cudaMemcpyToSymbol`                                      | `hipMemcpyToSymbol`           |
| `cudaMemcpyToSymbolAsync`                                 | `hipMemcpyToSymbolAsync`      |
| `cudaMemset`                                              | `hipMemset`                   |
| `cudaMemset2D`                                            | `hipMemset2D`                 |
| `cudaMemset2DAsync`                                       |                               |
| `cudaMemset3D`                                            |                               |
| `cudaMemset3DAsync`                                       |                               |
| `cudaMemsetAsync`                                         | `hipMemsetAsync`              |
| `make_cudaExtent`                                         | `make_hipExtent`              |
| `make_cudaPitchedPtr`                                     | `make_hipPitchedPtr`          |
| `make_cudaPos`                                            | `make_hipPos`                 |

## **10. Unified Addressing**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaPointerGetAttributes`                                | `hipPointerGetAttributes`     |

## **11. Peer Device Memory Access**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaDeviceCanAccessPeer`                                 | `hipDeviceCanAccessPeer`      |
| `cudaDeviceDisablePeerAccess`                             | `hipDeviceDisablePeerAccess`  |
| `cudaDeviceEnablePeerAccess`                              | `hipDeviceEnablePeerAccess`   |

## **12. OpenGL Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaGLGetDevices`                                        |                               |
| `cudaGraphicsGLRegisterBuffer`                            |                               |
| `cudaGraphicsGLRegisterImage`                             |                               |
| `cudaWGLGetDevice`                                        |                               |

## **13. OpenGL Interoperability [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaGLMapBufferObject`                                   |                               |
| `cudaGLMapBufferObjectAsync`                              |                               |
| `cudaGLRegisterBufferObject`                              |                               |
| `cudaGLSetBufferObjectMapFlags`                           |                               |
| `cudaGLSetGLDevice`                                       |                               |
| `cudaGLUnmapBufferObject`                                 |                               |
| `cudaGLUnmapBufferObjectAsync`                            |                               |
| `cudaGLUnregisterBufferObject`                            |                               |

## **14. Direct3D 9 Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaD3D9GetDevice`                                       |                               |
| `cudaD3D9GetDevices`                                      |                               |
| `cudaD3D9GetDirect3DDevice`                               |                               |
| `cudaD3D9SetDirect3DDevice`                               |                               |
| `cudaGraphicsD3D9RegisterResource`                        |                               |

## **15. Direct3D 9 Interoperability [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaD3D9MapResources`                                    |                               |
| `cudaD3D9RegisterResource`                                |                               |
| `cudaD3D9ResourceGetMappedArray`                          |                               |
| `cudaD3D9ResourceGetMappedPitch`                          |                               |
| `cudaD3D9ResourceGetMappedPointer`                        |                               |
| `cudaD3D9ResourceGetMappedSize`                           |                               |
| `cudaD3D9ResourceGetSurfaceDimensions`                    |                               |
| `cudaD3D9ResourceSetMapFlags`                             |                               |
| `cudaD3D9UnmapResources`                                  |                               |
| `cudaD3D9UnregisterResource`                              |                               |

## **16. Direct3D 10 Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaD3D10GetDevice`                                      |                               |
| `cudaD3D10GetDevices`                                     |                               |
| `cudaGraphicsD3D10RegisterResource`                       |                               |

## **17. Direct3D 10 Interoperability [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaD3D10GetDirect3DDevice`                              |                               |
| `cudaD3D10MapResources`                                   |                               |
| `cudaD3D10RegisterResource`                               |                               |
| `cudaD3D10ResourceGetMappedArray`                         |                               |
| `cudaD3D10ResourceGetMappedPitch`                         |                               |
| `cudaD3D10ResourceGetMappedPointer`                       |                               |
| `cudaD3D10ResourceGetMappedSize`                          |                               |
| `cudaD3D10ResourceGetSurfaceDimensions`                   |                               |
| `cudaD3D10ResourceSetMapFlags`                            |                               |
| `cudaD3D10SetDirect3DDevice`                              |                               |
| `cudaD3D10UnmapResources`                                 |                               |
| `cudaD3D10UnregisterResource`                             |                               |

## **18. Direct3D 11 Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaD3D11GetDevice`                                      |                               |
| `cudaD3D11GetDevices`                                     |                               |
| `cudaGraphicsD3D11RegisterResource`                       |                               |

## **19. Direct3D 11 Interoperability [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaD3D11GetDirect3DDevice`                              |                               |
| `cudaD3D11SetDirect3DDevice`                              |                               |

## **20. VDPAU Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaGraphicsVDPAURegisterOutputSurface`                  |                               |
| `cudaGraphicsVDPAURegisterVideoSurface`                   |                               |
| `cudaVDPAUGetDevice`                                      |                               |
| `cudaVDPAUSetVDPAUDevice`                                 |                               |

## **21. EGL Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaEGLStreamConsumerAcquireFrame`                       |                               |
| `cudaEGLStreamConsumerConnect`                            |                               |
| `cudaEGLStreamConsumerConnectWithFlags`                   |                               |
| `cudaEGLStreamConsumerReleaseFrame`                       |                               |
| `cudaEGLStreamProducerConnect`                            |                               |
| `cudaEGLStreamProducerDisconnect`                         |                               |
| `cudaEGLStreamProducerPresentFrame`                       |                               |
| `cudaEGLStreamProducerReturnFrame`                        |                               |
| `cudaGraphicsEGLRegisterImage`                            |                               |
| `cudaGraphicsResourceGetMappedEglFrame`                   |                               |

## **22. Graphics Interoperability**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaGraphicsMapResources`                                |                               |
| `cudaGraphicsResourceGetMappedMipmappedArray`             |                               |
| `cudaGraphicsResourceGetMappedPointer`                    |                               |
| `cudaGraphicsResourceSetMapFlags`                         |                               |
| `cudaGraphicsSubResourceGetMappedArray`                   |                               |
| `cudaGraphicsUnmapResources`                              |                               |
| `cudaGraphicsUnregisterResource`                          |                               |

## **23. Texture Reference Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaBindTexture`                                         | `hipBindTexture`              |
| `cudaBindTexture2D`                                       | `hipBindTexture2D`            |
| `cudaBindTextureToArray`                                  | `hipBindTextureToArray`       |
| `cudaBindTextureToMipmappedArray`                         |                               |
| `cudaCreateChannelDesc`                                   | `hipCreateChannelDesc`        |
| `cudaGetChannelDesc`                                      | `hipGetChannelDesc`           |
| `cudaGetTextureAlignmentOffset`                           |                               |
| `cudaGetTextureReference`                                 |                               |
| `cudaUnbindTexture`                                       | `hipUnbindTexture`            |

## **24. Surface Reference Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaBindSurfaceToArray`                                  |                               |
| `cudaGetSurfaceReference`                                 |                               |

## **25. Texture Object Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaCreateTextureObject`                                 |`hipCreateTextureObject`       |
| `cudaDestroyTextureObject`                                |`hipDestroyTextureObject`      |
| `cudaGetTextureObjectResourceDesc`                        |`hipGetTextureObjectResourceDesc` |
| `cudaGetTextureObjectResourceViewDesc`                    |`hipGetTextureObjectResourceViewDesc` |
| `cudaGetTextureObjectTextureDesc`                         |`hipGetTextureObjectTextureDesc` |

## **26. Surface Object Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaCreateSurfaceObject`                                 |                               |
| `cudaDestroySurfaceObject`                                |                               |
| `cudaGetSurfaceObjectResourceDesc`                        |                               |

## **27. Version Management**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaDriverGetVersion`                                    | `hipDriverGetVersion`         |
| `cudaRuntimeGetVersion`                                   | `hipRuntimeGetVersion`        |

## **28. C++ API Routines**
*(7.0 contains, 7.5 doesn’t)*

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaBindSurfaceToArray`                                  |                               |
| `cudaBindTexture`                                         | `hipBindTexture`              |
| `cudaBindTexture2D`                                       |                               |
| `cudaBindTextureToArray`                                  |                               |
| `cudaBindTextureToMipmappedArray`                         |                               |
| `cudaCreateChannelDesc`                                   | `hipCreateChannelDesc`        |
| `cudaFuncGetAttributes`                                   |                               |
| `cudaFuncSetCacheConfig`                                  |                               |
| `cudaGetSymbolAddress`                                    |                               |
| `cudaGetSymbolSize`                                       |                               |
| `cudaGetTextureAlignmentOffset`                           |                               |
| `cudaLaunch`                                              |                               |
| `cudaLaunchKernel`                                        |                               |
| `cudaMallocHost`                                          |                               |
| `cudaMallocManaged`                                       |                               |
| `cudaMemcpyFromSymbol`                                    |                               |
| `cudaMemcpyFromSymbolAsync`                               |                               |
| `cudaMemcpyToSymbol`                                      |                               |
| `cudaMemcpyToSymbolAsync`                                 |                               |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor`           | `hipOccupancyMaxActiveBlocksPerMultiprocessor` |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`  |                               |
| `cudaOccupancyMaxPotentialBlockSize`                      | `hipOccupancyMaxPotentialBlockSize` |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMem`          |                               |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags` |                               |
| `cudaOccupancyMaxPotentialBlockSizeWithFlags`             |                               |
| `cudaSetupArgument`                                       |                               |
| `cudaStreamAttachMemAsync`                                |                               |
| `cudaUnbindTexture`                                       | `hipUnbindTexture`            |

## **30. Profiler Control**

|   **CUDA**                                                |   **HIP**                     |
|-----------------------------------------------------------|-------------------------------|
| `cudaProfilerInitialize`                                  |                               |
| `cudaProfilerStart`                                       | `hipProfilerStart`            |
| `cudaProfilerStop`                                        | `hipProfilerStop`             |

# Data types used by CUDA Runtime API and supported by HIP

## **31. Data types**

| **type**     |   **CUDA**                                    |   **HIP**                                            |
|-------------:|-----------------------------------------------|------------------------------------------------------|
| struct       | `cudaChannelFormatDesc`                       | `hipChannelFormatDesc`                               |
| struct       | `cudaDeviceProp`                              | `hipDeviceProp_t`                                    |
| struct       | `cudaExtent`                                  | `hipExtent`                                          |
| struct       | `cudaFuncAttributes`                          |                                                      |
| struct       | `cudaIpcEventHandle_t`                        | `hipIpcEventHandle_t`                                |
| struct       | `cudaIpcMemHandle_t`                          | `hipIpcMemHandle_t`                                  |
| struct       | `cudaMemcpy3DParms`                           | `hipMemcpy3DParms`                                   |
| struct       | `cudaMemcpy3DPeerParms`                       |                                                      |
| struct       | `cudaPitchedPtr`                              | `hipPitchedPtr`                                      |
| struct       | `cudaPointerAttributes`                       | `hipPointerAttribute_t`                              |
| struct       | `cudaPos`                                     | `hipPos`                                             |
| struct       | `cudaResourceDesc`                            | `hipResourceDesc`                                    |
| struct       | `cudaResourceViewDesc`                        | `hipResourceViewDesc`                                |
| struct       | `cudaTextureDesc`                             | `hipTextureDesc`                                     |
| struct       | `surfaceReference`                            |                                                      |
| struct       | `textureReference`                            | `textureReference`                                   |
| enum         |***`cudaChannelFormatKind`***                  |***`hipChannelFormatKind`***                          |
|            0 |*`cudaChannelFormatKindSigned`*                |*`hipChannelFormatKindSigned`*                        |
|            1 |*`cudaChannelFormatKindUnsigned`*              |*`hipChannelFormatKindUnsigned`*                      |
|            2 |*`cudaChannelFormatKindFloat`*                 |*`hipChannelFormatKindFloat`*                         |
|            3 |*`cudaChannelFormatKindNone`*                  |*`hipChannelFormatKindNone`*                          |
| enum         |***`cudaComputeMode`***                        |                                                      |
|            0 |*`cudaComputeModeDefault`*                     |                                                      |
|            1 |*`cudaComputeModeExclusive`*                   |                                                      |
|            2 |*`cudaComputeModeProhibited`*                  |                                                      |
|            3 |*`cudaComputeModeExclusiveProcess`*            |                                                      |
| enum         |***`cudaDeviceAttr`***                         |***`hipDeviceAttribute_t`***                          |
|            1 |*`cudaDevAttrMaxThreadsPerBlock`*              |*`hipDeviceAttributeMaxThreadsPerBlock`*              |
|            2 |*`cudaDevAttrMaxBlockDimX`*                    |*`hipDeviceAttributeMaxBlockDimX`*                    |
|            3 |*`cudaDevAttrMaxBlockDimY`*                    |*`hipDeviceAttributeMaxBlockDimY`*                    |
|            4 |*`cudaDevAttrMaxBlockDimZ`*                    |*`hipDeviceAttributeMaxBlockDimZ`*                    |
|            5 |*`cudaDevAttrMaxGridDimX`*                     |*`hipDeviceAttributeMaxGridDimX`*                     |
|            6 |*`cudaDevAttrMaxGridDimY`*                     |*`hipDeviceAttributeMaxGridDimY`*                     |
|            7 |*`cudaDevAttrMaxGridDimZ`*                     |*`hipDeviceAttributeMaxGridDimZ`*                     |
|            8 |*`cudaDevAttrMaxSharedMemoryPerBlock`*         |*`hipDeviceAttributeMaxSharedMemoryPerBlock`*         |
|            9 |*`cudaDevAttrTotalConstantMemory`*             |*`hipDeviceAttributeTotalConstantMemory`*             |
|           10 |*`cudaDevAttrWarpSize`*                        |*`hipDeviceAttributeWarpSize`*                        |
|           11 |*`cudaDevAttrMaxPitch`*                        |                                                      |
|           12 |*`cudaDevAttrMaxRegistersPerBlock`*            |*`hipDeviceAttributeMaxRegistersPerBlock`*            |
|           13 |*`cudaDevAttrClockRate`*                       |*`hipDeviceAttributeClockRate`*                       |
|           14 |*`cudaDevAttrTextureAlignment`*                |                                                      |
|           15 |*`cudaDevAttrGpuOverlap`*                      |                                                      |
|           16 |*`cudaDevAttrMultiProcessorCount`*             |*`hipDeviceAttributeMultiprocessorCount`*             |
|           17 |*`cudaDevAttrKernelExecTimeout`*               |                                                      |
|           18 |*`cudaDevAttrIntegrated`*                      |                                                      |
|           19 |*`cudaDevAttrCanMapHostMemory`*                |                                                      |
|           20 |*`cudaDevAttrComputeMode`*                     |*`hipDeviceAttributeComputeMode`*                     |
|           21 |*`cudaDevAttrMaxTexture1DWidth`*               |                                                      |
|           22 |*`cudaDevAttrMaxTexture2DWidth`*               |                                                      |
|           23 |*`cudaDevAttrMaxTexture2DHeight`*              |                                                      |
|           24 |*`cudaDevAttrMaxTexture3DWidth`*               |                                                      |
|           25 |*`cudaDevAttrMaxTexture3DHeight`*              |                                                      |
|           26 |*`cudaDevAttrMaxTexture3DDepth`*               |                                                      |
|           27 |*`cudaDevAttrMaxTexture2DLayeredWidth`*        |                                                      |
|           28 |*`cudaDevAttrMaxTexture2DLayeredHeight`*       |                                                      |
|           29 |*`cudaDevAttrMaxTexture2DLayeredLayers`*       |                                                      |
|           30 |*`cudaDevAttrSurfaceAlignment`*                |                                                      |
|           31 |*`cudaDevAttrConcurrentKernels`*               |*`hipDeviceAttributeConcurrentKernels`*               |
|           32 |*`cudaDevAttrEccEnabled`*                      |                                                      |
|           33 |*`cudaDevAttrPciBusId`*                        |*`hipDeviceAttributePciBusId`*                        |
|           34 |*`cudaDevAttrPciDeviceId`*                     |*`hipDeviceAttributePciDeviceId`*                     |
|           35 |*`cudaDevAttrTccDriver`*                       |                                                      |
|           36 |*`cudaDevAttrMemoryClockRate`*                 |*`hipDeviceAttributeMemoryClockRate`*                 |
|           37 |*`cudaDevAttrGlobalMemoryBusWidth`*            |*`hipDeviceAttributeMemoryBusWidth`*                  |
|           38 |*`cudaDevAttrL2CacheSize`*                     |*`hipDeviceAttributeL2CacheSize`*                     |
|           39 |*`cudaDevAttrMaxThreadsPerMultiProcessor`*     |*`hipDeviceAttributeMaxThreadsPerMultiProcessor`*     |
|           40 |*`cudaDevAttrAsyncEngineCount`*                |                                                      |
|           41 |*`cudaDevAttrUnifiedAddressing`*               |                                                      |
|           42 |*`cudaDevAttrMaxTexture1DLayeredWidth`*        |                                                      |
|           43 |*`cudaDevAttrMaxTexture1DLayeredLayers`*       |                                                      |
|           44 |                                               |                                                      |
|           45 |*`cudaDevAttrMaxTexture2DGatherWidth`*         |                                                      |
|           46 |*`cudaDevAttrMaxTexture2DGatherHeight`*        |                                                      |
|           47 |*`cudaDevAttrMaxTexture3DWidthAlt`*            |                                                      |
|           48 |*`cudaDevAttrMaxTexture3DHeightAlt`*           |                                                      |
|           49 |*`cudaDevAttrMaxTexture3DDepthAlt`*            |                                                      |
|           50 |*`cudaDevAttrPciDomainId`*                     |                                                      |
|           51 |*`cudaDevAttrTexturePitchAlignment`*           |                                                      |
|           52 |*`cudaDevAttrMaxTextureCubemapWidth`*          |                                                      |
|           53 |*`cudaDevAttrMaxTextureCubemapLayeredWidth`*   |                                                      |
|           54 |*`cudaDevAttrMaxTextureCubemapLayeredLayers`*  |                                                      |
|           55 |*`cudaDevAttrMaxSurface1DWidth`*               |                                                      |
|           56 |*`cudaDevAttrMaxSurface2DWidth`*               |                                                      |
|           57 |*`cudaDevAttrMaxSurface2DHeight`*              |                                                      |
|           58 |*`cudaDevAttrMaxSurface3DWidth`*               |                                                      |
|           59 |*`cudaDevAttrMaxSurface3DHeight`*              |                                                      |
|           60 |*`cudaDevAttrMaxSurface3DDepth`*               |                                                      |
|           61 |*`cudaDevAttrMaxSurface1DLayeredWidth`*        |                                                      |
|           62 |*`cudaDevAttrMaxSurface1DLayeredLayers`*       |                                                      |
|           63 |*`cudaDevAttrMaxSurface2DLayeredWidth`*        |                                                      |
|           64 |*`cudaDevAttrMaxSurface2DLayeredHeight`*       |                                                      |
|           65 |*`cudaDevAttrMaxSurface2DLayeredLayers`*       |                                                      |
|           66 |*`cudaDevAttrMaxSurfaceCubemapWidth`*          |                                                      |
|           67 |*`cudaDevAttrMaxSurfaceCubemapLayeredWidth`*   |                                                      |
|           68 |*`cudaDevAttrMaxSurfaceCubemapLayeredLayers`*  |                                                      |
|           69 |*`cudaDevAttrMaxTexture1DLinearWidth`*         |                                                      |
|           70 |*`cudaDevAttrMaxTexture2DLinearWidth`*         |                                                      |
|           71 |*`cudaDevAttrMaxTexture2DLinearHeight`*        |                                                      |
|           72 |*`cudaDevAttrMaxTexture2DLinearPitch`*         |                                                      |
|           73 |*`cudaDevAttrMaxTexture2DMipmappedWidth`*      |                                                      |
|           74 |*`cudaDevAttrMaxTexture2DMipmappedHeight`*     |                                                      |
|           75 |*`cudaDevAttrComputeCapabilityMajor`*          |*`hipDeviceAttributeComputeCapabilityMajor`*          |
|           76 |*`cudaDevAttrComputeCapabilityMinor`*          |*`hipDeviceAttributeComputeCapabilityMinor`*          |
|           77 |*`cudaDevAttrMaxTexture1DMipmappedWidth`*      |                                                      |
|           78 |*`cudaDevAttrStreamPrioritiesSupported`*       |                                                      |
|           79 |*`cudaDevAttrGlobalL1CacheSupported`*          |                                                      |
|           80 |*`cudaDevAttrLocalL1CacheSupported`*           |                                                      |
|           81 |*`cudaDevAttrMaxSharedMemoryPerMultiprocessor`*|*`hipDeviceAttributeMaxSharedMemoryPerMultiprocessor`*|
|           82 |*`cudaDevAttrMaxRegistersPerMultiprocessor`*   |                                                      |
|           83 |*`cudaDevAttrManagedMemory`*                   |                                                      |
|           84 |*`cudaDevAttrIsMultiGpuBoard`*                 |*`hipDeviceAttributeIsMultiGpuBoard`*                 |
|           85 |*`cudaDevAttrMultiGpuBoardGroupID`*            |                                                      |
| enum         |***`cudaError`***                              |***`hipError_t`***                                    |
| enum         |***`cudaError_t`***                            |***`hipError_t`***                                    |
|            0 |*`cudaSuccess`*                                |*`hipSuccess`*                                        |
|            1 |*`cudaErrorMissingConfiguration`*              |                                                      |
|            2 |*`cudaErrorMemoryAllocation`*                  |*`hipErrorMemoryAllocation`*                          |
|            3 |*`cudaErrorInitializationError`*               |*`hipErrorInitializationError`*                       |
|            4 |*`cudaErrorLaunchFailure`*                     |                                                      |
|            5 |*`cudaErrorPriorLaunchFailure`*                |                                                      |
|            6 |*`cudaErrorLaunchTimeout`*                     |                                                      |
|            7 |*`cudaErrorLaunchOutOfResources`*              |*`hipErrorLaunchOutOfResources`*                      |
|            8 |*`cudaErrorInvalidDeviceFunction`*             |                                                      |
|            9 |*`cudaErrorInvalidConfiguration`*              |                                                      |
|           10 |*`cudaErrorInvalidDevice`*                     |*`hipErrorInvalidDevice`*                             |
|           11 |*`cudaErrorInvalidValue`*                      |*`hipErrorInvalidValue`*                              |
|           12 |*`cudaErrorInvalidPitchValue`*                 |                                                      |
|           13 |*`cudaErrorInvalidSymbol`*                     |                                                      |
|           14 |*`cudaErrorMapBufferObjectFailed`*             |                                                      |
|           15 |*`cudaErrorUnmapBufferObjectFailed`*           |                                                      |
|           16 |*`cudaErrorInvalidHostPointer`*                |                                                      |
|           17 |*`cudaErrorInvalidDevicePointer`*              |*`hipErrorInvalidDevicePointer`*                      |
|           18 |*`cudaErrorInvalidTexture`*                    |                                                      |
|           19 |*`cudaErrorInvalidTextureBinding`*             |                                                      |
|           20 |*`cudaErrorInvalidChannelDescriptor`*          |                                                      |
|           21 |*`cudaErrorInvalidMemcpyDirection`*            |                                                      |
|           22 |*`cudaErrorAddressOfConstant`*                 |                                                      |
|           23 |*`cudaErrorTextureFetchFailed`*                |                                                      |
|           24 |*`cudaErrorTextureNotBound`*                   |                                                      |
|           25 |*`cudaErrorSynchronizationError`*              |                                                      |
|           26 |*`cudaErrorInvalidFilterSetting`*              |                                                      |
|           27 |*`cudaErrorInvalidNormSetting`*                |                                                      |
|           28 |*`cudaErrorMixedDeviceExecution`*              |                                                      |
|           29 |*`cudaErrorCudartUnloading`*                   |                                                      |
|           30 |*`cudaErrorUnknown`*                           |*`hipErrorUnknown`*                                   |
|           31 |*`cudaErrorNotYetImplemented`*                 |                                                      |
|           32 |*`cudaErrorMemoryValueTooLarge`*               |                                                      |
|           33 |*`cudaErrorInvalidResourceHandle`*             |*`hipErrorInvalidResourceHandle`*                     |
|           34 |*`cudaErrorNotReady`*                          |*`hipErrorNotReady`*                                  |
|           35 |*`cudaErrorInsufficientDriver`*                |                                                      |
|           36 |*`cudaErrorSetOnActiveProcess`*                |                                                      |
|           37 |*`cudaErrorInvalidSurface`*                    |                                                      |
|           38 |*`cudaErrorNoDevice`*                          |*`hipErrorNoDevice`*                                  |
|           39 |*`cudaErrorECCUncorrectable`*                  |                                                      |
|           40 |*`cudaErrorSharedObjectSymbolNotFound`*        |                                                      |
|           41 |*`cudaErrorSharedObjectInitFailed`*            |                                                      |
|           42 |*`cudaErrorUnsupportedLimit`*                  |*`hipErrorUnsupportedLimit`*                          |
|           43 |*`cudaErrorDuplicateVariableName`*             |                                                      |
|           44 |*`cudaErrorDuplicateTextureName`*              |                                                      |
|           45 |*`cudaErrorDuplicateSurfaceName`*              |                                                      |
|           46 |*`cudaErrorDevicesUnavailable`*                |                                                      |
|           47 |*`cudaErrorInvalidKernelImage`*                |                                                      |
|           48 |*`cudaErrorNoKernelImageForDevice`*            |                                                      |
|           49 |*`cudaErrorIncompatibleDriverContext`*         |                                                      |
|           50 |*`cudaErrorPeerAccessAlreadyEnabled`*          |*`hipErrorPeerAccessAlreadyEnabled`*                  |
|           51 |*`cudaErrorPeerAccessNotEnabled`*              |*`hipErrorPeerAccessNotEnabled`*                      |
|           52 |                                               |                                                      |
|           53 |                                               |                                                      |
|           54 |*`cudaErrorDeviceAlreadyInUse`*                |                                                      |
|           55 |*`cudaErrorProfilerDisabled`*                  |                                                      |
|           56 |*`cudaErrorProfilerNotInitialized`*            |                                                      |
|           57 |*`cudaErrorProfilerAlreadyStarted`*            |                                                      |
|           58 |*`cudaErrorProfilerAlreadyStopped`*            |                                                      |
|           59 |*`cudaErrorAssert`*                            |                                                      |
|           60 |*`cudaErrorTooManyPeers`*                      |                                                      |
|           61 |*`cudaErrorHostMemoryAlreadyRegistered`*       | *`hipErrorHostMemoryAlreadyRegistered`*              |
|           62 |*`cudaErrorHostMemoryNotRegistered`*           | *`hipErrorHostMemoryNotRegistered`*                  |
|           63 |*`cudaErrorOperatingSystem`*                   |                                                      |
|           64 |*`cudaErrorPeerAccessUnsupported`*             |                                                      |
|           65 |*`cudaErrorLaunchMaxDepthExceeded`*            |                                                      |
|           66 |*`cudaErrorLaunchFileScopedTex`*               |                                                      |
|           67 |*`cudaErrorLaunchFileScopedSurf`*              |                                                      |
|           68 |*`cudaErrorSyncDepthExceeded`*                 |                                                      |
|           69 |*`cudaErrorLaunchPendingCountExceeded`*        |                                                      |
|           70 |*`cudaErrorNotPermitted`*                      |                                                      |
|           71 |*`cudaErrorNotSupported`*                      |                                                      |
|           72 |*`cudaErrorHardwareStackError`*                |                                                      |
|           73 |*`cudaErrorIllegalInstruction`*                |                                                      |
|           74 |*`cudaErrorMisalignedAddress`*                 |                                                      |
|           75 |*`cudaErrorInvalidAddressSpace`*               |                                                      |
|           76 |*`cudaErrorInvalidPc`*                         |                                                      |
|           77 |*`cudaErrorIllegalAddress`*                    |                                                      |
|           78 |*`cudaErrorInvalidPtx`*                        |                                                      |
|           79 |*`cudaErrorInvalidGraphicsContext`*            |                                                      |
|         0x7f |*`cudaErrorStartupFailure`*                    |                                                      |
|         1000 |*`cudaErrorApiFailureBase`*                    |                                                      |
| enum         |***`cudaFuncCache`***                          |***`hipFuncCache_t`***                                |
|            0 |*`cudaFuncCachePreferNone`*                    |*`hipFuncCachePreferNone`*                            |
|            1 |*`cudaFuncCachePreferShared`*                  |*`hipFuncCachePreferShared`*                          |
|            2 |*`cudaFuncCachePreferL1`*                      |*`hipFuncCachePreferL1`*                              |
|            3 |*`cudaFuncCachePreferEqual`*                   |*`hipFuncCachePreferEqual`*                           |
| enum         |***`cudaGraphicsCubeFace`***                   |                                                      |
|         0x00 |*`cudaGraphicsCubeFacePositiveX`*              |                                                      |
|         0x01 |*`cudaGraphicsCubeFaceNegativeX`*              |                                                      |
|         0x02 |*`cudaGraphicsCubeFacePositiveY`*              |                                                      |
|         0x03 |*`cudaGraphicsCubeFaceNegativeY`*              |                                                      |
|         0x04 |*`cudaGraphicsCubeFacePositiveZ`*              |                                                      |
|         0x05 |*`cudaGraphicsCubeFaceNegativeZ`*              |                                                      |
| enum         |***`cudaGraphicsMapFlags`***                   |                                                      |
|            0 |*`cudaGraphicsMapFlagsNone`*                   |                                                      |
|            1 |*`cudaGraphicsMapFlagsReadOnly`*               |                                                      |
|            2 |*`cudaGraphicsMapFlagsWriteDiscard`*           |                                                      |
| enum         |***`cudaGraphicsRegisterFlags`***              |                                                      |
|            0 |*`cudaGraphicsRegisterFlagsNone`*              |                                                      |
|            1 |*`cudaGraphicsRegisterFlagsReadOnly`*          |                                                      |
|            2 |*`cudaGraphicsRegisterFlagsWriteDiscard`*      |                                                      |
|            4 |*`cudaGraphicsRegisterFlagsSurfaceLoadStore`*  |                                                      |
|            8 |*`cudaGraphicsRegisterFlagsTextureGather`*     |                                                      |
| enum         |***`cudaLimit`***                              |***`hipLimit_t`***                                    |
|         0x00 |*`cudaLimitStackSize`*                         |                                                      |
|         0x01 |*`cudaLimitPrintfFifoSize`*                    |                                                      |
|         0x02 |*`cudaLimitMallocHeapSize`*                    |*`hipLimitMallocHeapSize`*                            |
|         0x03 |*`cudaLimitDevRuntimeSyncDepth`*               |                                                      |
|         0x04 |*`cudaLimitDevRuntimePendingLaunchCount`*      |                                                      |
| enum         |***`cudaMemcpyKind`***                         |***`hipMemcpyKind`***                                 |
|            0 |*`cudaMemcpyHostToHost`*                       |*`hipMemcpyHostToHost`*                               |
|            1 |*`cudaMemcpyHostToDevice`*                     |*`hipMemcpyHostToDevice`*                             |
|            2 |*`cudaMemcpyDeviceToHost`*                     |*`hipMemcpyDeviceToHost`*                             |
|            3 |*`cudaMemcpyDeviceToDevice`*                   |*`hipMemcpyDeviceToDevice`*                           |
|            4 |*`cudaMemcpyDefault`*                          |*`hipMemcpyDefault`*                                  |
| enum         |***`cudaMemoryType`***                         |***`hipMemoryType`***                                 |
|            1 |*`cudaMemoryTypeHost`*                         |*`hipMemoryTypeHost`*                                 |
|            2 |*`cudaMemoryTypeDevice`*                       |*`hipMemoryTypeDevice`*                               |
| enum         |***`cudaResourceType`***                       |***`hipResourceType`***                               |
|            0 |*`cudaResourceTypeArray`*                      |*`hipResourceTypeArray`*                              |
|            1 |*`cudaResourceTypeMipmappedArray`*             |*`hipResourceTypeMipmappedArray`*                     |
|            2 |*`cudaResourceTypeLinear`*                     |*`hipResourceTypeLinear`*                             |
|            3 |*`cudaResourceTypePitch2D`*                    |*`hipResourceTypePitch2D`*                            |
| enum         |***`cudaResourceViewFormat`***                 |***`hipResourceViewFormat`***                         |
|         0x00 |*`cudaResViewFormatNone`*                      |*`hipResViewFormatNone`*                              |
|         0x01 |*`cudaResViewFormatUnsignedChar1`*             |*`hipResViewFormatUnsignedChar1`*                     |
|         0x02 |*`cudaResViewFormatUnsignedChar2`*             |*`hipResViewFormatUnsignedChar2`*                     |
|         0x03 |*`cudaResViewFormatUnsignedChar4`*             |*`hipResViewFormatUnsignedChar4`*                     |
|         0x04 |*`cudaResViewFormatSignedChar1`*               |*`hipResViewFormatSignedChar1`*                       |
|         0x05 |*`cudaResViewFormatSignedChar2`*               |*`hipResViewFormatSignedChar2`*                       |
|         0x06 |*`cudaResViewFormatSignedChar4`*               |*`hipResViewFormatSignedChar4`*                       |
|         0x07 |*`cudaResViewFormatUnsignedShort1`*            |*`hipResViewFormatUnsignedShort1`*                    |
|         0x08 |*`cudaResViewFormatUnsignedShort2`*            |*`hipResViewFormatUnsignedShort2`*                    |
|         0x09 |*`cudaResViewFormatUnsignedShort4`*            |*`hipResViewFormatUnsignedShort4`*                    |
|         0x0a |*`cudaResViewFormatSignedShort1`*              |*`hipResViewFormatSignedShort1`*                      |
|         0x0b |*`cudaResViewFormatSignedShort2`*              |*`hipResViewFormatSignedShort2`*                      |
|         0x0c |*`cudaResViewFormatSignedShort4`*              |*`hipResViewFormatSignedShort4`*                      |
|         0x0d |*`cudaResViewFormatUnsignedInt1`*              |*`hipResViewFormatUnsignedInt1`*                      |
|         0x0e |*`cudaResViewFormatUnsignedInt2`*              |*`hipResViewFormatUnsignedInt2`*                      |
|         0x0f |*`cudaResViewFormatUnsignedInt4`*              |*`hipResViewFormatUnsignedInt4`*                      |
|         0x10 |*`cudaResViewFormatSignedInt1`*                |*`hipResViewFormatSignedInt1`*                        |
|         0x11 |*`cudaResViewFormatSignedInt2`*                |*`hipResViewFormatSignedInt2`*                        |
|         0x12 |*`cudaResViewFormatSignedInt4`*                |*`hipResViewFormatSignedInt4`*                        |
|         0x13 |*`cudaResViewFormatHalf1`*                     |*`hipResViewFormatHalf1`*                             |
|         0x14 |*`cudaResViewFormatHalf2`*                     |*`hipResViewFormatHalf2`*                             |
|         0x15 |*`cudaResViewFormatHalf4`*                     |*`hipResViewFormatHalf4`*                             |
|         0x16 |*`cudaResViewFormatFloat1`*                    |*`hipResViewFormatFloat1`*                            |
|         0x17 |*`cudaResViewFormatFloat2`*                    |*`hipResViewFormatFloat2`*                            |
|         0x18 |*`cudaResViewFormatFloat4`*                    |*`hipResViewFormatFloat4`*                            |
|         0x19 |*`cudaResViewFormatUnsignedBlockCompressed1`*  |*`hipResViewFormatUnsignedBlockCompressed1`*          |
|         0x1a |*`cudaResViewFormatUnsignedBlockCompressed2`*  |*`hipResViewFormatUnsignedBlockCompressed2`*          |
|         0x1b |*`cudaResViewFormatUnsignedBlockCompressed3`*  |*`hipResViewFormatUnsignedBlockCompressed3`*          |
|         0x1c |*`cudaResViewFormatUnsignedBlockCompressed4`*  |*`hipResViewFormatUnsignedBlockCompressed4`*          |
|         0x1d |*`cudaResViewFormatSignedBlockCompressed4`*    |*`hipResViewFormatSignedBlockCompressed4`*            |
|         0x1e |*`cudaResViewFormatUnsignedBlockCompressed5`*  |*`hipResViewFormatUnsignedBlockCompressed5`*          |
|         0x1f |*`cudaResViewFormatSignedBlockCompressed5`*    |*`hipResViewFormatSignedBlockCompressed5`*            |
|         0x20 |*`cudaResViewFormatUnsignedBlockCompressed6H`* |*`hipResViewFormatUnsignedBlockCompressed6H`*         |
|         0x21 |*`cudaResViewFormatSignedBlockCompressed6H`*   |*`hipResViewFormatSignedBlockCompressed6H`*           |
|         0x22 |*`cudaResViewFormatUnsignedBlockCompressed7`*  |*`hipResViewFormatUnsignedBlockCompressed7`*          |
| enum         |***`cudaSharedMemConfig`***                    |***`hipSharedMemConfig`***                            |
|            0 |*`cudaSharedMemBankSizeDefault`*               |*`hipSharedMemBankSizeDefault`*                       |
|            1 |*`cudaSharedMemBankSizeFourByte`*              |*`hipSharedMemBankSizeFourByte`*                      |
|            2 |*`cudaSharedMemBankSizeEightByte`*             |*`hipSharedMemBankSizeEightByte`*                     |
| enum         |***`cudaSurfaceBoundaryMode`***                |                                                      |
|            0 |*`cudaBoundaryModeZero`*                       |                                                      |
|            1 |*`cudaBoundaryModeClamp`*                      |                                                      |
|            2 |*`cudaBoundaryModeTrap`*                       |                                                      |
| enum         |***`cudaSurfaceFormatMode`***                  |                                                      |
|            0 |*`cudaFormatModeForced`*                       |                                                      |
|            1 |*`cudaFormatModeAuto`*                         |                                                      |
| enum         |***`cudaTextureAddressMode`***                 |***`hipTextureAddressMode`***                         |
|            0 |*`cudaAddressModeWrap`*                        |*`hipAddressModeWrap`*                                |
|            1 |*`cudaAddressModeClamp`*                       |*`hipAddressModeClamp`*                               |
|            2 |*`cudaAddressModeMirror`*                      |*`hipAddressModeMirror`*                              |
|            3 |*`cudaAddressModeBorder`*                      |*`hipAddressModeBorder`*                              |
| enum         |***`cudaTextureFilterMode`***                  |***`hipTextureFilterMode`***                          |
|            0 |*`cudaFilterModePoint`*                        |*`hipFilterModePoint`*                                |
|            1 |*`cudaFilterModeLinear`*                       |*`hipFilterModeLinear`*                               |
| enum         |***`cudaTextureReadMode`***                    |***`hipTextureReadMode`***                            |
|            0 |*`cudaReadModeElementType`*                    |*`hipReadModeElementType`*                            |
|            1 |*`cudaReadModeNormalizedFloat`*                |*`hipReadModeNormalizedFloat`*                        |
| enum         |***`cudaGLDeviceList`***                       |                                                      |
|         0x01 |*`cudaGLDeviceListAll`*                        |                                                      |
|         0x02 |*`cudaGLDeviceListCurrentFrame`*               |                                                      |
|         0x03 |*`cudaGLDeviceListNextFrame`*                  |                                                      |
| enum         |***`cudaGLMapFlags`***                         |                                                      |
|         0x00 |*`cudaGLMapFlagsNone`*                         |                                                      |
|         0x01 |*`cudaGLMapFlagsReadOnly`*                     |                                                      |
|         0x02 |*`cudaGLMapFlagsWriteDiscard`*                 |                                                      |
| enum         |***`cudaD3D9DeviceList`***                     |                                                      |
|            1 |*`cudaD3D9DeviceListAll`*                      |                                                      |
|            2 |*`cudaD3D9DeviceListCurrentFrame`*             |                                                      |
|            3 |*`cudaD3D9DeviceListNextFrame`*                |                                                      |
| enum         |***`cudaD3D9MapFlags`***                       |                                                      |
|            0 |*`cudaD3D9MapFlagsNone`*                       |                                                      |
|            1 |*`cudaD3D9MapFlagsReadOnly`*                   |                                                      |
|            2 |*`cudaD3D9MapFlagsWriteDiscard`*               |                                                      |
| enum         |***`cudaD3D9RegisterFlags`***                  |                                                      |
|            0 |*`cudaD3D9RegisterFlagsNone`*                  |                                                      |
|            1 |*`cudaD3D9RegisterFlagsArray`*                 |                                                      |
| enum         |***`cudaD3D10DeviceList`***                    |                                                      |
|            1 |*`cudaD3D10DeviceListAll`*                     |                                                      |
|            2 |*`cudaD3D10DeviceListCurrentFrame`*            |                                                      |
|            3 |*`cudaD3D10DeviceListNextFrame`*               |                                                      |
| enum         |***`cudaD3D10MapFlags`***                      |                                                      |
|            0 |*`cudaD3D10MapFlagsNone`*                      |                                                      |
|            1 |*`cudaD3D10MapFlagsReadOnly`*                  |                                                      |
|            2 |*`cudaD3D10MapFlagsWriteDiscard`*              |                                                      |
| enum         |***`cudaD3D10RegisterFlags`***                 |                                                      |
|            0 |*`cudaD3D10RegisterFlagsNone`*                 |                                                      |
|            1 |*`cudaD3D10RegisterFlagsArray`*                |                                                      |
| enum         |***`cudaD3D11DeviceList`***                    |                                                      |
|            1 |*`cudaD3D11DeviceListAll`*                     |                                                      |
|            2 |*`cudaD3D11DeviceListCurrentFrame`*            |                                                      |
|            3 |*`cudaD3D11DeviceListNextFrame`*               |                                                      |
| struct       | `cudaArray`                                   | `hipArray`                                           |
| typedef      | `cudaArray_t`                                 | `hipArray_t`                                         |
| typedef      | `cudaArray_const_t`                           | `hipArray_const_t`                                   |
| enum         | `cudaError`                                   | `hipError_t`                                         |
| typedef      | `cudaError_t`                                 | `hipError_t`                                         |
| typedef      | `cudaEvent_t`                                 | `hipEvent_t`                                         |
| typedef      | `cudaGraphicsResource_t`                      |                                                      |
| typedef      | `cudaMipmappedArray_t`                        | `hipMipmappedArray_t`                                |
| typedef      | `cudaMipmappedArray_const_t`                  | `hipMipmappedArray_const_t`                          |
| enum         |***`cudaOutputMode`***                         |                                                      |
|         0x00 |*`cudaKeyValuePair`*                           |                                                      |
|         0x01 |*`cudaCSV`*                                    |                                                      |
| typedef      | `cudaOutputMode_t`                            |                                                      |
| typedef      | `cudaStream_t`                                | `hipStream_t`                                        |
| typedef      | `cudaStreamCallback_t`                        | `hipStreamCallback_t`                                |
| typedef      | `cudaSurfaceObject_t`                         |                                                      |
| typedef      | `cudaTextureObject_t`                         |                                                      |
| typedef      | `CUuuid_stcudaUUID_t`                         |                                                      |
| define       | `CUDA_IPC_HANDLE_SIZE`                        |                                                      |
| define       | `cudaArrayCubemap`                            |                                                      |
| define       | `cudaArrayDefault`                            |                                                      |
| define       | `cudaArrayLayered`                            |                                                      |
| define       | `cudaArraySurfaceLoadStore`                   |                                                      |
| define       | `cudaArrayTextureGather`                      |                                                      |
| define       | `cudaDeviceBlockingSync`                      | `hipDeviceScheduleBlockingSync`                      |
| define       | `cudaDeviceLmemResizeToMax`                   |                                                      |
| define       | `cudaDeviceMapHost`                           |                                                      |
| define       | `cudaDeviceMask`                              |                                                      |
| define       | `cudaDevicePropDontCare`                      |                                                      |
| define       | `cudaDeviceScheduleAuto`                      | `hipDeviceScheduleAuto`                              |
| define       | `cudaDeviceScheduleBlockingSync`              | `hipDeviceScheduleBlockingSync`                      |
| define       | `cudaDeviceScheduleMask`                      | `hipDeviceScheduleMask`                              |
| define       | `cudaDeviceScheduleSpin`                      | `hipDeviceScheduleSpin`                              |
| define       | `cudaDeviceScheduleYield`                     | `hipDeviceScheduleYield`                             |
| define       | `cudaEventDefault`                            | `hipEventDefault`                                    |
| define       | `cudaEventDisableTiming`                      | `hipEventDisableTiming`                              |
| define       | `cudaEventInterprocess`                       | `hipEventInterprocess`                               |
| define       | `cudaHostAllocDefault`                        | `hipHostMallocDefault`                               |
| define       | `cudaHostAllocMapped`                         | `hipHostMallocMapped`                                |
| define       | `cudaHostAllocPortable`                       | `hipHostMallocPortable`                              |
| define       | `cudaHostAllocWriteCombined`                  | `hipHostMallocWriteCombined`                         |
| define       | `cudaHostRegisterDefault`                     | `hipHostRegisterDefault`                             |
| define       | `cudaHostRegisterIoMemory`                    | `hipHostRegisterIoMemory`                            |
| define       | `cudaHostRegisterMapped`                      | `hipHostRegisterMapped`                              |
| define       | `cudaHostRegisterPortable`                    | `hipHostRegisterPortable`                            |
| define       | `cudaIpcMemLazyEnablePeerAccess`              | `hipIpcMemLazyEnablePeerAccess`                      |
| define       | `cudaMemAttachGlobal`                         |                                                      |
| define       | `cudaMemAttachHost`                           |                                                      |
| define       | `cudaMemAttachSingle`                         |                                                      |
| define       | `cudaOccupancyDefault`                        |                                                      |
| define       | `cudaOccupancyDisableCachingOverride`         |                                                      |
| define       | `cudaPeerAccessDefault`                       |                                                      |
| define       | `cudaStreamDefault`                           | `hipStreamDefault`                                   |
| define       | `cudaStreamLegacy`                            |                                                      |
| define       | `cudaStreamNonBlocking`                       | `hipStreamNonBlocking`                               |
| define       | `cudaStreamPerThread`                         |                                                      |
