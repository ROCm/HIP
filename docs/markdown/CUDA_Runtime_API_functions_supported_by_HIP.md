# CUDA Runtime API functions supported by HIP

## **1. Device Management**

|   **CUDA**                                                |   **HIP**                         |**CUDA version\***|
|-----------------------------------------------------------|-----------------------------------|:----------------:|
| `cudaChooseDevice`                                        | `hipChooseDevice`                 |
| `cudaDeviceGetAttribute`                                  | `hipDeviceGetAttribute`           |
| `cudaDeviceGetByPCIBusId`                                 | `hipDeviceGetByPCIBusId`          |
| `cudaDeviceGetCacheConfig`                                | `hipDeviceGetCacheConfig`         |
| `cudaDeviceGetLimit`                                      | `hipDeviceGetLimit`               |
| `cudaDeviceGetPCIBusId`                                   | `hipDeviceGetPCIBusId`            |
| `cudaDeviceGetSharedMemConfig`                            | `hipDeviceGetSharedMemConfig`     |
| `cudaDeviceGetStreamPriorityRange`                        | `hipDeviceGetStreamPriorityRange` |
| `cudaDeviceReset`                                         | `hipDeviceReset`                  |
| `cudaDeviceSetCacheConfig`                                | `hipDeviceSetCacheConfig`         |
| `cudaDeviceSetLimit`                                      | `hipDeviceSetLimit`               |
| `cudaDeviceSetSharedMemConfig`                            | `hipDeviceSetSharedMemConfig`     |
| `cudaDeviceSynchronize`                                   | `hipDeviceSynchronize`            |
| `cudaGetDevice`                                           | `hipGetDevice`                    |
| `cudaGetDeviceCount`                                      | `hipGetDeviceCount`               |
| `cudaGetDeviceFlags`                                      | `hipCtxGetFlags`                  |
| `cudaGetDeviceProperties`                                 | `hipGetDeviceProperties`          |
| `cudaIpcCloseMemHandle`                                   | `hipIpcCloseMemHandle`            |
| `cudaIpcGetEventHandle`                                   | `hipIpcGetEventHandle`            |
| `cudaIpcGetMemHandle`                                     | `hipIpcGetMemHandle`              |
| `cudaIpcOpenEventHandle`                                  | `hipIpcOpenEventHandle`           |
| `cudaIpcOpenMemHandle`                                    | `hipIpcOpenMemHandle`             |
| `cudaSetDevice`                                           | `hipSetDevice`                    |
| `cudaSetDeviceFlags`                                      | `hipSetDeviceFlags`               |
| `cudaSetValidDevices`                                     |                                   |
| `cudaDeviceGetP2PAttribute`                               |                                   | 8.0              |

## **2. Thread Management [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaThreadExit`                                          | `hipDeviceReset`              |
| `cudaThreadGetCacheConfig`                                | `hipDeviceGetCacheConfig`     |
| `cudaThreadGetLimit`                                      |                               |
| `cudaThreadSetCacheConfig`                                | `hipDeviceSetCacheConfig`     |
| `cudaThreadSetLimit`                                      |                               |
| `cudaThreadSynchronize`                                   | `hipDeviceSynchronize`        |

## **3. Error Handling**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaGetErrorName`                                        | `hipGetErrorName`             |
| `cudaGetErrorString`                                      | `hipGetErrorString`           |
| `cudaGetLastError`                                        | `hipGetLastError`             |
| `cudaPeekAtLastError`                                     | `hipPeekAtLastError`          |

## **4. Stream Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaStreamAddCallback`                                   | `hipStreamAddCallback`        |
| `cudaStreamAttachMemAsync`                                |                               |
| `cudaStreamBeginCapture`                                  |                               | 10.0             |
| `cudaStreamEndCapture`                                    |                               | 10.0             |
| `cudaStreamIsCapturing`                                   |                               | 10.0             |
| `cudaStreamGetCaptureInfo`                                |                               | 10.1             |
| `cudaStreamCreate`                                        | `hipStreamCreate`             |
| `cudaStreamCreateWithFlags`                               | `hipStreamCreateWithFlags`    |
| `cudaStreamCreateWithPriority`                            | `hipStreamCreateWithPriority` |
| `cudaStreamDestroy`                                       | `hipStreamDestroy`            |
| `cudaStreamGetFlags`                                      | `hipStreamGetFlags`           |
| `cudaStreamGetPriority`                                   | `hipStreamGetPriority`        |
| `cudaStreamQuery`                                         | `hipStreamQuery`              |
| `cudaStreamSynchronize`                                   | `hipStreamSynchronize`        |
| `cudaStreamWaitEvent`                                     | `hipStreamWaitEvent`          |
| `cudaThreadExchangeStreamCaptureMode`                     |                               | 10.1             |

## **5. Event Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaEventCreate`                                         | `hipEventCreate`              |
| `cudaEventCreateWithFlags`                                | `hipEventCreateWithFlags`     |
| `cudaEventDestroy`                                        | `hipEventDestroy`             |
| `cudaEventElapsedTime`                                    | `hipEventElapsedTime`         |
| `cudaEventQuery`                                          | `hipEventQuery`               |
| `cudaEventRecord`                                         | `hipEventRecord`              |
| `cudaEventSynchronize`                                    | `hipEventSynchronize`         |

## **6. External Resource Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaSignalExternalSemaphoresAsync`                       |                               | 10.0             |
| `cudaWaitExternalSemaphoresAsync`                         |                               | 10.0             |
| `cudaImportExternalMemory`                                |                               | 10.0             |
| `cudaExternalMemoryGetMappedBuffer`                       |                               | 10.0             |
| `cudaExternalMemoryGetMappedMipmappedArray`               |                               | 10.0             |
| `cudaDestroyExternalMemory`                               |                               | 10.0             |
| `cudaImportExternalSemaphore`                             |                               | 10.0             |
| `cudaDestroyExternalSemaphore`                            |                               | 10.0             |

## **7. Execution Control**

|   **CUDA**                                                |   **HIP**                             |**CUDA version\***|
|-----------------------------------------------------------|---------------------------------------|:----------------:|
| `cudaFuncGetAttributes`                                   |                                       |
| `cudaFuncSetAttribute`                                    |                                       | 9.0              |
| `cudaFuncSetCacheConfig`                                  |`hipFuncSetCacheConfig`                |
| `cudaFuncSetSharedMemConfig`                              |                                       |
| `cudaGetParameterBuffer`                                  |                                       |
| `cudaGetParameterBufferV2`                                |                                       |
| `cudaLaunchKernel`                                        |`hipLaunchKernel`                      |
| `cudaSetDoubleForDevice`                                  |                                       |
| `cudaSetDoubleForHost`                                    |                                       |
| `cudaLaunchCooperativeKernel`                             |`hipLaunchCooperativeKernel`           | 9.0              |
| `cudaLaunchCooperativeKernelMultiDevice`                  |`hipLaunchCooperativeKernelMultiDevice`| 9.0              |
| `cudaLaunchHostFunc`                                      |                                       | 10.0             |

## **8. Occupancy**

|   **CUDA**                                                |   **HIP**                                             |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------------|:----------------:|
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor`           |`hipOccupancyMaxActiveBlocksPerMultiprocessor`         |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`  |`hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`|

## **9. Execution Control [DEPRECATED since 7.0]**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaConfigureCall`                                       | `hipConfigureCall`            |
| `cudaLaunch`                                              | `hipLaunchByPtr`              |
| `cudaSetupArgument`                                       | `hipSetupArgument`            | 7.0 - 10.0       |

## **10. Memory Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaArrayGetInfo`                                        |                               |
| `cudaFree`                                                | `hipFree`                     |
| `cudaFreeArray`                                           | `hipFreeArray`                |
| `cudaFreeHost`                                            | `hipHostFree`                 |
| `cudaFreeMipmappedArray`                                  |                               |
| `cudaGetMipmappedArrayLevel`                              |                               |
| `cudaGetSymbolAddress`                                    | `hipGetSymbolAddress`         |
| `cudaGetSymbolSize`                                       | `hipGetSymbolSize`            |
| `cudaHostAlloc`                                           | `hipHostMalloc`               |
| `cudaHostGetDevicePointer`                                | `hipHostGetDevicePointer`     |
| `cudaHostGetFlags`                                        | `hipHostGetFlags`             |
| `cudaHostRegister`                                        | `hipHostRegister`             |
| `cudaHostUnregister`                                      | `hipHostUnregister`           |
| `cudaMalloc`                                              | `hipMalloc`                   |
| `cudaMalloc3D`                                            | `hipMalloc3D`                 |
| `cudaMalloc3DArray`                                       | `hipMalloc3DArray`            |
| `cudaMallocArray`                                         | `hipMallocArray`              |
| `cudaMallocHost`                                          | `hipHostMalloc`               |
| `cudaMallocManaged`                                       | `hipMallocManaged`            |
| `cudaMallocMipmappedArray`                                |                               |
| `cudaMallocPitch`                                         |                               |
| `cudaMemGetInfo`                                          | `hipMemGetInfo`               |
| `cudaMemPrefetchAsync`                                    |                               | 8.0              |
| `cudaMemcpy`                                              | `hipMemcpy`                   |
| `cudaMemcpy2D`                                            | `hipMemcpy2D`                 |
| `cudaMemcpy2DArrayToArray`                                |                               |
| `cudaMemcpy2DAsync`                                       | `hipMemcpy2DAsync`            |
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
| `cudaMemcpyFromArray`                                     | `hipMemcpyFromArray`          |
| `cudaMemcpyFromArrayAsync`                                |                               |
| `cudaMemcpyFromSymbol`                                    | `hipMemcpyFromSymbol`         |
| `cudaMemcpyFromSymbolAsync`                               | `hipMemcpyFromSymbolAsync`    |
| `cudaMemcpyPeer`                                          | `hipMemcpyPeer`               |
| `cudaMemcpyPeerAsync`                                     | `hipMemcpyPeerAsync`          |
| `cudaMemcpyToArray`                                       | `hipMemcpyToArray`            |
| `cudaMemcpyToArrayAsync`                                  |                               |
| `cudaMemcpyToSymbol`                                      | `hipMemcpyToSymbol`           |
| `cudaMemcpyToSymbolAsync`                                 | `hipMemcpyToSymbolAsync`      |
| `cudaMemset`                                              | `hipMemset`                   |
| `cudaMemset2D`                                            | `hipMemset2D`                 |
| `cudaMemset2DAsync`                                       | `hipMemset2DAsync`            |
| `cudaMemset3D`                                            |                               |
| `cudaMemset3DAsync`                                       |                               |
| `cudaMemsetAsync`                                         | `hipMemsetAsync`              |
| `make_cudaExtent`                                         | `make_hipExtent`              |
| `make_cudaPitchedPtr`                                     | `make_hipPitchedPtr`          |
| `make_cudaPos`                                            | `make_hipPos`                 |

## **11. Unified Addressing**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaPointerGetAttributes`                                | `hipPointerGetAttributes`     |

## **12. Peer Device Memory Access**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaDeviceCanAccessPeer`                                 | `hipDeviceCanAccessPeer`      |
| `cudaDeviceDisablePeerAccess`                             | `hipDeviceDisablePeerAccess`  |
| `cudaDeviceEnablePeerAccess`                              | `hipDeviceEnablePeerAccess`   |

## **13. OpenGL Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaGLGetDevices`                                        |                               |
| `cudaGraphicsGLRegisterBuffer`                            |                               |
| `cudaGraphicsGLRegisterImage`                             |                               |
| `cudaWGLGetDevice`                                        |                               |

## **14. OpenGL Interoperability [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaGLMapBufferObject`                                   |                               |
| `cudaGLMapBufferObjectAsync`                              |                               |
| `cudaGLRegisterBufferObject`                              |                               |
| `cudaGLSetBufferObjectMapFlags`                           |                               |
| `cudaGLSetGLDevice`                                       |                               |
| `cudaGLUnmapBufferObject`                                 |                               |
| `cudaGLUnmapBufferObjectAsync`                            |                               |
| `cudaGLUnregisterBufferObject`                            |                               |

## **15. Direct3D 9 Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaD3D9GetDevice`                                       |                               |
| `cudaD3D9GetDevices`                                      |                               |
| `cudaD3D9GetDirect3DDevice`                               |                               |
| `cudaD3D9SetDirect3DDevice`                               |                               |
| `cudaGraphicsD3D9RegisterResource`                        |                               |

## **16. Direct3D 9 Interoperability [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
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

## **17. Direct3D 10 Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaD3D10GetDevice`                                      |                               |
| `cudaD3D10GetDevices`                                     |                               |
| `cudaGraphicsD3D10RegisterResource`                       |                               |

## **18. Direct3D 10 Interoperability [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
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

## **19. Direct3D 11 Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaD3D11GetDevice`                                      |                               |
| `cudaD3D11GetDevices`                                     |                               |
| `cudaGraphicsD3D11RegisterResource`                       |                               |

## **20. Direct3D 11 Interoperability [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaD3D11GetDirect3DDevice`                              |                               |
| `cudaD3D11SetDirect3DDevice`                              |                               |

## **21. VDPAU Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaGraphicsVDPAURegisterOutputSurface`                  |                               |
| `cudaGraphicsVDPAURegisterVideoSurface`                   |                               |
| `cudaVDPAUGetDevice`                                      |                               |
| `cudaVDPAUSetVDPAUDevice`                                 |                               |

## **22. EGL Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaEGLStreamConsumerAcquireFrame`                       |                               | 8.0              |
| `cudaEGLStreamConsumerConnect`                            |                               | 8.0              |
| `cudaEGLStreamConsumerConnectWithFlags`                   |                               | 8.0              |
| `cudaEGLStreamConsumerDisconnect`                         |                               | 8.0              |
| `cudaEGLStreamConsumerReleaseFrame`                       |                               | 8.0              |
| `cudaEGLStreamProducerConnect`                            |                               | 8.0              |
| `cudaEGLStreamProducerDisconnect`                         |                               | 8.0              |
| `cudaEGLStreamProducerPresentFrame`                       |                               | 8.0              |
| `cudaEGLStreamProducerReturnFrame`                        |                               | 8.0              |
| `cudaEventCreateFromEGLSync`                              |                               | 9.0              |
| `cudaGraphicsEGLRegisterImage`                            |                               | 8.0              |
| `cudaGraphicsResourceGetMappedEglFrame`                   |                               | 8.0              |

## **23. Graphics Interoperability**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaGraphicsMapResources`                                |                               |
| `cudaGraphicsResourceGetMappedMipmappedArray`             |                               |
| `cudaGraphicsResourceGetMappedPointer`                    |                               |
| `cudaGraphicsResourceSetMapFlags`                         |                               |
| `cudaGraphicsSubResourceGetMappedArray`                   |                               |
| `cudaGraphicsUnmapResources`                              |                               |
| `cudaGraphicsUnregisterResource`                          |                               |

## **24. Texture Reference Management**

|   **CUDA**                                                |   **HIP**                        |**CUDA version\***|
|-----------------------------------------------------------|----------------------------------|:----------------:|
| `cudaBindTexture`                                         | `hipBindTexture`                 |
| `cudaBindTexture2D`                                       | `hipBindTexture2D`               |
| `cudaBindTextureToArray`                                  | `hipBindTextureToArray`          |
| `cudaBindTextureToMipmappedArray`                         | `hipBindTextureToMipmappedArray` |
| `cudaCreateChannelDesc`                                   | `hipCreateChannelDesc`           |
| `cudaGetChannelDesc`                                      | `hipGetChannelDesc`              |
| `cudaGetTextureAlignmentOffset`                           | `hipGetTextureAlignmentOffset`   |
| `cudaGetTextureReference`                                 | `hipGetTextureReference`         |
| `cudaUnbindTexture`                                       | `hipUnbindTexture`               |

## **25. Surface Reference Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaBindSurfaceToArray`                                  |                               |
| `cudaGetSurfaceReference`                                 |                               |

## **26. Texture Object Management**

|   **CUDA**                                                |   **HIP**                            |**CUDA version\***|
|-----------------------------------------------------------|--------------------------------------|:----------------:|
| `cudaCreateTextureObject`                                 |`hipCreateTextureObject`              |
| `cudaDestroyTextureObject`                                |`hipDestroyTextureObject`             |
| `cudaGetTextureObjectResourceDesc`                        |`hipGetTextureObjectResourceDesc`     |
| `cudaGetTextureObjectResourceViewDesc`                    |`hipGetTextureObjectResourceViewDesc` |
| `cudaGetTextureObjectTextureDesc`                         |`hipGetTextureObjectTextureDesc`      |

## **27. Surface Object Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaCreateSurfaceObject`                                 | `hipCreateSurfaceObject`      |
| `cudaDestroySurfaceObject`                                | `hipDestroySurfaceObject`     |
| `cudaGetSurfaceObjectResourceDesc`                        |                               |

## **28. Version Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaDriverGetVersion`                                    | `hipDriverGetVersion`         |
| `cudaRuntimeGetVersion`                                   | `hipRuntimeGetVersion`        |

## **29. Graph Management**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaGraphAddChildGraphNode`                              |                               | 10.0             |
| `cudaGraphAddDependencies`                                |                               | 10.0             |
| `cudaGraphAddEmptyNode`                                   |                               | 10.0             |
| `cudaGraphAddHostNode`                                    |                               | 10.0             |
| `cudaGraphAddKernelNode`                                  |                               | 10.0             |
| `cudaGraphAddMemcpyNode`                                  |                               | 10.0             |
| `cudaGraphAddMemsetNode`                                  |                               | 10.0             |
| `cudaGraphChildGraphNodeGetGraph`                         |                               | 10.0             |
| `cudaGraphClone`                                          |                               | 10.0             |
| `cudaGraphCreate`                                         |                               | 10.0             |
| `cudaGraphDestroy`                                        |                               | 10.0             |
| `cudaGraphDestroyNode`                                    |                               | 10.0             |
| `cudaGraphExecDestroy`                                    |                               | 10.0             |
| `cudaGraphGetEdges`                                       |                               | 10.0             |
| `cudaGraphGetNodes`                                       |                               | 10.0             |
| `cudaGraphGetRootNodes`                                   |                               | 10.0             |
| `cudaGraphHostNodeGetParams`                              |                               | 10.0             |
| `cudaGraphHostNodeSetParams`                              |                               | 10.0             |
| `cudaGraphInstantiate`                                    |                               | 10.0             |
| `cudaGraphExecKernelNodeSetParams`                        |                               | 10.1             |
| `cudaGraphKernelNodeGetParams`                            |                               | 10.0             |
| `cudaGraphKernelNodeSetParams`                            |                               | 10.0             |
| `cudaGraphLaunch`                                         |                               | 10.0             |
| `cudaGraphMemcpyNodeGetParams`                            |                               | 10.0             |
| `cudaGraphMemcpyNodeSetParams`                            |                               | 10.0             |
| `cudaGraphMemsetNodeGetParams`                            |                               | 10.0             |
| `cudaGraphMemsetNodeSetParams`                            |                               | 10.0             |
| `cudaGraphNodeFindInClone`                                |                               | 10.0             |
| `cudaGraphNodeGetDependencies`                            |                               | 10.0             |
| `cudaGraphNodeGetDependentNodes`                          |                               | 10.0             |
| `cudaGraphNodeGetType`                                    |                               | 10.0             |
| `cudaGraphRemoveDependencies`                             |                               | 10.0             |

## **30. C++ API Routines**
*(7.0 contains, 7.5 doesn't)*

|   **CUDA**                                                |   **HIP**                                             |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------------------------------|:----------------:|
| `cudaBindSurfaceToArray`                                  |                                                       |
| `cudaBindTexture`                                         |`hipBindTexture`                                       |
| `cudaBindTexture2D`                                       |                                                       |
| `cudaBindTextureToArray`                                  |                                                       |
| `cudaBindTextureToMipmappedArray`                         |                                                       |
| `cudaCreateChannelDesc`                                   |`hipCreateChannelDesc`                                 |
| `cudaEventCreate`                                         |                                                       |
| `cudaFuncGetAttributes`                                   |                                                       |
| `cudaFuncSetAttribute`                                    |                                                       |
| `cudaFuncSetCacheConfig`                                  |                                                       |
| `cudaGetSymbolAddress`                                    |`hipGetSymbolAddress`                                  |
| `cudaGetSymbolSize`                                       |`hipGetSymbolSize`                                     |
| `cudaGetTextureAlignmentOffset`                           |                                                       |
| `cudaLaunch`                                              |                                                       |
| `cudaLaunchCooperativeKernel`                             |`hipLaunchCooperativeKernel`                           |
| `cudaLaunchCooperativeKernelMultiDevice`                  |`hipLaunchCooperativeKernelMultiDevice`                |
| `cudaLaunchKernel`                                        |                                                       |
| `cudaMallocHost`                                          |                                                       |
| `cudaMallocManaged`                                       |                                                       |
| `cudaMemcpyFromSymbol`                                    |                                                       |
| `cudaMemcpyFromSymbolAsync`                               |                                                       |
| `cudaMemcpyToSymbol`                                      |                                                       |
| `cudaMemcpyToSymbolAsync`                                 |                                                       |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor`           |`hipOccupancyMaxActiveBlocksPerMultiprocessor`         |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`  |`hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`|
| `cudaOccupancyMaxPotentialBlockSize`                      |`hipOccupancyMaxPotentialBlockSize`                    |
| `cudaOccupancyMaxPotentialBlockSizeWithFlags`             |                                                       |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMem`          |                                                       |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags` |                                                       |
| `cudaSetupArgument`                                       |                                                       |
| `cudaStreamAttachMemAsync`                                |                                                       |
| `cudaUnbindTexture`                                       |`hipUnbindTexture`                                     |

## **32. Profiler Control**

|   **CUDA**                                                |   **HIP**                     |**CUDA version\***|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaProfilerInitialize`                                  |                               |
| `cudaProfilerStart`                                       | `hipProfilerStart`            |
| `cudaProfilerStop`                                        | `hipProfilerStop`             |

# Data types used by CUDA Runtime API and supported by HIP

## **33. Data types**

| **type**     |   **CUDA**                                          |**CUDA version\***|   **HIP**                                                  |**HIP value** (if differs) |
|-------------:|-----------------------------------------------------|:----------------:|------------------------------------------------------------|---------------------------|
| struct       |`cudaChannelFormatDesc`                              |                  |`hipChannelFormatDesc`                                      |
| struct       |`cudaDeviceProp`                                     |                  |`hipDeviceProp_t`                                           |
| struct       |`cudaEglFrame`                                       | 9.1              |                                                            |
| typedef      |`cudaEglFrame_st`                                    | 9.1              |                                                            |
| struct       |`cudaEglPlaneDesc`                                   | 9.1              |                                                            |
| typedef      |`cudaEglPlaneDesc_st`                                | 9.1              |                                                            |
| struct       |`cudaExtent`                                         |                  |`hipExtent`                                                 |
| struct       |`cudaFuncAttributes`                                 |                  |`hipFuncAttributes`                                         |
| struct       |`cudaIpcEventHandle_t`                               |                  |`hipIpcEventHandle_t`                                       |
| struct       |`cudaIpcMemHandle_t`                                 |                  |`hipIpcMemHandle_t`                                         |
| struct       |`cudaMemcpy3DParms`                                  |                  |`hipMemcpy3DParms`                                          |
| struct       |`cudaMemcpy3DPeerParms`                              |                  |                                                            |
| struct       |`cudaPitchedPtr`                                     |                  |`hipPitchedPtr`                                             |
| struct       |`cudaPointerAttributes`                              |                  |`hipPointerAttribute_t`                                     |
| struct       |`cudaPos`                                            |                  |`hipPos`                                                    |
| struct       |`cudaResourceDesc`                                   |                  |`hipResourceDesc`                                           |
| struct       |`cudaResourceViewDesc`                               |                  |`hipResourceViewDesc`                                       |
| struct       |`cudaTextureDesc`                                    |                  |`hipTextureDesc`                                            |
| struct       |`textureReference`                                   |                  |`textureReference`                                          |
| struct       |`surfaceReference`                                   |                  |                                                            |
| struct       |`CUuuid_st`                                          |                  |                                                            |
| enum         |***`cudaCGScope`***                                  | 9.0              |                                                            |
|            0 |*`cudaCGScopeInvalid`*                               | 9.0              |                                                            |
|            1 |*`cudaCGScopeGrid`*                                  | 9.0              |                                                            |
|            2 |*`cudaCGScopeMultiGrid`*                             | 9.0              |                                                            |
| enum         |***`cudaChannelFormatKind`***                        |                  |***`hipChannelFormatKind`***                                |
|            0 |*`cudaChannelFormatKindSigned`*                      |                  |*`hipChannelFormatKindSigned`*                              |
|            1 |*`cudaChannelFormatKindUnsigned`*                    |                  |*`hipChannelFormatKindUnsigned`*                            |
|            2 |*`cudaChannelFormatKindFloat`*                       |                  |*`hipChannelFormatKindFloat`*                               |
|            3 |*`cudaChannelFormatKindNone`*                        |                  |*`hipChannelFormatKindNone`*                                |
| enum         |***`cudaComputeMode`***                              |                  |***`hipComputeMode`***                                      |
|            0 |*`cudaComputeModeDefault`*                           |                  |*`hipComputeModeDefault`*                                   |
|            1 |*`cudaComputeModeExclusive`*                         |                  |*`hipComputeModeExclusive`*                                 |
|            2 |*`cudaComputeModeProhibited`*                        |                  |*`hipComputeModeProhibited`*                                |
|            3 |*`cudaComputeModeExclusiveProcess`*                  |                  |*`hipComputeModeExclusiveProcess`*                          |
| enum         |***`cudaDeviceAttr`***                               |                  |***`hipDeviceAttribute_t`***                                |
|            1 |*`cudaDevAttrMaxThreadsPerBlock`*                    |                  |*`hipDeviceAttributeMaxThreadsPerBlock`*                    |
|            2 |*`cudaDevAttrMaxBlockDimX`*                          |                  |*`hipDeviceAttributeMaxBlockDimX`*                          |
|            3 |*`cudaDevAttrMaxBlockDimY`*                          |                  |*`hipDeviceAttributeMaxBlockDimY`*                          |
|            4 |*`cudaDevAttrMaxBlockDimZ`*                          |                  |*`hipDeviceAttributeMaxBlockDimZ`*                          |
|            5 |*`cudaDevAttrMaxGridDimX`*                           |                  |*`hipDeviceAttributeMaxGridDimX`*                           |
|            6 |*`cudaDevAttrMaxGridDimY`*                           |                  |*`hipDeviceAttributeMaxGridDimY`*                           |
|            7 |*`cudaDevAttrMaxGridDimZ`*                           |                  |*`hipDeviceAttributeMaxGridDimZ`*                           |
|            8 |*`cudaDevAttrMaxSharedMemoryPerBlock`*               |                  |*`hipDeviceAttributeMaxSharedMemoryPerBlock`*               |
|            9 |*`cudaDevAttrTotalConstantMemory`*                   |                  |*`hipDeviceAttributeTotalConstantMemory`*                   |
|           10 |*`cudaDevAttrWarpSize`*                              |                  |*`hipDeviceAttributeWarpSize`*                              |
|           11 |*`cudaDevAttrMaxPitch`*                              |                  |                                                            |
|           12 |*`cudaDevAttrMaxRegistersPerBlock`*                  |                  |*`hipDeviceAttributeMaxRegistersPerBlock`*                  |
|           13 |*`cudaDevAttrClockRate`*                             |                  |*`hipDeviceAttributeClockRate`*                             |
|           14 |*`cudaDevAttrTextureAlignment`*                      |                  |                                                            |
|           15 |*`cudaDevAttrGpuOverlap`*                            |                  |                                                            |
|           16 |*`cudaDevAttrMultiProcessorCount`*                   |                  |*`hipDeviceAttributeMultiprocessorCount`*                   |
|           17 |*`cudaDevAttrKernelExecTimeout`*                     |                  |                                                            |
|           18 |*`cudaDevAttrIntegrated`*                            |                  |*`hipDeviceAttributeIntegrated`*                            |
|           19 |*`cudaDevAttrCanMapHostMemory`*                      |                  |                                                            |
|           20 |*`cudaDevAttrComputeMode`*                           |                  |*`hipDeviceAttributeComputeMode`*                           |
|           21 |*`cudaDevAttrMaxTexture1DWidth`*                     |                  |*`hipDeviceAttributeMaxTexture1DWidth`*                     |
|           22 |*`cudaDevAttrMaxTexture2DWidth`*                     |                  |*`hipDeviceAttributeMaxTexture2DWidth`*                     |
|           23 |*`cudaDevAttrMaxTexture2DHeight`*                    |                  |*`hipDeviceAttributeMaxTexture2DHeight`*                    |
|           24 |*`cudaDevAttrMaxTexture3DWidth`*                     |                  |*`hipDeviceAttributeMaxTexture3DWidth`*                     |
|           25 |*`cudaDevAttrMaxTexture3DHeight`*                    |                  |*`hipDeviceAttributeMaxTexture3DHeight`*                    |
|           26 |*`cudaDevAttrMaxTexture3DDepth`*                     |                  |*`hipDeviceAttributeMaxTexture3DDepth`*                     |
|           27 |*`cudaDevAttrMaxTexture2DLayeredWidth`*              |                  |                                                            |
|           28 |*`cudaDevAttrMaxTexture2DLayeredHeight`*             |                  |                                                            |
|           29 |*`cudaDevAttrMaxTexture2DLayeredLayers`*             |                  |                                                            |
|           30 |*`cudaDevAttrSurfaceAlignment`*                      |                  |                                                            |
|           31 |*`cudaDevAttrConcurrentKernels`*                     |                  |*`hipDeviceAttributeConcurrentKernels`*                     |
|           32 |*`cudaDevAttrEccEnabled`*                            |                  |                                                            |
|           33 |*`cudaDevAttrPciBusId`*                              |                  |*`hipDeviceAttributePciBusId`*                              |
|           34 |*`cudaDevAttrPciDeviceId`*                           |                  |*`hipDeviceAttributePciDeviceId`*                           |
|           35 |*`cudaDevAttrTccDriver`*                             |                  |                                                            |
|           36 |*`cudaDevAttrMemoryClockRate`*                       |                  |*`hipDeviceAttributeMemoryClockRate`*                       |
|           37 |*`cudaDevAttrGlobalMemoryBusWidth`*                  |                  |*`hipDeviceAttributeMemoryBusWidth`*                        |
|           38 |*`cudaDevAttrL2CacheSize`*                           |                  |*`hipDeviceAttributeL2CacheSize`*                           |
|           39 |*`cudaDevAttrMaxThreadsPerMultiProcessor`*           |                  |*`hipDeviceAttributeMaxThreadsPerMultiProcessor`*           |
|           40 |*`cudaDevAttrAsyncEngineCount`*                      |                  |                                                            |
|           41 |*`cudaDevAttrUnifiedAddressing`*                     |                  |                                                            |
|           42 |*`cudaDevAttrMaxTexture1DLayeredWidth`*              |                  |                                                            |
|           43 |*`cudaDevAttrMaxTexture1DLayeredLayers`*             |                  |                                                            |
|           44 |                                                     |                  |                                                            |
|           45 |*`cudaDevAttrMaxTexture2DGatherWidth`*               |                  |                                                            |
|           46 |*`cudaDevAttrMaxTexture2DGatherHeight`*              |                  |                                                            |
|           47 |*`cudaDevAttrMaxTexture3DWidthAlt`*                  |                  |                                                            |
|           48 |*`cudaDevAttrMaxTexture3DHeightAlt`*                 |                  |                                                            |
|           49 |*`cudaDevAttrMaxTexture3DDepthAlt`*                  |                  |                                                            |
|           50 |*`cudaDevAttrPciDomainId`*                           |                  |                                                            |
|           51 |*`cudaDevAttrTexturePitchAlignment`*                 |                  |                                                            |
|           52 |*`cudaDevAttrMaxTextureCubemapWidth`*                |                  |                                                            |
|           53 |*`cudaDevAttrMaxTextureCubemapLayeredWidth`*         |                  |                                                            |
|           54 |*`cudaDevAttrMaxTextureCubemapLayeredLayers`*        |                  |                                                            |
|           55 |*`cudaDevAttrMaxSurface1DWidth`*                     |                  |                                                            |
|           56 |*`cudaDevAttrMaxSurface2DWidth`*                     |                  |                                                            |
|           57 |*`cudaDevAttrMaxSurface2DHeight`*                    |                  |                                                            |
|           58 |*`cudaDevAttrMaxSurface3DWidth`*                     |                  |                                                            |
|           59 |*`cudaDevAttrMaxSurface3DHeight`*                    |                  |                                                            |
|           60 |*`cudaDevAttrMaxSurface3DDepth`*                     |                  |                                                            |
|           61 |*`cudaDevAttrMaxSurface1DLayeredWidth`*              |                  |                                                            |
|           62 |*`cudaDevAttrMaxSurface1DLayeredLayers`*             |                  |                                                            |
|           63 |*`cudaDevAttrMaxSurface2DLayeredWidth`*              |                  |                                                            |
|           64 |*`cudaDevAttrMaxSurface2DLayeredHeight`*             |                  |                                                            |
|           65 |*`cudaDevAttrMaxSurface2DLayeredLayers`*             |                  |                                                            |
|           66 |*`cudaDevAttrMaxSurfaceCubemapWidth`*                |                  |                                                            |
|           67 |*`cudaDevAttrMaxSurfaceCubemapLayeredWidth`*         |                  |                                                            |
|           68 |*`cudaDevAttrMaxSurfaceCubemapLayeredLayers`*        |                  |                                                            |
|           69 |*`cudaDevAttrMaxTexture1DLinearWidth`*               |                  |                                                            |
|           70 |*`cudaDevAttrMaxTexture2DLinearWidth`*               |                  |                                                            |
|           71 |*`cudaDevAttrMaxTexture2DLinearHeight`*              |                  |                                                            |
|           72 |*`cudaDevAttrMaxTexture2DLinearPitch`*               |                  |                                                            |
|           73 |*`cudaDevAttrMaxTexture2DMipmappedWidth*             |                  |                                                            |
|           74 |*`cudaDevAttrMaxTexture2DMipmappedHeight`*           |                  |                                                            |
|           75 |*`cudaDevAttrComputeCapabilityMajor`*                |                  |*`hipDeviceAttributeComputeCapabilityMajor`*                |
|           76 |*`cudaDevAttrComputeCapabilityMinor`*                |                  |*`hipDeviceAttributeComputeCapabilityMinor`*                |
|           77 |*`cudaDevAttrMaxTexture1DMipmappedWidth`*            |                  |                                                            |
|           78 |*`cudaDevAttrStreamPrioritiesSupported`*             |                  |                                                            |
|           79 |*`cudaDevAttrGlobalL1CacheSupported`*                |                  |                                                            |
|           80 |*`cudaDevAttrLocalL1CacheSupported`*                 |                  |                                                            |
|           81 |*`cudaDevAttrMaxSharedMemoryPerMultiprocessor`*      |                  |*`hipDeviceAttributeMaxSharedMemoryPerMultiprocessor`*      |
|           82 |*`cudaDevAttrMaxRegistersPerMultiprocessor`*         |                  |                                                            |
|           83 |*`cudaDevAttrManagedMemory`*                         |                  |                                                            |
|           84 |*`cudaDevAttrIsMultiGpuBoard`*                       |                  |*`hipDeviceAttributeIsMultiGpuBoard`*                       |
|           85 |*`cudaDevAttrMultiGpuBoardGroupID`*                  |                  |                                                            |
|           86 |*`cudaDevAttrHostNativeAtomicSupported`*             | 8.0              |                                                            |
|           87 |*`cudaDevAttrSingleToDoublePrecisionPerfRatio`*      | 8.0              |                                                            |
|           88 |*`cudaDevAttrPageableMemoryAccess`*                  | 8.0              |                                                            |
|           89 |*`cudaDevAttrConcurrentManagedAccess`*               | 8.0              |                                                            |
|           90 |*`cudaDevAttrComputePreemptionSupported`*            | 8.0              |                                                            |
|           91 |*`cudaDevAttrCanUseHostPointerForRegisteredMem`*     | 8.0              |                                                            |
|           92 |*`cudaDevAttrReserved92`*                            | 9.0              |                                                            |
|           93 |*`cudaDevAttrReserved93`*                            | 9.0              |                                                            |
|           94 |*`cudaDevAttrReserved94`*                            | 9.0              |                                                            |
|           95 |*`cudaDevAttrCooperativeLaunch`*                     | 9.0              |*`hipDeviceAttributeCooperativeLaunch`*                     |
|           96 |*`cudaDevAttrCooperativeMultiDeviceLaunch`*          | 9.0              |*`hipDeviceAttributeCooperativeMultiDeviceLaunch`*          |
|           97 |*`cudaDevAttrMaxSharedMemoryPerBlockOptin`*          | 9.0              |                                                            |
|           98 |*`cudaDevAttrCanFlushRemoteWrites`*                  | 9.2              |                                                            |
|           99 |*`cudaDevAttrHostRegisterSupported`*                 | 9.2              |                                                            |
|          100 |*`cudaDevAttrPageableMemoryAccessUsesHostPageTables`*| 9.2              |                                                            |
|          101 |*`cudaDevAttrDirectManagedMemAccessFromHost`*        | 9.2              |                                                            |
| enum         |***`cudaDeviceP2PAttr`***                            | 8.0              |                                                            |
|            1 |*`cudaDevP2PAttrPerformanceRank`*                    | 8.0              |                                                            |
|            2 |*`cudaDevP2PAttrAccessSupported`*                    | 8.0              |                                                            |
|            3 |*`cudaDevP2PAttrNativeAtomicSupported`*              | 8.0              |                                                            |
|            4 |*`cudaDevP2PAttrCudaArrayAccessSupported`*           | 9.2              |                                                            |
| enum         |***`cudaEglColorFormat`***                           | 9.1              |                                                            |
|            0 |*`cudaEglColorFormatYUV420Planar`*                   | 9.1              |                                                            |
|            1 |*`cudaEglColorFormatYUV420SemiPlanar`*               | 9.1              |                                                            |
|            2 |*`cudaEglColorFormatYUV422Planar`*                   | 9.1              |                                                            |
|            3 |*`cudaEglColorFormatYUV422SemiPlanar`*               | 9.1              |                                                            |
|            4 |*`cudaEglColorFormatRGB`*                            | 9.1              |                                                            |
|            5 |*`cudaEglColorFormatBGR`*                            | 9.1              |                                                            |
|            6 |*`cudaEglColorFormatARGB`*                           | 9.1              |                                                            |
|            7 |*`cudaEglColorFormatRGBA`*                           | 9.1              |                                                            |
|            8 |*`cudaEglColorFormatL`*                              | 9.1              |                                                            |
|            9 |*`cudaEglColorFormatR`*                              | 9.1              |                                                            |
|           10 |*`cudaEglColorFormatYUV444Planar`*                   | 9.1              |                                                            |
|           11 |*`cudaEglColorFormatYUV444SemiPlanar`*               | 9.1              |                                                            |
|           12 |*`cudaEglColorFormatYUYV422`*                        | 9.1              |                                                            |
|           13 |*`cudaEglColorFormatUYVY422`*                        | 9.1              |                                                            |
|           14 |*`cudaEglColorFormatABGR`*                           | 9.1              |                                                            |
|           15 |*`cudaEglColorFormatBGRA`*                           | 9.1              |                                                            |
|           16 |*`cudaEglColorFormatA`*                              | 9.1              |                                                            |
|           17 |*`cudaEglColorFormatRG`*                             | 9.1              |                                                            |
|           18 |*`cudaEglColorFormatAYUV`*                           | 9.1              |                                                            |
|           19 |*`cudaEglColorFormatYVU444SemiPlanar`*               | 9.1              |                                                            |
|           20 |*`cudaEglColorFormatYVU422SemiPlanar`*               | 9.1              |                                                            |
|           21 |*`cudaEglColorFormatYVU420SemiPlanar`*               | 9.1              |                                                            |
|           22 |*`cudaEglColorFormatY10V10U10_444SemiPlanar`*        | 9.1              |                                                            |
|           23 |*`cudaEglColorFormatY10V10U10_420SemiPlanar`*        | 9.1              |                                                            |
|           24 |*`cudaEglColorFormatY12V12U12_444SemiPlanar`*        | 9.1              |                                                            |
|           25 |*`cudaEglColorFormatY12V12U12_420SemiPlanar`*        | 9.1              |                                                            |
|           26 |*`cudaEglColorFormatVYUY_ER`*                        | 9.1              |                                                            |
|           27 |*`cudaEglColorFormatUYVY_ER`*                        | 9.1              |                                                            |
|           28 |*`cudaEglColorFormatYUYV_ER`*                        | 9.1              |                                                            |
|           29 |*`cudaEglColorFormatYVYU_ER`*                        | 9.1              |                                                            |
|           30 |*`cudaEglColorFormatYUV_ER`*                         | 9.1              |                                                            |
|           31 |*`cudaEglColorFormatYUVA_ER`*                        | 9.1              |                                                            |
|           32 |*`cudaEglColorFormatAYUV_ER`*                        | 9.1              |                                                            |
|           33 |*`cudaEglColorFormatYUV444Planar_ER`*                | 9.1              |                                                            |
|           34 |*`cudaEglColorFormatYUV422Planar_ER`*                | 9.1              |                                                            |
|           35 |*`cudaEglColorFormatYUV420Planar_ER`*                | 9.1              |                                                            |
|           36 |*`cudaEglColorFormatYUV444SemiPlanar_ER`*            | 9.1              |                                                            |
|           37 |*`cudaEglColorFormatYUV422SemiPlanar_ER`*            | 9.1              |                                                            |
|           38 |*`cudaEglColorFormatYUV420SemiPlanar_ER`*            | 9.1              |                                                            |
|           39 |*`cudaEglColorFormatYVU444Planar_ER`*                | 9.1              |                                                            |
|           40 |*`cudaEglColorFormatYVU422Planar_ER`*                | 9.1              |                                                            |
|           41 |*`cudaEglColorFormatYVU420Planar_ER`*                | 9.1              |                                                            |
|           42 |*`cudaEglColorFormatYVU444SemiPlanar_ER`*            | 9.1              |                                                            |
|           43 |*`cudaEglColorFormatYVU422SemiPlanar_ER`*            | 9.1              |                                                            |
|           44 |*`cudaEglColorFormatYVU420SemiPlanar_ER`*            | 9.1              |                                                            |
|           45 |*`cudaEglColorFormatBayerRGGB`*                      | 9.1              |                                                            |
|           46 |*`cudaEglColorFormatBayerBGGR`*                      | 9.1              |                                                            |
|           47 |*`cudaEglColorFormatBayerGRBG`*                      | 9.1              |                                                            |
|           48 |*`cudaEglColorFormatBayerGBRG`*                      | 9.1              |                                                            |
|           49 |*`cudaEglColorFormatBayer10RGGB`*                    | 9.1              |                                                            |
|           50 |*`cudaEglColorFormatBayer10BGGR`*                    | 9.1              |                                                            |
|           51 |*`cudaEglColorFormatBayer10GRBG`*                    | 9.1              |                                                            |
|           52 |*`cudaEglColorFormatBayer10GBRG`*                    | 9.1              |                                                            |
|           53 |*`cudaEglColorFormatBayer12RGGB`*                    | 9.1              |                                                            |
|           54 |*`cudaEglColorFormatBayer12BGGR`*                    | 9.1              |                                                            |
|           55 |*`cudaEglColorFormatBayer12GRBG`*                    | 9.1              |                                                            |
|           56 |*`cudaEglColorFormatBayer12GBRG`*                    | 9.1              |                                                            |
|           57 |*`cudaEglColorFormatBayer14RGGB`*                    | 9.1              |                                                            |
|           58 |*`cudaEglColorFormatBayer14BGGR`*                    | 9.1              |                                                            |
|           59 |*`cudaEglColorFormatBayer14GRBG`*                    | 9.1              |                                                            |
|           60 |*`cudaEglColorFormatBayer14GBRG`*                    | 9.1              |                                                            |
|           61 |*`cudaEglColorFormatBayer20RGGB`*                    | 9.1              |                                                            |
|           62 |*`cudaEglColorFormatBayer20BGGR`*                    | 9.1              |                                                            |
|           63 |*`cudaEglColorFormatBayer20GRBG`*                    | 9.1              |                                                            |
|           64 |*`cudaEglColorFormatBayer20GBRG`*                    | 9.1              |                                                            |
|           65 |*`cudaEglColorFormatYVU444Planar`*                   | 9.1              |                                                            |
|           66 |*`cudaEglColorFormatYVU422Planar`*                   | 9.1              |                                                            |
|           67 |*`cudaEglColorFormatYVU420Planar`*                   | 9.1              |                                                            |
|           68 |*`cudaEglColorFormatBayerIspRGGB`*                   | 9.2              |                                                            |
|           69 |*`cudaEglColorFormatBayerIspBGGR`*                   | 9.2              |                                                            |
|           70 |*`cudaEglColorFormatBayerIspGRBG`*                   | 9.2              |                                                            |
|           71 |*`cudaEglColorFormatBayerIspGBRG`*                   | 9.2              |                                                            |
| enum         |***`cudaEglFrameType`***                             | 9.1              |                                                            |
|            0 |*`cudaEglFrameTypeArray`*                            | 9.1              |                                                            |
|            1 |*`cudaEglFrameTypePitch`*                            | 9.1              |                                                            |
| enum         |***`cudaExternalMemoryHandleType`***                 | 10.0             |                                                            |
|            1 |*`cudaExternalMemoryHandleTypeOpaqueFd`*             | 10.0             |                                                            |
|            2 |*`cudaExternalMemoryHandleTypeOpaqueWin32`*          | 10.0             |                                                            |
|            3 |*`cudaExternalMemoryHandleTypeOpaqueWin32Kmt`*       | 10.0             |                                                            |
|            4 |*`cudaExternalMemoryHandleTypeD3D12Heap`*            | 10.0             |                                                            |
|            5 |*`cudaExternalMemoryHandleTypeD3D12Resource`*        | 10.0             |                                                            |
| enum         |***`cudaExternalSemaphoreHandleType`***              | 10.0             |                                                            |
|            1 |*`cudaExternalSemaphoreHandleTypeOpaqueFd`*          | 10.0             |                                                            |
|            2 |*`cudaExternalSemaphoreHandleTypeOpaqueWin32`*       | 10.0             |                                                            |
|            3 |*`cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt`*    | 10.0             |                                                            |
|            4 |*`cudaExternalSemaphoreHandleTypeD3D12Fence`*        | 10.0             |                                                            |
| enum         |***`cudaFuncAttribute`***                            | 9.0              |                                                            |
|            8 |*`cudaFuncAttributeMaxDynamicSharedMemorySize`*      | 9.0              |                                                            |
|            9 |*`cudaFuncAttributePreferredSharedMemoryCarveout`*   | 9.0              |                                                            |
|           10 |*`cudaFuncAttributeMax`*                             | 9.0              |                                                            |
| enum         |***`cudaEglResourceLocationFlags`***                 | 9.1              |                                                            |
|         0x00 |*`cudaEglResourceLocationSysmem`*                    | 9.1              |                                                            |
|         0x01 |*`cudaEglResourceLocationVidmem`*                    | 9.1              |                                                            |
| enum         |***`cudaError`***                                    |                  |***`hipError_t`***                                          |
| typedef      |***`cudaError_t`***                                  |                  |***`hipError_t`***                                          |
|            0 |*`cudaSuccess`*                                      |                  |*`hipSuccess`*                                              |
|            1 |*`cudaErrorMissingConfiguration`*                    |                  |*`hipErrorMissingConfiguration`*                            | 1001                      |
|            2 |*`cudaErrorMemoryAllocation`*                        |                  |*`hipErrorMemoryAllocation`*                                | 1002                      |
|            3 |*`cudaErrorInitializationError`*                     |                  |*`hipErrorInitializationError`*                             | 1003                      |
|            4 |*`cudaErrorLaunchFailure`*                           |                  |*`hipErrorLaunchFailure`*                                   | 1004                      |
|            5 |*`cudaErrorPriorLaunchFailure`*                      |                  |*`hipErrorPriorLaunchFailure`*                              | 1005                      |
|            6 |*`cudaErrorLaunchTimeout`*                           |                  |*`hipErrorLaunchTimeOut`*                                   | 1006                      |
|            7 |*`cudaErrorLaunchOutOfResources`*                    |                  |*`hipErrorLaunchOutOfResources`*                            | 1007                      |
|            8 |*`cudaErrorInvalidDeviceFunction`*                   |                  |*`hipErrorInvalidDeviceFunction`*                           | 1008                      |
|            9 |*`cudaErrorInvalidConfiguration`*                    |                  |*`hipErrorInvalidConfiguration`*                            | 1009                      |
|           10 |*`cudaErrorInvalidDevice`*                           |                  |*`hipErrorInvalidDevice`*                                   | 1010                      |
|           11 |*`cudaErrorInvalidValue`*                            |                  |*`hipErrorInvalidValue`*                                    | 1011                      |
|           12 |*`cudaErrorInvalidPitchValue`*                       |                  |                                                            |
|           13 |*`cudaErrorInvalidSymbol`*                           |                  |*`hipErrorInvalidSymbol`*                                   | 701                       |
|           14 |*`cudaErrorMapBufferObjectFailed`*                   |                  |*`hipErrorMapFailed`*                                       | 205                       |
|           15 |*`cudaErrorUnmapBufferObjectFailed`*                 |                  |*`hipErrorUnmapFailed`*                                     | 206                       |
|           16 |*`cudaErrorInvalidHostPointer`*                      |                  |                                                            |
|           17 |*`cudaErrorInvalidDevicePointer`*                    |                  |*`hipErrorInvalidDevicePointer`*                            | 1017                      |
|           18 |*`cudaErrorInvalidTexture`*                          |                  |                                                            |
|           19 |*`cudaErrorInvalidTextureBinding`*                   |                  |                                                            |
|           20 |*`cudaErrorInvalidChannelDescriptor`*                |                  |                                                            |
|           21 |*`cudaErrorInvalidMemcpyDirection`*                  |                  |*`hipErrorInvalidMemcpyDirection`*                          | 1021                      |
|           22 |*`cudaErrorAddressOfConstant`*                       |                  |                                                            |
|           23 |*`cudaErrorTextureFetchFailed`*                      |                  |                                                            |
|           24 |*`cudaErrorTextureNotBound`*                         |                  |                                                            |
|           25 |*`cudaErrorSynchronizationError`*                    |                  |                                                            |
|           26 |*`cudaErrorInvalidFilterSetting`*                    |                  |                                                            |
|           27 |*`cudaErrorInvalidNormSetting`*                      |                  |                                                            |
|           28 |*`cudaErrorMixedDeviceExecution`*                    |                  |                                                            |
|           29 |*`cudaErrorCudartUnloading`*                         |                  |*`hipErrorDeinitialized`*                                   | 4                         |
|           30 |*`cudaErrorUnknown`*                                 |                  |*`hipErrorUnknown`*                                         | 1030                      |
|           31 |*`cudaErrorNotYetImplemented`*                       |                  |                                                            |
|           32 |*`cudaErrorMemoryValueTooLarge`*                     |                  |                                                            |
|           33 |*`cudaErrorInvalidResourceHandle`*                   |                  |*`hipErrorInvalidResourceHandle`*                           | 1033                      |
|           34 |*`cudaErrorNotReady`*                                |                  |*`hipErrorNotReady`*                                        | 1034                      |
|           35 |*`cudaErrorInsufficientDriver`*                      |                  |*`hipErrorInsufficientDriver`*                              |
|           36 |*`cudaErrorSetOnActiveProcess`*                      |                  |                                                            |
|           37 |*`cudaErrorInvalidSurface`*                          |                  |                                                            |
|           38 |*`cudaErrorNoDevice`*                                |                  |*`hipErrorNoDevice`*                                        | 1038                      |
|           39 |*`cudaErrorECCUncorrectable`*                        |                  |*`hipErrorECCNotCorrectable`*                               | 214                       |
|           40 |*`cudaErrorSharedObjectSymbolNotFound`*              |                  |*`hipErrorSharedObjectSymbolNotFound`*                      | 302                       |
|           41 |*`cudaErrorSharedObjectInitFailed`*                  |                  |*`hipErrorSharedObjectInitFailed`*                          | 303                       |
|           42 |*`cudaErrorUnsupportedLimit`*                        |                  |*`hipErrorUnsupportedLimit`*                                | 215                       |
|           43 |*`cudaErrorDuplicateVariableName`*                   |                  |                                                            |
|           44 |*`cudaErrorDuplicateTextureName`*                    |                  |                                                            |
|           45 |*`cudaErrorDuplicateSurfaceName`*                    |                  |                                                            |
|           46 |*`cudaErrorDevicesUnavailable`*                      |                  |                                                            |
|           47 |*`cudaErrorInvalidKernelImage`*                      |                  |*`hipErrorInvalidImage`*                                    | 200                       |
|           48 |*`cudaErrorNoKernelImageForDevice`*                  |                  |*`hipErrorNoBinaryForGpu`*                                  | 209                       |
|           49 |*`cudaErrorIncompatibleDriverContext`*               |                  |                                                            |
|           50 |*`cudaErrorPeerAccessAlreadyEnabled`*                |                  |*`hipErrorPeerAccessAlreadyEnabled`*                        | 1050                      |
|           51 |*`cudaErrorPeerAccessNotEnabled`*                    |                  |*`hipErrorPeerAccessNotEnabled`*                            | 1051                      |
|           52 |                                                     |                  |                                                            |
|           53 |                                                     |                  |                                                            |
|           54 |*`cudaErrorDeviceAlreadyInUse`*                      |                  |                                                            |
|           55 |*`cudaErrorProfilerDisabled`*                        |                  |*`hipErrorProfilerDisabled`*                                | 5                         |
|           56 |*`cudaErrorProfilerNotInitialized`*                  |                  |*`hipErrorProfilerNotInitialized`*                          | 6                         |
|           57 |*`cudaErrorProfilerAlreadyStarted`*                  |                  |*`hipErrorProfilerAlreadyStarted`*                          | 7                         |
|           58 |*`cudaErrorProfilerAlreadyStopped`*                  |                  |*`hipErrorProfilerAlreadyStopped`*                          | 8                         |
|           59 |*`cudaErrorAssert`*                                  |                  |*`hipErrorAssert`*                                          | 1081                      |
|           60 |*`cudaErrorTooManyPeers`*                            |                  |                                                            |
|           61 |*`cudaErrorHostMemoryAlreadyRegistered`*             |                  |*`hipErrorHostMemoryAlreadyRegistered`*                     | 1061                      |
|           62 |*`cudaErrorHostMemoryNotRegistered`*                 |                  |*`hipErrorHostMemoryNotRegistered`*                         | 1062                      |
|           63 |*`cudaErrorOperatingSystem`*                         |                  |*`hipErrorOperatingSystem`*                                 | 304                       |
|           64 |*`cudaErrorPeerAccessUnsupported`*                   |                  |*`hipErrorPeerAccessUnsupported`*                           | 217                       |
|           65 |*`cudaErrorLaunchMaxDepthExceeded`*                  |                  |                                                            |
|           66 |*`cudaErrorLaunchFileScopedTex`*                     |                  |                                                            |
|           67 |*`cudaErrorLaunchFileScopedSurf`*                    |                  |                                                            |
|           68 |*`cudaErrorSyncDepthExceeded`*                       |                  |                                                            |
|           69 |*`cudaErrorLaunchPendingCountExceeded`*              |                  |                                                            |
|           70 |*`cudaErrorNotPermitted`*                            |                  |                                                            |
|           71 |*`cudaErrorNotSupported`*                            |                  |                                                            |
|           72 |*`cudaErrorHardwareStackError`*                      |                  |                                                            |
|           73 |*`cudaErrorIllegalInstruction`*                      |                  |                                                            |
|           74 |*`cudaErrorMisalignedAddress`*                       |                  |                                                            |
|           75 |*`cudaErrorInvalidAddressSpace`*                     |                  |                                                            |
|           76 |*`cudaErrorInvalidPc`*                               |                  |                                                            |
|           77 |*`cudaErrorIllegalAddress`*                          |                  |*`hipErrorIllegalAddress`*                                  | 700                       |
|           78 |*`cudaErrorInvalidPtx`*                              |                  |*`hipErrorInvalidKernelFile`*                               | 218                       |
|           79 |*`cudaErrorInvalidGraphicsContext`*                  |                  |*`hipErrorInvalidGraphicsContext`*                          | 219                       |
|           80 |*`cudaErrorNvlinkUncorrectable`*                     | 8.0              |                                                            |
|           81 |*`cudaErrorJitCompilerNotFound`*                     | 9.0              |                                                            |
|           82 |*`cudaErrorCooperativeLaunchTooLarge`*               | 9.0              |                                                            |
|           83 |*`cudaErrorSystemNotReady`*                          | 10.0             |                                                            |
|           84 |*`cudaErrorIllegalState`*                            | 10.0             |                                                            |
|          127 |*`cudaErrorStartupFailure`*                          | 10.0             |                                                            |
|          803 |*`cudaErrorSystemDriverMismatch`*                    | 10.0             |                                                            |
|          804 |*`cudaErrorCompatNotSupportedOnDevice`*              | 10.0             |                                                            |
|          900 |*`cudaErrorStreamCaptureUnsupported`*                | 10.0             |                                                            |
|          901 |*`cudaErrorStreamCaptureInvalidated`*                | 10.0             |                                                            |
|          902 |*`cudaErrorStreamCaptureMerge`*                      | 10.0             |                                                            |
|          903 |*`cudaErrorStreamCaptureUnmatched`*                  | 10.0             |                                                            |
|          904 |*`cudaErrorStreamCaptureUnjoined`*                   | 10.0             |                                                            |
|          905 |*`cudaErrorStreamCaptureIsolation`*                  | 10.0             |                                                            |
|          906 |*`cudaErrorStreamCaptureImplicit`*                   | 10.0             |                                                            |
|          907 |*`cudaErrorCapturedEvent`*                           | 10.0             |                                                            |
|          908 |*`cudaErrorStreamCaptureWrongThread`*                | 10.1             |                                                            |
|        10000 |*`cudaErrorApiFailureBase`*                          |                  |                                                            |
| enum         |***`cudaFuncCache`***                                |                  |***`hipFuncCache_t`***                                      |
|            0 |*`cudaFuncCachePreferNone`*                          |                  |*`hipFuncCachePreferNone`*                                  |
|            1 |*`cudaFuncCachePreferShared`*                        |                  |*`hipFuncCachePreferShared`*                                |
|            2 |*`cudaFuncCachePreferL1`*                            |                  |*`hipFuncCachePreferL1`*                                    |
|            3 |*`cudaFuncCachePreferEqual`*                         |                  |*`hipFuncCachePreferEqual`*                                 |
| enum         |***`cudaGraphicsCubeFace`***                         |                  |                                                            |
|         0x00 |*`cudaGraphicsCubeFacePositiveX`*                    |                  |                                                            |
|         0x01 |*`cudaGraphicsCubeFaceNegativeX`*                    |                  |                                                            |
|         0x02 |*`cudaGraphicsCubeFacePositiveY`*                    |                  |                                                            |
|         0x03 |*`cudaGraphicsCubeFaceNegativeY`*                    |                  |                                                            |
|         0x04 |*`cudaGraphicsCubeFacePositiveZ`*                    |                  |                                                            |
|         0x05 |*`cudaGraphicsCubeFaceNegativeZ`*                    |                  |                                                            |
| enum         |***`cudaGraphicsMapFlags`***                         |                  |                                                            |
|            0 |*`cudaGraphicsMapFlagsNone`*                         |                  |                                                            |
|            1 |*`cudaGraphicsMapFlagsReadOnly`*                     |                  |                                                            |
|            2 |*`cudaGraphicsMapFlagsWriteDiscard`*                 |                  |                                                            |
| enum         |***`cudaGraphicsRegisterFlags`***                    |                  |                                                            |
|            0 |*`cudaGraphicsRegisterFlagsNone`*                    |                  |                                                            |
|            1 |*`cudaGraphicsRegisterFlagsReadOnly`*                |                  |                                                            |
|            2 |*`cudaGraphicsRegisterFlagsWriteDiscard`*            |                  |                                                            |
|            4 |*`cudaGraphicsRegisterFlagsSurfaceLoadStore`*        |                  |                                                            |
|            8 |*`cudaGraphicsRegisterFlagsTextureGather`*           |                  |                                                            |
| enum         |***`cudaGraphNodeType`***                            | 10.0             |                                                            |
|         0x00 |*`cudaGraphNodeTypeKernel`*                          | 10.0             |                                                            |
|         0x01 |*`cudaGraphNodeTypeMemcpy`*                          | 10.0             |                                                            |
|         0x02 |*`cudaGraphNodeTypeMemset`*                          | 10.0             |                                                            |
|         0x03 |*`cudaGraphNodeTypeHost`*                            | 10.0             |                                                            |
|         0x04 |*`cudaGraphNodeTypeGraph`*                           | 10.0             |                                                            |
|         0x05 |*`cudaGraphNodeTypeEmpty`*                           | 10.0             |                                                            |
|              |*`cudaGraphNodeTypeCount`*                           | 10.0             |                                                            |
| enum         |***`cudaLimit`***                                    |                  |***`hipLimit_t`***                                          |
|         0x00 |*`cudaLimitStackSize`*                               |                  |                                                            |
|         0x01 |*`cudaLimitPrintfFifoSize`*                          |                  |                                                            |
|         0x02 |*`cudaLimitMallocHeapSize`*                          |                  |*`hipLimitMallocHeapSize`*                                  |
|         0x03 |*`cudaLimitDevRuntimeSyncDepth`*                     |                  |                                                            |
|         0x04 |*`cudaLimitDevRuntimePendingLaunchCount`*            |                  |                                                            |
|         0x05 |*`cudaLimitMaxL2FetchGranularity`*                   | 10.0             |                                                            |
| enum         |***`cudaMemcpyKind`***                               |                  |***`hipMemcpyKind`***                                       |
|            0 |*`cudaMemcpyHostToHost`*                             |                  |*`hipMemcpyHostToHost`*                                     |
|            1 |*`cudaMemcpyHostToDevice`*                           |                  |*`hipMemcpyHostToDevice`*                                   |
|            2 |*`cudaMemcpyDeviceToHost`*                           |                  |*`hipMemcpyDeviceToHost`*                                   |
|            3 |*`cudaMemcpyDeviceToDevice`*                         |                  |*`hipMemcpyDeviceToDevice`*                                 |
|            4 |*`cudaMemcpyDefault`*                                |                  |*`hipMemcpyDefault`*                                        |
| enum         |***`cudaMemoryAdvise`***                             | 8.0              |                                                            |
|            1 |*`cudaMemAdviseSetReadMostly`*                       | 8.0              |                                                            |
|            2 |*`cudaMemAdviseUnsetReadMostly`*                     | 8.0              |                                                            |
|            3 |*`cudaMemAdviseSetPreferredLocation`*                | 8.0              |                                                            |
|            4 |*`cudaMemAdviseUnsetPreferredLocation`*              | 8.0              |                                                            |
|            5 |*`cudaMemAdviseSetAccessedBy`*                       | 8.0              |                                                            |
|            6 |*`cudaMemAdviseUnsetAccessedBy`*                     | 8.0              |                                                            |
| enum         |***`cudaMemoryType`***                               |                  |                                                            |
|            0 |*`cudaMemoryTypeUnregistered`*                       |                  |                                                            |
|            1 |*`cudaMemoryTypeHost`*                               |                  |                                                            |
|            2 |*`cudaMemoryTypeDevice`*                             |                  |                                                            |
|            3 |*`cudaMemoryTypeManaged`*                            | 10.0             |                                                            |
| enum         |***`cudaMemRangeAttribute`***                        | 8.0              |                                                            |
|            1 |*`cudaMemRangeAttributeReadMostly`*                  | 8.0              |                                                            |
|            2 |*`cudaMemRangeAttributePreferredLocation`*           | 8.0              |                                                            |
|            3 |*`cudaMemRangeAttributeAccessedBy`*                  | 8.0              |                                                            |
|            4 |*`cudaMemRangeAttributeLastPrefetchLocation`*        | 8.0              |                                                            |
| enum         |***`cudaResourceType`***                             |                  |***`hipResourceType`***                                     |
|         0x00 |*`cudaResourceTypeArray`*                            |                  |*`hipResourceTypeArray`*                                    |
|         0x01 |*`cudaResourceTypeMipmappedArray`*                   |                  |*`hipResourceTypeMipmappedArray`*                           |
|         0x02 |*`cudaResourceTypeLinear`*                           |                  |*`hipResourceTypeLinear`*                                   |
|         0x03 |*`cudaResourceTypePitch2D`*                          |                  |*`hipResourceTypePitch2D`*                                  |
| enum         |***`cudaResourceViewFormat`***                       |                  |***`hipResourceViewFormat`***                               |
|         0x00 |*`cudaResViewFormatNone`*                            |                  |*`hipResViewFormatNone`*                                    |
|         0x01 |*`cudaResViewFormatUnsignedChar1`*                   |                  |*`hipResViewFormatUnsignedChar1`*                           |
|         0x02 |*`cudaResViewFormatUnsignedChar2`*                   |                  |*`hipResViewFormatUnsignedChar2`*                           |
|         0x03 |*`cudaResViewFormatUnsignedChar4`*                   |                  |*`hipResViewFormatUnsignedChar4`*                           |
|         0x04 |*`cudaResViewFormatSignedChar1`*                     |                  |*`hipResViewFormatSignedChar1`*                             |
|         0x05 |*`cudaResViewFormatSignedChar2`*                     |                  |*`hipResViewFormatSignedChar2`*                             |
|         0x06 |*`cudaResViewFormatSignedChar4`*                     |                  |*`hipResViewFormatSignedChar4`*                             |
|         0x07 |*`cudaResViewFormatUnsignedShort1`*                  |                  |*`hipResViewFormatUnsignedShort1`*                          |
|         0x08 |*`cudaResViewFormatUnsignedShort2`*                  |                  |*`hipResViewFormatUnsignedShort2`*                          |
|         0x09 |*`cudaResViewFormatUnsignedShort4`*                  |                  |*`hipResViewFormatUnsignedShort4`*                          |
|         0x0a |*`cudaResViewFormatSignedShort1`*                    |                  |*`hipResViewFormatSignedShort1`*                            |
|         0x0b |*`cudaResViewFormatSignedShort2`*                    |                  |*`hipResViewFormatSignedShort2`*                            |
|         0x0c |*`cudaResViewFormatSignedShort4`*                    |                  |*`hipResViewFormatSignedShort4`*                            |
|         0x0d |*`cudaResViewFormatUnsignedInt1`*                    |                  |*`hipResViewFormatUnsignedInt1`*                            |
|         0x0e |*`cudaResViewFormatUnsignedInt2`*                    |                  |*`hipResViewFormatUnsignedInt2`*                            |
|         0x0f |*`cudaResViewFormatUnsignedInt4`*                    |                  |*`hipResViewFormatUnsignedInt4`*                            |
|         0x10 |*`cudaResViewFormatSignedInt1`*                      |                  |*`hipResViewFormatSignedInt1`*                              |
|         0x11 |*`cudaResViewFormatSignedInt2`*                      |                  |*`hipResViewFormatSignedInt2`*                              |
|         0x12 |*`cudaResViewFormatSignedInt4`*                      |                  |*`hipResViewFormatSignedInt4`*                              |
|         0x13 |*`cudaResViewFormatHalf1`*                           |                  |*`hipResViewFormatHalf1`*                                   |
|         0x14 |*`cudaResViewFormatHalf2`*                           |                  |*`hipResViewFormatHalf2`*                                   |
|         0x15 |*`cudaResViewFormatHalf4`*                           |                  |*`hipResViewFormatHalf4`*                                   |
|         0x16 |*`cudaResViewFormatFloat1`*                          |                  |*`hipResViewFormatFloat1`*                                  |
|         0x17 |*`cudaResViewFormatFloat2`*                          |                  |*`hipResViewFormatFloat2`*                                  |
|         0x18 |*`cudaResViewFormatFloat4`*                          |                  |*`hipResViewFormatFloat4`*                                  |
|         0x19 |*`cudaResViewFormatUnsignedBlockCompressed1`*        |                  |*`hipResViewFormatUnsignedBlockCompressed1`*                |
|         0x1a |*`cudaResViewFormatUnsignedBlockCompressed2`*        |                  |*`hipResViewFormatUnsignedBlockCompressed2`*                |
|         0x1b |*`cudaResViewFormatUnsignedBlockCompressed3`*        |                  |*`hipResViewFormatUnsignedBlockCompressed3`*                |
|         0x1c |*`cudaResViewFormatUnsignedBlockCompressed4`*        |                  |*`hipResViewFormatUnsignedBlockCompressed4`*                |
|         0x1d |*`cudaResViewFormatSignedBlockCompressed4`*          |                  |*`hipResViewFormatSignedBlockCompressed4`*                  |
|         0x1e |*`cudaResViewFormatUnsignedBlockCompressed5`*        |                  |*`hipResViewFormatUnsignedBlockCompressed5`*                |
|         0x1f |*`cudaResViewFormatSignedBlockCompressed5`*          |                  |*`hipResViewFormatSignedBlockCompressed5`*                  |
|         0x20 |*`cudaResViewFormatUnsignedBlockCompressed6H`*       |                  |*`hipResViewFormatUnsignedBlockCompressed6H`*               |
|         0x21 |*`cudaResViewFormatSignedBlockCompressed6H`*         |                  |*`hipResViewFormatSignedBlockCompressed6H`*                 |
|         0x22 |*`cudaResViewFormatUnsignedBlockCompressed7`*        |                  |*`hipResViewFormatUnsignedBlockCompressed7`*                |
| enum         |***`cudaSharedMemConfig`***                          |                  |***`hipSharedMemConfig`***                                  |
|            0 |*`cudaSharedMemBankSizeDefault`*                     |                  |*`hipSharedMemBankSizeDefault`*                             |
|            1 |*`cudaSharedMemBankSizeFourByte`*                    |                  |*`hipSharedMemBankSizeFourByte`*                            |
|            2 |*`cudaSharedMemBankSizeEightByte`*                   |                  |*`hipSharedMemBankSizeEightByte`*                           |
| enum         |***`cudaSharedCarveout`***                           | 9.0              |                                                            |
|           -1 |*`cudaSharedmemCarveoutDefault`*                     | 9.0              |                                                            |
|          100 |*`cudaSharedmemCarveoutMaxShared`*                   | 9.0              |                                                            |
|            0 |*`cudaSharedmemCarveoutMaxL1`*                       | 9.0              |                                                            |
| enum         |***`cudaStreamCaptureStatus`***                      | 10.0             |                                                            |
|            0 |*`cudaStreamCaptureStatusNone`*                      | 10.0             |                                                            |
|            1 |*`cudaStreamCaptureStatusActive`*                    | 10.0             |                                                            |
|            2 |*`cudaStreamCaptureStatusInvalidated`*               | 10.0             |                                                            |
| enum         |***`cudaStreamCaptureMode`***                        | 10.1             |                                                            |
|            0 |*`cudaStreamCaptureModeGlobal`*                      | 10.1             |                                                            |
|            1 |*`cudaStreamCaptureModeThreadLocal`*                 | 10.1             |                                                            |
|            2 |*`cudaStreamCaptureModeRelaxed`*                     | 10.1             |                                                            |
| enum         |***`cudaSurfaceBoundaryMode`***                      |                  |***`hipSurfaceBoundaryMode`***                              |
|            0 |*`cudaBoundaryModeZero`*                             |                  |*`hipBoundaryModeZero`*                                     |
|            1 |*`cudaBoundaryModeClamp`*                            |                  |*`hipBoundaryModeClamp`*                                    |
|            2 |*`cudaBoundaryModeTrap`*                             |                  |*`hipBoundaryModeTrap`*                                     |
| enum         |***`cudaSurfaceFormatMode`***                        |                  |                                                            |
|            0 |*`cudaFormatModeForced`*                             |                  |                                                            |
|            1 |*`cudaFormatModeAuto`*                               |                  |                                                            |
| enum         |***`cudaTextureAddressMode`***                       |                  |***`hipTextureAddressMode`***                               |
|            0 |*`cudaAddressModeWrap`*                              |                  |*`hipAddressModeWrap`*                                      |
|            1 |*`cudaAddressModeClamp`*                             |                  |*`hipAddressModeClamp`*                                     |
|            2 |*`cudaAddressModeMirror`*                            |                  |*`hipAddressModeMirror`*                                    |
|            3 |*`cudaAddressModeBorder`*                            |                  |*`hipAddressModeBorder`*                                    |
| enum         |***`cudaTextureFilterMode`***                        |                  |***`hipTextureFilterMode`***                                |
|            0 |*`cudaFilterModePoint`*                              |                  |*`hipFilterModePoint`*                                      |
|            1 |*`cudaFilterModeLinear`*                             |                  |*`hipFilterModeLinear`*                                     |
| enum         |***`cudaTextureReadMode`***                          |                  |***`hipTextureReadMode`***                                  |
|            0 |*`cudaReadModeElementType`*                          |                  |*`hipReadModeElementType`*                                  |
|            1 |*`cudaReadModeNormalizedFloat`*                      |                  |*`hipReadModeNormalizedFloat`*                              |
| enum         |***`cudaGLDeviceList`***                             |                  |                                                            |
|            1 |*`cudaGLDeviceListAll`*                              |                  |                                                            |
|            2 |*`cudaGLDeviceListCurrentFrame`*                     |                  |                                                            |
|            3 |*`cudaGLDeviceListNextFrame`*                        |                  |                                                            |
| enum         |***`cudaGLMapFlags`***                               |                  |                                                            |
|            0 |*`cudaGLMapFlagsNone`*                               |                  |                                                            |
|            1 |*`cudaGLMapFlagsReadOnly`*                           |                  |                                                            |
|            2 |*`cudaGLMapFlagsWriteDiscard`*                       |                  |                                                            |
| enum         |***`cudaD3D9DeviceList`***                           |                  |                                                            |
|            1 |*`cudaD3D9DeviceListAll`*                            |                  |                                                            |
|            2 |*`cudaD3D9DeviceListCurrentFrame`*                   |                  |                                                            |
|            3 |*`cudaD3D9DeviceListNextFrame`*                      |                  |                                                            |
| enum         |***`cudaD3D9MapFlags`***                             |                  |                                                            |
|            0 |*`cudaD3D9MapFlagsNone`*                             |                  |                                                            |
|            1 |*`cudaD3D9MapFlagsReadOnly`*                         |                  |                                                            |
|            2 |*`cudaD3D9MapFlagsWriteDiscard`*                     |                  |                                                            |
| enum         |***`cudaD3D9RegisterFlags`***                        |                  |                                                            |
|            0 |*`cudaD3D9RegisterFlagsNone`*                        |                  |                                                            |
|            1 |*`cudaD3D9RegisterFlagsArray`*                       |                  |                                                            |
| enum         |***`cudaD3D10DeviceList`***                          |                  |                                                            |
|            1 |*`cudaD3D10DeviceListAll`*                           |                  |                                                            |
|            2 |*`cudaD3D10DeviceListCurrentFrame`*                  |                  |                                                            |
|            3 |*`cudaD3D10DeviceListNextFrame`*                     |                  |                                                            |
| enum         |***`cudaD3D10MapFlags`***                            |                  |                                                            |
|            0 |*`cudaD3D10MapFlagsNone`*                            |                  |                                                            |
|            1 |*`cudaD3D10MapFlagsReadOnly`*                        |                  |                                                            |
|            2 |*`cudaD3D10MapFlagsWriteDiscard`*                    |                  |                                                            |
| enum         |***`cudaD3D10RegisterFlags`***                       |                  |                                                            |
|            0 |*`cudaD3D10RegisterFlagsNone`*                       |                  |                                                            |
|            1 |*`cudaD3D10RegisterFlagsArray`*                      |                  |                                                            |
| enum         |***`cudaD3D11DeviceList`***                          |                  |                                                            |
|            1 |*`cudaD3D11DeviceListAll`*                           |                  |                                                            |
|            2 |*`cudaD3D11DeviceListCurrentFrame`*                  |                  |                                                            |
|            3 |*`cudaD3D11DeviceListNextFrame`*                     |                  |                                                            |
| struct       |`cudaArray`                                          |                  |`hipArray`                                                  |
| typedef      |`cudaArray_t`                                        |                  |`hipArray_t`                                                |
| typedef      |`cudaArray_const_t`                                  |                  |`hipArray_const_t`                                          |
| typedef      |`cudaEvent_t`                                        |                  |`hipEvent_t`                                                |
| struct       |`CUevent_st`                                         |                  |`ihipEvent_t`                                               |
| typedef      |`cudaGraphicsResource_t`                             |                  |                                                            |
| struct       |`cudaMipmappedArray`                                 |                  |`hipMipmappedArray`                                         |
| typedef      |`cudaMipmappedArray_t`                               |                  |`hipMipmappedArray_t`                                       |
| typedef      |`cudaMipmappedArray_const_t`                         |                  |`hipMipmappedArray_const_t`                                 |
| enum         |***`cudaOutputMode`***                               |                  |                                                            |
| typedef      |***`cudaOutputMode_t`***                             |                  |                                                            |
|         0x00 |*`cudaKeyValuePair`*                                 |                  |                                                            |
|         0x01 |*`cudaCSV`*                                          |                  |                                                            |
| typedef      |`cudaStream_t`                                       |                  |`hipStream_t`                                               |
| struct       |`CUstream_st`                                        |                  |`ihipStream_t`                                              |
| typedef      |`cudaStreamCallback_t`                               |                  |`hipStreamCallback_t`                                       |
| typedef      |`cudaSurfaceObject_t`                                |                  |`hipSurfaceObject_t`                                        |
| typedef      |`cudaTextureObject_t`                                |                  |`hipTextureObject_t`                                        |
| struct       |`CUuuid_st`                                          |                  |                                                            |
| typedef      |`cudaUUID_t`                                         |                  |                                                            |
| define       |`CUDA_EGL_MAX_PLANES`                                | 9.1              |                                                            |
| define       |`CUDA_IPC_HANDLE_SIZE`                               |                  |                                                            |
| define       |`cudaArrayColorAttachment`                           | 10.0             |                                                            |
| define       |`cudaArrayCubemap`                                   |                  |`hipArrayCubemap`                                           |
| define       |`cudaArrayDefault`                                   |                  |`hipArrayDefault`                                           |
| define       |`cudaArrayLayered`                                   |                  |`hipArrayLayered`                                           |
| define       |`cudaArraySurfaceLoadStore`                          |                  |`hipArraySurfaceLoadStore`                                  |
| define       |`cudaArrayTextureGather`                             |                  |`hipArrayTextureGather`                                     |
| define       |`cudaCooperativeLaunchMultiDeviceNoPreSync`          | 9.0              |                                                            |
| define       |`cudaCooperativeLaunchMultiDeviceNoPostSync`         | 9.0              |                                                            |
| define       |`cudaCpuDeviceId`                                    | 8.0              |                                                            |
| define       |`cudaInvalidDeviceId`                                | 8.0              |                                                            |
| define       |`cudaDeviceBlockingSync`                             |                  |`hipDeviceScheduleBlockingSync`                             |
| define       |`cudaDeviceLmemResizeToMax`                          |                  |`hipDeviceLmemResizeToMax`                                  | 0x16                      |
| define       |`cudaDeviceMapHost`                                  |                  |`hipDeviceMapHost`                                          |
| define       |`cudaDeviceMask`                                     |                  |                                                            |
| define       |`cudaDevicePropDontCare`                             |                  |                                                            |
| define       |`cudaDeviceScheduleAuto`                             |                  |`hipDeviceScheduleAuto`                                     |
| define       |`cudaDeviceScheduleBlockingSync`                     |                  |`hipDeviceScheduleBlockingSync`                             |
| define       |`cudaDeviceScheduleMask`                             |                  |`hipDeviceScheduleMask`                                     |
| define       |`cudaDeviceScheduleSpin`                             |                  |`hipDeviceScheduleSpin`                                     |
| define       |`cudaDeviceScheduleYield`                            |                  |`hipDeviceScheduleYield`                                    |
| define       |`cudaEventDefault`                                   |                  |`hipEventDefault`                                           |
| define       |`cudaEventBlockingSync`                              |                  |`hipEventBlockingSync`                                      |
| define       |`cudaEventDisableTiming`                             |                  |`hipEventDisableTiming`                                     |
| define       |`cudaEventInterprocess`                              |                  |`hipEventInterprocess`                                      |
| define       |`cudaHostAllocDefault`                               |                  |`hipHostMallocDefault`                                      |
| define       |`cudaHostAllocMapped`                                |                  |`hipHostMallocMapped`                                       |
| define       |`cudaHostAllocPortable`                              |                  |`hipHostMallocPortable`                                     |
| define       |`cudaHostAllocWriteCombined`                         |                  |`hipHostMallocWriteCombined`                                |
| define       |`cudaHostRegisterDefault`                            |                  |`hipHostRegisterDefault`                                    |
| define       |`cudaHostRegisterIoMemory`                           | 7.5              |`hipHostRegisterIoMemory`                                   |
| define       |`cudaHostRegisterMapped`                             |                  |`hipHostRegisterMapped`                                     |
| define       |`cudaHostRegisterPortable`                           |                  |`hipHostRegisterPortable`                                   |
| define       |`cudaIpcMemLazyEnablePeerAccess`                     |                  |`hipIpcMemLazyEnablePeerAccess`                             | 0                         |
| define       |`cudaMemAttachGlobal`                                |                  |`hipMemAttachGlobal`                                        |
| define       |`cudaMemAttachHost`                                  |                  |`hipMemAttachHost`                                          |
| define       |`cudaMemAttachSingle`                                |                  |                                                            |
| define       |`cudaOccupancyDefault`                               |                  |                                                            |
| define       |`cudaOccupancyDisableCachingOverride`                |                  |                                                            |
| define       |`cudaPeerAccessDefault`                              |                  |                                                            |
| define       |`cudaStreamDefault`                                  |                  |`hipStreamDefault`                                          |
| define       |`cudaStreamNonBlocking`                              |                  |`hipStreamNonBlocking`                                      |
| define       |`cudaStreamLegacy`                                   |                  |                                                            |
| define       |`cudaStreamPerThread`                                |                  |                                                            |
| define       |`cudaTextureType1D`                                  |                  |`hipTextureType1D`                                          |
| define       |`cudaTextureType2D`                                  |                  |`hipTextureType2D`                                          |
| define       |`cudaTextureType3D`                                  |                  |`hipTextureType3D`                                          |
| define       |`cudaTextureTypeCubemap`                             |                  |`hipTextureTypeCubemap`                                     |
| define       |`cudaTextureType1DLayered`                           |                  |`hipTextureType1DLayered`                                   |
| define       |`cudaTextureType2DLayered`                           |                  |`hipTextureType2DLayered`                                   |
| define       |`cudaTextureTypeCubemapLayered`                      |                  |`hipTextureTypeCubemapLayered`                              |
| enum         |***`cudaDataType_t`***                               | 8.0              |***`hipblasDatatype_t`***                                   |
| enum         |***`cudaDataType`***                                 | 8.0              |***`hipblasDatatype_t`***                                   |
|            2 |*`CUDA_R_16F`*                                       | 8.0              |*`HIPBLAS_R_16F`*                                           | 150                       |
|            6 |*`CUDA_C_16F`*                                       | 8.0              |*`HIPBLAS_C_16F`*                                           | 153                       |
|            0 |*`CUDA_R_32F`*                                       | 8.0              |*`HIPBLAS_R_32F`*                                           | 151                       |
|            4 |*`CUDA_C_32F`*                                       | 8.0              |*`HIPBLAS_C_32F`*                                           | 154                       |
|            1 |*`CUDA_R_64F`*                                       | 8.0              |*`HIPBLAS_R_64F`*                                           | 152                       |
|            5 |*`CUDA_C_64F`*                                       | 8.0              |*`HIPBLAS_C_64F`*                                           | 155                       |
|            3 |*`CUDA_R_8I`*                                        | 8.0              |                                                            |
|            7 |*`CUDA_C_8I`*                                        | 8.0              |                                                            |
|            8 |*`CUDA_R_8U`*                                        | 8.0              |                                                            |
|            9 |*`CUDA_C_8U`*                                        | 8.0              |                                                            |
|           10 |*`CUDA_R_32I`*                                       | 8.0              |                                                            |
|           11 |*`CUDA_C_32I`*                                       | 8.0              |                                                            |
|           12 |*`CUDA_R_32U`*                                       | 8.0              |                                                            |
|           13 |*`CUDA_C_32U`*                                       | 8.0              |                                                            |
| struct       |`cudaExternalMemoryBufferDesc`                       | 10.0             |                                                            |
| struct       |`cudaExternalMemoryHandleDesc`                       | 10.0             |                                                            |
| struct       |`cudaExternalMemoryMipmappedArrayDesc`               | 10.0             |                                                            |
| struct       |`cudaExternalSemaphoreHandleDesc`                    | 10.0             |                                                            |
| struct       |`cudaExternalSemaphoreSignalParams`                  | 10.0             |                                                            |
| struct       |`cudaExternalSemaphoreWaitParams`                    | 10.0             |                                                            |
| struct       |`cudaHostNodeParams`                                 | 10.0             |                                                            |
| struct       |`cudaLaunchParams`                                   | 9.0              |`hipLaunchParams`                                           |
| struct       |`cudaMemsetParams`                                   | 10.0             |                                                            |
| struct       |`CUeglStreamConnection_st`                           | 9.1              |                                                            |
| typedef      |`cudaEglStreamConnection`                            | 9.1              |                                                            |
| define       |`cudaExternalMemoryDedicated`                        | 10.0             |                                                            |
| typedef      |`cudaExternalMemory_t`                               | 10.0             |                                                            |
| struct       |`CUexternalMemory_st`                                | 10.0             |                                                            |
| typedef      |`cudaExternalSemaphore_t`                            | 10.0             |                                                            |
| struct       |`CUexternalSemaphore_st`                             | 10.0             |                                                            |
| typedef      |`cudaGraph_t`                                        | 10.0             |                                                            |
| struct       |`CUgraph_st`                                         | 10.0             |                                                            |
| typedef      |`cudaGraphNode_t`                                    | 10.0             |                                                            |
| struct       |`CUgraphNode_st`                                     | 10.0             |                                                            |
| typedef      |`cudaGraphExec_t`                                    | 10.0             |                                                            |
| struct       |`CUgraphExec_st`                                     | 10.0             |                                                            |
| typedef      |`cudaGraphicsResource_t`                             |                  |                                                            |
| struct       |`cudaGraphicsResource`                               |                  |                                                            |
| typedef      |`cudaHostFn_t`                                       | 10.0             |                                                            |
| enum         |***`libraryPropertyType`***                          | 8.0              |                                                            |
| typedef      |***`libraryPropertyType_t`***                        | 8.0              |                                                            |
|            0 |*`MAJOR_VERSION`*                                    | 8.0              |                                                            |
|            1 |*`MINOR_VERSION`*                                    | 8.0              |                                                            |
|            2 |*`PATCH_LEVEL`*                                      | 8.0              |                                                            |

\* CUDA version, in which API has appeared and (optional) last version before abandoning it; no value in case of earlier versions < 7.5.
