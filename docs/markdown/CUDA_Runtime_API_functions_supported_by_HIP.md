# CUDA Runtime API functions supported by HIP

## **1. Device Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaChooseDevice`                                        | `hipChooseDevice`             | Select compute-device which best matches criteria.                                                                             |
| `cudaDeviceGetAttribute`                                  | `hipDeviceGetAttribute`       | Returns information about the device.                                                                                          |
| `cudaDeviceGetByPCIBusId`                                 | `hipDeviceGetByPCIBusId`      | Returns a handle to a compute device.                                                                                          |
| `cudaDeviceGetCacheConfig`                                | `hipDeviceGetCacheConfig`     | Returns the preferred cache configuration for the current device.                                                              |
| `cudaDeviceGetLimit`                                      | `hipDeviceGetLimit`           | Returns resource limits.                                                                                                       |
| `cudaDeviceGetPCIBusId`                                   | `hipDeviceGetPCIBusId`        | Returns a PCI Bus Id string for the device.                                                                                    |
| `cudaDeviceGetSharedMemConfig`                            | `hipDeviceGetSharedMemConfig` | Returns the shared memory configuration for the current device.                                                                |
| `cudaDeviceGetStreamPriorityRange`                        |                               | Returns numerical values that correspond to the least and greatest stream priorities.                                          |
| `cudaDeviceReset`                                         | `hipDeviceReset`              | Destroy all allocations and reset all state on the current device in the current process.                                      |
| `cudaDeviceSetCacheConfig`                                | `hipDeviceSetCacheConfig`     | Sets the preferred cache configuration for the current device.                                                                 |
| `cudaDeviceSetLimit`                                      | `hipDeviceSetLimit`           | Set resource limits.                                                                                                           |
| `cudaDeviceSetSharedMemConfig`                            | `hipDeviceSetSharedMemConfig` | Sets the shared memory configuration for the current device.                                                                   |
| `cudaDeviceSynchronize`                                   | `hipDeviceSynchronize`        | Wait for compute device to finish.                                                                                             |
| `cudaGetDevice`                                           | `hipGetDevice`                | Returns which device is currently being used.                                                                                  |
| `cudaGetDeviceCount`                                      | `hipGetDeviceCount`           | Returns the number of compute-capable devices.                                                                                 |
| `cudaGetDeviceFlags`                                      |                               | Gets the flags for the current device.                                                                                         |
| `cudaGetDeviceProperties`                                 | `hipGetDeviceProperties`      | Returns information about the compute-device.                                                                                  |
| `cudaIpcCloseMemHandle`                                   | `hipIpcCloseMemHandle`        | Close memory mapped with cudaIpcOpenMemHandle.                                                                                 |
| `cudaIpcGetEventHandle`                                   | `hipIpcGetEventHandle`        | Gets an interprocess handle for a previously allocated event.                                                                  |
| `cudaIpcGetMemHandle`                                     | `hipIpcGetMemHandle`          | Gets an interprocess memory handle for an existing device memory allocation.                                                   |
| `cudaIpcOpenEventHandle`                                  | `hipIpcOpenEventHandle`       | Opens an interprocess event handle for use in the current process.                                                             |
| `cudaIpcOpenMemHandle`                                    | `hipIpcOpenMemHandle`         | Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.    |
| `cudaSetDevice`                                           | `hipSetDevice`                | Set device to be used for GPU executions.                                                                                      |
| `cudaSetDeviceFlags`                                      | `hipSetDeviceFlags`           | Sets flags to be used for device executions.                                                                                   |
| `cudaSetValidDevices`                                     |                               | Set a list of devices that can be used for CUDA.                                                                               |

## **2. Error Handling**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaGetErrorName`                                        | `hipGetErrorName`             | Returns the string representation of an error code enum name.                                                                  |
| `cudaGetErrorString`                                      | `hipGetErrorString`           | Returns the description string for an error code.                                                                              |
| `cudaGetLastError`                                        | `hipGetLastError`             | Returns the last error from a runtime call.                                                                                    |
| `cudaPeekAtLastError`                                     | `hipPeekAtLastError`          | Returns the last error from a runtime call.                                                                                    |

## **3. Stream Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaStreamAddCallback`                                   | `hipStreamAddCallback`        | Add a callback to a compute stream.                                                                                            |
| `cudaStreamAttachMemAsync`                                |                               | Attach managed memory to a stream asynchronously.                                                                              |
| `cudaStreamCreate`                                        | `hipStreamCreate`             | Create an asynchronous stream.                                                                                                 |
| `cudaStreamCreateWithFlags`                               | `hipStreamCreateWithFlags`    | Create an asynchronous stream.                                                                                                 |
| `cudaStreamCreateWithPriority`                            |                               | Create an asynchronous stream with the specified priority.                                                                     |
| `cudaStreamDestroy`                                       | `hipStreamDestroy`            | Destroys and cleans up an asynchronous stream.                                                                                 |
| `cudaStreamGetFlags`                                      | `hipStreamGetFlags`           | Query the flags of a stream.                                                                                                   |
| `cudaStreamGetPriority`                                   |                               | Query the priority of a stream.                                                                                                |
| `cudaStreamQuery`                                         | `hipStreamQuery`              | Queries an asynchronous stream for completion status.                                                                          |
| `cudaStreamSynchronize`                                   | `hipStreamSynchronize`        | Waits for stream tasks to complete.                                                                                            |
| `cudaStreamWaitEvent`                                     | `hipStreamWaitEvent`          | Make a compute stream wait on an event.                                                                                        |

## **4. Event Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaEventCreate`                                         | `hipEventCreate`              | Creates an event object.                                                                                                       |
| `cudaEventCreateWithFlags`                                | `hipEventCreateWithFlags`     | Creates an event object with the specified flags.                                                                              |
| `cudaEventDestroy`                                        | `hipEventDestroy`             | Destroys an event object.                                                                                                      |
| `cudaEventElapsedTime`                                    | `hipEventElapsedTime`         | Computes the elapsed time between events.                                                                                      |
| `cudaEventQuery`                                          | `hipEventQuery`               | Queries an event's status.                                                                                                     |
| `cudaEventRecord`                                         | `hipEventRecord`              | Records an event.                                                                                                              |
| `cudaEventSynchronize`                                    | `hipEventSynchronize`         | Waits for an event to complete.                                                                                                |

## **5. Execution Control**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaFuncGetAttributes`                                   |                               | Find out attributes for a given function.                                                                                      |
| `cudaFuncSetCacheConfig`                                  | `hipFuncSetCacheConfig`       | Sets the preferred cache configuration for a device function.                                                                  |
| `cudaFuncSetSharedMemConfig`                              |                               | Sets the shared memory configuration for a device function.                                                                    |
| `cudaGetParameterBuffer`                                  |                               | Obtains a parameter buffer.                                                                                                    |
| `cudaGetParameterBufferV2`                                |                               | Launches a specified kernel.                                                                                                   |
| `cudaLaunchKernel`                                        | `hipLaunchKernel`             | Launches a device function.                                                                                                    |
| `cudaSetDoubleForDevice`                                  |                               | Converts a double argument to be executed on a device.                                                                         |
| `cudaSetDoubleForHost`                                    |                               | Converts a double argument after execution on a device.                                                                        |

## **6. Occupancy**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor`           | `hipOccupancyMaxActiveBlocksPerMultiprocessor`| Returns occupancy for a device function.                                                                       |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`  |                               | Returns occupancy for a device function with the specified flags.                                                              |

## **7. Execution Control [deprecated since 7.0]**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaConfigureCall`                                       |                               | Configure a device-launch.                                                                                                     |
| `cudaLaunch`                                              |                               | Launches a device function.                                                                                                    |
| `cudaSetupArgument`                                       |                               | Configure a device launch.                                                                                                     |

## **8. Memory Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaArrayGetInfo`                                        |                               | Gets info about the specified cudaArray.                                                                                       |
| `cudaFree`                                                | `hipFree`                     | Frees memory on the device.                                                                                                    |
| `cudaFreeArray`                                           | `hipFreeArray`                | Frees an array on the device.                                                                                                  |
| `cudaFreeHost`                                            | `hipHostFree`                 | Frees page-locked memory.                                                                                                      |
| `cudaFreeMipmappedArray`                                  |                               | Frees a mipmapped array on the device.                                                                                         |
| `cudaGetMipmappedArrayLevel`                              |                               | Gets a mipmap level of a CUDA mipmapped array.                                                                                 |
| `cudaGetSymbolAddress`                                    |                               | Finds the address associated with a CUDA symbol.                                                                               |
| `cudaGetSymbolSize`                                       |                               | Finds the size of the object associated with a CUDA symbol.                                                                    |
| `cudaHostAlloc`                                           | `hipHostMalloc`               | Allocates page-locked memory on the host.                                                                                      |
| `cudaHostGetDevicePointer`                                | `hipHostGetDevicePointer`     | Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister.                 |
| `cudaHostGetFlags`                                        | `hipHostGetFlags`             | Passes back flags used to allocate pinned host memory allocated by cudaHostAlloc.                                              |
| `cudaHostRegister`                                        | `hipHostRegister`             | Registers an existing host memory range for use by CUDA.                                                                       |
| `cudaHostUnregister`                                      | `hipHostUnregister`           | Unregisters a memory range that was registered with cudaHostRegister.                                                          |
| `cudaMalloc`                                              | `hipMalloc`                   | Allocate memory on the device.                                                                                                 |
| `cudaMalloc3D`                                            |                               | Allocates logical 1D, 2D, or 3D memory objects on the device.                                                                  |
| `cudaMalloc3DArray`                                       |                               | Allocate an array on the device.                                                                                               |
| `cudaMallocArray`                                         | `hipMallocArray`              | Allocate an array on the device.                                                                                               |
| `cudaMallocHost`                                          | `hipHostMalloc`               | Allocates page-locked memory on the host.                                                                                      |
| `cudaMallocManaged`                                       |                               | Allocates memory that will be automatically managed by the Unified Memory system.                                              |
| `cudaMallocMipmappedArray`                                |                               | Allocate a mipmapped array on the device.                                                                                      |
| `cudaMallocPitch`                                         |                               | Allocates pitched memory on the device.                                                                                        |
| `cudaMemGetInfo`                                          | `hipMemGetInfo`               | Gets free and total device memory.                                                                                             |
| `cudaMemcpy`                                              | `hipMemcpy`                   | Copies data between host and device.                                                                                           |
| `cudaMemcpy2D`                                            | `hipMemcpy2D`                 | Copies data between host and device.                                                                                           |
| `cudaMemcpy2DArrayToArray`                                |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpy2DAsync`                                       |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpy2DFromArray`                                   |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpy2DFromArrayAsync`                              |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpy2DToArray`                                     | `hipMemcpy2DToArray`          | Copies data between host and device.                                                                                           |
| `cudaMemcpy2DToArrayAsync`                                |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpy3D`                                            |                               | Copies data between 3D objects.                                                                                                |
| `cudaMemcpy3DAsync`                                       |                               | Copies data between 3D objects.                                                                                                |
| `cudaMemcpy3DPeer`                                        |                               | Copies memory between devices.                                                                                                 |
| `cudaMemcpy3DPeerAsync`                                   |                               | Copies memory between devices asynchronously.                                                                                  |
| `cudaMemcpyArrayToArray`                                  |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpyAsync`                                         | `hipMemcpyAsync`              | Copies data between host and device.                                                                                           |
| `cudaMemcpyFromArray`                                     | `MemcpyFromArray`             | Copies data between host and device.                                                                                           |
| `cudaMemcpyFromArrayAsync`                                |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpyFromSymbol`                                    | `hipMemcpyFromSymbol`         | Copies data from the given symbol on the device.                                                                               |
| `cudaMemcpyFromSymbolAsync`                               |                               | Copies data from the given symbol on the device.                                                                               |
| `cudaMemcpyPeer`                                          | `hipMemcpyPeer`               | Copies memory between two devices.                                                                                             |
| `cudaMemcpyPeerAsync`                                     | `hipMemcpyPeerAsync`          | Copies memory between two devices asynchronously.                                                                              |
| `cudaMemcpyToArray`                                       | `hipMemcpyToArray`            | Copies data between host and device.                                                                                           |
| `cudaMemcpyToArrayAsync`                                  |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpyToSymbol`                                      | `hipMemcpyToSymbol`           | Copies data to the given symbol on the device.                                                                                 |
| `cudaMemcpyToSymbolAsync`                                 | `hipMemcpyToSymbolAsync`      | Copies data to the given symbol on the device.                                                                                 |
| `cudaMemset`                                              | `hipMemset`                   | Initializes or sets device memory to a value.                                                                                  |
| `cudaMemset2D`                                            |                               | Initializes or sets device memory to a value.                                                                                  |
| `cudaMemset2DAsync`                                       |                               | Initializes or sets device memory to a value.                                                                                  |
| `cudaMemset3D`                                            |                               | Initializes or sets device memory to a value.                                                                                  |
| `cudaMemset3DAsync`                                       |                               | Initializes or sets device memory to a value.                                                                                  |
| `cudaMemsetAsync`                                         | `hipMemsetAsync`              | Initializes or sets device memory to a value.                                                                                  |
| `make_cudaExtent`                                         |                               | Returns a cudaExtent based on input parameters.                                                                                |
| `make_cudaPitchedPtr`                                     |                               | Returns a cudaPitchedPtr based on input parameters.                                                                            |
| `make_cudaPos`                                            |                               | Returns a cudaPos based on input parameters.                                                                                   |

## **9. Unified Addressing**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaPointerGetAttributes`                                | `hipPointerGetAttributes`     | Returns attributes about a specified pointer.                                                                                  |

## **10. Peer Device Memory Access**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaDeviceCanAccessPeer`                                 | `hipDeviceCanAccessPeer`      | Queries if a device may directly access a peer device's memory.                                                                |
| `cudaDeviceDisablePeerAccess`                             | `hipDeviceDisablePeerAccess`  | Disables direct access to memory allocations on a peer device.                                                                 |
| `cudaDeviceEnablePeerAccess`                              | `hipDeviceEnablePeerAccess`   | Enables direct access to memory allocations on a peer device.                                                                  |

## **11. OpenGL Interoperability**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaGLGetDevices`                                        |                               | Gets the CUDA devices associated with the current OpenGL context.                                                              |
| `cudaGraphicsGLRegisterBuffer`                            |                               | Registers an OpenGL buffer object.                                                                                             |
| `cudaGraphicsGLRegisterImage`                             |                               | Register an OpenGL texture or renderbuffer object.                                                                             |
| `cudaWGLGetDevice`                                        |                               | Gets the CUDA device associated with hGpu.                                                                                     |

## **12. Graphics Interoperability**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaGraphicsMapResources`                                |                               | Map graphics resources for access by CUDA.                                                                                     |
| `cudaGraphicsResourceGetMappedMipmappedArray`             |                               | Get a mipmapped array through which to access a mapped graphics resource.                                                      |
| `cudaGraphicsResourceGetMappedPointer`                    |                               | Get a device pointer through which to access a mapped graphics resource.                                                       |
| `cudaGraphicsResourceSetMapFlags`                         |                               | Set usage flags for mapping a graphics resource.                                                                               |
| `cudaGraphicsSubResourceGetMappedArray`                   |                               | Get an array through which to access a subresource of a mapped graphics resource.                                              |
| `cudaGraphicsUnmapResources`                              |                               | Unmap graphics resources.                                                                                                      |
| `cudaGraphicsUnregisterResource`                          |                               | Unregisters a graphics resource for access by CUDA.                                                                            |

## **13. Texture Reference Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaBindTexture`                                         |                               | Binds a memory area to a texture.                                                                                              |
| `cudaBindTexture2D`                                       |                               | Binds a 2D memory area to a texture.                                                                                           |
| `cudaBindTextureToArray`                                  |                               | Binds an array to a texture.                                                                                                   |
| `cudaBindTextureToMipmappedArray`                         |                               | Binds a mipmapped array to a texture.                                                                                          |
| `cudaCreateChannelDesc`                                   |                               | Returns a channel descriptor using the specified format.                                                                       |
| `cudaGetChannelDesc`                                      |                               | Get the channel descriptor of an array.                                                                                        |
| `cudaGetTextureAlignmentOffset`                           |                               | Get the alignment offset of a texture.                                                                                         |
| `cudaGetTextureReference`                                 |                               | Get the texture reference associated with a symbol.                                                                            |
| `cudaUnbindTexture`                                       |                               | Unbinds a texture.                                                                                                             |

## **14. Surface Reference Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaBindSurfaceToArray`                                  |                               | Binds an array to a surface.                                                                                                   |
| `cudaGetSurfaceReference`                                 |                               | Get the surface reference associated with a symbol.                                                                            |

## **15. Texture Object Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaCreateTextureObject`                                 |                               | Creates a texture object.                                                                                                      |
| `cudaDestroyTextureObject`                                |                               | Destroys a texture object.                                                                                                     |
| `cudaGetTextureObjectResourceDesc`                        |                               | Returns a texture object's resource descriptor.                                                                                |
| `cudaGetTextureObjectResourceViewDesc`                    |                               | Returns a texture object's resource view descriptor.                                                                           |
| `cudaGetTextureObjectTextureDesc`                         |                               | Returns a texture object's texture descriptor.                                                                                 |

## **16. Surface Object Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaCreateSurfaceObject`                                 |                               | Creates a surface object.                                                                                                      |
| `cudaDestroySurfaceObject`                                |                               | Destroys a surface object.                                                                                                     |
| `cudaGetSurfaceObjectResourceDesc`                        |                               | Returns a surface object's resource descriptor Returns the resource descriptor for the surface object specified by surfObject. |

## **17. Version Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaDriverGetVersion`                                    | `hipDriverGetVersion`         | Returns the CUDA driver version.                                                                                               |
| `cudaRuntimeGetVersion`                                   | `hipRuntimeGetVersion`        | Returns the CUDA Runtime version.                                                                                              |

## **18. C++ API Routines**
*(7.0 contains, 7.5 doesn’t)*

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaBindSurfaceToArray`                                  |                               | Binds an array to a surface.                                                                                                   |
| `cudaBindTexture`                                         | `hipBindTexture`              | Binds a memory area to a texture.                                                                                              |
| `cudaBindTexture2D`                                       |                               | Binds a 2D memory area to a texture.                                                                                           |
| `cudaBindTextureToArray`                                  |                               | Binds an array to a texture.                                                                                                   |
| `cudaBindTextureToMipmappedArray`                         |                               | Binds a mipmapped array to a texture.                                                                                          |
| `cudaCreateChannelDesc`                                   | `hipCreateChannelDesc`        | Returns a channel descriptor using the specified format.                                                                       |
| `cudaFuncGetAttributes`                                   |                               | Find out attributes for a given function.                                                                                      |
| `cudaFuncSetCacheConfig`                                  |                               | Sets the preferred cache configuration for a device function.                                                                  |
| `cudaGetSymbolAddress`                                    |                               | Finds the address associated with a CUDA symbol                                                                                |
| `cudaGetSymbolSize`                                       |                               | Finds the size of the object associated with a CUDA symbol.                                                                    |
| `cudaGetTextureAlignmentOffset`                           |                               | Get the alignment offset of a texture.                                                                                         |
| `cudaLaunch`                                              |                               | Launches a device function.                                                                                                    |
| `cudaLaunchKernel`                                        |                               | Launches a device function.                                                                                                    |
| `cudaMallocHost`                                          |                               | Allocates page-locked memory on the host                                                                                       |
| `cudaMallocManaged`                                       |                               | Allocates memory that will be automatically managed by the Unified Memory system.                                              |
| `cudaMemcpyFromSymbol`                                    |                               | Copies data from the given symbol on the device.                                                                               |
| `cudaMemcpyFromSymbolAsync`                               |                               | Copies data from the given symbol on the device.                                                                               |
| `cudaMemcpyToSymbol`                                      |                               | Copies data to the given symbol on the device.                                                                                 |
| `cudaMemcpyToSymbolAsync`                                 |                               | Async copies data to the given symbol on the device.                                                                           |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor`           | `hipOccupancyMaxActiveBlocksPerMultiprocessor` | Returns occupancy for a device function.                                                                      |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`  |                               | Returns occupancy for a device function with the specified flags.                                                              |
| `cudaOccupancyMaxPotentialBlockSize`                      | `hipOccupancyMaxPotentialBlockSize` | Returns grid and block size that achieves maximum potential occupancy for a device function.                             |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMem`          |                               | Returns grid and block size that achieves maximum potential occupancy for a device function.                                   |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags` |                               | Returns grid and block size that achieves maximum potential occupancy for a device function.                                   |
| `cudaOccupancyMaxPotentialBlockSizeWithFlags`             |                               | Returns grid and block size that achived maximum potential occupancy for a device function with the specified flags.           |
| `cudaSetupArgument`                                       |                               | Configure a device launch.                                                                                                     |
| `cudaStreamAttachMemAsync`                                |                               | Attach memory to a stream asynchronously.                                                                                      |
| `cudaUnbindTexture`                                       | `hipUnbindTexture`            | Unbinds a texture.                                                                                                             |

## **19. Profiler Control**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaProfilerInitialize`                                  |                               | Initialize the CUDA profiler.                                                                                                  |
| `cudaProfilerStart`                                       | `hipProfilerStart`            | Enable profiling.                                                                                                              |
| `cudaProfilerStop`                                        | `hipProfilerStop`             | Disable profiling.                                                                                                             |

# Data types used by CUDA Runtime API and supported by HIP

## **20. Data types**

| **type**     |   **CUDA**                                 |   **HIP**                     | **CUDA description**                                                                                                           |
|--------------|--------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| struct       | `cudaChannelFormatDesc`                    | `hipChannelFormatDesc`        | CUDA Channel format descriptor.                                                                                                |
| struct       | `cudaDeviceProp`                           | `hipDeviceProp_t`             | CUDA device properties.                                                                                                        |
| struct       | `cudaExtent`                               |                               | CUDA extent (width, height, depth).                                                                                            |
| struct       | `cudaFuncAttributes`                       |                               | CUDA function attributes.                                                                                                      |
| struct       | `cudaIpcEventHandle_t`                     | `hipIpcEventHandle_t`         | CUDA IPC event handle.                                                                                                         |
| struct       | `cudaIpcMemHandle_t`                       | `hipIpcMemHandle_t`           | CUDA IPC memory handle.                                                                                                        |
| struct       | `cudaMemcpy3DParms`                        |                               | CUDA 3D memory copying parameters.                                                                                             |
| struct       | `cudaMemcpy3DPeerParms`                    |                               | CUDA 3D cross-device memory copying parameters.                                                                                |
| struct       | `cudaPitchedPtr`                           |                               | CUDA Pitched memory pointer.                                                                                                   |
| struct       | `cudaPointerAttributes`                    | `hipPointerAttribute_t`       | CUDA pointer attributes.                                                                                                       |
| struct       | `cudaPos`                                  |                               | CUDA 3D position.                                                                                                              |
| struct       | `cudaResourceDesc`                         |                               | CUDA resource descriptor.                                                                                                      |
| struct       | `cudaResourceViewDesc`                     |                               | CUDA resource view descriptor.                                                                                                 |
| struct       | `cudaTextureDesc`                          |                               | CUDA texture descriptor.                                                                                                       |
| struct       | `surfaceReference`                         |                               | CUDA Surface reference.                                                                                                        |
| struct       | `textureReference`                         | `textureReference`            | CUDA texture reference.                                                                                                        |
| enum         | `cudaChannelFormatKind`                    | `hipChannelFormatKind`        | Channel format kind.                                                                                                           |
| enum         | `cudaComputeMode`                          |                               | CUDA device compute modes.                                                                                                     |
| enum         | `cudaDeviceAttr`                           | `hipDeviceAttribute_t`        | CUDA device attributes.                                                                                                        |
| enum         | `cudaError`                                | `hipError_t`                  | CUDA Error types.                                                                                                              |
| enum         | `cudaError_t`                              | `hipError_t`                  | CUDA Error types.                                                                                                              |
| enum         | `cudaFuncCache`                            | `hipFuncCache_t`              | CUDA function cache configurations.                                                                                            |
| enum         | `cudaGraphicsCubeFace`                     |                               | CUDA graphics interop array indices for cube maps.                                                                             |
| enum         | `cudaGraphicsMapFlags`                     |                               | CUDA graphics interop map flags.                                                                                               |
| enum         | `cudaGraphicsRegisterFlags`                |                               | CUDA graphics interop register flags.                                                                                          |
| enum         | `cudaMemcpyKind`                           | `hipMemcpyKind`               | CUDA memory copy types.                                                                                                        |
| enum         | `cudaMemoryType`                           | `hipMemoryType`               | CUDA memory types.                                                                                                             |
| enum         | `cudaOutputMode`                           |                               | CUDA Profiler Output modes.                                                                                                    |
| enum         | `cudaResourceType`                         |                               | CUDA resource types.                                                                                                           |
| enum         | `cudaResourceViewFormat`                   |                               | CUDA texture resource view formats.                                                                                            |
| enum         | `cudaSharedMemConfig`                      | `hipSharedMemConfig`          | CUDA shared memory configuration.                                                                                              |
| enum         | `cudaSurfaceBoundaryMode`                  |                               | CUDA Surface boundary modes.                                                                                                   |
| enum         | `cudaSurfaceFormatMode`                    |                               | CUDA Surface format modes.                                                                                                     |
| enum         | `cudaTextureAddressMode`                   |                               | CUDA texture address modes.                                                                                                    |
| enum         | `cudaTextureFilterMode`                    | `hipTextureFilterMode`        | CUDA texture filter modes.                                                                                                     |
| enum         | `cudaTextureReadMode`                      | `hipTextureReadMode`          | CUDA texture read modes.                                                                                                       |
| struct       | `cudaArray`                                | `hipArray`                    | CUDA array [opaque].                                                                                                           |
| typedef      | `cudaArray_t`                              | `hipArray *`                  | CUDA array pointer.                                                                                                            |
| typedef      | `cudaArray_const_t`                        | `const hipArray *`            | CUDA array (as source copy argument).                                                                                          |
| enum         | `cudaError`                                | `hipError_t`                  | CUDA Error types.                                                                                                              |
| typedef      | `cudaError_t`                              | `hipError_t`                  | CUDA Error types.                                                                                                              |
| typedef      | `cudaEvent_t`                              | `hipEvent_t`                  | CUDA event types.                                                                                                              |
| typedef      | `cudaGraphicsResource_t`                   |                               | CUDA graphics resource types.                                                                                                  |
| typedef      | `cudaMipmappedArray_t`                     |                               | CUDA mipmapped array.                                                                                                          |
| typedef      | `cudaMipmappedArray_const_t`               |                               | CUDA mipmapped array (as source argument).                                                                                     |
| enum         | `cudaOutputMode`                           |                               | CUDA output file modes.                                                                                                        |
| typedef      | `cudaOutputMode_t`                         |                               | CUDA output file modes.                                                                                                        |
| typedef      | `cudaStream_t`                             | `hipStream_t`                 | CUDA stream.                                                                                                                   |
| typedef      | `cudaSurfaceObject_t`                      |                               | An opaque value that represents a CUDA Surface object.                                                                         |
| typedef      | `cudaTextureObject_t`                      |                               | An opaque value that represents a CUDA texture object.                                                                         |
| typedef      | `CUuuid_stcudaUUID_t`                      |                               | CUDA UUID types.                                                                                                               |
| define       | `CUDA_IPC_HANDLE_SIZE`                     |                               | CUDA IPC Handle Size.                                                                                                          |
| define       | `cudaArrayCubemap`                         |                               | Must be set in cudaMalloc3DArray to create a cubemap CUDA array.                                                               |
| define       | `cudaArrayDefault`                         |                               | Default CUDA array allocation flag.                                                                                            |
| define       | `cudaArrayLayered`                         |                               | Must be set in cudaMalloc3DArray to create a layered CUDA array.                                                               |
| define       | `cudaArraySurfaceLoadStore`                |                               | Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array.                               |
| define       | `cudaArrayTextureGather`                   |                               | Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array.           |
| define       | `cudaDeviceBlockingSync`                   | `hipDeviceScheduleBlockingSync` | Device flag - Use blocking synchronization. Deprecated as of CUDA 4.0 and replaced with cudaDeviceScheduleBlockingSync.      |
| define       | `cudaDeviceLmemResizeToMax`                |                               | Device flag - Keep local memory allocation after launch.                                                                       |
| define       | `cudaDeviceMapHost`                        |                               | Device flag - Support mapped pinned allocations.                                                                               |
| define       | `cudaDeviceMask`                           |                               | Device flags mask.                                                                                                             |
| define       | `cudaDevicePropDontCare`                   |                               | Empty device properties.                                                                                                       |
| define       | `cudaDeviceScheduleAuto`                   | `hipDeviceScheduleAuto`       | Device flag - Automatic scheduling.                                                                                            |
| define       | `cudaDeviceScheduleBlockingSync`           | `hipDeviceScheduleBlockingSync` | Device flag - Use blocking synchronization.                                                                                  |
| define       | `cudaDeviceScheduleMask`                   | `hipDeviceScheduleMask`       | Device schedule flags mask.                                                                                                    |
| define       | `cudaDeviceScheduleSpin`                   | `hipDeviceScheduleSpin`       | Device flag - Spin default scheduling.                                                                                         |
| define       | `cudaDeviceScheduleYield`                  | `hipDeviceScheduleYield`      | Device flag - Yield default scheduling.                                                                                        |
| define       | `cudaEventBlockingSync`                    | `hipEventBlockingSync`        | Event uses blocking synchronization.                                                                                           |
| define       | `cudaEventDefault`                         | `hipEventDefault`             | Default event flag.                                                                                                            |
| define       | `cudaEventDisableTiming`                   | `hipEventDisableTiming`       | Event will not record timing data.                                                                                             |
| define       | `cudaEventInterprocess`                    | `hipEventInterprocess`        | Event is suitable for interprocess use. cudaEventDisableTiming must be set.                                                    |
| define       | `cudaHostAllocDefault`                     | `hipHostMallocDefault`        | Default page-locked allocation flag.                                                                                           |
| define       | `cudaHostAllocMapped`                      | `hipHostMallocMapped`         | Map allocation into device space.                                                                                              |
| define       | `cudaHostAllocPortable`                    | `hipHostMallocPortable`       | Pinned memory accessible by all CUDA contexts.                                                                                 |
| define       | `cudaHostAllocWriteCombined`               | `hipHostMallocWriteCombined`  | Write-combined memory.                                                                                                         |
| define       | `cudaHostRegisterDefault`                  | `hipHostRegisterDefault`      | Default host memory registration flag.                                                                                         |
| define       | `cudaHostRegisterIoMemory`                 | `hipHostRegisterIoMemory`     | Memory-mapped I/O space.                                                                                                       |
| define       | `cudaHostRegisterMapped`                   | `hipHostRegisterMapped`       | Map registered memory into device space.                                                                                       |
| define       | `cudaHostRegisterPortable`                 | `hipHostRegisterPortable`     | Pinned memory accessible by all CUDA contexts.                                                                                 |
| define       | `cudaIpcMemLazyEnablePeerAccess`           | `hipIpcMemLazyEnablePeerAccess` | Automatically enable peer access between remote devices as needed.                                                           |
| define       | `cudaMemAttachGlobal`                      |                               | Memory can be accessed by any stream on any device.                                                                            |
| define       | `cudaMemAttachHost`                        |                               | Memory cannot be accessed by any stream on any device.                                                                         |
| define       | `cudaMemAttachSingle`                      |                               | Memory can only be accessed by a single stream on the associated device.                                                       |
| define       | `cudaOccupancyDefault`                     |                               | Default behavior.                                                                                                              |
| define       | `cudaOccupancyDisableCachingOverride`      |                               | Assume global caching is enabled and cannot be automatically turned off.                                                       |
| define       | `cudaPeerAccessDefault`                    |                               | Default peer addressing enable flag.                                                                                           |
| define       | `cudaStreamDefault`                        | `hipStreamDefault`            | Default stream flag.                                                                                                           |
| define       | `cudaStreamLegacy`                         |                               | Default stream flag.                                                                                                           |
| define       | `cudaStreamNonBlocking`                    | `hipStreamNonBlocking`        | Stream does not synchronize with stream 0 (the NULL stream).                                                                   |
| define       | `cudaStreamPerThread`                      |                               | Per-thread stream handle.                                                                                                      |
