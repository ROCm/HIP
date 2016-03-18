**1. Device Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaChooseDevice`                                        |                               | Select compute-device which best matches criteria.                                                                             |
| `cudaDeviceGetAttribute`                                  |                               | Returns information about the device.                                                                                          |
| `cudaDeviceGetByPCIBusId`                                 |                               | Returns a handle to a compute device.                                                                                          |
| `cudaDeviceGetCacheConfig`                                | `hipDeviceGetCacheConfig`     | Returns the preferred cache configuration for the current device.                                                              |
| `cudaDeviceGetLimit`                                      |                               | Returns resource limits.                                                                                                       |
| `cudaDeviceGetPCIBusId`                                   |                               | Returns a PCI Bus Id string for the device.                                                                                    |
| `cudaDeviceGetSharedMemConfig`                            | `hipDeviceGetSharedMemConfig` | Returns the shared memory configuration for the current device.                                                                |
| `cudaDeviceGetStreamPriorityRange`                        |                               | Returns numerical values that correspond to the least and greatest stream priorities.                                          |
| `cudaDeviceReset`                                         | `hipDeviceReset`              | Destroy all allocations and reset all state on the current device in the current process.                                      |
| `cudaDeviceSetCacheConfig`                                | `hipDeviceSetCacheConfig`     | Sets the preferred cache configuration for the current device.                                                                 |
| `cudaDeviceSetLimit`                                      |                               | Set resource limits.                                                                                                           |
| `cudaDeviceSetSharedMemConfig`                            | `hipDeviceSetSharedMemConfig` | Sets the shared memory configuration for the current device.                                                                   |
| `cudaDeviceSynchronize`                                   | `hipDeviceSynchronize`        | Wait for compute device to finish.                                                                                             |
| `cudaGetDevice`                                           | `hipGetDevice`                | Returns which device is currently being used.                                                                                  |
| `cudaGetDeviceCount`                                      | `hipGetDeviceCount`           | Returns the number of compute-capable devices.                                                                                 |
| `cudaGetDeviceFlags`                                      |                               | Gets the flags for the current device.                                                                                         |
| `cudaGetDeviceProperties`                                 | `hipDeviceGetProperties`      | Returns information about the compute-device.                                                                                  |
| `cudaIpcCloseMemHandle`                                   |                               | Close memory mapped with cudaIpcOpenMemHandle.                                                                                 |
| `cudaIpcGetEventHandle`                                   |                               | Gets an interprocess handle for a previously allocated event.                                                                  |
| `cudaIpcGetMemHandle`                                     |                               | Gets an interprocess memory handle for an existing device memory allocation.                                                   |
| `cudaIpcOpenEventHandle`                                  |                               | Opens an interprocess event handle for use in the current process.                                                             |
| `cudaIpcOpenMemHandle`                                    |                               | Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.    |
| `cudaSetDevice`                                           | `hipSetDevice`                | Set device to be used for GPU executions.                                                                                      |
| `cudaSetDeviceFlags`                                      |                               | Sets flags to be used for device executions.                                                                                   |
| `cudaSetValidDevices`                                     |                               | Set a list of devices that can be used for CUDA.                                                                               |`

**2. Error Handling**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaGetErrorName`                                        | `hipGetErrorName`             | Returns the string representation of an error code enum name.                                                                  |
| `cudaGetErrorString`                                      | `hipGetErrorString`           | Returns the description string for an error code.                                                                              |
| `cudaGetLastError`                                        | `hipGetLastError`             | Returns the last error from a runtime call.                                                                                    |
| `cudaPeekAtLastError`                                     | `hipPeekAtLastError`          | Returns the last error from a runtime call.                                                                                    |

**3. Stream Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaStreamAddCallback`                                   |                               | Add a callback to a compute stream.                                                                                            |
| `cudaStreamAttachMemAsync`                                |                               | Attach memory to a stream asynchronously.                                                                                      |
| `cudaStreamCreate`                                        | `hipStreamCreate`             | Create an asynchronous stream.                                                                                                 |
| `cudaStreamCreateWithFlags`                               | `hipStreamCreateWithFlags`    | Create an asynchronous stream.                                                                                                 |
| `cudaStreamCreateWithPriority`                            |                               | Create an asynchronous stream with the specified priority.                                                                     |
| `cudaStreamDestroy`                                       | `hipStreamDestroy`            | Destroys and cleans up an asynchronous stream.                                                                                 |
| `cudaStreamGetFlags`                                      |                               | Query the flags of a stream.                                                                                                   |
| `cudaStreamGetPriority`                                   |                               | Query the priority of a stream.                                                                                                |
| `cudaStreamQuery`                                         |                               | Queries an asynchronous stream for completion status.                                                                          |
| `cudaStreamSynchronize`                                   | `hipStreamSynchronize`        | Waits for stream tasks to complete.                                                                                            |
| `cudaStreamWaitEvent`                                     | `hipStreamWaitEvent`          | Make a compute stream wait on an event.                                                                                        |

**4. Event Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaEventCreate`                                         | `hipEventCreate`              | Creates an event object.                                                                                                       |
| `cudaEventCreateWithFlags`                                | `hipEventCreateWithFlags`     | Creates an event object with the specified flags.                                                                              |
| `cudaEventDestroy`                                        | `hipEventDestroy`             | Destroys an event object.                                                                                                      |
| `cudaEventElapsedTime`                                    | `hipEventElapsedTime`         | Computes the elapsed time between events.                                                                                      |
| `cudaEventQuery`                                          | `hipEventQuery`               | Queries an event's status.                                                                                                     |
| `cudaEventRecord`                                         | `hipEventRecord`              | Records an event.                                                                                                              |
| `cudaEventSynchronize`                                    | `hipEventSynchronize`         | Waits for an event to complete.                                                                                                |

**5. Execution Control**

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

**6. Occupancy**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor`           |                               | Returns occupancy for a device function.                                                                                       |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`  |                               | Returns occupancy for a device function with the specified flags.                                                              |

**7. Memory Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaArrayGetInfo`                                        |                               | Gets info about the specified cudaArray.                                                                                       |
| `cudaFree`                                                | `hipFree`                     | Frees memory on the device.                                                                                                    |
| `cudaFreeArray`                                           |                               | Frees an array on the device.                                                                                                  |
| `cudaFreeHost`                                            | `hipHostFree`                 | Frees page-locked memory.                                                                                                      |
| `cudaFreeMipmappedArray`                                  |                               | Frees a mipmapped array on the device.                                                                                         |
| `cudaGetMipmappedArrayLevel`                              |                               | Gets a mipmap level of a CUDA mipmapped array.                                                                                 |
| `cudaGetSymbolAddress`                                    |                               | Finds the address associated with a CUDA symbol.                                                                               |
| `cudaGetSymbolSize`                                       |                               | Finds the size of the object associated with a CUDA symbol.                                                                    |
| `cudaHostAlloc`                                           |                               | Allocates page-locked memory on the host.                                                                                      |
| `cudaHostGetDevicePointer`                                |                               | Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister.                 |
| `cudaHostGetFlags`                                        |                               | Passes back flags used to allocate pinned host memory allocated by cudaHostAlloc.                                              |
| `cudaHostRegister`                                        |                               | Registers an existing host memory range for use by CUDA.                                                                       |
| `cudaHostUnregister`                                      |                               | Unregisters a memory range that was registered with cudaHostRegister.                                                          |
| `cudaMalloc`                                              | `hipMalloc`                   | Allocate memory on the device.                                                                                                 |
| `cudaMalloc3D`                                            |                               | Allocates logical 1D, 2D, or 3D memory objects on the device.                                                                  |
| `cudaMalloc3DArray`                                       |                               | Allocate an array on the device.                                                                                               |
| `cudaMallocArray`                                         |                               | Allocate an array on the device.                                                                                               |
| `cudaMallocHost`                                          | `hipHostAlloc`                | Allocates page-locked memory on the host.                                                                                      |
| `cudaMallocManaged`                                       |                               | Allocates memory that will be automatically managed by the Unified Memory system.                                              |
| `cudaMallocMipmappedArray`                                |                               | Allocate a mipmapped array on the device.                                                                                      |
| `cudaMallocPitch`                                         |                               | Allocates pitched memory on the device.                                                                                        |
| `cudaMemGetInfo`                                          |                               | Gets free and total device memory.                                                                                             |
| `cudaMemcpy`                                              | `hipMemcpy`                   | Copies data between host and device.                                                                                           |
| `cudaMemcpy2D`                                            |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpy2DArrayToArray`                                |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpy2DAsync`                                       |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpy2DFromArray`                                   |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpy2DFromArrayAsync`                              |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpy2DToArray`                                     |                               | Copies data between host and device.                                                                                           |
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
| `cudaMemcpyToArray`                                       |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpyToArrayAsync`                                  |                               | Copies data between host and device.                                                                                           |
| `cudaMemcpyToSymbol`                                      | `hipMemcpyToSymbol`           | Copies data to the given symbol on the device.                                                                                 |
| `cudaMemcpyToSymbolAsync`                                 |                               | Copies data to the given symbol on the device.                                                                                 |
| `cudaMemset`                                              | `hipMemset`                   | Initializes or sets device memory to a value.                                                                                  |
| `cudaMemset2D`                                            |                               | Initializes or sets device memory to a value.                                                                                  |
| `cudaMemset2DAsync`                                       |                               | Initializes or sets device memory to a value.                                                                                  |
| `cudaMemset3D`                                            |                               | Initializes or sets device memory to a value.                                                                                  |
| `cudaMemset3DAsync`                                       |                               | Initializes or sets device memory to a value.                                                                                  |
| `cudaMemsetAsync`                                         | `hipMemsetAsync`              | Initializes or sets device memory to a value.                                                                                  |
| `make\_cudaExtent`                                        |                               | Returns a cudaExtent based on input parameters.                                                                                |
| `make\_cudaPitchedPtr`                                    |                               | Returns a cudaPitchedPtr based on input parameters.                                                                            |
| `make\_cudaPos`                                           |                               | Returns a cudaPos based on input parameters.                                                                                   |

**8. Unified Addressing**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaPointerGetAttributes`                                |                               | Returns attributes about a specified pointer.                                                                                  |

**9. Peer Device Memory Access**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaDeviceCanAccessPeer`                                 | `hipDeviceCanAccessPeer`      | Queries if a device may directly access a peer device's memory.                                                                |
| `cudaDeviceDisablePeerAccess`                             | `hipDeviceDisablePeerAccess`  | Disables direct access to memory allocations on a peer device.                                                                 |
| `cudaDeviceEnablePeerAccess`                              | `hipDeviceEnablePeerAccess`   | Enables direct access to memory allocations on a peer device.                                                                  |

**10. OpenGL Interoperability**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaGLGetDevices`                                        |                               | Gets the CUDA devices associated with the current OpenGL context.                                                              |
| `cudaGraphicsGLRegisterBuffer`                            |                               | Registers an OpenGL buffer object.                                                                                             |
| `cudaGraphicsGLRegisterImage`                             |                               | Register an OpenGL texture or renderbuffer object.                                                                             |
| `cudaWGLGetDevice`                                        |                               | Gets the CUDA device associated with hGpu.                                                                                     |

**11. Graphics Interoperability**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaGraphicsMapResources`                                |                               | Map graphics resources for access by CUDA.                                                                                     |
| `cudaGraphicsResourceGetMappedMipmappedArray`             |                               | Get a mipmapped array through which to access a mapped graphics resource.                                                      |
| `cudaGraphicsResourceGetMappedPointer`                    |                               | Get a device pointer through which to access a mapped graphics resource.                                                       |
| `cudaGraphicsResourceSetMapFlags`                         |                               | Set usage flags for mapping a graphics resource.                                                                               |
| `cudaGraphicsSubResourceGetMappedArray`                   |                               | Get an array through which to access a subresource of a mapped graphics resource.                                              |
| `cudaGraphicsUnmapResources`                              |                               | Unmap graphics resources.                                                                                                      |
| `cudaGraphicsUnregisterResource`                          |                               | Unregisters a graphics resource for access by CUDA.                                                                            |

**12. Texture Reference Management**

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

**13. Surface Reference Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaBindSurfaceToArray`                                  |                               | Binds an array to a surface.                                                                                                   |
| `cudaGetSurfaceReference`                                 |                               | Get the surface reference associated with a symbol.                                                                            |

**14. Texture Object Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaCreateTextureObject`                                 |                               | Creates a texture object.                                                                                                      |
| `cudaDestroyTextureObject`                                |                               | Destroys a texture object.                                                                                                     |
| `cudaGetTextureObjectResourceDesc`                        |                               | Returns a texture object's resource descriptor.                                                                                |
| `cudaGetTextureObjectResourceViewDesc`                    |                               | Returns a texture object's resource view descriptor.                                                                           |
| `cudaGetTextureObjectTextureDesc`                         |                               | Returns a texture object's texture descriptor.                                                                                 |

**15. Surface Object Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaCreateSurfaceObject`                                 |                               | Creates a surface object.                                                                                                      |
| `cudaDestroySurfaceObject`                                |                               | Destroys a surface object.                                                                                                     |
| `cudaGetSurfaceObjectResourceDesc`                        |                               | Returns a surface object's resource descriptor Returns the resource descriptor for the surface object specified by surfObject. |

**16. Version Management**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaDriverGetVersion`                                    | `hipDriverGetVersion`         | Returns the CUDA driver version.                                                                                               |
| `cudaRuntimeGetVersion`                                   |                               | Returns the CUDA Runtime version.                                                                                              |

**17. C++ API Routines (7.0 contains, 7.5 doesn’t)**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaBindSurfaceToArra`y                                  |                               | Binds an array to a surface.                                                                                                   |
| `cudaBindTexture`                                         |                               | Binds a memory area to a texture.                                                                                              |
| `cudaBindTexture2D`                                       |                               | Binds a 2D memory area to a texture.                                                                                           |
| `cudaBindTextureToArray`                                  |                               | Binds an array to a texture.                                                                                                   |
| `cudaBindTextureToMipmappedArray`                         |                               | Binds a mipmapped array to a texture.                                                                                          |
| `cudaCreateChannelDesc`                                   |                               | Returns a channel descriptor using the specified format.                                                                       |
| `cudaEventCreate`                                         |                               | Creates an event object with the specified flags.                                                                              |
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
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor`           |                               | Returns occupancy for a device function.                                                                                       |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`  |                               | Returns occupancy for a device function with the specified flags.                                                              |
| `cudaOccupancyMaxPotentialBlockSize`                      |                               | Returns grid and block size that achieves maximum potential occupancy for a device function.                                   |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMem`          |                               | Returns grid and block size that achieves maximum potential occupancy for a device function.                                   |
| `cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags` |                               | Returns grid and block size that achieves maximum potential occupancy for a device function.                                   |
| `cudaOccupancyMaxPotentialBlockSizeWithFlags`             |                               | Returns grid and block size that achived maximum potential occupancy for a device function with the specified flags.           |
| `cudaSetupArgument`                                       |                               | Configure a device launch.                                                                                                     |
| `cudaStreamAttachMemAsync`                                |                               | Attach memory to a stream asynchronously.                                                                                      |
| `cudaUnbindTexture`                                       |                               | Unbinds a texture.                                                                                                             |

**18. Profiler Control**

|   **CUDA**                                                |   **HIP**                     | **CUDA description**                                                                                                           |
|-----------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cudaProfilerInitialize`                                  |                               | Initialize the CUDA profiler.                                                                                                  |
| `cudaProfilerStart`                                       | `hipProfilerStart`            | Enable profiling.                                                                                                              |
| `cudaProfilerStop`                                        | `hipProfilerStop`             | Disable profiling.                                                                                                             |
