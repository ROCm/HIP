# CUDA Runtime API supported by HIP

## **1. Device Management**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaChooseDevice`|  |  |  |`hipChooseDevice`|
|`cudaDeviceGetAttribute`|  |  |  |`hipDeviceGetAttribute`|
|`cudaDeviceGetByPCIBusId`|  |  |  |`hipDeviceGetByPCIBusId`|
|`cudaDeviceGetCacheConfig`|  |  |  |`hipDeviceGetCacheConfig`|
|`cudaDeviceGetLimit`|  |  |  |`hipDeviceGetLimit`|
|`cudaDeviceGetNvSciSyncAttributes`| 10.2 |  |  ||
|`cudaDeviceGetP2PAttribute`| 8.0 |  |  |`hipDeviceGetP2PAttribute`|
|`cudaDeviceGetPCIBusId`|  |  |  |`hipDeviceGetPCIBusId`|
|`cudaDeviceGetSharedMemConfig`|  |  |  |`hipDeviceGetSharedMemConfig`|
|`cudaDeviceGetStreamPriorityRange`|  |  |  |`hipDeviceGetStreamPriorityRange`|
|`cudaDeviceGetTexture1DLinearMaxWidth`| 11.1 |  |  ||
|`cudaDeviceReset`|  |  |  |`hipDeviceReset`|
|`cudaDeviceSetCacheConfig`|  |  |  |`hipDeviceSetCacheConfig`|
|`cudaDeviceSetLimit`|  |  |  |`hipDeviceSetLimit`|
|`cudaDeviceSetSharedMemConfig`|  |  |  |`hipDeviceSetSharedMemConfig`|
|`cudaDeviceSynchronize`|  |  |  |`hipDeviceSynchronize`|
|`cudaGetDevice`|  |  |  |`hipGetDevice`|
|`cudaGetDeviceCount`|  |  |  |`hipGetDeviceCount`|
|`cudaGetDeviceFlags`|  |  |  |`hipGetDeviceFlags`|
|`cudaGetDeviceProperties`|  |  |  |`hipGetDeviceProperties`|
|`cudaIpcCloseMemHandle`|  |  |  |`hipIpcCloseMemHandle`|
|`cudaIpcGetEventHandle`|  |  |  |`hipIpcGetEventHandle`|
|`cudaIpcGetMemHandle`|  |  |  |`hipIpcGetMemHandle`|
|`cudaIpcOpenEventHandle`|  |  |  |`hipIpcOpenEventHandle`|
|`cudaIpcOpenMemHandle`|  |  |  |`hipIpcOpenMemHandle`|
|`cudaSetDevice`|  |  |  |`hipSetDevice`|
|`cudaSetDeviceFlags`|  |  |  |`hipSetDeviceFlags`|
|`cudaSetValidDevices`|  |  |  ||

## **2. Thread Management [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaThreadExit`|  | 10.0 |  |`hipDeviceReset`|
|`cudaThreadGetCacheConfig`|  | 10.0 |  |`hipDeviceGetCacheConfig`|
|`cudaThreadGetLimit`|  | 10.0 |  ||
|`cudaThreadSetCacheConfig`|  | 10.0 |  |`hipDeviceSetCacheConfig`|
|`cudaThreadSetLimit`|  | 10.0 |  ||
|`cudaThreadSynchronize`|  | 10.0 |  |`hipDeviceSynchronize`|

## **3. Error Handling**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaGetErrorName`|  |  |  |`hipGetErrorName`|
|`cudaGetErrorString`|  |  |  |`hipGetErrorString`|
|`cudaGetLastError`|  |  |  |`hipGetLastError`|
|`cudaPeekAtLastError`|  |  |  |`hipPeekAtLastError`|

## **4. Stream Management**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaCtxResetPersistingL2Cache`| 11.0 |  |  ||
|`cudaStreamAddCallback`|  |  |  |`hipStreamAddCallback`|
|`cudaStreamAttachMemAsync`|  |  |  |`hipStreamAttachMemAsync`|
|`cudaStreamBeginCapture`| 10.0 |  |  ||
|`cudaStreamCopyAttributes`| 11.0 |  |  ||
|`cudaStreamCreate`|  |  |  |`hipStreamCreate`|
|`cudaStreamCreateWithFlags`|  |  |  |`hipStreamCreateWithFlags`|
|`cudaStreamCreateWithPriority`|  |  |  |`hipStreamCreateWithPriority`|
|`cudaStreamDestroy`|  |  |  |`hipStreamDestroy`|
|`cudaStreamEndCapture`| 10.0 |  |  ||
|`cudaStreamGetAttribute`| 11.0 |  |  ||
|`cudaStreamGetCaptureInfo`| 10.1 |  |  ||
|`cudaStreamGetFlags`|  |  |  |`hipStreamGetFlags`|
|`cudaStreamGetPriority`|  |  |  |`hipStreamGetPriority`|
|`cudaStreamIsCapturing`| 10.0 |  |  ||
|`cudaStreamQuery`|  |  |  |`hipStreamQuery`|
|`cudaStreamSetAttribute`| 11.0 |  |  ||
|`cudaStreamSynchronize`|  |  |  |`hipStreamSynchronize`|
|`cudaStreamWaitEvent`|  |  |  |`hipStreamWaitEvent`|
|`cudaThreadExchangeStreamCaptureMode`| 10.1 |  |  ||

## **5. Event Management**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaEventCreate`|  |  |  |`hipEventCreate`|
|`cudaEventCreateWithFlags`|  |  |  |`hipEventCreateWithFlags`|
|`cudaEventDestroy`|  |  |  |`hipEventDestroy`|
|`cudaEventElapsedTime`|  |  |  |`hipEventElapsedTime`|
|`cudaEventQuery`|  |  |  |`hipEventQuery`|
|`cudaEventRecord`|  |  |  |`hipEventRecord`|
|`cudaEventRecordWithFlags`| 11.1 |  |  ||
|`cudaEventSynchronize`|  |  |  |`hipEventSynchronize`|

## **6. External Resource Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaDestroyExternalMemory`| 10.0 |  |  ||
|`cudaDestroyExternalSemaphore`| 10.0 |  |  ||
|`cudaExternalMemoryGetMappedBuffer`| 10.0 |  |  ||
|`cudaExternalMemoryGetMappedMipmappedArray`| 10.0 |  |  ||
|`cudaImportExternalMemory`| 10.0 |  |  ||
|`cudaImportExternalSemaphore`| 10.0 |  |  ||
|`cudaSignalExternalSemaphoresAsync`| 10.0 |  |  ||
|`cudaWaitExternalSemaphoresAsync`| 10.0 |  |  ||

## **7. Execution Control**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaFuncGetAttributes`|  |  |  |`hipFuncGetAttributes`|
|`cudaFuncSetAttribute`| 9.0 |  |  |`hipFuncSetAttribute`|
|`cudaFuncSetCacheConfig`|  |  |  |`hipFuncSetCacheConfig`|
|`cudaFuncSetSharedMemConfig`|  |  |  |`hipFuncSetSharedMemConfig`|
|`cudaGetParameterBuffer`|  |  |  ||
|`cudaGetParameterBufferV2`|  |  |  ||
|`cudaLaunchCooperativeKernel`| 9.0 |  |  |`hipLaunchCooperativeKernel`|
|`cudaLaunchCooperativeKernelMultiDevice`| 9.0 |  |  |`hipLaunchCooperativeKernelMultiDevice`|
|`cudaLaunchHostFunc`| 10.0 |  |  ||
|`cudaLaunchKernel`|  |  |  |`hipLaunchKernel`|
|`cudaSetDoubleForDevice`|  | 10.0 |  ||
|`cudaSetDoubleForHost`|  | 10.0 |  ||

## **8. Occupancy**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaOccupancyAvailableDynamicSMemPerBlock`| 11.0 |  |  ||
|`cudaOccupancyMaxActiveBlocksPerMultiprocessor`|  |  |  |`hipOccupancyMaxActiveBlocksPerMultiprocessor`|
|`cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`|  |  |  |`hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`|
|`cudaOccupancyMaxPotentialBlockSize`|  |  |  |`hipOccupancyMaxPotentialBlockSize`|
|`cudaOccupancyMaxPotentialBlockSizeVariableSMem`|  |  |  ||
|`cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags`|  |  |  ||
|`cudaOccupancyMaxPotentialBlockSizeWithFlags`|  |  |  |`hipOccupancyMaxPotentialBlockSizeWithFlags`|

## **9. Memory Management**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaArrayGetInfo`|  |  |  ||
|`cudaArrayGetSparseProperties`| 11.1 |  |  ||
|`cudaFree`|  |  |  |`hipFree`|
|`cudaFreeArray`|  |  |  |`hipFreeArray`|
|`cudaFreeHost`|  |  |  |`hipHostFree`|
|`cudaFreeMipmappedArray`|  |  |  |`hipFreeMipmappedArray`|
|`cudaGetMipmappedArrayLevel`|  |  |  |`hipGetMipmappedArrayLevel`|
|`cudaGetSymbolAddress`|  |  |  |`hipGetSymbolAddress`|
|`cudaGetSymbolSize`|  |  |  |`hipGetSymbolSize`|
|`cudaHostAlloc`|  |  |  |`hipHostMalloc`|
|`cudaHostGetDevicePointer`|  |  |  |`hipHostGetDevicePointer`|
|`cudaHostGetFlags`|  |  |  |`hipHostGetFlags`|
|`cudaHostRegister`|  |  |  |`hipHostRegister`|
|`cudaHostUnregister`|  |  |  |`hipHostUnregister`|
|`cudaMalloc`|  |  |  |`hipMalloc`|
|`cudaMalloc3D`|  |  |  |`hipMalloc3D`|
|`cudaMalloc3DArray`|  |  |  |`hipMalloc3DArray`|
|`cudaMallocArray`|  |  |  |`hipMallocArray`|
|`cudaMallocHost`|  |  |  |`hipHostMalloc`|
|`cudaMallocManaged`|  |  |  |`hipMallocManaged`|
|`cudaMallocMipmappedArray`|  |  |  |`hipMallocMipmappedArray`|
|`cudaMallocPitch`|  |  |  |`hipMallocPitch`|
|`cudaMemAdvise`| 8.0 |  |  |`hipMemAdvise`|
|`cudaMemGetInfo`|  |  |  |`hipMemGetInfo`|
|`cudaMemPrefetchAsync`| 8.0 |  |  |`hipMemPrefetchAsync`|
|`cudaMemRangeGetAttribute`| 8.0 |  |  |`hipMemRangeGetAttribute`|
|`cudaMemRangeGetAttributes`| 8.0 |  |  |`hipMemRangeGetAttributes`|
|`cudaMemcpy`|  |  |  |`hipMemcpy`|
|`cudaMemcpy2D`|  |  |  |`hipMemcpy2D`|
|`cudaMemcpy2DArrayToArray`|  |  |  ||
|`cudaMemcpy2DAsync`|  |  |  |`hipMemcpy2DAsync`|
|`cudaMemcpy2DFromArray`|  |  |  |`hipMemcpy2DFromArray`|
|`cudaMemcpy2DFromArrayAsync`|  |  |  |`hipMemcpy2DFromArrayAsync`|
|`cudaMemcpy2DToArray`|  |  |  |`hipMemcpy2DToArray`|
|`cudaMemcpy2DToArrayAsync`|  |  |  ||
|`cudaMemcpy3D`|  |  |  |`hipMemcpy3D`|
|`cudaMemcpy3DAsync`|  |  |  |`hipMemcpy3DAsync`|
|`cudaMemcpy3DPeer`|  |  |  ||
|`cudaMemcpy3DPeerAsync`|  |  |  ||
|`cudaMemcpyAsync`|  |  |  |`hipMemcpyAsync`|
|`cudaMemcpyFromSymbol`|  |  |  |`hipMemcpyFromSymbol`|
|`cudaMemcpyFromSymbolAsync`|  |  |  |`hipMemcpyFromSymbolAsync`|
|`cudaMemcpyPeer`|  |  |  |`hipMemcpyPeer`|
|`cudaMemcpyPeerAsync`|  |  |  |`hipMemcpyPeerAsync`|
|`cudaMemcpyToSymbol`|  |  |  |`hipMemcpyToSymbol`|
|`cudaMemcpyToSymbolAsync`|  |  |  |`hipMemcpyToSymbolAsync`|
|`cudaMemset`|  |  |  |`hipMemset`|
|`cudaMemset2D`|  |  |  |`hipMemset2D`|
|`cudaMemset2DAsync`|  |  |  |`hipMemset2DAsync`|
|`cudaMemset3D`|  |  |  |`hipMemset3D`|
|`cudaMemset3DAsync`|  |  |  |`hipMemset3DAsync`|
|`cudaMemsetAsync`|  |  |  |`hipMemsetAsync`|
|`make_cudaExtent`|  |  |  |`make_hipExtent`|
|`make_cudaPitchedPtr`|  |  |  |`make_hipPitchedPtr`|
|`make_cudaPos`|  |  |  |`make_hipPos`|

## **10. Memory Management [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaMemcpyArrayToArray`|  | 10.1 |  ||
|`cudaMemcpyFromArray`|  | 10.1 |  |`hipMemcpyFromArray`|
|`cudaMemcpyFromArrayAsync`|  | 10.1 |  ||
|`cudaMemcpyToArray`|  | 10.1 |  |`hipMemcpyToArray`|
|`cudaMemcpyToArrayAsync`|  | 10.1 |  |`hipMemcpyToArrayAsync`|

## **11. Unified Addressing**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaPointerGetAttributes`|  |  |  |`hipPointerGetAttributes`|

## **12. Peer Device Memory Access**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaDeviceCanAccessPeer`|  |  |  |`hipDeviceCanAccessPeer`|
|`cudaDeviceDisablePeerAccess`|  |  |  |`hipDeviceDisablePeerAccess`|
|`cudaDeviceEnablePeerAccess`|  |  |  |`hipDeviceEnablePeerAccess`|

## **13. OpenGL Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaGLGetDevices`|  |  |  ||
|`cudaGraphicsGLRegisterBuffer`|  |  |  ||
|`cudaGraphicsGLRegisterImage`|  |  |  ||
|`cudaWGLGetDevice`|  |  |  ||

## **14. OpenGL Interoperability [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaGLMapBufferObject`|  | 10.0 |  ||
|`cudaGLMapBufferObjectAsync`|  | 10.0 |  ||
|`cudaGLRegisterBufferObject`|  | 10.0 |  ||
|`cudaGLSetBufferObjectMapFlags`|  | 10.0 |  ||
|`cudaGLSetGLDevice`|  | 10.0 |  ||
|`cudaGLUnmapBufferObject`|  | 10.0 |  ||
|`cudaGLUnmapBufferObjectAsync`|  | 10.0 |  ||
|`cudaGLUnregisterBufferObject`|  | 10.0 |  ||

## **15. Direct3D 9 Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaD3D9GetDevice`|  |  |  ||
|`cudaD3D9GetDevices`|  |  |  ||
|`cudaD3D9GetDirect3DDevice`|  |  |  ||
|`cudaD3D9SetDirect3DDevice`|  |  |  ||
|`cudaGraphicsD3D9RegisterResource`|  |  |  ||

## **16. Direct3D 9 Interoperability [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaD3D9MapResources`|  | 10.0 |  ||
|`cudaD3D9RegisterResource`|  |  |  ||
|`cudaD3D9ResourceGetMappedArray`|  | 10.0 |  ||
|`cudaD3D9ResourceGetMappedPitch`|  | 10.0 |  ||
|`cudaD3D9ResourceGetMappedPointer`|  | 10.0 |  ||
|`cudaD3D9ResourceGetMappedSize`|  | 10.0 |  ||
|`cudaD3D9ResourceGetSurfaceDimensions`|  | 10.0 |  ||
|`cudaD3D9ResourceSetMapFlags`|  | 10.0 |  ||
|`cudaD3D9UnmapResources`|  | 10.0 |  ||
|`cudaD3D9UnregisterResource`|  | 10.0 |  ||

## **17. Direct3D 10 Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaD3D10GetDevice`|  |  |  ||
|`cudaD3D10GetDevices`|  |  |  ||
|`cudaGraphicsD3D10RegisterResource`|  |  |  ||

## **18. Direct3D 10 Interoperability [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaD3D10GetDirect3DDevice`|  | 10.0 |  ||
|`cudaD3D10MapResources`|  | 10.0 |  ||
|`cudaD3D10RegisterResource`|  | 10.0 |  ||
|`cudaD3D10ResourceGetMappedArray`|  | 10.0 |  ||
|`cudaD3D10ResourceGetMappedPitch`|  | 10.0 |  ||
|`cudaD3D10ResourceGetMappedPointer`|  | 10.0 |  ||
|`cudaD3D10ResourceGetMappedSize`|  | 10.0 |  ||
|`cudaD3D10ResourceGetSurfaceDimensions`|  | 10.0 |  ||
|`cudaD3D10ResourceSetMapFlags`|  | 10.0 |  ||
|`cudaD3D10SetDirect3DDevice`|  | 10.0 |  ||
|`cudaD3D10UnmapResources`|  | 10.0 |  ||
|`cudaD3D10UnregisterResource`|  | 10.0 |  ||

## **19. Direct3D 11 Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaD3D11GetDevice`|  |  |  ||
|`cudaD3D11GetDevices`|  |  |  ||
|`cudaGraphicsD3D11RegisterResource`|  |  |  ||

## **20. Direct3D 11 Interoperability [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaD3D11GetDirect3DDevice`|  | 10.0 |  ||
|`cudaD3D11SetDirect3DDevice`|  | 10.0 |  ||

## **21. VDPAU Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaGraphicsVDPAURegisterOutputSurface`|  |  |  ||
|`cudaGraphicsVDPAURegisterVideoSurface`|  |  |  ||
|`cudaVDPAUGetDevice`|  |  |  ||
|`cudaVDPAUSetVDPAUDevice`|  |  |  ||

## **22. EGL Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaEGLStreamConsumerAcquireFrame`| 9.1 |  |  ||
|`cudaEGLStreamConsumerConnect`| 9.1 |  |  ||
|`cudaEGLStreamConsumerConnectWithFlags`| 9.1 |  |  ||
|`cudaEGLStreamConsumerDisconnect`| 9.1 |  |  ||
|`cudaEGLStreamConsumerReleaseFrame`| 9.1 |  |  ||
|`cudaEGLStreamProducerConnect`| 9.1 |  |  ||
|`cudaEGLStreamProducerDisconnect`| 9.1 |  |  ||
|`cudaEGLStreamProducerPresentFrame`| 9.1 |  |  ||
|`cudaEGLStreamProducerReturnFrame`| 9.1 |  |  ||
|`cudaEventCreateFromEGLSync`| 9.1 |  |  ||
|`cudaGraphicsEGLRegisterImage`| 9.1 |  |  ||
|`cudaGraphicsResourceGetMappedEglFrame`| 9.1 |  |  ||

## **23. Graphics Interoperability**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaGraphicsMapResources`|  |  |  ||
|`cudaGraphicsResourceGetMappedMipmappedArray`|  |  |  ||
|`cudaGraphicsResourceGetMappedPointer`|  |  |  ||
|`cudaGraphicsResourceSetMapFlags`|  |  |  ||
|`cudaGraphicsSubResourceGetMappedArray`|  |  |  ||
|`cudaGraphicsUnmapResources`|  |  |  ||
|`cudaGraphicsUnregisterResource`|  |  |  ||

## **24. Texture Reference Management [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaBindTexture`|  | 11.0 |  |`hipBindTexture`|
|`cudaBindTexture2D`|  | 11.0 |  |`hipBindTexture2D`|
|`cudaBindTextureToArray`|  | 11.0 |  |`hipBindTextureToArray`|
|`cudaBindTextureToMipmappedArray`|  | 11.0 |  |`hipBindTextureToMipmappedArray`|
|`cudaCreateChannelDesc`|  |  |  |`hipCreateChannelDesc`|
|`cudaGetChannelDesc`|  |  |  |`hipGetChannelDesc`|
|`cudaGetTextureAlignmentOffset`|  | 11.0 |  |`hipGetTextureAlignmentOffset`|
|`cudaGetTextureReference`|  | 11.0 |  |`hipGetTextureReference`|
|`cudaUnbindTexture`|  | 11.0 |  |`hipUnbindTexture`|

## **25. Surface Reference Management [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaBindSurfaceToArray`|  | 11.0 |  ||
|`cudaGetSurfaceReference`|  | 11.0 |  ||

## **26. Texture Object Management**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cuTexObjectGetTextureDesc`| 9.0 |  |  |`hipGetTextureObjectTextureDesc`|
|`cudaCreateTextureObject`|  |  |  |`hipCreateTextureObject`|
|`cudaDestroyTextureObject`|  |  |  |`hipDestroyTextureObject`|
|`cudaGetTextureObjectResourceDesc`|  |  |  |`hipGetTextureObjectResourceDesc`|
|`cudaGetTextureObjectResourceViewDesc`|  |  |  |`hipGetTextureObjectResourceViewDesc`|

## **27. Surface Object Management**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaCreateSurfaceObject`| 9.0 |  |  |`hipCreateSurfaceObject`|
|`cudaDestroySurfaceObject`| 9.0 |  |  |`hipDestroySurfaceObject`|
|`cudaGetSurfaceObjectResourceDesc`| 9.0 |  |  ||

## **28. Version Management**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaDriverGetVersion`| 9.0 |  |  |`hipDriverGetVersion`|
|`cudaRuntimeGetVersion`| 9.0 |  |  |`hipRuntimeGetVersion`|

## **29. Graph Management**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaGraphAddChildGraphNode`| 10.0 |  |  ||
|`cudaGraphAddDependencies`| 10.0 |  |  ||
|`cudaGraphAddEmptyNode`| 10.0 |  |  ||
|`cudaGraphAddEventRecordNode`| 11.1 |  |  ||
|`cudaGraphAddEventWaitNode`| 11.1 |  |  ||
|`cudaGraphAddHostNode`| 10.0 |  |  ||
|`cudaGraphAddKernelNode`| 10.0 |  |  ||
|`cudaGraphAddMemcpyNode`| 10.0 |  |  ||
|`cudaGraphAddMemcpyNode1D`| 11.1 |  |  ||
|`cudaGraphAddMemcpyNodeFromSymbol`| 11.1 |  |  ||
|`cudaGraphAddMemcpyNodeToSymbol`| 11.1 |  |  ||
|`cudaGraphAddMemsetNode`| 10.0 |  |  ||
|`cudaGraphChildGraphNodeGetGraph`| 10.0 |  |  ||
|`cudaGraphClone`| 10.0 |  |  ||
|`cudaGraphCreate`| 10.0 |  |  ||
|`cudaGraphDestroy`| 10.0 |  |  ||
|`cudaGraphDestroyNode`| 10.0 |  |  ||
|`cudaGraphEventRecordNodeGetEvent`| 11.1 |  |  ||
|`cudaGraphEventRecordNodeSetEvent`| 11.1 |  |  ||
|`cudaGraphEventWaitNodeGetEvent`| 11.1 |  |  ||
|`cudaGraphEventWaitNodeSetEvent`| 11.1 |  |  ||
|`cudaGraphExecChildGraphNodeSetParams`| 11.1 |  |  ||
|`cudaGraphExecDestroy`| 10.0 |  |  ||
|`cudaGraphExecEventRecordNodeSetEvent`| 11.1 |  |  ||
|`cudaGraphExecEventWaitNodeSetEvent`| 11.1 |  |  ||
|`cudaGraphExecHostNodeSetParams`| 11.0 |  |  ||
|`cudaGraphExecKernelNodeSetParams`| 11.0 |  |  ||
|`cudaGraphExecMemcpyNodeSetParams`| 11.0 |  |  ||
|`cudaGraphExecMemcpyNodeSetParams1D`| 11.1 |  |  ||
|`cudaGraphExecMemcpyNodeSetParamsFromSymbol`| 11.1 |  |  ||
|`cudaGraphExecMemcpyNodeSetParamsToSymbol`| 11.1 |  |  ||
|`cudaGraphExecMemsetNodeSetParams`| 11.0 |  |  ||
|`cudaGraphExecUpdate`| 11.0 |  |  ||
|`cudaGraphGetEdges`| 10.0 |  |  ||
|`cudaGraphGetNodes`| 10.0 |  |  ||
|`cudaGraphGetRootNodes`| 10.0 |  |  ||
|`cudaGraphHostNodeGetParams`| 10.0 |  |  ||
|`cudaGraphHostNodeSetParams`| 10.0 |  |  ||
|`cudaGraphInstantiate`| 10.0 |  |  ||
|`cudaGraphKernelNodeCopyAttributes`| 11.0 |  |  ||
|`cudaGraphKernelNodeGetAttribute`| 11.0 |  |  ||
|`cudaGraphKernelNodeGetParams`| 11.0 |  |  ||
|`cudaGraphKernelNodeSetAttribute`| 11.0 |  |  ||
|`cudaGraphKernelNodeSetParams`| 11.0 |  |  ||
|`cudaGraphLaunch`| 11.0 |  |  ||
|`cudaGraphMemcpyNodeGetParams`| 11.0 |  |  ||
|`cudaGraphMemcpyNodeSetParams`| 11.0 |  |  ||
|`cudaGraphMemcpyNodeSetParams1D`| 11.1 |  |  ||
|`cudaGraphMemcpyNodeSetParamsFromSymbol`| 11.1 |  |  ||
|`cudaGraphMemcpyNodeSetParamsToSymbol`| 11.1 |  |  ||
|`cudaGraphMemsetNodeGetParams`| 11.0 |  |  ||
|`cudaGraphMemsetNodeSetParams`| 11.0 |  |  ||
|`cudaGraphNodeFindInClone`| 11.0 |  |  ||
|`cudaGraphNodeGetDependencies`| 11.0 |  |  ||
|`cudaGraphNodeGetDependentNodes`| 11.0 |  |  ||
|`cudaGraphNodeGetType`| 11.0 |  |  ||
|`cudaGraphRemoveDependencies`| 11.0 |  |  ||
|`cudaGraphUpload`| 11.1 |  |  ||

## **30. C++ API Routines**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|

## **31. Interactions with the CUDA Driver API**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaGetFuncBySymbol`| 11.0 |  |  ||

## **32. Profiler Control [DEPRECATED]**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaProfilerInitialize`|  | 11.0 |  ||

## **33. Profiler Control**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaProfilerStart`|  |  |  |`hipProfilerStart`|
|`cudaProfilerStop`|  |  |  |`hipProfilerStop`|

## **34. Data types used by CUDA Runtime**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`CUDA_EGL_MAX_PLANES`| 9.1 |  |  ||
|`CUDA_IPC_HANDLE_SIZE`|  |  |  |`HIP_IPC_HANDLE_SIZE`|
|`CUeglStreamConnection_st`| 9.1 |  |  ||
|`CUevent_st`|  |  |  |`ihipEvent_t`|
|`CUexternalMemory_st`| 10.0 |  |  ||
|`CUexternalSemaphore_st`| 10.0 |  |  ||
|`CUgraphExec_st`| 10.0 |  |  ||
|`CUgraphNode_st`| 10.0 |  |  ||
|`CUgraph_st`| 10.0 |  |  ||
|`CUstream_st`|  |  |  |`ihipStream_t`|
|`CUuuid_st`|  |  |  ||
|`MAJOR_VERSION`| 8.0 |  |  ||
|`MINOR_VERSION`| 8.0 |  |  ||
|`PATCH_LEVEL`| 8.0 |  |  ||
|`cudaAccessPolicyWindow`| 11.0 |  |  ||
|`cudaAccessProperty`| 11.0 |  |  ||
|`cudaAccessPropertyNormal`| 11.0 |  |  ||
|`cudaAccessPropertyPersisting`| 11.0 |  |  ||
|`cudaAccessPropertyStreaming`| 11.0 |  |  ||
|`cudaAddressModeBorder`|  |  |  |`hipAddressModeBorder`|
|`cudaAddressModeClamp`|  |  |  |`hipAddressModeClamp`|
|`cudaAddressModeMirror`|  |  |  |`hipAddressModeMirror`|
|`cudaAddressModeWrap`|  |  |  |`hipAddressModeWrap`|
|`cudaArray`|  |  |  |`hipArray`|
|`cudaArrayColorAttachment`| 10.0 |  |  ||
|`cudaArrayCubemap`|  |  |  |`hipArrayCubemap`|
|`cudaArrayDefault`|  |  |  |`hipArrayDefault`|
|`cudaArrayLayered`|  |  |  |`hipArrayLayered`|
|`cudaArraySparse`| 11.1 |  |  ||
|`cudaArraySparseProperties`| 11.1 |  |  ||
|`cudaArraySparsePropertiesSingleMipTail`| 11.1 |  |  ||
|`cudaArraySurfaceLoadStore`|  |  |  |`hipArraySurfaceLoadStore`|
|`cudaArrayTextureGather`|  |  |  |`hipArrayTextureGather`|
|`cudaArray_const_t`|  |  |  |`hipArray_const_t`|
|`cudaArray_t`|  |  |  |`hipArray_t`|
|`cudaBoundaryModeClamp`|  |  |  |`hipBoundaryModeClamp`|
|`cudaBoundaryModeTrap`|  |  |  |`hipBoundaryModeTrap`|
|`cudaBoundaryModeZero`|  |  |  |`hipBoundaryModeZero`|
|`cudaCGScope`| 9.0 |  |  ||
|`cudaCGScopeGrid`| 9.0 |  |  ||
|`cudaCGScopeInvalid`| 9.0 |  |  ||
|`cudaCGScopeMultiGrid`| 9.0 |  |  ||
|`cudaCSV`|  |  |  ||
|`cudaChannelFormatDesc`|  |  |  |`hipChannelFormatDesc`|
|`cudaChannelFormatKind`|  |  |  |`hipChannelFormatKind`|
|`cudaChannelFormatKindFloat`|  |  |  |`hipChannelFormatKindFloat`|
|`cudaChannelFormatKindNone`|  |  |  |`hipChannelFormatKindNone`|
|`cudaChannelFormatKindSigned`|  |  |  |`hipChannelFormatKindSigned`|
|`cudaChannelFormatKindUnsigned`|  |  |  |`hipChannelFormatKindUnsigned`|
|`cudaComputeMode`|  |  |  |`hipComputeMode`|
|`cudaComputeModeDefault`|  |  |  |`hipComputeModeDefault`|
|`cudaComputeModeExclusive`|  |  |  |`hipComputeModeExclusive`|
|`cudaComputeModeExclusiveProcess`|  |  |  |`hipComputeModeExclusiveProcess`|
|`cudaComputeModeProhibited`|  |  |  |`hipComputeModeProhibited`|
|`cudaCooperativeLaunchMultiDeviceNoPostSync`| 9.0 |  |  |`hipCooperativeLaunchMultiDeviceNoPostSync`|
|`cudaCooperativeLaunchMultiDeviceNoPreSync`| 9.0 |  |  |`hipCooperativeLaunchMultiDeviceNoPreSync`|
|`cudaCpuDeviceId`| 8.0 |  |  |`hipCpuDeviceId`|
|`cudaD3D10DeviceList`|  |  |  ||
|`cudaD3D10DeviceListAll`|  |  |  ||
|`cudaD3D10DeviceListCurrentFrame`|  |  |  ||
|`cudaD3D10DeviceListNextFrame`|  |  |  ||
|`cudaD3D10MapFlags`|  |  |  ||
|`cudaD3D10MapFlagsNone`|  |  |  ||
|`cudaD3D10MapFlagsReadOnly`|  |  |  ||
|`cudaD3D10MapFlagsWriteDiscard`|  |  |  ||
|`cudaD3D10RegisterFlags`|  |  |  ||
|`cudaD3D10RegisterFlagsArray`|  |  |  ||
|`cudaD3D10RegisterFlagsNone`|  |  |  ||
|`cudaD3D11DeviceList`|  |  |  ||
|`cudaD3D11DeviceListAll`|  |  |  ||
|`cudaD3D11DeviceListCurrentFrame`|  |  |  ||
|`cudaD3D11DeviceListNextFrame`|  |  |  ||
|`cudaD3D9DeviceList`|  |  |  ||
|`cudaD3D9DeviceListAll`|  |  |  ||
|`cudaD3D9DeviceListCurrentFrame`|  |  |  ||
|`cudaD3D9DeviceListNextFrame`|  |  |  ||
|`cudaD3D9MapFlags`|  |  |  ||
|`cudaD3D9MapFlagsNone`|  |  |  ||
|`cudaD3D9MapFlagsReadOnly`|  |  |  ||
|`cudaD3D9MapFlagsWriteDiscard`|  |  |  ||
|`cudaD3D9RegisterFlags`|  |  |  ||
|`cudaD3D9RegisterFlagsArray`|  |  |  ||
|`cudaD3D9RegisterFlagsNone`|  |  |  ||
|`cudaDevAttrAsyncEngineCount`|  |  |  ||
|`cudaDevAttrCanFlushRemoteWrites`| 9.2 |  |  ||
|`cudaDevAttrCanMapHostMemory`|  |  |  |`hipDeviceAttributeCanMapHostMemory`|
|`cudaDevAttrCanUseHostPointerForRegisteredMem`| 8.0 |  |  ||
|`cudaDevAttrClockRate`|  |  |  |`hipDeviceAttributeClockRate`|
|`cudaDevAttrComputeCapabilityMajor`|  |  |  |`hipDeviceAttributeComputeCapabilityMajor`|
|`cudaDevAttrComputeCapabilityMinor`|  |  |  |`hipDeviceAttributeComputeCapabilityMinor`|
|`cudaDevAttrComputeMode`|  |  |  |`hipDeviceAttributeComputeMode`|
|`cudaDevAttrComputePreemptionSupported`| 8.0 |  |  ||
|`cudaDevAttrConcurrentKernels`|  |  |  |`hipDeviceAttributeConcurrentKernels`|
|`cudaDevAttrConcurrentManagedAccess`| 8.0 |  |  |`hipDeviceAttributeConcurrentManagedAccess`|
|`cudaDevAttrCooperativeLaunch`| 9.0 |  |  |`hipDeviceAttributeCooperativeLaunch`|
|`cudaDevAttrCooperativeMultiDeviceLaunch`| 9.0 |  |  |`hipDeviceAttributeCooperativeMultiDeviceLaunch`|
|`cudaDevAttrDirectManagedMemAccessFromHost`| 9.2 |  |  |`hipDeviceAttributeDirectManagedMemAccessFromHost`|
|`cudaDevAttrEccEnabled`|  |  |  |`hipDeviceAttributeEccEnabled`|
|`cudaDevAttrGlobalL1CacheSupported`|  |  |  ||
|`cudaDevAttrGlobalMemoryBusWidth`|  |  |  |`hipDeviceAttributeMemoryBusWidth`|
|`cudaDevAttrGpuOverlap`|  |  |  ||
|`cudaDevAttrHostNativeAtomicSupported`| 8.0 |  |  ||
|`cudaDevAttrHostRegisterReadOnlySupported`| 11.1 |  |  ||
|`cudaDevAttrHostRegisterSupported`| 9.2 |  |  ||
|`cudaDevAttrIntegrated`|  |  |  |`hipDeviceAttributeIntegrated`|
|`cudaDevAttrIsMultiGpuBoard`|  |  |  |`hipDeviceAttributeIsMultiGpuBoard`|
|`cudaDevAttrKernelExecTimeout`|  |  |  |`hipDeviceAttributeKernelExecTimeout`|
|`cudaDevAttrL2CacheSize`|  |  |  |`hipDeviceAttributeL2CacheSize`|
|`cudaDevAttrLocalL1CacheSupported`|  |  |  ||
|`cudaDevAttrManagedMemory`|  |  |  |`hipDeviceAttributeManagedMemory`|
|`cudaDevAttrMaxBlockDimX`|  |  |  |`hipDeviceAttributeMaxBlockDimX`|
|`cudaDevAttrMaxBlockDimY`|  |  |  |`hipDeviceAttributeMaxBlockDimY`|
|`cudaDevAttrMaxBlockDimZ`|  |  |  |`hipDeviceAttributeMaxBlockDimZ`|
|`cudaDevAttrMaxBlocksPerMultiprocessor`| 11.0 |  |  ||
|`cudaDevAttrMaxGridDimX`|  |  |  |`hipDeviceAttributeMaxGridDimX`|
|`cudaDevAttrMaxGridDimY`|  |  |  |`hipDeviceAttributeMaxGridDimY`|
|`cudaDevAttrMaxGridDimZ`|  |  |  |`hipDeviceAttributeMaxGridDimZ`|
|`cudaDevAttrMaxPitch`|  |  |  |`hipDeviceAttributeMaxPitch`|
|`cudaDevAttrMaxRegistersPerBlock`|  |  |  |`hipDeviceAttributeMaxRegistersPerBlock`|
|`cudaDevAttrMaxRegistersPerMultiprocessor`|  |  |  ||
|`cudaDevAttrMaxSharedMemoryPerBlock`|  |  |  |`hipDeviceAttributeMaxSharedMemoryPerBlock`|
|`cudaDevAttrMaxSharedMemoryPerBlockOptin`| 9.0 |  |  ||
|`cudaDevAttrMaxSharedMemoryPerMultiprocessor`|  |  |  |`hipDeviceAttributeMaxSharedMemoryPerMultiprocessor`|
|`cudaDevAttrMaxSurface1DLayeredLayers`|  |  |  ||
|`cudaDevAttrMaxSurface1DLayeredWidth`|  |  |  ||
|`cudaDevAttrMaxSurface1DWidth`|  |  |  ||
|`cudaDevAttrMaxSurface2DHeight`|  |  |  ||
|`cudaDevAttrMaxSurface2DLayeredHeight`|  |  |  ||
|`cudaDevAttrMaxSurface2DLayeredLayers`|  |  |  ||
|`cudaDevAttrMaxSurface2DLayeredWidth`|  |  |  ||
|`cudaDevAttrMaxSurface2DWidth`|  |  |  ||
|`cudaDevAttrMaxSurface3DDepth`|  |  |  ||
|`cudaDevAttrMaxSurface3DHeight`|  |  |  ||
|`cudaDevAttrMaxSurface3DWidth`|  |  |  ||
|`cudaDevAttrMaxSurfaceCubemapLayeredLayers`|  |  |  ||
|`cudaDevAttrMaxSurfaceCubemapLayeredWidth`|  |  |  ||
|`cudaDevAttrMaxSurfaceCubemapWidth`|  |  |  ||
|`cudaDevAttrMaxTexture1DLayeredLayers`|  |  |  ||
|`cudaDevAttrMaxTexture1DLayeredWidth`|  |  |  ||
|`cudaDevAttrMaxTexture1DLinearWidth`|  |  |  ||
|`cudaDevAttrMaxTexture1DMipmappedWidth`|  |  |  ||
|`cudaDevAttrMaxTexture1DWidth`|  |  |  |`hipDeviceAttributeMaxTexture1DWidth`|
|`cudaDevAttrMaxTexture2DGatherHeight`|  |  |  ||
|`cudaDevAttrMaxTexture2DGatherWidth`|  |  |  ||
|`cudaDevAttrMaxTexture2DHeight`|  |  |  |`hipDeviceAttributeMaxTexture2DHeight`|
|`cudaDevAttrMaxTexture2DLayeredHeight`|  |  |  ||
|`cudaDevAttrMaxTexture2DLayeredLayers`|  |  |  ||
|`cudaDevAttrMaxTexture2DLayeredWidth`|  |  |  ||
|`cudaDevAttrMaxTexture2DLinearHeight`|  |  |  ||
|`cudaDevAttrMaxTexture2DLinearPitch`|  |  |  ||
|`cudaDevAttrMaxTexture2DLinearWidth`|  |  |  ||
|`cudaDevAttrMaxTexture2DMipmappedHeight`|  |  |  ||
|`cudaDevAttrMaxTexture2DMipmappedWidth`|  |  |  ||
|`cudaDevAttrMaxTexture2DWidth`|  |  |  |`hipDeviceAttributeMaxTexture2DWidth`|
|`cudaDevAttrMaxTexture3DDepth`|  |  |  |`hipDeviceAttributeMaxTexture3DDepth`|
|`cudaDevAttrMaxTexture3DDepthAlt`|  |  |  ||
|`cudaDevAttrMaxTexture3DHeight`|  |  |  |`hipDeviceAttributeMaxTexture3DHeight`|
|`cudaDevAttrMaxTexture3DHeightAlt`|  |  |  ||
|`cudaDevAttrMaxTexture3DWidth`|  |  |  |`hipDeviceAttributeMaxTexture3DWidth`|
|`cudaDevAttrMaxTexture3DWidthAlt`|  |  |  ||
|`cudaDevAttrMaxTextureCubemapLayeredLayers`|  |  |  ||
|`cudaDevAttrMaxTextureCubemapLayeredWidth`|  |  |  ||
|`cudaDevAttrMaxTextureCubemapWidth`|  |  |  ||
|`cudaDevAttrMaxThreadsPerBlock`|  |  |  |`hipDeviceAttributeMaxThreadsPerBlock`|
|`cudaDevAttrMaxThreadsPerMultiProcessor`|  |  |  |`hipDeviceAttributeMaxThreadsPerMultiProcessor`|
|`cudaDevAttrMemoryClockRate`|  |  |  |`hipDeviceAttributeMemoryClockRate`|
|`cudaDevAttrMultiGpuBoardGroupID`|  |  |  ||
|`cudaDevAttrMultiProcessorCount`|  |  |  |`hipDeviceAttributeMultiprocessorCount`|
|`cudaDevAttrPageableMemoryAccess`| 8.0 |  |  |`hipDeviceAttributePageableMemoryAccess`|
|`cudaDevAttrPageableMemoryAccessUsesHostPageTables`| 9.2 |  |  |`hipDeviceAttributePageableMemoryAccessUsesHostPageTables`|
|`cudaDevAttrPciBusId`|  |  |  |`hipDeviceAttributePciBusId`|
|`cudaDevAttrPciDeviceId`|  |  |  |`hipDeviceAttributePciDeviceId`|
|`cudaDevAttrPciDomainId`|  |  |  ||
|`cudaDevAttrReserved92`| 9.0 |  |  ||
|`cudaDevAttrReserved93`| 9.0 |  |  ||
|`cudaDevAttrReserved94`| 9.0 |  |  ||
|`cudaDevAttrReservedSharedMemoryPerBlock`| 11.0 |  |  ||
|`cudaDevAttrSingleToDoublePrecisionPerfRatio`| 8.0 |  |  ||
|`cudaDevAttrSparseCudaArraySupported`| 11.1 |  |  ||
|`cudaDevAttrStreamPrioritiesSupported`|  |  |  ||
|`cudaDevAttrSurfaceAlignment`|  |  |  ||
|`cudaDevAttrTccDriver`|  |  |  ||
|`cudaDevAttrTextureAlignment`|  |  |  |`hipDeviceAttributeTextureAlignment`|
|`cudaDevAttrTexturePitchAlignment`|  |  |  ||
|`cudaDevAttrTotalConstantMemory`|  |  |  |`hipDeviceAttributeTotalConstantMemory`|
|`cudaDevAttrUnifiedAddressing`|  |  |  ||
|`cudaDevAttrWarpSize`|  |  |  |`hipDeviceAttributeWarpSize`|
|`cudaDevP2PAttrAccessSupported`| 8.0 |  |  |`hipDevP2PAttrAccessSupported`|
|`cudaDevP2PAttrCudaArrayAccessSupported`| 9.2 |  |  |`hipDevP2PAttrHipArrayAccessSupported`|
|`cudaDevP2PAttrNativeAtomicSupported`| 8.0 |  |  |`hipDevP2PAttrNativeAtomicSupported`|
|`cudaDevP2PAttrPerformanceRank`| 8.0 |  |  |`hipDevP2PAttrPerformanceRank`|
|`cudaDeviceAttr`|  |  |  |`hipDeviceAttribute_t`|
|`cudaDeviceBlockingSync`|  |  |  |`hipDeviceScheduleBlockingSync`|
|`cudaDeviceLmemResizeToMax`|  |  |  |`hipDeviceLmemResizeToMax`|
|`cudaDeviceMapHost`|  |  |  |`hipDeviceMapHost`|
|`cudaDeviceMask`|  |  |  ||
|`cudaDeviceP2PAttr`| 8.0 |  |  |`hipDeviceP2PAttr`|
|`cudaDeviceProp`|  |  |  |`hipDeviceProp_t`|
|`cudaDevicePropDontCare`|  |  |  ||
|`cudaDeviceScheduleAuto`|  |  |  |`hipDeviceScheduleAuto`|
|`cudaDeviceScheduleBlockingSync`|  |  |  |`hipDeviceScheduleBlockingSync`|
|`cudaDeviceScheduleMask`|  |  |  |`hipDeviceScheduleMask`|
|`cudaDeviceScheduleSpin`|  |  |  |`hipDeviceScheduleSpin`|
|`cudaDeviceScheduleYield`|  |  |  |`hipDeviceScheduleYield`|
|`cudaEglColorFormat`| 9.1 |  |  ||
|`cudaEglColorFormatA`| 9.1 |  |  ||
|`cudaEglColorFormatABGR`| 9.1 |  |  ||
|`cudaEglColorFormatARGB`| 9.1 |  |  ||
|`cudaEglColorFormatAYUV`| 9.1 |  |  ||
|`cudaEglColorFormatAYUV_ER`| 9.1 |  |  ||
|`cudaEglColorFormatBGR`| 9.1 |  |  ||
|`cudaEglColorFormatBGRA`| 9.1 |  |  ||
|`cudaEglColorFormatBayer10BGGR`| 9.1 |  |  ||
|`cudaEglColorFormatBayer10GBRG`| 9.1 |  |  ||
|`cudaEglColorFormatBayer10GRBG`| 9.1 |  |  ||
|`cudaEglColorFormatBayer10RGGB`| 9.1 |  |  ||
|`cudaEglColorFormatBayer12BGGR`| 9.1 |  |  ||
|`cudaEglColorFormatBayer12GBRG`| 9.1 |  |  ||
|`cudaEglColorFormatBayer12GRBG`| 9.1 |  |  ||
|`cudaEglColorFormatBayer12RGGB`| 9.1 |  |  ||
|`cudaEglColorFormatBayer14BGGR`| 9.1 |  |  ||
|`cudaEglColorFormatBayer14GBRG`| 9.1 |  |  ||
|`cudaEglColorFormatBayer14GRBG`| 9.1 |  |  ||
|`cudaEglColorFormatBayer14RGGB`| 9.1 |  |  ||
|`cudaEglColorFormatBayer20BGGR`| 9.1 |  |  ||
|`cudaEglColorFormatBayer20GBRG`| 9.1 |  |  ||
|`cudaEglColorFormatBayer20GRBG`| 9.1 |  |  ||
|`cudaEglColorFormatBayer20RGGB`| 9.1 |  |  ||
|`cudaEglColorFormatBayerBGGR`| 9.1 |  |  ||
|`cudaEglColorFormatBayerGBRG`| 9.1 |  |  ||
|`cudaEglColorFormatBayerGRBG`| 9.1 |  |  ||
|`cudaEglColorFormatBayerIspBGGR`| 9.2 |  |  ||
|`cudaEglColorFormatBayerIspGBRG`| 9.2 |  |  ||
|`cudaEglColorFormatBayerIspGRBG`| 9.2 |  |  ||
|`cudaEglColorFormatBayerIspRGGB`| 9.2 |  |  ||
|`cudaEglColorFormatBayerRGGB`| 9.1 |  |  ||
|`cudaEglColorFormatL`| 9.1 |  |  ||
|`cudaEglColorFormatR`| 9.1 |  |  ||
|`cudaEglColorFormatRG`| 9.1 |  |  ||
|`cudaEglColorFormatRGB`| 9.1 |  |  ||
|`cudaEglColorFormatRGBA`| 9.1 |  |  ||
|`cudaEglColorFormatUYVY422`| 9.1 |  |  ||
|`cudaEglColorFormatUYVY_ER`| 9.1 |  |  ||
|`cudaEglColorFormatVYUY_ER`| 9.1 |  |  ||
|`cudaEglColorFormatY10V10U10_420SemiPlanar`| 9.1 |  |  ||
|`cudaEglColorFormatY10V10U10_444SemiPlanar`| 9.1 |  |  ||
|`cudaEglColorFormatY12V12U12_420SemiPlanar`| 9.1 |  |  ||
|`cudaEglColorFormatY12V12U12_444SemiPlanar`| 9.1 |  |  ||
|`cudaEglColorFormatYUV420Planar`| 9.1 |  |  ||
|`cudaEglColorFormatYUV420Planar_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYUV420SemiPlanar`| 9.1 |  |  ||
|`cudaEglColorFormatYUV420SemiPlanar_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYUV422Planar`| 9.1 |  |  ||
|`cudaEglColorFormatYUV422Planar_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYUV422SemiPlanar`| 9.1 |  |  ||
|`cudaEglColorFormatYUV422SemiPlanar_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYUV444Planar`| 9.1 |  |  ||
|`cudaEglColorFormatYUV444Planar_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYUV444SemiPlanar`| 9.1 |  |  ||
|`cudaEglColorFormatYUV444SemiPlanar_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYUVA_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYUV_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYUYV422`| 9.1 |  |  ||
|`cudaEglColorFormatYUYV_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYVU420Planar`| 9.1 |  |  ||
|`cudaEglColorFormatYVU420Planar_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYVU420SemiPlanar`| 9.1 |  |  ||
|`cudaEglColorFormatYVU420SemiPlanar_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYVU422Planar`| 9.1 |  |  ||
|`cudaEglColorFormatYVU422Planar_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYVU422SemiPlanar`| 9.1 |  |  ||
|`cudaEglColorFormatYVU422SemiPlanar_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYVU444Planar`| 9.1 |  |  ||
|`cudaEglColorFormatYVU444Planar_ER`| 9.1 |  |  ||
|`cudaEglColorFormatYVU444SemiPlanar`| 9.1 |  |  ||
|`cudaEglColorFormatYVU444SemiPlanar_ER`|  |  |  ||
|`cudaEglColorFormatYVYU_ER`| 9.1 |  |  ||
|`cudaEglFrame`| 9.1 |  |  ||
|`cudaEglFrameType`| 9.1 |  |  ||
|`cudaEglFrameTypeArray`| 9.1 |  |  ||
|`cudaEglFrameTypePitch`| 9.1 |  |  ||
|`cudaEglFrame_st`| 9.1 |  |  ||
|`cudaEglPlaneDesc`| 9.1 |  |  ||
|`cudaEglPlaneDesc_st`| 9.1 |  |  ||
|`cudaEglResourceLocationFlags`| 9.1 |  |  ||
|`cudaEglResourceLocationSysmem`| 9.1 |  |  ||
|`cudaEglResourceLocationVidmem`| 9.1 |  |  ||
|`cudaEglStreamConnection`| 9.1 |  |  ||
|`cudaError`|  |  |  |`hipError_t`|
|`cudaErrorAddressOfConstant`|  | 3.1 |  ||
|`cudaErrorAlreadyAcquired`| 10.1 |  |  |`hipErrorAlreadyAcquired`|
|`cudaErrorAlreadyMapped`| 10.1 |  |  |`hipErrorAlreadyMapped`|
|`cudaErrorApiFailureBase`|  | 4.1 |  ||
|`cudaErrorArrayIsMapped`| 10.1 |  |  |`hipErrorArrayIsMapped`|
|`cudaErrorAssert`|  |  |  |`hipErrorAssert`|
|`cudaErrorCallRequiresNewerDriver`| 11.1 |  |  ||
|`cudaErrorCapturedEvent`| 10.0 |  |  ||
|`cudaErrorCompatNotSupportedOnDevice`| 10.1 |  |  ||
|`cudaErrorContextIsDestroyed`| 10.1 |  |  ||
|`cudaErrorCooperativeLaunchTooLarge`| 9.0 |  |  |`hipErrorCooperativeLaunchTooLarge`|
|`cudaErrorCudartUnloading`|  |  |  |`hipErrorDeinitialized`|
|`cudaErrorDeviceAlreadyInUse`|  |  |  |`hipErrorContextAlreadyInUse`|
|`cudaErrorDeviceNotLicensed`| 11.1 |  |  ||
|`cudaErrorDeviceUninitialized`| 10.2 |  |  |`hipErrorInvalidContext`|
|`cudaErrorDeviceUninitilialized`| 10.1 |  | 10.2 |`hipErrorInvalidContext`|
|`cudaErrorDevicesUnavailable`|  |  |  ||
|`cudaErrorDuplicateSurfaceName`|  |  |  ||
|`cudaErrorDuplicateTextureName`|  |  |  ||
|`cudaErrorDuplicateVariableName`|  |  |  ||
|`cudaErrorECCUncorrectable`|  |  |  |`hipErrorECCNotCorrectable`|
|`cudaErrorFileNotFound`| 10.1 |  |  |`hipErrorFileNotFound`|
|`cudaErrorGraphExecUpdateFailure`| 10.2 |  |  ||
|`cudaErrorHardwareStackError`|  |  |  ||
|`cudaErrorHostMemoryAlreadyRegistered`|  |  |  |`hipErrorHostMemoryAlreadyRegistered`|
|`cudaErrorHostMemoryNotRegistered`|  |  |  |`hipErrorHostMemoryNotRegistered`|
|`cudaErrorIllegalAddress`|  |  |  |`hipErrorIllegalAddress`|
|`cudaErrorIllegalInstruction`|  |  |  ||
|`cudaErrorIllegalState`| 10.0 |  |  ||
|`cudaErrorIncompatibleDriverContext`|  |  |  ||
|`cudaErrorInitializationError`|  |  |  |`hipErrorNotInitialized`|
|`cudaErrorInsufficientDriver`|  |  |  |`hipErrorInsufficientDriver`|
|`cudaErrorInvalidAddressSpace`|  |  |  ||
|`cudaErrorInvalidChannelDescriptor`|  |  |  ||
|`cudaErrorInvalidConfiguration`|  |  |  |`hipErrorInvalidConfiguration`|
|`cudaErrorInvalidDevice`|  |  |  |`hipErrorInvalidDevice`|
|`cudaErrorInvalidDeviceFunction`|  |  |  |`hipErrorInvalidDeviceFunction`|
|`cudaErrorInvalidDevicePointer`|  | 10.1 |  |`hipErrorInvalidDevicePointer`|
|`cudaErrorInvalidFilterSetting`|  |  |  ||
|`cudaErrorInvalidGraphicsContext`|  |  |  |`hipErrorInvalidGraphicsContext`|
|`cudaErrorInvalidHostPointer`|  | 10.1 |  ||
|`cudaErrorInvalidKernelImage`|  |  |  |`hipErrorInvalidImage`|
|`cudaErrorInvalidMemcpyDirection`|  |  |  |`hipErrorInvalidMemcpyDirection`|
|`cudaErrorInvalidNormSetting`|  |  |  ||
|`cudaErrorInvalidPc`|  |  |  ||
|`cudaErrorInvalidPitchValue`|  |  |  ||
|`cudaErrorInvalidPtx`|  |  |  |`hipErrorInvalidKernelFile`|
|`cudaErrorInvalidResourceHandle`|  |  |  |`hipErrorInvalidHandle`|
|`cudaErrorInvalidSource`| 10.1 |  |  |`hipErrorInvalidSource`|
|`cudaErrorInvalidSurface`|  |  |  ||
|`cudaErrorInvalidSymbol`|  |  |  |`hipErrorInvalidSymbol`|
|`cudaErrorInvalidTexture`|  |  |  ||
|`cudaErrorInvalidTextureBinding`|  |  |  ||
|`cudaErrorInvalidValue`|  |  |  |`hipErrorInvalidValue`|
|`cudaErrorJitCompilerNotFound`| 9.0 |  |  ||
|`cudaErrorLaunchFailure`|  |  |  |`hipErrorLaunchFailure`|
|`cudaErrorLaunchFileScopedSurf`|  |  |  ||
|`cudaErrorLaunchFileScopedTex`|  |  |  ||
|`cudaErrorLaunchIncompatibleTexturing`| 10.1 |  |  ||
|`cudaErrorLaunchMaxDepthExceeded`|  |  |  ||
|`cudaErrorLaunchOutOfResources`|  |  |  |`hipErrorLaunchOutOfResources`|
|`cudaErrorLaunchPendingCountExceeded`|  |  |  ||
|`cudaErrorLaunchTimeout`|  |  |  |`hipErrorLaunchTimeOut`|
|`cudaErrorMapBufferObjectFailed`|  |  |  |`hipErrorMapFailed`|
|`cudaErrorMemoryAllocation`|  |  |  |`hipErrorOutOfMemory`|
|`cudaErrorMemoryValueTooLarge`|  | 3.1 |  ||
|`cudaErrorMisalignedAddress`|  |  |  ||
|`cudaErrorMissingConfiguration`|  |  |  |`hipErrorMissingConfiguration`|
|`cudaErrorMixedDeviceExecution`|  | 3.1 |  ||
|`cudaErrorNoDevice`|  |  |  |`hipErrorNoDevice`|
|`cudaErrorNoKernelImageForDevice`|  |  |  |`hipErrorNoBinaryForGpu`|
|`cudaErrorNotMapped`| 10.1 |  |  |`hipErrorNotMapped`|
|`cudaErrorNotMappedAsArray`| 10.1 |  |  |`hipErrorNotMappedAsArray`|
|`cudaErrorNotMappedAsPointer`| 10.1 |  |  |`hipErrorNotMappedAsPointer`|
|`cudaErrorNotPermitted`|  |  |  ||
|`cudaErrorNotReady`|  |  |  |`hipErrorNotReady`|
|`cudaErrorNotSupported`|  |  |  |`hipErrorNotSupported`|
|`cudaErrorNotYetImplemented`|  | 4.1 |  ||
|`cudaErrorNvlinkUncorrectable`| 8.0 |  |  ||
|`cudaErrorOperatingSystem`|  |  |  |`hipErrorOperatingSystem`|
|`cudaErrorPeerAccessAlreadyEnabled`|  |  |  |`hipErrorPeerAccessAlreadyEnabled`|
|`cudaErrorPeerAccessNotEnabled`|  |  |  |`hipErrorPeerAccessNotEnabled`|
|`cudaErrorPeerAccessUnsupported`|  |  |  |`hipErrorPeerAccessUnsupported`|
|`cudaErrorPriorLaunchFailure`|  | 3.1 |  |`hipErrorPriorLaunchFailure`|
|`cudaErrorProfilerAlreadyStarted`|  | 5.0 |  |`hipErrorProfilerAlreadyStarted`|
|`cudaErrorProfilerAlreadyStopped`|  | 5.0 |  |`hipErrorProfilerAlreadyStopped`|
|`cudaErrorProfilerDisabled`|  |  |  |`hipErrorProfilerDisabled`|
|`cudaErrorProfilerNotInitialized`|  | 5.0 |  |`hipErrorProfilerNotInitialized`|
|`cudaErrorSetOnActiveProcess`|  |  |  |`hipErrorSetOnActiveProcess`|
|`cudaErrorSharedObjectInitFailed`|  |  |  |`hipErrorSharedObjectInitFailed`|
|`cudaErrorSharedObjectSymbolNotFound`|  |  |  |`hipErrorSharedObjectSymbolNotFound`|
|`cudaErrorStartupFailure`|  |  |  ||
|`cudaErrorStreamCaptureImplicit`| 10.0 |  |  ||
|`cudaErrorStreamCaptureInvalidated`| 10.0 |  |  ||
|`cudaErrorStreamCaptureIsolation`| 10.0 |  |  ||
|`cudaErrorStreamCaptureMerge`| 10.0 |  |  ||
|`cudaErrorStreamCaptureUnjoined`| 10.0 |  |  ||
|`cudaErrorStreamCaptureUnmatched`| 10.0 |  |  ||
|`cudaErrorStreamCaptureUnsupported`| 10.0 |  |  ||
|`cudaErrorStreamCaptureWrongThread`| 10.1 |  |  ||
|`cudaErrorStubLibrary`| 11.1 |  |  ||
|`cudaErrorSymbolNotFound`| 10.1 |  |  |`hipErrorNotFound`|
|`cudaErrorSyncDepthExceeded`|  |  |  ||
|`cudaErrorSynchronizationError`|  | 3.1 |  ||
|`cudaErrorSystemDriverMismatch`| 10.1 |  |  ||
|`cudaErrorSystemNotReady`| 10.0 |  |  ||
|`cudaErrorTextureFetchFailed`|  | 3.1 |  ||
|`cudaErrorTextureNotBound`|  | 3.1 |  ||
|`cudaErrorTimeout`| 10.2 |  |  ||
|`cudaErrorTooManyPeers`|  |  |  ||
|`cudaErrorUnknown`|  |  |  |`hipErrorUnknown`|
|`cudaErrorUnmapBufferObjectFailed`|  |  |  |`hipErrorUnmapFailed`|
|`cudaErrorUnsupportedLimit`|  |  |  |`hipErrorUnsupportedLimit`|
|`cudaErrorUnsupportedPtxVersion`| 11.1 |  |  ||
|`cudaError_t`|  |  |  |`hipError_t`|
|`cudaEventBlockingSync`|  |  |  |`hipEventBlockingSync`|
|`cudaEventDefault`|  |  |  |`hipEventDefault`|
|`cudaEventDisableTiming`|  |  |  |`hipEventDisableTiming`|
|`cudaEventInterprocess`|  |  |  |`hipEventInterprocess`|
|`cudaEventRecordDefault`| 11.1 |  |  ||
|`cudaEventRecordExternal`| 11.1 |  |  ||
|`cudaEventWaitDefault`| 11.1 |  |  ||
|`cudaEventWaitExternal`|  |  |  ||
|`cudaEvent_t`|  |  |  |`hipEvent_t`|
|`cudaExtent`|  |  |  |`hipExtent`|
|`cudaExternalMemoryBufferDesc`| 10.0 |  |  ||
|`cudaExternalMemoryDedicated`| 10.0 |  |  ||
|`cudaExternalMemoryHandleDesc`| 10.0 |  |  ||
|`cudaExternalMemoryHandleType`| 10.0 |  |  ||
|`cudaExternalMemoryHandleTypeD3D11Resource`| 10.0 |  |  ||
|`cudaExternalMemoryHandleTypeD3D11ResourceKmt`| 10.2 |  |  ||
|`cudaExternalMemoryHandleTypeD3D12Heap`| 10.0 |  |  ||
|`cudaExternalMemoryHandleTypeD3D12Resource`| 10.0 |  |  ||
|`cudaExternalMemoryHandleTypeNvSciBuf`| 10.2 |  |  ||
|`cudaExternalMemoryHandleTypeOpaqueFd`| 10.0 |  |  ||
|`cudaExternalMemoryHandleTypeOpaqueWin32`| 10.0 |  |  ||
|`cudaExternalMemoryHandleTypeOpaqueWin32Kmt`| 10.0 |  |  ||
|`cudaExternalMemoryMipmappedArrayDesc`| 10.0 |  |  ||
|`cudaExternalMemory_t`| 10.0 |  |  ||
|`cudaExternalSemaphoreHandleDesc`| 10.0 |  |  ||
|`cudaExternalSemaphoreHandleType`| 10.0 |  |  ||
|`cudaExternalSemaphoreHandleTypeD3D11Fence`| 10.2 |  |  ||
|`cudaExternalSemaphoreHandleTypeD3D12Fence`| 10.0 |  |  ||
|`cudaExternalSemaphoreHandleTypeKeyedMutex`| 10.2 |  |  ||
|`cudaExternalSemaphoreHandleTypeKeyedMutexKmt`| 10.2 |  |  ||
|`cudaExternalSemaphoreHandleTypeNvSciSync`| 10.2 |  |  ||
|`cudaExternalSemaphoreHandleTypeOpaqueFd`| 10.0 |  |  ||
|`cudaExternalSemaphoreHandleTypeOpaqueWin32`| 10.0 |  |  ||
|`cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt`| 10.0 |  |  ||
|`cudaExternalSemaphoreSignalParams`| 10.0 |  |  ||
|`cudaExternalSemaphoreSignalSkipNvSciBufMemSync`| 10.2 |  |  ||
|`cudaExternalSemaphoreWaitParams`| 10.0 |  |  ||
|`cudaExternalSemaphoreWaitSkipNvSciBufMemSync`| 10.2 |  |  ||
|`cudaExternalSemaphore_t`| 10.0 |  |  ||
|`cudaFilterModeLinear`|  |  |  |`hipFilterModeLinear`|
|`cudaFilterModePoint`|  |  |  |`hipFilterModePoint`|
|`cudaFormatModeAuto`|  |  |  ||
|`cudaFormatModeForced`|  |  |  ||
|`cudaFuncAttribute`| 9.0 |  |  |`hipFuncAttribute`|
|`cudaFuncAttributeMax`| 9.0 |  |  |`hipFuncAttributeMax`|
|`cudaFuncAttributeMaxDynamicSharedMemorySize`| 9.0 |  |  |`hipFuncAttributeMaxDynamicSharedMemorySize`|
|`cudaFuncAttributePreferredSharedMemoryCarveout`| 9.0 |  |  |`hipFuncAttributePreferredSharedMemoryCarveout`|
|`cudaFuncAttributes`|  |  |  |`hipFuncAttributes`|
|`cudaFuncCache`|  |  |  |`hipFuncCache_t`|
|`cudaFuncCachePreferEqual`|  |  |  |`hipFuncCachePreferEqual`|
|`cudaFuncCachePreferL1`|  |  |  |`hipFuncCachePreferL1`|
|`cudaFuncCachePreferNone`|  |  |  |`hipFuncCachePreferNone`|
|`cudaFuncCachePreferShared`|  |  |  |`hipFuncCachePreferShared`|
|`cudaGLDeviceList`|  |  |  ||
|`cudaGLDeviceListAll`|  |  |  ||
|`cudaGLDeviceListCurrentFrame`|  |  |  ||
|`cudaGLDeviceListNextFrame`|  |  |  ||
|`cudaGLMapFlags`|  |  |  ||
|`cudaGLMapFlagsNone`|  |  |  ||
|`cudaGLMapFlagsReadOnly`|  |  |  ||
|`cudaGLMapFlagsWriteDiscard`|  |  |  ||
|`cudaGraphExecUpdateError`| 10.2 |  |  ||
|`cudaGraphExecUpdateErrorFunctionChanged`| 10.2 |  |  ||
|`cudaGraphExecUpdateErrorNodeTypeChanged`| 10.2 |  |  ||
|`cudaGraphExecUpdateErrorNotSupported`| 10.2 |  |  ||
|`cudaGraphExecUpdateErrorParametersChanged`| 10.2 |  |  ||
|`cudaGraphExecUpdateErrorTopologyChanged`| 10.2 |  |  ||
|`cudaGraphExecUpdateResult`| 10.2 |  |  ||
|`cudaGraphExecUpdateSuccess`| 10.2 |  |  ||
|`cudaGraphExec_t`| 10.0 |  |  ||
|`cudaGraphNodeType`| 10.0 |  |  ||
|`cudaGraphNodeTypeCount`| 10.0 |  |  ||
|`cudaGraphNodeTypeEmpty`| 10.0 |  |  ||
|`cudaGraphNodeTypeEventRecord`| 11.1 |  |  ||
|`cudaGraphNodeTypeGraph`| 10.0 |  |  ||
|`cudaGraphNodeTypeHost`| 10.0 |  |  ||
|`cudaGraphNodeTypeKernel`| 10.0 |  |  ||
|`cudaGraphNodeTypeMemcpy`| 10.0 |  |  ||
|`cudaGraphNodeTypeMemset`| 10.0 |  |  ||
|`cudaGraphNodeTypeWaitEvent`| 11.1 |  |  ||
|`cudaGraphNode_t`| 10.0 |  |  ||
|`cudaGraph_t`| 10.0 |  |  ||
|`cudaGraphicsCubeFace`|  |  |  ||
|`cudaGraphicsCubeFaceNegativeX`|  |  |  ||
|`cudaGraphicsCubeFaceNegativeY`|  |  |  ||
|`cudaGraphicsCubeFaceNegativeZ`|  |  |  ||
|`cudaGraphicsCubeFacePositiveX`|  |  |  ||
|`cudaGraphicsCubeFacePositiveY`|  |  |  ||
|`cudaGraphicsCubeFacePositiveZ`|  |  |  ||
|`cudaGraphicsMapFlags`|  |  |  ||
|`cudaGraphicsMapFlagsNone`|  |  |  ||
|`cudaGraphicsMapFlagsReadOnly`|  |  |  ||
|`cudaGraphicsMapFlagsWriteDiscard`|  |  |  ||
|`cudaGraphicsRegisterFlags`|  |  |  ||
|`cudaGraphicsRegisterFlagsNone`|  |  |  ||
|`cudaGraphicsRegisterFlagsReadOnly`|  |  |  ||
|`cudaGraphicsRegisterFlagsSurfaceLoadStore`|  |  |  ||
|`cudaGraphicsRegisterFlagsTextureGather`|  |  |  ||
|`cudaGraphicsRegisterFlagsWriteDiscard`|  |  |  ||
|`cudaGraphicsResource`|  |  |  ||
|`cudaGraphicsResource_t`|  |  |  ||
|`cudaHostAllocDefault`|  |  |  |`hipHostMallocDefault`|
|`cudaHostAllocMapped`|  |  |  |`hipHostMallocMapped`|
|`cudaHostAllocPortable`|  |  |  |`hipHostMallocPortable`|
|`cudaHostAllocWriteCombined`|  |  |  |`hipHostMallocWriteCombined`|
|`cudaHostFn_t`| 10.0 |  |  ||
|`cudaHostNodeParams`| 10.0 |  |  ||
|`cudaHostRegisterDefault`|  |  |  |`hipHostRegisterDefault`|
|`cudaHostRegisterIoMemory`| 7.5 |  |  |`hipHostRegisterIoMemory`|
|`cudaHostRegisterMapped`|  |  |  |`hipHostRegisterMapped`|
|`cudaHostRegisterPortable`|  |  |  |`hipHostRegisterPortable`|
|`cudaHostRegisterReadOnly`| 11.1 |  |  ||
|`cudaInvalidDeviceId`| 8.0 |  |  |`hipInvalidDeviceId`|
|`cudaIpcEventHandle_st`|  |  |  |`hipIpcEventHandle_st`|
|`cudaIpcEventHandle_t`|  |  |  |`hipIpcEventHandle_t`|
|`cudaIpcMemHandle_st`|  |  |  |`hipIpcMemHandle_st`|
|`cudaIpcMemHandle_t`|  |  |  |`hipIpcMemHandle_t`|
|`cudaIpcMemLazyEnablePeerAccess`|  |  |  |`hipIpcMemLazyEnablePeerAccess`|
|`cudaKernelNodeAttrID`| 11.0 |  |  ||
|`cudaKernelNodeAttrValue`| 11.0 |  |  ||
|`cudaKernelNodeAttributeAccessPolicyWindow`| 11.0 |  |  ||
|`cudaKernelNodeAttributeCooperative`| 11.0 |  |  ||
|`cudaKernelNodeParams`| 10.0 |  |  ||
|`cudaKeyValuePair`|  |  |  ||
|`cudaLaunchParams`| 9.0 |  |  |`hipLaunchParams`|
|`cudaLimit`|  |  |  |`hipLimit_t`|
|`cudaLimitDevRuntimePendingLaunchCount`|  |  |  ||
|`cudaLimitDevRuntimeSyncDepth`|  |  |  ||
|`cudaLimitMallocHeapSize`|  |  |  |`hipLimitMallocHeapSize`|
|`cudaLimitMaxL2FetchGranularity`| 10.0 |  |  ||
|`cudaLimitPersistingL2CacheSize`| 11.0 |  |  ||
|`cudaLimitPrintfFifoSize`|  |  |  ||
|`cudaLimitStackSize`|  |  |  ||
|`cudaMemAdviseSetAccessedBy`| 8.0 |  |  |`hipMemAdviseSetAccessedBy`|
|`cudaMemAdviseSetPreferredLocation`| 8.0 |  |  |`hipMemAdviseSetPreferredLocation`|
|`cudaMemAdviseSetReadMostly`| 8.0 |  |  |`hipMemAdviseSetReadMostly`|
|`cudaMemAdviseUnsetAccessedBy`| 8.0 |  |  |`hipMemAdviseUnsetAccessedBy`|
|`cudaMemAdviseUnsetPreferredLocation`| 8.0 |  |  |`hipMemAdviseUnsetPreferredLocation`|
|`cudaMemAdviseUnsetReadMostly`| 8.0 |  |  |`hipMemAdviseUnsetReadMostly`|
|`cudaMemAttachGlobal`|  |  |  |`hipMemAttachGlobal`|
|`cudaMemAttachHost`|  |  |  |`hipMemAttachHost`|
|`cudaMemAttachSingle`|  |  |  |`hipMemAttachSingle`|
|`cudaMemRangeAttribute`| 8.0 |  |  |`hipMemRangeAttribute`|
|`cudaMemRangeAttributeAccessedBy`| 8.0 |  |  |`hipMemRangeAttributeAccessedBy`|
|`cudaMemRangeAttributeLastPrefetchLocation`| 8.0 |  |  |`hipMemRangeAttributeLastPrefetchLocation`|
|`cudaMemRangeAttributePreferredLocation`| 8.0 |  |  |`hipMemRangeAttributePreferredLocation`|
|`cudaMemRangeAttributeReadMostly`| 8.0 |  |  |`hipMemRangeAttributeReadMostly`|
|`cudaMemcpy3DParms`|  |  |  |`hipMemcpy3DParms`|
|`cudaMemcpy3DPeerParms`|  |  |  ||
|`cudaMemcpyDefault`|  |  |  |`hipMemcpyDefault`|
|`cudaMemcpyDeviceToDevice`|  |  |  |`hipMemcpyDeviceToDevice`|
|`cudaMemcpyDeviceToHost`|  |  |  |`hipMemcpyDeviceToHost`|
|`cudaMemcpyHostToDevice`|  |  |  |`hipMemcpyHostToDevice`|
|`cudaMemcpyHostToHost`|  |  |  |`hipMemcpyHostToHost`|
|`cudaMemcpyKind`|  |  |  |`hipMemcpyKind`|
|`cudaMemoryAdvise`| 8.0 |  |  |`hipMemoryAdvise`|
|`cudaMemoryType`|  |  |  ||
|`cudaMemoryTypeDevice`|  |  |  ||
|`cudaMemoryTypeHost`|  |  |  ||
|`cudaMemoryTypeManaged`| 10.0 |  |  ||
|`cudaMemoryTypeUnregistered`|  |  |  ||
|`cudaMemsetParams`| 10.0 |  |  ||
|`cudaMipmappedArray`|  |  |  |`hipMipmappedArray`|
|`cudaMipmappedArray_const_t`|  |  |  |`hipMipmappedArray_const_t`|
|`cudaMipmappedArray_t`|  |  |  |`hipMipmappedArray_t`|
|`cudaNvSciSyncAttrSignal`| 10.2 |  |  ||
|`cudaNvSciSyncAttrWait`| 10.2 |  |  ||
|`cudaOccupancyDefault`|  |  |  |`hipOccupancyDefault`|
|`cudaOccupancyDisableCachingOverride`|  |  |  ||
|`cudaOutputMode`|  |  |  ||
|`cudaOutputMode_t`|  |  |  ||
|`cudaPitchedPtr`|  |  |  |`hipPitchedPtr`|
|`cudaPointerAttributes`|  |  |  |`hipPointerAttribute_t`|
|`cudaPos`|  |  |  |`hipPos`|
|`cudaReadModeElementType`|  |  |  |`hipReadModeElementType`|
|`cudaReadModeNormalizedFloat`|  |  |  |`hipReadModeNormalizedFloat`|
|`cudaResViewFormatFloat1`|  |  |  |`hipResViewFormatFloat1`|
|`cudaResViewFormatFloat2`|  |  |  |`hipResViewFormatFloat2`|
|`cudaResViewFormatFloat4`|  |  |  |`hipResViewFormatFloat4`|
|`cudaResViewFormatHalf1`|  |  |  |`hipResViewFormatHalf1`|
|`cudaResViewFormatHalf2`|  |  |  |`hipResViewFormatHalf2`|
|`cudaResViewFormatHalf4`|  |  |  |`hipResViewFormatHalf4`|
|`cudaResViewFormatNone`|  |  |  |`hipResViewFormatNone`|
|`cudaResViewFormatSignedBlockCompressed4`|  |  |  |`hipResViewFormatSignedBlockCompressed4`|
|`cudaResViewFormatSignedBlockCompressed5`|  |  |  |`hipResViewFormatSignedBlockCompressed5`|
|`cudaResViewFormatSignedBlockCompressed6H`|  |  |  |`hipResViewFormatSignedBlockCompressed6H`|
|`cudaResViewFormatSignedChar1`|  |  |  |`hipResViewFormatSignedChar1`|
|`cudaResViewFormatSignedChar2`|  |  |  |`hipResViewFormatSignedChar2`|
|`cudaResViewFormatSignedChar4`|  |  |  |`hipResViewFormatSignedChar4`|
|`cudaResViewFormatSignedInt1`|  |  |  |`hipResViewFormatSignedInt1`|
|`cudaResViewFormatSignedInt2`|  |  |  |`hipResViewFormatSignedInt2`|
|`cudaResViewFormatSignedInt4`|  |  |  |`hipResViewFormatSignedInt4`|
|`cudaResViewFormatSignedShort1`|  |  |  |`hipResViewFormatSignedShort1`|
|`cudaResViewFormatSignedShort2`|  |  |  |`hipResViewFormatSignedShort2`|
|`cudaResViewFormatSignedShort4`|  |  |  |`hipResViewFormatSignedShort4`|
|`cudaResViewFormatUnsignedBlockCompressed1`|  |  |  |`hipResViewFormatUnsignedBlockCompressed1`|
|`cudaResViewFormatUnsignedBlockCompressed2`|  |  |  |`hipResViewFormatUnsignedBlockCompressed2`|
|`cudaResViewFormatUnsignedBlockCompressed3`|  |  |  |`hipResViewFormatUnsignedBlockCompressed3`|
|`cudaResViewFormatUnsignedBlockCompressed4`|  |  |  |`hipResViewFormatUnsignedBlockCompressed4`|
|`cudaResViewFormatUnsignedBlockCompressed5`|  |  |  |`hipResViewFormatUnsignedBlockCompressed5`|
|`cudaResViewFormatUnsignedBlockCompressed6H`|  |  |  |`hipResViewFormatUnsignedBlockCompressed6H`|
|`cudaResViewFormatUnsignedBlockCompressed7`|  |  |  |`hipResViewFormatUnsignedBlockCompressed7`|
|`cudaResViewFormatUnsignedChar1`|  |  |  |`hipResViewFormatUnsignedChar1`|
|`cudaResViewFormatUnsignedChar2`|  |  |  |`hipResViewFormatUnsignedChar2`|
|`cudaResViewFormatUnsignedChar4`|  |  |  |`hipResViewFormatUnsignedChar4`|
|`cudaResViewFormatUnsignedInt1`|  |  |  |`hipResViewFormatUnsignedInt1`|
|`cudaResViewFormatUnsignedInt2`|  |  |  |`hipResViewFormatUnsignedInt2`|
|`cudaResViewFormatUnsignedInt4`|  |  |  |`hipResViewFormatUnsignedInt4`|
|`cudaResViewFormatUnsignedShort1`|  |  |  |`hipResViewFormatUnsignedShort1`|
|`cudaResViewFormatUnsignedShort2`|  |  |  |`hipResViewFormatUnsignedShort2`|
|`cudaResViewFormatUnsignedShort4`|  |  |  |`hipResViewFormatUnsignedShort4`|
|`cudaResourceDesc`|  |  |  |`hipResourceDesc`|
|`cudaResourceType`|  |  |  |`hipResourceType`|
|`cudaResourceTypeArray`|  |  |  |`hipResourceTypeArray`|
|`cudaResourceTypeLinear`|  |  |  |`hipResourceTypeLinear`|
|`cudaResourceTypeMipmappedArray`|  |  |  |`hipResourceTypeMipmappedArray`|
|`cudaResourceTypePitch2D`|  |  |  |`hipResourceTypePitch2D`|
|`cudaResourceViewDesc`|  |  |  |`hipResourceViewDesc`|
|`cudaResourceViewFormat`|  |  |  |`hipResourceViewFormat`|
|`cudaSharedCarveout`| 9.0 |  |  ||
|`cudaSharedMemBankSizeDefault`|  |  |  |`hipSharedMemBankSizeDefault`|
|`cudaSharedMemBankSizeEightByte`|  |  |  |`hipSharedMemBankSizeEightByte`|
|`cudaSharedMemBankSizeFourByte`|  |  |  |`hipSharedMemBankSizeFourByte`|
|`cudaSharedMemConfig`|  |  |  |`hipSharedMemConfig`|
|`cudaSharedmemCarveoutDefault`| 9.0 |  |  ||
|`cudaSharedmemCarveoutMaxL1`| 9.0 |  |  ||
|`cudaSharedmemCarveoutMaxShared`| 9.0 |  |  ||
|`cudaStreamAttrID`| 11.0 |  |  ||
|`cudaStreamAttrValue`| 11.0 |  |  ||
|`cudaStreamAttributeAccessPolicyWindow`| 11.0 |  |  ||
|`cudaStreamAttributeSynchronizationPolicy`| 11.0 |  |  ||
|`cudaStreamCallback_t`|  |  |  |`hipStreamCallback_t`|
|`cudaStreamCaptureMode`| 10.1 |  |  ||
|`cudaStreamCaptureModeGlobal`| 10.1 |  |  ||
|`cudaStreamCaptureModeRelaxed`| 10.1 |  |  ||
|`cudaStreamCaptureModeThreadLocal`| 10.1 |  |  ||
|`cudaStreamCaptureStatus`| 10.0 |  |  ||
|`cudaStreamCaptureStatusActive`| 10.0 |  |  ||
|`cudaStreamCaptureStatusInvalidated`| 10.0 |  |  ||
|`cudaStreamCaptureStatusNone`| 10.0 |  |  ||
|`cudaStreamDefault`|  |  |  |`hipStreamDefault`|
|`cudaStreamLegacy`|  |  |  ||
|`cudaStreamNonBlocking`|  |  |  |`hipStreamNonBlocking`|
|`cudaStreamPerThread`|  |  |  ||
|`cudaStream_t`|  |  |  |`hipStream_t`|
|`cudaSuccess`|  |  |  |`hipSuccess`|
|`cudaSurfaceBoundaryMode`|  |  |  |`hipSurfaceBoundaryMode`|
|`cudaSurfaceFormatMode`|  |  |  ||
|`cudaSurfaceObject_t`|  |  |  |`hipSurfaceObject_t`|
|`cudaSyncPolicyAuto`| 11.0 |  |  ||
|`cudaSyncPolicyBlockingSync`| 11.0 |  |  ||
|`cudaSyncPolicySpin`| 11.0 |  |  ||
|`cudaSyncPolicyYield`| 11.0 |  |  ||
|`cudaSynchronizationPolicy`| 11.0 |  |  ||
|`cudaTextureAddressMode`|  |  |  |`hipTextureAddressMode`|
|`cudaTextureDesc`|  |  |  |`hipTextureDesc`|
|`cudaTextureFilterMode`|  |  |  |`hipTextureFilterMode`|
|`cudaTextureObject_t`|  |  |  |`hipTextureObject_t`|
|`cudaTextureReadMode`|  |  |  |`hipTextureReadMode`|
|`cudaTextureType1D`|  |  |  |`hipTextureType1D`|
|`cudaTextureType1DLayered`|  |  |  |`hipTextureType1DLayered`|
|`cudaTextureType2D`|  |  |  |`hipTextureType2D`|
|`cudaTextureType2DLayered`|  |  |  |`hipTextureType2DLayered`|
|`cudaTextureType3D`|  |  |  |`hipTextureType3D`|
|`cudaTextureTypeCubemap`|  |  |  |`hipTextureTypeCubemap`|
|`cudaTextureTypeCubemapLayered`|  |  |  |`hipTextureTypeCubemapLayered`|
|`cudaUUID_t`|  |  |  ||
|`libraryPropertyType`| 8.0 |  |  ||
|`libraryPropertyType_t`| 8.0 |  |  ||
|`surfaceReference`|  |  |  ||

## **35. Execution Control [REMOVED]**

| **CUDA** | **A** | **D** | **R** | **HIP** |
|:--|:-:|:-:|:-:|:--|
|`cudaConfigureCall`|  |  | 10.1 |`hipConfigureCall`|
|`cudaLaunch`|  |  | 10.1 |`hipLaunchByPtr`|
|`cudaSetupArgument`|  |  | 10.1 |`hipSetupArgument`|


\* A - Added, D - Deprecated, R - Removed