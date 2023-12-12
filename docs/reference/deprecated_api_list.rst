**********************************************************************************************
HIP deprecated runtime API functions
**********************************************************************************************

Several of our API functions have been flagged for deprecation. Using the following functions results in
errors and unexpected results, so we encourage you to update your code accordingly.

Context management
============================================================

CUDA supports cuCtx API, which is the driver API that defines "Context" and "Devices" as separate
entities. Context contains a single device, and a device can theoretically have multiple contexts. HIP
initially added limited support for these APIs in order to facilitate porting from existing driver codes.
These APIs are now marked as deprecated because there are better alternate interfaces (such as
``hipSetDevice`` or the stream API) to achieve these functions.

* ``hipCtxCreate``
* ``hipCtxDestroy``
* ``hipCtxPopCurrent``
* ``hipCtxPushCurrent``
* ``hipCtxSetCurrent``
* ``hipCtxGetCurrent``
* ``hipCtxGetDevice``
* ``hipCtxGetApiVersion``
* ``hipCtxGetCacheConfig``
* ``hipCtxSetCacheConfig``
* ``hipCtxSetSharedMemConfig``
* ``hipCtxGetSharedMemConfig``
* ``hipCtxSynchronize``
* ``hipCtxGetFlags``
* ``hipCtxEnablePeerAccess``
* ``hipCtxDisablePeerAccess``

Memory management
============================================================

* ``hipMallocHost`` (replaced with ``hipHostMalloc``)
* ``hipMemAllocHost`` (replaced with ``hipHostMalloc``)
* ``hipHostAlloc`` (replaced with ``hipHostMalloc``)
* ``hipFreeHost`` (replaced with ``hipHostFree``)
* ``hipMemcpyToArray``
* ``hipMemcpyFromArray``

Profiler control
============================================================

* ``hipProfilerStart`` (use roctracer/rocTX)
* ``hipProfilerStop`` (use roctracer/rocTX)


Texture management
============================================================

* ``hipGetTextureReference``
* ``hipTexRefSetAddressMode``
* ``hipTexRefSetArray``
* ``hipTexRefSetFilterMode``
* ``hipTexRefSetFlags``
* ``hipTexRefSetFormat``
* ``hipBindTexture``
* ``hipBindTexture2D``
* ``hipBindTextureToArray``
* ``hipGetTextureAlignmentOffset``
* ``hipUnbindTexture``
* ``hipTexRefGetAddress``
* ``hipTexRefGetAddressMode``
* ``hipTexRefGetFilterMode``
* ``hipTexRefGetFlags``
* ``hipTexRefGetFormat``
* ``hipTexRefGetMaxAnisotropy``
* ``hipTexRefGetMipmapFilterMode``
* ``hipTexRefGetMipmapLevelBias``
* ``hipTexRefGetMipmapLevelClamp``
* ``hipTexRefGetMipMappedArray``
* ``hipTexRefSetAddress``
* ``hipTexRefSetAddress2D``
* ``hipTexRefSetMaxAnisotropy``
* ``hipTexRefSetBorderColor``
* ``hipTexRefSetMipmapFilterMode``
* ``hipTexRefSetMipmapLevelBias``
* ``hipTexRefSetMipmapLevelClamp``
* ``hipTexRefSetMipmappedArray``
