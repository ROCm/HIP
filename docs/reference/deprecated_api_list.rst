.. meta::
   :description: HIP deprecated runtime API functions.
   :keywords: AMD, ROCm, HIP, deprecated, API

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
* ``hipDevicePrimaryCtxGetState``
* ``hipDevicePrimaryCtxRelease``
* ``hipDevicePrimaryCtxRetain``
* ``hipDevicePrimaryCtxReset``
* ``hipDevicePrimaryCtxSetFlags``

Memory management
============================================================

* ``hipMallocHost`` (replaced with ``hipHostMalloc``)
* ``hipMemAllocHost`` (replaced with ``hipHostMalloc``)
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
* ``hipTexRefGetBorderColor``
* ``hipTexRefGetArray``
* ``hipBindTexture``
* ``hipBindTexture2D``
* ``hipBindTextureToArray``
* ``hipGetTextureAlignmentOffset``
* ``hipUnbindTexture``
* ``hipBindTextureToMipmappedArray``
