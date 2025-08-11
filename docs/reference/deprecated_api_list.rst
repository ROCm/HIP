.. meta::
   :description: HIP deprecated runtime API functions.
   :keywords: AMD, ROCm, HIP, deprecated, API

**********************************************************************************************
HIP deprecated runtime API functions
**********************************************************************************************

Several of our API functions have been flagged for deprecation. Using the
following functions results in errors and unexpected results, so we encourage
you to update your code accordingly.

Deprecated since ROCm 6.1.0
============================================================

Deprecated texture management functions.

.. list-table::
   :widths: 40
   :header-rows: 1
   :align: left

   * - function
   * - :cpp:func:`hipTexRefGetBorderColor`
   * - :cpp:func:`hipTexRefGetArray`

Deprecated since ROCm 5.7.0
============================================================

Deprecated texture management functions.

.. list-table::
   :widths: 40
   :header-rows: 1
   :align: left

   * - function
   * - :cpp:func:`hipBindTextureToMipmappedArray`

Deprecated since ROCm 5.3.0
============================================================

Deprecated texture management functions.

.. list-table::
   :widths: 40
   :header-rows: 1
   :align: left

   * - function
   * - :cpp:func:`hipGetTextureReference`
   * - :cpp:func:`hipTexRefSetAddressMode`
   * - :cpp:func:`hipTexRefSetArray`
   * - :cpp:func:`hipTexRefSetFlags`
   * - :cpp:func:`hipTexRefSetFilterMode`
   * - :cpp:func:`hipTexRefSetFormat`
   * - :cpp:func:`hipTexRefSetMipmapFilterMode`
   * - :cpp:func:`hipTexRefSetMipmapLevelBias`
   * - :cpp:func:`hipTexRefSetMipmapLevelClamp`
   * - :cpp:func:`hipTexRefSetMipmappedArray`

Deprecated since ROCm 4.3.0
============================================================

Deprecated texture management functions.

.. list-table::
   :widths: 40
   :header-rows: 1
   :align: left

   * - function
   * - :cpp:func:`hipTexRefGetAddress`
   * - :cpp:func:`hipTexRefGetAddressMode`
   * - :cpp:func:`hipTexRefGetFilterMode`
   * - :cpp:func:`hipTexRefGetFlags`
   * - :cpp:func:`hipTexRefGetFormat`
   * - :cpp:func:`hipTexRefGetMaxAnisotropy`
   * - :cpp:func:`hipTexRefGetMipmapFilterMode`
   * - :cpp:func:`hipTexRefGetMipmapLevelBias`
   * - :cpp:func:`hipTexRefGetMipmapLevelClamp`
   * - :cpp:func:`hipTexRefGetMipMappedArray`
   * - :cpp:func:`hipTexRefSetAddress`
   * - :cpp:func:`hipTexRefSetAddress2D`
   * - :cpp:func:`hipTexRefSetBorderColor`
   * - :cpp:func:`hipTexRefSetMaxAnisotropy`

Deprecated since ROCm 3.8.0
============================================================

Deprecated memory management and texture management functions.

.. list-table::
   :widths: 40
   :header-rows: 1
   :align: left

   * - function
   * - :cpp:func:`hipBindTexture`
   * - :cpp:func:`hipBindTexture2D`
   * - :cpp:func:`hipBindTextureToArray`
   * - :cpp:func:`hipGetTextureAlignmentOffset`
   * - :cpp:func:`hipUnbindTexture`
   * - :cpp:func:`hipMemcpyToArray`
   * - :cpp:func:`hipMemcpyFromArray`

Deprecated since ROCm 3.1.0
============================================================

Deprecated memory management functions.

.. list-table::
   :widths: 40, 60
   :header-rows: 1
   :align: left

   * - function
     -
   * - :cpp:func:`hipMallocHost`
     - replaced with :cpp:func:`hipHostAlloc`
   * - :cpp:func:`hipMemAllocHost`
     - replaced with :cpp:func:`hipHostAlloc`

Deprecated since ROCm 3.0.0
============================================================

The ``hipProfilerStart`` and ``hipProfilerStop`` functions are deprecated.
Instead, you can use ``roctracer`` or ``rocTX`` for profiling which provide more
flexibility and detailed profiling capabilities.

.. list-table::
   :widths: 40
   :header-rows: 1
   :align: left

   * - function
   * - :cpp:func:`hipProfilerStart`
   * - :cpp:func:`hipProfilerStop`

Deprecated since ROCm 1.9.0
============================================================

CUDA supports cuCtx API, which is the driver API that defines "Context" and
"Devices" as separate entities. Context contains a single device, and a device
can theoretically have multiple contexts. HIP initially added limited support
for context APIs in order to facilitate porting from existing driver codes. These
APIs are now marked as deprecated because there are better alternate interfaces
(such as ``hipSetDevice`` or the stream API) to achieve these functions.

.. list-table::
   :widths: 40
   :header-rows: 1
   :align: left

   * - function
   * -  :cpp:func:`hipCtxCreate`
   * -  :cpp:func:`hipCtxDestroy`
   * -  :cpp:func:`hipCtxPopCurrent`
   * -  :cpp:func:`hipCtxPushCurrent`
   * -  :cpp:func:`hipCtxSetCurrent`
   * -  :cpp:func:`hipCtxGetCurrent`
   * -  :cpp:func:`hipCtxGetDevice`
   * -  :cpp:func:`hipCtxGetApiVersion`
   * -  :cpp:func:`hipCtxGetCacheConfig`
   * -  :cpp:func:`hipCtxSetCacheConfig`
   * -  :cpp:func:`hipCtxSetSharedMemConfig`
   * -  :cpp:func:`hipCtxGetSharedMemConfig`
   * -  :cpp:func:`hipCtxSynchronize`
   * -  :cpp:func:`hipCtxGetFlags`
   * -  :cpp:func:`hipCtxEnablePeerAccess`
   * -  :cpp:func:`hipCtxDisablePeerAccess`
   * -  :cpp:func:`hipDevicePrimaryCtxGetState`
   * -  :cpp:func:`hipDevicePrimaryCtxRelease`
   * -  :cpp:func:`hipDevicePrimaryCtxRetain`
   * -  :cpp:func:`hipDevicePrimaryCtxReset`
   * -  :cpp:func:`hipDevicePrimaryCtxSetFlags`
