.. meta::
  :description: This topic discusses the changes introduced in HIP 7.0
  :keywords: AMD, ROCm, HIP, HIP changes, CUDA, C++ language extensions

.. _compatibility-changes:

*******************************************************************************
HIP API 7.0 changes
*******************************************************************************

To improve code portability between AMD and NVIDIA GPU programming models, changes were made to the HIP API in ROCm 7.0 to simplify cross-platform programming. These changes align HIP C++ even more closely with NVIDIA CUDA. These changes are incompatible with prior releases, and might require recompiling existing HIP applications for use with ROCm 7.0, or editing and recompiling code in some cases. In the best case, the change requires no modification of existing applications. These changes were made available in a preview release based on ROCm 6.4.1 to help you prepare.

Behavior changes in HIP Runtime API
===================================

Update ``hipGetLastError``
--------------------------

Prior to the 7.0 release of the HIP API, :cpp:func:`hipGetLastError` was not fully compliant with CUDA's behavior. The purpose of this change is to have ``hipGetLastError`` return the last actual error caught in the current thread during the application execution. Neither ``hipSuccess`` nor ``hipErrorNotReady`` is considered an error. Take the following code as an example:

.. code:: cpp

  1: hipError_t err = hipMalloc(...); // returns hipOutOfMemory
  2: err = hipSetDevice(0); // returns hipSuccess
  3: err = hipGetLastError();

The prior behavior was for ``hipGetLastError`` at line 3 to return ``hipSuccess`` from line 2. In the 7.0 release, the value of ``err`` at line 3 is ``hipOutOfMemory`` which is the error returned in Line 1, rather than simply the result returned in line 2. This matches CUDA behavior.

You can still use the prior functionality by using the ``hipExtGetLastError`` function. Notice that the function begins with ``hipExt`` which denotes a function call that is unique to HIP, without correlation to CUDA. This function was introduced with the 6.0 release.

Cooperative groups changes
--------------------------

For :cpp:func:`hipLaunchCooperativeKernelMultiDevice` function, HIP now includes additional input parameter validation checks.

* If the input launch stream is a NULLPTR or it is ``hipStreamLegacy``, the function now returns ``hipErrorInvalidResourceHandle``.
* If the stream capturing is active, the function returns the error code ``hipErrorStreamCaptureUnsupported``.
* If the stream capture status is invalidated, the function returns the error ``hipErrorStreamCaptureInvalidated``.

The :cpp:func:`hipLaunchCooperativeKernel` function now checks the input stream handle. If it's invalid, the returned error is changed to ``hipErrorInvalidHandle`` from ``hipErrorContextIsDestroyed``.

Update ``hipPointerGetAttributes``
----------------------------------

:cpp:func:`hipPointerGetAttributes` now matches the functionality of ``cudaPointerGetAttributes`` which changed in CUDA 11. If a NULL host or attribute pointer is passed as input parameter, ``hipPointerGetAttributes`` now returns ``hipSuccess`` instead of the error code ``hipErrorInvalidValue``.

Any application which is expecting the API to return an error instead of success could be impacted and a code change may need to handle the error properly.

Update ``hipFree``
------------------

:cpp:func:`hipFree` previously had an implicit wait for synchronization purpose which is applicable for all memory allocations. This wait has been disabled in the HIP 7.0 runtime for allocations made with ``hipMallocAsync`` and ``hipMallocFromPoolAsync`` to match the behavior of CUDA API ``cudaFree``

Update ``hipFreeAsync``
-----------------------

The API returns ``hipSuccess`` when the input pointer is NULL, instead of ``hipErrorInvalidValue``, to be consistent with :cpp:func:`hipFree`.

Exceptions effect during kernel execution changes
-------------------------------------------------

Exceptions that occur during kernel execution will no longer abort the process, but will instead return an error, unless core dumping is enabled.

HIP runtime compiler (hipRTC) changes
=====================================

Runtime compilation for HIP is available through the ``hipRTC`` library as described in :ref:`hip_runtime_compiler_how-to`. The library grew organically within the main HIP runtime code. However, segregation of the ``hipRTC`` code is now needed to ensure better compatibility and easier code portability.

Removal of ``hipRTC`` symbols from HIP Runtime Library
------------------------------------------------------

``hipRTC`` has been an independent library since the 6.0 release, but the ``hipRTC`` symbols were still available in the HIP runtime library. Starting with the 7.0 release ``hipRTC`` is no longer included in the HIP runtime, and any application using ``hipRTC`` APIs should link explicitly with the ``hipRTC`` library.

This change makes the usage of ``hipRTC`` library on Linux the same as on Windows and matches the behavior of CUDA ``nvRTC``.

``hipRTC`` compilation
----------------------

The device code compilation via ``hipRTC`` now uses namespace ``__hip_internal``, instead of the standard headers ``std``, to avoid namespace collision. These changes are made in the HIP header files.

No code change is required in any application, but rebuilding is necessary.

Removal of datatypes from ``hipRTC``
------------------------------------

In ``hipRTC``, datatype definitions such as ``int64_t``, ``uint64_t``, ``int32_t``, and ``uint32_t`` could result in conflicts in some applications, as they use their own definitions for these types. ``nvRTC`` doesn't define these datatypes either.
These datatypes are removed and replaced by HIP internal datatypes prefixed with ``__hip``, for example, ``__hip_int64_t``.

Any application relying on HIP internal datatypes during ``hipRTC`` compilation might be affected.
These changes have no impact on any application if it compiles as expected using ``nvRTC``.

HIP header clean up
===================

HIP header files previously included unnecessary Standard Template Libraries (STL) headers.
With the 7.0 release, unnecessary STL headers are no longer included, and only the required STL headers
are included. 

Applications relying on HIP runtime header files might need to be updated to include STL header
files that have been removed in 7.0.

API signature and struct changes
================================

API signature changes
---------------------

Signatures in some APIs have been modified to match corresponding CUDA APIs, as described below.

The RTC method definition is changed in the following ``hipRTC`` APIs:

* :cpp:func:`hiprtcCreateProgram`
* :cpp:func:`hiprtcCompileProgram`

In these APIs, the input parameter type changes from ``const char**`` to ``const char* const*``.

In addition, the following APIs have signature changes:

* :cpp:func:`hipMemcpyHtoD`, the type of the second argument pointer changes from ``const void*`` to ``void*``.
* :cpp:func:`hipCtxGetApiVersion`, the type of second argument is changed from ``int*`` to ``unsigned int*``.

These signature changes do not require code modifications but do require rebuilding the application.

Deprecated struct ``HIP_MEMSET_NODE_PARAMS`` 
--------------------------------------------

The deprecated structure ``HIP_MEMSET_NODE_PARAMS`` is removed.
You can use the definition :cpp:struct:`hipMemsetParams` instead, as input parameter, while using these two APIs:

* :cpp:func:`hipDrvGraphAddMemsetNode`
* :cpp:func:`hipDrvGraphExecMemsetNodeSetParams`

``hipMemsetParams`` struct change
---------------------------------

The struct :cpp:struct:`hipMemsetParams` is updated to be compatible with CUDA.
The change is from the old struct definition shown below:

.. code:: cpp

  typedef struct hipMemsetParams {
    void* dst;
    unsigned int elementSize;
    size_t height;
    size_t pitch;
    unsigned int value;
    size_t width;
  } hipMemsetParams;

To the new struct definition as follows:

.. code:: cpp

  typedef struct hipMemsetParams {
    void* dst;
    size_t pitch;
    unsigned int value;
    unsigned int elementSize;
    size_t width;
    size_t height;
  } hipMemsetParams;

No code change is required in any application using this structure, but rebuilding is necessary.

HIP vector constructor change
-----------------------------

Changes have been made to HIP vector constructors for ``hipComplex`` initialization to generate values in alignment with CUDA. The affected constructors are small vector types such as ``float2`` and ``int4`` for example. If your code previously relied on a single value to initialize all components within a vector or complex type, you might need to update your code. Otherwise, rebuilding the application is necessary but no code change is required in any application using these constructors.

Stream capture updates
======================

Restrict stream capture modes
-----------------------------

Stream capture mode has been restricted in the following APIs to relaxed (``hipStreamCaptureModeRelaxed``) mode:

* :cpp:func:`hipMallocManaged`
* :cpp:func:`hipMemAdvise`

These APIs are allowed only in relaxed stream capture mode. If the functions are used with stream capture, the HIP runtime the will return ``hipErrorStreamCaptureUnsupported`` on unsupported stream capture modes.

Check stream capture mode
-------------------------

The following APIs will check the stream capture mode and return error codes to match the behavior of CUDA. No impact if stream capture is working correctly on CUDA. Otherwise, the application would need to modify the graph being captured.

* :cpp:func:`hipLaunchCooperativeKernelMultiDevice` - Returns error code while stream capture status is active. The usage is restricted during stream capture
* :cpp:func:`hipEventQuery` - Returns an error ``hipErrorStreamCaptureUnsupported`` in global capture mode
* :cpp:func:`hipStreamAddCallback` - The stream capture behavior is updated. The function now checks if any of the blocking streams are capturing. If so, it returns an error and invalidates all capturing streams. The usage of this API is restricted during stream capture to match CUDA.

Stream capture error return 
---------------------------

During stream capture, the following HIP APIs return the ``hipErrorStreamCaptureUnsupported`` error on the HIP runtime, but not always ``hipSuccess``, to match behavior with CUDA.

* :cpp:func:`hipDeviceSetMemPool`
* :cpp:func:`hipMemPoolCreate`
* :cpp:func:`hipMemPoolDestroy`
* :cpp:func:`hipDeviceSetSharedMemConfig`
* :cpp:func:`hipDeviceSetCacheConfig`
* :cpp:func:`hipMemcpyWithStream`

The usage of these APIs is restricted during stream capture. No impact if stream capture is working fine on CUDA.

Error code changes
==================

The following HIP APIs have been updated to return new or additional error codes to match the corresponding
CUDA APIs. Most existing applications just check if ``hipSuccess`` is returned and no change is needed.
However, if an application checks for a specific error code, the application code may need to be updated
to match/handle the new error code accordingly.

Module management related APIs
------------------------------

Kernel launch APIs
^^^^^^^^^^^^^^^^^^

The following APIs have updated implementations:

* :cpp:func:`hipModuleLaunchKernel`
* :cpp:func:`hipExtModuleLaunchKernel`
* :cpp:func:`hipExtLaunchKernel`
* :cpp:func:`hipDrvLaunchKernelEx`
* :cpp:func:`hipLaunchKernel`
* :cpp:func:`hipLaunchKernelExC`

More conditional checks are added in the API implementation, and the return errors are added or changed in the following scenarios:

* If the input stream handle is invalid, the returned error is changed to ``hipErrorContextIsDestroyed`` from ``hipErrorInvalidValue``
* Adds a grid dimension check, if any input global work size dimension is zero, returns ``hipErrorInvalidValue``
* Adds extra shared memory size check, if exceeds the size limit, returns ``hipErrorInvalidValue``
* If the total number of threads per block exceeds the maximum work group limit during a kernel launch, the return value is changed to ``hipErrorInvalidConfiguration`` from ``hipErrorInvalidValue``

``hipModuleLaunchCooperativeKernel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conditions are added in the API implementation of :cpp:func:`hipModuleLaunchCooperativeKernel`, and the returned errors are added in the following scenarios:

* If the input stream is invalid, returns ``hipErrorContextIsDestroyed``, instead of ``hipErrorInvalidValue``
* If any grid dimension or block dimension is zero, returns ``hipErrorInvalidValue``
* If any grid dimension exceeds the maximum dimension limit, or work group size exceeds the maximum size, returns ``hipErrorInvalidConfiguration`` , instead of ``hipErrorInvalidValue`` 
* If shared memory size in bytes exceeds the device local memory size per CU, returns ``hipErrorCooperativeLaunchTooLarge``

``hipModuleLoad``
^^^^^^^^^^^^^^^^^^

The API updates the negative return of :cpp:func:`hipModuleLoad` to match the CUDA behavior. In cases where the file name exists but the file size is 0, the function returns ``hipErrorInvalidImage`` instead of ``hipErrorInvalidValue``.

Texture management related APIs
-------------------------------

The following APIs have updated the return codes to match the CUDA behavior:

* :cpp:func:`hipTexObjectCreate`, supports zero width and height for 2D image. If either width or height are zero the function will not return ``false``.
* :cpp:func:`hipBindTexture2D`, adds extra check, if pointer for texture reference or device is NULL, returns ``hipErrorNotFound``.
* :cpp:func:`hipBindTextureToArray`, if any NULL pointer is input for texture object, resource descriptor, or texture descriptor, returns error ``hipErrorInvalidChannelDescriptor``, instead of ``hipErrorInvalidValue``.
* :cpp:func:`hipGetTextureAlignmentOffset`, adds a return code ``hipErrorInvalidTexture`` when the texture reference pointer is NULL.

Cooperative group related APIs
-------------------------------

``hipLaunchCooperativeKernelMultiDevice``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validations are added to the API implementation of :cpp:func:`hipLaunchCooperativeKernelMultiDevice`, as follows:

* If input launch stream is NULLPTR or it is ``hipStreamLegacy``, returns ``hipErrorInvalidResourceHandle``.
* If the stream capturing is active, returns the error ``hipErrorStreamCaptureUnsupported``.
* If the stream capture status is invalidated, returns the error ``hipErrorStreamCaptureInvalidated``
* If the total number of threads per block exceeds the maximum work group limit during a kernel launch, the return value is changed to ``hipErrorInvalidConfiguration``  from ``hipErrorInvalidValue``.

``hipLaunchCooperativeKernel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validation are added to the API implementation of :cpp:func:`hipLaunchCooperativeKernel`, as follows:

* If the input stream handle is invalid, the returned error is changed to ``hipErrorInvalidHandle`` from ``hipErrorContextIsDestroyed``.
* If the total number of threads per block exceeds the maximum work group limit during a kernel launch, the return value is changed to ``hipErrorInvalidConfiguration`` from ``hipErrorInvalidValue`` .

Invalid stream input parameter handling matches CUDA
====================================================

In order to match the CUDA runtime behavior more closely, HIP APIs with streams passed as input parameters no longer check the stream validity. Prior to the 7.0 release, the HIP runtime returns an error code ``hipErrorContextIsDestroyed`` if the stream is invalid. In CUDA 12 and later, the equivalent behavior is to raise a segmentation fault. With HIP 7.0, the HIP runtime matches CUDA by causing a segmentation fault. The list of APIs impacted by this change are as follows:

* Stream management related APIs

  * :cpp:func:`hipStreamGetCaptureInfo`
  * :cpp:func:`hipStreamGetPriority`
  * :cpp:func:`hipStreamGetFlags`
  * :cpp:func:`hipStreamDestroy`
  * :cpp:func:`hipStreamAddCallback`
  * :cpp:func:`hipStreamQuery`
  * :cpp:func:`hipLaunchHostFunc`

* Graph management related APIs

  * :cpp:func:`hipGraphUpload`
  * :cpp:func:`hipGraphLaunch`
  * :cpp:func:`hipStreamBeginCaptureToGraph`
  * :cpp:func:`hipStreamBeginCapture`
  * :cpp:func:`hipStreamIsCapturing`
  * :cpp:func:`hipStreamGetCaptureInfo`
  * :cpp:func:`hipGraphInstantiateWithParams`

* Memory management related APIs

  * :cpp:func:`hipMemcpyPeerAsync`
  * :cpp:func:`hipMallocFromPoolAsync`
  * :cpp:func:`hipFreeAsync`
  * :cpp:func:`hipMallocAsync`
  * :cpp:func:`hipMemcpyAsync`
  * :cpp:func:`hipMemcpyToSymbolAsync`
  * :cpp:func:`hipStreamAttachMemAsync`
  * :cpp:func:`hipMemPrefetchAsync`
  * :cpp:func:`hipDrvMemcpy3D`
  * :cpp:func:`hipDrvMemcpy3DAsync`
  * :cpp:func:`hipDrvMemcpy2DUnaligned`
  * :cpp:func:`hipMemcpyParam2D`
  * :cpp:func:`hipMemcpyParam2DAsync`
  * :cpp:func:`hipMemcpy2DArrayToArray`
  * :cpp:func:`hipMemcpy2D`
  * :cpp:func:`hipMemcpy2DAsync`
  * :cpp:func:`hipDrvMemcpy2DUnaligned`
  * :cpp:func:`hipMemcpy3D`

* Event management related APIs

  * :cpp:func:`hipEventRecord`
  * :cpp:func:`hipEventRecordWithFlags`

Developers porting CUDA code to HIP no longer need to modify their error handling code. However,
if you have come to expect the HIP runtime to return the error code ``hipErrorContextIsDestroyed``,
you might need to adjust your code.

warpSize Change
===============

To match the CUDA specification, ``warpSize`` is no longer a ``constexpr``.
In general, this should be a transparent change. However, if an application was using ``warpSize``
as a compile-time constant, it will have to be updated to handle the new definition.
For more information, see `warpSize <./how-to/hip_cpp_language_extensions.html#warpsize>`_
in :doc:`./how-to/hip_cpp_language_extensions`.
