.. meta::
    :description: HIP error codes reference
    :keywords: AMD, HIP, error codes, error, reference

.. _hip_error_codes:

********************************************************************************
HIP error codes
********************************************************************************

This page lists all HIP runtime error codes and their descriptions. These error codes are
returned by HIP API functions to indicate various runtime conditions and errors.

For more details, see :ref:`Error handling functions <error_handling_reference>`.

.. _basic_runtime_errors:

Basic Runtime Errors
====================

.. list-table::
    :header-rows: 1
    :widths: 30 10 60

    * - Error Code
      - Value
      - Description

    * - :term:`hipSuccess`
      - ``0``
      - No error

    * - :term:`hipErrorUnknown`
      - ``999``
      - Unknown error

    * - :term:`hipErrorNotReady`
      - ``600``
      - Device not ready

    * - :term:`hipErrorIllegalState`
      - ``401``
      - The operation cannot be performed in the present state

    * - :term:`hipErrorNotSupported`
      - ``801``
      - Operation not supported

    * - :term:`hipErrorTbd`
      - ``1054``
      - To be determined/implemented

.. glossary::

  hipSuccess

    No error. Operation completed successfully. This is returned when a HIP function completes
    without any errors and indicates normal execution.

  hipErrorUnknown

    Unknown error. This is a general error code returned when no other error code is applicable or
    when the specific error condition cannot be determined. This may indicate an unexpected
    internal error in the HIP runtime or driver.

  hipErrorNotReady

    Device not ready. This error occurs when asynchronous operations have not completed.
    Common scenarios include:

    * Attempting to access results of an asynchronous operation that is still in progress
    * Querying the status of a device that is still processing commands
    * Attempting to synchronize with an event that hasn't occurred yet

  hipErrorIllegalState

    The operation cannot be performed in the present state. This error occurs when a valid operation
    is attempted at an inappropriate time or when the system is in a state that doesn't allow the
    requested action. Common scenarios include:

    * Attempting to modify resources that are in use by an active operation
    * Calling functions in an incorrect sequence
    * State machine violations in the HIP runtime
    * Attempting operations on a device that is in an error state
    * Trying to change configurations that can only be set during initialization
    * Calling APIs in the wrong order for multi-step operations

  hipErrorNotSupported

    Operation not supported. This error indicates that the requested operation is not supported by the
    current hardware, driver, or HIP implementation.

  hipErrorTbd

    To be determined/implemented. This is a placeholder error code for functionality that is planned
    but not yet fully implemented. It indicates that:

    * The feature or API may be documented but not fully functional
    * The error handling for a particular edge case is not yet defined
    * The functionality is under development and will be available in future releases

    If this error is encountered, it generally means the API or feature is not fully supported in the
    current version.

.. _memory_management_errors:

Memory Management Errors
========================

.. list-table::
    :header-rows: 1
    :widths: 30 10 60

    * - Error Code
      - Value
      - Description

    * - :term:`hipErrorOutOfMemory`
      - ``2``
      - Out of memory

    * - :term:`hipErrorInvalidDevicePointer`
      - ``17``
      - Invalid device pointer

    * - :term:`hipErrorHostMemoryAlreadyRegistered`
      - ``712``
      - Part or all of the requested memory range is already mapped

    * - :term:`hipErrorHostMemoryNotRegistered`
      - ``713``
      - Pointer does not correspond to a registered memory region

    * - :term:`hipErrorInvalidMemcpyDirection`
      - ``21``
      - Invalid copy direction for memcpy

    * - :term:`hipErrorIllegalAddress`
      - ``700``
      - An illegal memory access was encountered

    * - :term:`hipErrorRuntimeMemory`
      - ``1052``
      - Runtime memory call returned error

    * - :term:`hipErrorInvalidChannelDescriptor`
      - ``911``
      - Input for texture object, resource descriptor, or texture descriptor is a NULL pointer or invalid

    * - :term:`hipErrorInvalidTexture`
      - ``912``
      - Texture reference pointer is NULL or invalid

.. glossary::

  hipErrorOutOfMemory

    Out of memory. This error occurs when the HIP runtime cannot allocate enough memory to perform the
    requested operation. Common scenarios include:

    * Device memory exhaustion during :cpp:func:`hipMalloc()` or similar allocation functions
    * Allocating more memory than is available on the device
    * Fragmentation of device memory preventing allocation of a contiguous block
    * Multiple concurrent allocations exceeding available memory

  hipErrorInvalidDevicePointer

    Invalid device pointer. This error occurs when:

    * Using a host pointer where a device pointer is expected
    * Using an unallocated device pointer
    * Using a device pointer that has been freed
    * Using a device pointer from a different context

  hipErrorHostMemoryAlreadyRegistered

    Part or all of the requested memory range is already mapped. This error occurs when attempting to
    register host memory that has already been registered. Common scenarios include:

    * Calling :cpp:func:`hipHostRegister()` on a memory region that was previously registered
    * Overlapping memory ranges where part of the new range is already registered
    * Multiple registration attempts of the same pointer in different parts of the application
    * Attempting to register memory that was allocated with :cpp:func:`hipHostMalloc()` (which is already registered)

    This error is distinct from general allocation errors as it specifically deals with the
    page-locking/registration of host memory for faster GPU access.

  hipErrorHostMemoryNotRegistered

    Pointer does not correspond to a registered memory region. This error occurs when operations that
    require registered host memory are performed on unregistered memory. Common scenarios include:

    * Calling :cpp:func:`hipHostUnregister()` on a pointer that was not previously registered
    * Using :cpp:func:`hipHostGetDevicePointer()` on an unregistered host pointer
    * Attempting to use :cpp:func:`hipHostGetFlags()` on an unregistered pointer
    * Expecting zero-copy behavior with memory that hasn't been properly registered

    This error is the complement to ``hipErrorHostMemoryAlreadyRegistered`` and indicates that an operation
    expected registered memory but received a standard host allocation.

  hipErrorInvalidMemcpyDirection

    Invalid copy direction for memcpy. This error occurs when an invalid direction parameter is specified
    for memory copy operations. Valid directions include:

    * ``hipMemcpyHostToHost``
    * ``hipMemcpyHostToDevice``
    * ``hipMemcpyDeviceToHost``
    * ``hipMemcpyDeviceToDevice``
    * ``hipMemcpyDefault``

    The error typically occurs when:

    * Using an undefined direction value
    * Using ``hipMemcpyDeviceToDevice`` when copying between incompatible devices
    * Using a direction that doesn't match the actual source and destination pointer types

  hipErrorIllegalAddress

    An illegal memory access was encountered. This error indicates that a memory access violation occurred
    during kernel execution. Common causes include:

    * Dereferencing a null pointer in device code
    * Out-of-bounds access to an array or buffer
    * Using an unallocated memory address
    * Accessing memory after it has been freed
    * Misaligned memory access for types requiring specific alignment
    * Writing to read-only memory
    * Race conditions in multi-threaded kernels

    This error typically terminates the kernel execution and may provide additional debugging information
    when running with GPU debugging tools enabled.

  hipErrorRuntimeMemory

    Runtime memory call returned error. This is a general error indicating that a memory management operation
    within the HIP runtime has failed. Common scenarios include:

    * Internal memory allocation failures within the HIP runtime
    * Memory corruption affecting the runtime's internal data structures
    * System-wide memory pressure affecting runtime operations
    * Resource limitations preventing memory operations
    * Driver-level memory management errors bubbling up to the application

    This error differs from ``hipErrorOutOfMemory`` in that it relates to memory operations internal to the HIP
    runtime rather than explicit application requests for memory allocation.

  hipErrorInvalidChannelDescriptor

    This error indicates that an invalid channel descriptor is used to define the format and layout of data
    in memory, particularly when working with textures or arrays. This could happen if the descriptor is
    incorrectly set up or if it does not match the expected format for the operation being performed.

  hipErrorInvalidTexture

    The error code is returned when an invalid texture object is used in a function call. This typically
    occurs when a texture object is not properly initialized or configured before being used in operations
    that require valid texture data. If you encounter this error, it suggests that the texture object
    might be missing necessary configuration details or has been corrupted.

.. _device_context_errors:

Device and Context Errors
=========================

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Error Code
     - Value
     - Description

   * - :term:`hipErrorNoDevice`
     - ``100``
     - No ROCm-capable device is detected

   * - :term:`hipErrorInvalidDevice`
     - ``101``
     - Invalid device ordinal

   * - :term:`hipErrorInvalidContext`
     - ``201``
     - Invalid device context

   * - :term:`hipErrorContextAlreadyCurrent`
     - ``202``
     - Context is already current context

   * - :term:`hipErrorContextAlreadyInUse`
     - ``216``
     - Exclusive-thread device already in use by a different thread

   * - :term:`hipErrorContextIsDestroyed`
     - ``709``
     - Context is destroyed

   * - :term:`hipErrorInvalidHandle`
     - ``400``
     - Invalid resource handle

   * - :term:`hipErrorSetOnActiveProcess`
     - ``708``
     - Cannot set while device is active in this process

   * - :term:`hipErrorDeinitialized`
     - ``4``
     - Driver shutting down

   * - :term:`hipErrorNotInitialized`
     - ``3``
     - Initialization error

   * - :term:`hipErrorInsufficientDriver`
     - ``35``
     - Driver version is insufficient for runtime version

.. glossary::

  hipErrorNoDevice

    No ROCm-capable device is detected. This error occurs when the system does not have any compatible GPU devices
    that support the HIP runtime. Common scenarios include:

    * No physical GPU is installed in the system
    * Installed GPUs are not supported by the current HIP/ROCm version
    * GPU drivers are missing, outdated, or corrupted
    * GPU hardware failure or disconnection
    * System configuration prevents GPU detection (e.g., BIOS settings, virtualization limitations)
    * On Linux with ``HIP_PLATFORM=amd``, insufficient user permissions - the user must belong to both the ``render`` and ``video`` groups

  hipErrorInvalidDevice

    Invalid device ordinal. This error occurs when a function is called with a device index that doesn't correspond to
    a valid device. Common scenarios include:

    * Using a device index greater than or equal to the number of available devices
    * Using a negative device index
    * Using a device that has been removed or disabled
    * Attempting to access a device after system configuration changes

    Unlike ``hipErrorNoDevice`` which indicates no devices are available at all, this error occurs when trying to access
    a specific invalid device index while other valid devices might still be present.

  hipErrorInvalidContext

    Invalid device context. This error occurs when an operation is attempted with an invalid or destroyed context.
    Common scenarios include:

    * Using a context after calling :cpp:func:`hipCtxDestroy()` on it
    * Context corruption due to previous errors
    * Using a context associated with a device that has been reset
    * Mixing contexts improperly between different HIP API calls
    * Context handle that was never properly created or initialized
    * Using a context from a different process or thread incorrectly

    Context errors often indicate improper resource management in the application or incorrect context handling
    in multi-GPU or multi-threaded applications.

  hipErrorContextAlreadyCurrent

    Context is already current context. This error occurs when attempting to make a context current when it is already
    the current context for the calling thread.

  hipErrorContextAlreadyInUse

    Exclusive-thread device already in use by a different thread. This error occurs when attempting to access a device or
    context that has been allocated in exclusive thread mode from a thread other than the one that created it.

  hipErrorContextIsDestroyed

    Context is destroyed. This error occurs when attempting to use a context that has been previously destroyed.

  hipErrorInvalidHandle

    Invalid resource handle. This error occurs when an invalid handle is provided to a HIP API function. Common scenarios
    include using handles that have been destroyed or were never properly initialized.

  hipErrorSetOnActiveProcess

    Cannot set while device is active in this process. This error occurs when attempting to change settings
    that cannot be modified while the device is active.

  hipErrorDeinitialized

    Driver shutting down. This error occurs when attempting to use HIP functionality when the driver is in the
    process of shutting down or has been deinitialized. Common scenarios include:

    * Using HIP functions after calling :cpp:func:`hipDeviceReset()`
    * System is in the process of shutdown or reboot
    * Driver crash or unexpected termination
    * Another process has triggered driver reset

  hipErrorNotInitialized

    Initialization error. This occurs when attempting to use HIP functionality before the runtime has been
    properly initialized. Common scenarios include:

    * Calling HIP API functions before calling :cpp:func:`hipInit()`
    * Driver or runtime initialization failure
    * System configuration issues preventing proper initialization of the HIP runtime
    * Hardware initialization problems

  hipErrorInsufficientDriver

    Driver version is insufficient for runtime version. This error occurs when the installed GPU driver is too
    old to support the current HIP runtime version. This version mismatch can cause compatibility issues.
    Common scenarios include:

    * Using a newer HIP SDK with older driver installations
    * System updates that upgraded the HIP runtime but not the GPU drivers
    * Custom build environments with mismatched components
    * Partial upgrades of the ROCm stack

.. _kernel_launch_errors:

Kernel and Launch Errors
========================

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Error Code
     - Value
     - Description

   * - :term:`hipErrorInvalidValue``
     - ``1``
     - Invalid input value

   * - :term:`hipErrorInvalidDeviceFunction`
     - ``98``
     - Invalid device function

   * - :term:`hipErrorContextIsDestroyed`
     - ``709``
     - Invalid stream handle

   * - :term:`hipErrorInvalidConfiguration`
     - ``9``
     - Invalid configuration argument

   * - :term:`hipErrorInvalidSymbol`
     - ``13``
     - Invalid device symbol

   * - :term:`hipErrorMissingConfiguration`
     - ``52``
     - ``__global__`` function call is not configured

   * - :term:`hipErrorNoBinaryForGpu`
     - ``209``
     - No kernel image is available for execution on the device

   * - :term:`hipErrorInvalidKernelFile`
     - ``218``
     - Invalid kernel file

   * - :term:`hipErrorInvalidImage`
     - ``200``
     - Device kernel image is invalid

   * - :term:`hipErrorLaunchFailure`
     - ``719``
     - Unspecified launch failure

   * - :term:`hipErrorLaunchTimeOut`
     - ``702``
     - The launch timed out and was terminated

   * - :term:`hipErrorLaunchOutOfResources`
     - ``701``
     - Too many resources requested for launch

   * - :term:`hipErrorCooperativeLaunchTooLarge`
     - ``720``
     - Too many blocks in cooperative launch

   * - :term:`hipErrorPriorLaunchFailure`
     - ``53``
     - Unspecified launch failure in prior launch

.. glossary::

  hipErrorInvalidValue

    Error returned when a grid dimension check finds any input global work size
    dimension is zero, or a shared memory size check finds the size exceeds the size limit. 

  hipErrorInvalidDeviceFunction

    Invalid device function. This error occurs when attempting to use a function that is not a valid device
    function or is not available for the current device. Common scenarios include:

    * Code compiled for a specific GPU architecture (using ``--offload-arch``) but executed on an different/incompatible GPU

  hipErrorContextIsDestroyed

    This error is returned when the input stream or input stream handle is invalid. 

  hipErrorInvalidConfiguration

    Invalid configuration argument. This error occurs when the configuration specified for a kernel launch
    or other configurable operation contains invalid parameters. Common scenarios include:

    * Block dimensions exceeding hardware limits (too many threads per block)
    * Grid dimensions that are invalid (zero size or exceeding limits)
    * Invalid shared memory configuration
    * Incompatible combination of launch parameters
    * Block dimensions that don't match kernel requirements
    * Attempting to use more resources per block than available on the device

    This error typically requires adjusting kernel launch parameters to stay within the limits of the target
    device. Device properties and specific hardware constraints can be queried using :cpp:func:`hipGetDeviceProperties()`.

  hipErrorInvalidSymbol

    Invalid device symbol. This error occurs when a referenced symbol (variable or function) cannot be found
    or is improperly specified. Common scenarios include:

    * Referencing a symbol that doesn't exist in the compiled kernel
    * Symbol name typos or case mismatches
    * Attempting to access a host symbol as if it were a device symbol
    * Symbol not properly decorated with ``__device__`` or other required attributes
    * Symbol not visible due to scope/namespace issues

  hipErrorMissingConfiguration

    ``__global__`` function call is not configured. This error occurs when a kernel launch is attempted
    without proper configuration. Common scenarios include:

    * Calling a kernel without specifying execution configuration (grid and block dimensions)
    * Invalid or incomplete kernel configuration
    * Calling a ``__global__`` function directly as if it were a regular CPU function
    * Using a function pointer to a ``__global__`` function incorrectly

    This error is specific to improper kernel invocation syntax and is different from general configuration
    errors (``hipErrorInvalidConfiguration``) which relate to the values provided in a properly formed
    launch configuration.

  hipErrorNoBinaryForGpu

    No kernel image is available for execution on the device. This error occurs when attempting to run a
    kernel on a device for which no compatible compiled binary exists. Common scenarios include:

    * Attempting to run code compiled for a different GPU architecture
    * Missing or corrupted kernel binary for the target device
    * Kernel was compiled without support for the target device architecture
    * Using pre-compiled kernels that don't support the installed hardware
    * JIT compilation failure during runtime

  hipErrorInvalidKernelFile

    Invalid kernel file. This error occurs when the kernel file or module being loaded is corrupted or in
    an invalid format, for example the file name exists but the file size is 0.

  hipErrorInvalidImage

    Device kernel image is invalid. This error occurs when the device code image is corrupted or in an
    unsupported format.

  hipErrorLaunchFailure

    Unspecified launch failure. This is a general error that occurs when a kernel launch fails.
    Common causes include:

    * Mismatch between block size configuration and block size specified in launch bounds parameter
    * Invalid memory access in kernel
    * Kernel execution timeout
    * Hardware-specific failures

  hipErrorLaunchTimeOut

    The launch timed out and was terminated. This error occurs when a kernel execution exceeds the
    system's watchdog timeout limit. Common scenarios include:

    * Infinite loops in kernel code
    * Extremely long-running computations exceeding system limits
    * Deadlocks in kernel execution
    * Complex kernels that legitimately need more time than the watchdog allows
    * Hardware or driver issues preventing normal kernel termination

    The GPU's watchdog timer is a safety mechanism to prevent a hanging kernel from making the system
    unresponsive.

  hipErrorLaunchOutOfResources

    Too many resources requested for launch. This occurs when kernel resource requirements exceed
    device limits, such as:

    * Exceeding maximum threads per block
    * Exceeding maximum shared memory per block
    * Exceeding maximum register count per thread
    * Insufficient hardware resources for parallel execution

  hipErrorCooperativeLaunchTooLarge

    Too many blocks in cooperative launch. This error occurs when a cooperative kernel launch requests
    more thread blocks than the device can support for cooperative groups functionality.
    Common scenarios include:

    * Launching a cooperative kernel with grid dimensions that exceed hardware limits
    * Requesting more resources than available for synchronization across thread blocks
    * The shared memory size in bytes exceeds the device local memory size per CU
    * Using cooperative groups on hardware with limited support
    * Not accounting for cooperative launch limitations in kernel configuration

    Cooperative kernels allow thread blocks to synchronize with each other, but this requires special
    hardware support with specific limitations on the maximum number of blocks that can participate
    in synchronization operations.

  hipErrorPriorLaunchFailure

    Unspecified launch failure in prior launch. This error indicates that a previous kernel launch failed
    and affected the current HIP context state. Common scenarios include:

    * Launching a new kernel after a previous kernel crashed without resetting the device
    * Context contamination from previous failed operations
    * Resource leaks from previous launches affecting current operations
    * Attempting to use results from a previous failed kernel execution

    When this error occurs, it may be necessary to reset the device or create a new context to continue
    normal operation. Additional debugging of the previous failed launch may be required to identify
    the root cause.

.. _stream_capture_errors:

Stream Capture Errors
=====================

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Error Code
     - Value
     - Description

   * - :term:`hipErrorStreamCaptureUnsupported`
     - ``900``
     - Operation not permitted when stream is capturing

   * - :term:`hipErrorStreamCaptureInvalidated`
     - ``901``
     - Operation failed due to a previous error during capture

   * - :term:`hipErrorStreamCaptureMerge`
     - ``902``
     - Operation would result in a merge of separate capture sequences

   * - :term:`hipErrorStreamCaptureUnmatched`
     - ``903``
     - Capture was not ended in the same stream as it began

   * - :term:`hipErrorStreamCaptureUnjoined`
     - ``904``
     - Capturing stream has unjoined work

   * - :term:`hipErrorStreamCaptureIsolation`
     - ``905``
     - Dependency created on uncaptured work in another stream

   * - :term:`hipErrorStreamCaptureImplicit`
     - ``906``
     - Operation would make the legacy stream depend on a capturing blocking stream

   * - :term:`hipErrorStreamCaptureWrongThread`
     - ``908``
     - Attempt to terminate a thread-local capture sequence from another thread

   * - :term:`hipErrorCapturedEvent`
     - ``907``
     - Operation not permitted on an event last recorded in a capturing stream

   * - :term:`hipErrorInvalidResourceHandle`
     - ``400``
     - Input launch stream is ``NULL`` or is ``hipStreamLegacy``

.. glossary::

  hipErrorStreamCaptureUnsupported

    Operation not permitted when stream is capturing. This error occurs when attempting to perform an
    operation that is incompatible with stream capture mode. Common scenarios include:

    * Calling synchronization functions like :cpp:func:`hipDeviceSynchronize()` during capture
    * Using operations that implicitly synchronize during stream capture
    * Attempting to use features that cannot be captured as part of a graph
    * Trying to perform operations on different devices during capture
    * Using driver APIs that are incompatible with the stream capture mechanism

    Stream capture is used to record operations for later replay as a graph. Certain operations that
    affect global state or rely on host-device synchronization cannot be properly captured in this
    execution model.

  hipErrorStreamCaptureInvalidated

    Operation failed due to a previous error during capture. This error occurs when a stream capture
    has been invalidated by a prior error but capture operations are still being attempted.
    Common scenarios include:

    * Continuing to add operations to a stream after a capture-invalidating error
    * Not checking return codes from previous capture operations
    * Attempting to end a capture after invalidation
    * System or resource conditions changing during capture

    Once a stream capture has been invalidated, the entire capture sequence should be aborted and
    restarted from the beginning after resolving the cause of the initial failure.

  hipErrorStreamCaptureMerge

    Operation would result in a merge of separate capture sequences. This error occurs when an operation
    would cause independent capture sequences to merge, which is not supported. Common scenarios include:

    * A stream that is being captured interacting with another capturing stream
    * Operations creating implicit dependencies between separate capture sequences
    * Using events or other synchronization primitives that would link separate captures
    * Resource sharing between different capture sequences

    Stream captures must remain independent of each other to be converted into separate executable graphs.
    Operations that would create dependencies between separate captures are not allowed.

  hipErrorStreamCaptureUnmatched

    Capture was not ended in the same stream as it began. This error occurs when trying to end a stream
    capture in a different stream than the one where it was started. Common scenarios include:

    * Calling :cpp:func:`hipStreamEndCapture()` on a different stream than :cpp:func:`hipStreamBeginCapture()`
    * Confusing stream handles in multi-stream applications
    * Not properly tracking which streams have active captures
    * Programming errors in capture sequence management

    Stream captures must begin and end in the same stream to maintain the integrity of the captured
    operation sequence. The same stream handle must be used for beginning and ending a capture sequence.

  hipErrorStreamCaptureUnjoined

    Capturing stream has unjoined work. This error occurs when attempting to end a stream capture
    when there are still pending operations from other streams that have not been joined back to
    the capturing stream. Common scenarios include:

    * Forgetting to properly join forked work before ending capture
    * Missing :cpp:func:`hipEventRecord()` / :cpp:func:`hipStreamWaitEvent()` pairs for joined streams
    * Complex stream dependencies that are not fully resolved at capture end
    * Attempting to end a capture before all child operations complete

    When a stream capture forks work to other streams, those operations must be explicitly joined
    back to the capturing stream before the capture can be ended. This ensures that all dependencies
    are properly represented in the resulting graph.

  hipErrorStreamCaptureIsolation

    Dependency created on uncaptured work in another stream. This error occurs when a capturing stream
    becomes dependent on operations in a non-capturing stream. Common scenarios include:

    * A capturing stream waiting on an event recorded in a non-capturing stream
    * Creating dependencies on the default stream or other streams outside the capture
    * Using synchronization primitives that create implicit dependencies
    * Operations that depend on host-side or uncaptured GPU work

    Stream capture requires that all dependencies be explicitly captured as part of the graph. Operations
    that would make the captured sequence dependent on work outside the capture cannot be represented
    in the resulting graph and are therefore not allowed.

  hipErrorStreamCaptureImplicit

    Operation would make the legacy stream depend on a capturing blocking stream. This error occurs when
    an operation would create a dependency from the default (legacy/null) stream to a stream that is
    being captured in blocking mode. Common scenarios include:

    * Using the default stream during capture in ways that would create dependencies
    * Operations that would cause implicit synchronization with the null stream
    * Mixing legacy stream synchronization behavior with stream capture
    * Not properly managing stream relationships in applications using both explicit streams and the
      default stream

    This error is related to the implicit synchronization behavior of the default stream in HIP,
    which can conflict with the explicit dependency tracking needed for stream capture.

  hipErrorStreamCaptureWrongThread

    Attempt to terminate a thread-local capture sequence from another thread. This error occurs when
    a thread tries to end a stream capture that was begun by a different thread when using
    thread-local capture mode. Common scenarios include:

    * Multi-threaded applications incorrectly managing stream capture
    * Attempting to end a capture from a different thread than the one that started it
    * Thread pool or worker thread designs that don't properly track capture ownership
    * Misunderstanding the thread locality requirements of certain capture modes

    When using ``hipStreamCaptureModeThreadLocal``, stream captures are associated with the specific
    thread that started them and can only be ended by that same thread.

  hipErrorCapturedEvent

    Operation not permitted on an event last recorded in a capturing stream. This error occurs
    when attempting to perform operations on an event that was last recorded in a stream that
    is being captured. Common scenarios include:

    * Calling :cpp:func:`hipEventQuery()` or :cpp:func:`hipEventSynchronize()` on an event recorded during capture
    * Using events for host synchronization that are part of a stream capture
    * Attempting to reuse events across capturing and non-capturing contexts
    * Mixing event usage between graph capture and immediate execution modes

    Events that are part of a stream capture sequence are handled differently than regular events
    and cannot be used for host-side synchronization until the capture is complete and the graph
    is executed.

  hipErrorInvalidResourceHandle

    This error is returned when the input launch stream is a NULL pointer, is invalid, or is ``hipStreamLegacy``.
    If you encounter this error, you should check the validity of the resource handle being used in your HIP
    API calls. Ensure that the handle was correctly obtained and has not been freed or invalidated before use.

.. _profiler_errors:

Profiler Errors
===============

.. warning::

  The HIP Profiler Control APIs (:cpp:func:`hipProfilerStart()`, :cpp:func:`hipProfilerStop()`) are deprecated.
  It is recommended to use the ROCm profiling tools such as rocprof, roctracer, or AMD Radeon GPU Profiler
  for performance analysis instead.

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Error Code
     - Value
     - Description

   * - :term:`hipErrorProfilerDisabled`
     - ``5``
     - Profiler disabled while using external profiling tool

   * - :term:`hipErrorProfilerNotInitialized`
     - ``6``
     - Profiler is not initialized

   * - :term:`hipErrorProfilerAlreadyStarted`
     - ``7``
     - Profiler already started

   * - :term:`hipErrorProfilerAlreadyStopped`
     - ``8``
     - Profiler already stopped

.. glossary::

  hipErrorProfilerDisabled

    Profiler disabled while using external profiling tool. This error occurs when attempting to use
    the built-in HIP profiling functionality while an external profiling tool has taken control of
    the profiling interface. Common scenarios include:

    * Using :cpp:func:`hipProfilerStart()` / :cpp:func:`hipProfilerStop()` while running under tools like rocprof
      or AMD Radeon GPU Profiler
    * Conflicting profiling requests from different parts of an application
    * Attempting to use the HIP profiling API when profiling has been disabled at the driver level
    * Environment configurations that disable internal profiling in favor of external tools

    When external performance analysis tools are in use, they typically take exclusive control of
    the profiling interface, preventing the application from using the built-in profiling functions.

  hipErrorProfilerNotInitialized

    Profiler is not initialized. This error occurs when attempting to use profiling functions before the
    profiler has been properly initialized. Common scenarios include:

    * Calling :cpp:func:`hipProfilerStop()` without first calling :cpp:func:`hipProfilerStart()`
    * Using profiling functions before the HIP runtime has fully initialized
    * Configuration issues preventing proper profiler initialization
    * Missing required profiler components or drivers

    The HIP profiler requires proper initialization before it can collect performance data. The
    :cpp:func:`hipProfilerStart()` function must be called successfully before using other profiling functions
    or attempting to collect profile data.

  hipErrorProfilerAlreadyStarted

    Profiler already started. This error occurs when attempting to start the HIP profiler when it
    has already been started. Common scenarios include:

    * Multiple calls to :cpp:func:`hipProfilerStart()` without intervening :cpp:func:`hipProfilerStop()`
    * Attempting to restart profiling in different parts of code without coordination
    * Nested profiling sections that don't properly track profiler state
    * Mismanagement of profiler state in complex applications

    The HIP profiler can only be started once and must be stopped before it can be started again.
    This error is informational and indicates that the profiler is already in the desired active
    state.

  hipErrorProfilerAlreadyStopped

    Profiler already stopped. This error occurs when attempting to stop the HIP profiler when it is
    not currently running. Common scenarios include:

    * Calling :cpp:func:`hipProfilerStop()` multiple times without intervening :cpp:func:`hipProfilerStart()`
    * Mismanagement of profiler state in code with multiple profiling sections
    * Attempting to stop profiling in error handling paths when it wasn't started
    * Improper profiler state tracking in complex applications

    The HIP profiler must be in an active state before it can be stopped. This error is informational
    and indicates that the profiler is already in the desired inactive state.

.. _resource_mapping_errors:

Resource Mapping Errors
=======================

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Error Code
     - Value
     - Description

   * - :term:`hipErrorMapFailed`
     - ``205``
     - Mapping of buffer object failed

   * - :term:`hipErrorUnmapFailed`
     - ``206``
     - Unmapping of buffer object failed

   * - :term:`hipErrorArrayIsMapped`
     - ``207``
     - Array is mapped

   * - :term:`hipErrorAlreadyMapped`
     - ``208``
     - Resource already mapped

   * - :term:`hipErrorNotMapped`
     - ``211``
     - Resource not mapped

   * - :term:`hipErrorNotMappedAsArray`
     - ``212``
     - Resource not mapped as array

   * - :term:`hipErrorNotMappedAsPointer`
     - ``213``
     - Resource not mapped as pointer

.. glossary::

  hipErrorMapFailed

    Mapping of buffer object failed. This error occurs when the system fails to map device memory to
    host-accessible memory space. Common scenarios include:

    * Insufficient system resources for mapping
    * Attempting to map too much memory simultaneously
    * Mapping memory that is in an invalid state (e.g., already mapped or in use)
    * Trying to map memory with incompatible access flags or properties
    * System-level memory mapping constraints or limitations
    * Attempting to map special memory types that don't support mapping
    * Memory pressure affecting the operating system's ability to establish mappings

    This error typically occurs with functions like :cpp:func:`hipHostRegister()`, :cpp:func:`hipGLMapBufferObject()`,
    or similar functions that attempt to make device memory accessible to the host through memory
    mapping mechanisms.

  hipErrorUnmapFailed

    Unmapping of buffer object failed. This error occurs when the system fails to unmap previously
    mapped memory. Common scenarios include:

    * Attempting to unmap memory that is not currently mapped
    * Resources being in use by an active operation
    * System or driver issues affecting memory management
    * Invalid handle or pointer provided to unmap function
    * Corrupted mapping state due to application errors
    * Operating system resource constraints or failures

    This error is the counterpart to ``hipErrorMapFailed`` and occurs during cleanup operations when
    releasing mappings between host and device memory spaces. It may indicate resource leaks or
    state inconsistencies if not properly handled.

  hipErrorArrayIsMapped

    Array is mapped. This error occurs when attempting an operation that is not permitted on a
    mapped array or buffer. Common scenarios include:

    * Trying to free or modify a mapped array
    * Performing certain operations that require exclusive access to mapped resources
    * Attempting to re-map an already mapped array
    * Using mapped arrays in ways that conflict with their current mapped state
    * API calls that are incompatible with the current mapping state

    Arrays or buffers that are currently mapped to host memory have certain restrictions on the
    operations that can be performed on them. They must be unmapped before certain operations
    are allowed.

  hipErrorAlreadyMapped

    Resource already mapped. This error occurs when attempting to map a resource that is already
    in a mapped state. Common scenarios include:

    * Calling mapping functions multiple times on the same resource
    * Improper tracking of resource mapping state in complex applications
    * Race conditions in multi-threaded applications accessing the same resources
    * Attempting to map a resource with different flags when it's already mapped

    This error is similar to ``hipErrorArrayIsMapped`` but is more general and can apply to various
    mappable resources, not just arrays. Resources must be unmapped before they can be mapped
    again, possibly with different properties.

  hipErrorNotMapped

    Resource not mapped. This error occurs when attempting to perform an operation that requires
    a resource to be in a mapped state, but the resource is not currently mapped.
    Common scenarios include:

    * Trying to unmap a resource that is not mapped
    * Attempting to access host pointers for unmapped resources
    * Using mapping-dependent functions on unmapped resources
    * Mismanaging mapping state in complex applications
    * Attempting to use mapping-specific features with resources that don't support mapping

    This error indicates that a resource must be explicitly mapped before certain operations
    can be performed on it.

  hipErrorNotMappedAsArray

    Resource not mapped as array. This error occurs when attempting to use a mapped resource
    as an array when it was not mapped with the appropriate array mapping type. Common scenarios include:

    * Attempting to use a resource as an array when it was mapped with a different mapping type
    * Using :cpp:func:`hipArrayGetInfo()` or similar functions on resources not mapped as arrays
    * Type confusion in complex applications using multiple mapping types
    * Mismatched mapping and usage patterns for shared resources

    Different mapping types provide access to resources in different ways, and operations specific
    to one mapping type cannot be used with resources mapped using a different type. This error
    specifically indicates that an array-specific operation was attempted on a resource that was
    not mapped as an array.

  hipErrorNotMappedAsPointer

    Resource not mapped as pointer. This error occurs when attempting to use a mapped resource as
    a pointer when it was not mapped with the appropriate pointer mapping type. Common scenarios include:

    * Attempting to use a resource as a pointer when it was mapped with a different mapping type
    * Trying to perform pointer arithmetic or pointer-based access on inappropriately mapped resources
    * Type confusion in complex applications using multiple mapping types
    * Mismatched mapping and usage patterns for shared resources

    This error is complementary to ``hipErrorNotMappedAsArray`` and indicates that a pointer-specific
    operation was attempted on a resource that was not mapped as a pointer. Resources must be mapped
    with the appropriate mapping type for the operations that will be performed on them.

.. _peer_access_errors:

Peer Access Errors
==================

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Error Code
     - Value
     - Description

   * - :term:`hipErrorPeerAccessUnsupported`
     - ``217``
     - Peer access is not supported between these two devices

   * - :term:`hipErrorPeerAccessAlreadyEnabled`
     - ``704``
     - Peer access is already enabled

   * - :term:`hipErrorPeerAccessNotEnabled`
     - ``705``
     - Peer access has not been enabled

.. glossary::

  hipErrorPeerAccessUnsupported

    Peer access is not supported between these two devices. This error occurs when attempting to enable peer
    access between devices that cannot physically support direct access to each other's memory.
    Common scenarios include:

    * Devices connected to different PCIe root complexes without required hardware support
    * Different types or generations of GPUs that are incompatible for peer access
    * System configurations (BIOS, chipset) that don't allow peer-to-peer transfers
    * Virtualized environments that restrict direct hardware access
    * Attempting peer access on systems where the hardware interconnect doesn't support it

    This error indicates a hardware or system limitation, not an application error. To work around it,
    use regular host-mediated memory transfers instead of direct peer access. Device compatibility should
    be verified with :cpp:func:`hipDeviceCanAccessPeer()` before enabling peer access.

  hipErrorPeerAccessAlreadyEnabled

    Peer access is already enabled. This error occurs when attempting to enable peer access between two
    devices when that access has already been enabled. Common scenarios include:

    * Multiple calls to :cpp:func:`hipDeviceEnablePeerAccess()` for the same device pair
    * Enabling peer access in different parts of code without tracking the current state
    * Attempting to re-enable peer access after a context change without checking status

    This error is informational and typically doesn't indicate a problem that needs to be fixed,
    but rather that the requested state is already in effect.

  hipErrorPeerAccessNotEnabled

    Peer access has not been enabled. This error occurs when operations requiring peer access between
    devices are attempted without first enabling that access. Common scenarios include:

    * Attempting peer-to-peer memory copies without calling :cpp:func:`hipDeviceEnablePeerAccess()`
    * Kernel launches that access memory on peer devices without proper access rights
    * Accessing peer memory after peer access has been disabled

    To fix this error, call :cpp:func:`hipDeviceEnablePeerAccess()` before attempting operations that require direct
    access between peer devices. Not all device combinations support peer access. Compatibility can be
    determined with :cpp:func:`hipDeviceCanAccessPeer()`.

.. _system_file_errors:

System and File Errors
======================

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Error Code
     - Value
     - Description

   * - :term:`hipErrorFileNotFound`
     - ``301``
     - File not found

   * - :term:`hipErrorSharedObjectSymbolNotFound`
     - ``302``
     - Shared object symbol not found

   * - :term:`hipErrorSharedObjectInitFailed`
     - ``303``
     - Shared object initialization failed

   * - :term:`hipErrorOperatingSystem`
     - ``304``
     - OS call failed or operation not supported on this OS

   * - :term:`hipErrorNotFound`
     - ``500``
     - Named symbol not found

   * - :term:`hipErrorRuntimeOther`
     - ``1053``
     - Runtime call other than memory returned error

.. glossary::

  hipErrorFileNotFound

    File not found. This error occurs when HIP attempts to load a file that doesn't exist in the
    specified location. Common scenarios include:

    * Missing kernel source or binary files
    * Incorrect file paths provided to API functions
    * Missing shared libraries or dependencies
    * Files deleted or moved after initial configuration
    * Permission issues preventing file access

    This error typically occurs with operations like loading external kernels, modules, or shared
    libraries required by HIP applications.

  hipErrorSharedObjectSymbolNotFound

    Shared object symbol not found. This error occurs when attempting to access a symbol in a shared
    library or module that doesn't exist or isn't exported. Common scenarios include:

    * Misspelled symbol names
    * Using symbols that exist in the source code but weren't exported in the compiled library
    * Versioning mismatches between headers and implementation
    * Mangled C++ symbol names not properly accounted for
    * Library compiled with different visibility settings than expected
    * Using a function or variable name that exists but is in a different namespace

    This error is commonly encountered when using :cpp:func:`hipModuleGetFunction()` or similar functions to obtain
    handles to functions in dynamically loaded modules.

  hipErrorSharedObjectInitFailed

    Shared object initialization failed. This error occurs when a shared library or module fails during
    its initialization routine. Common scenarios include:

    * Dependencies of the shared object are missing
    * Incompatible library versions
    * Library initialization code encountering errors
    * Resource allocation failures during initialization
    * Incompatible compilation settings between application and shared object
    * Issues with static constructors in C++ libraries

    This error indicates that while the shared object was found and could be loaded, something prevented
    its proper initialization, making its functions and resources unavailable for use.

  hipErrorOperatingSystem

    OS call failed or operation not supported on this OS. This error indicates a system-level failure
    outside of the HIP runtime's direct control. Common scenarios include:

    * Insufficient permissions for requested operations
    * OS resource limits reached (file descriptors, memory limits, etc.)
    * System calls returning failure codes
    * Attempting operations not supported by the current OS or OS version
    * Driver or hardware interactions failing at the OS level
    * File system errors or permission issues

    This is a general error that can occur when HIP interacts with the operating system and encounters
    problems that prevent successful completion of the requested operation.

  hipErrorNotFound

    Named symbol not found. This error is returned when a requested named entity (such as a symbol,
    texture, surface, etc.) cannot be found. Common scenarios include:

    * Referencing a kernel function that doesn't exist in the module
    * Looking up a texture that hasn't been bound or created
    * Searching for a device with specific properties that no installed device has
    * Referencing a stream or event that has been destroyed
    * Using a name for a resource that was never created
    * Typos in symbol names

    This error is similar to ``hipErrorSharedObjectSymbolNotFound`` but is more general and applies to
    various named entities beyond just symbols in shared objects.

  hipErrorRuntimeOther

    Runtime call other than memory returned error. This is a general error code for failures in the
    HIP runtime that don't fit into other more specific categories. Common scenarios include:

    * Internal runtime function failures
    * Unexpected conditions encountered during HIP API execution
    * Driver-level errors not covered by more specific error codes
    * Hardware interaction issues
    * State inconsistencies within the runtime

    This is a catch-all error that may require looking at system logs or using additional
    debugging tools to identify the root cause.

.. _graphics_content_errors:

Graphics Context Errors
=======================

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Error Code
     - Value
     - Description

   * - :term:`hipErrorInvalidGraphicsContext`
     - ``219``
     - Invalid OpenGL or DirectX context

   * - :term:`hipErrorGraphExecUpdateFailure`
     - ``910``
     - | The graph update was not performed because it included changes which violated
       | constraints specific to instantiated graph update

.. glossary::

  hipErrorInvalidGraphicsContext

    Invalid OpenGL or DirectX context. This error occurs when attempting to perform interoperability
    operations with an invalid or incompatible graphics context.

  hipErrorGraphExecUpdateFailure

    The graph update was not performed because it included changes which violated constraints specific to
    instantiated graph update. This error occurs when attempting to update an already instantiated
    graph with changes that are not allowed.

.. _hardware_errors:

Hardware Errors
===============

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Error Code
     - Value
     - Description

   * - :term:`hipErrorECCNotCorrectable`
     - ``214``
     - Uncorrectable ECC error encountered

   * - :term:`hipErrorUnsupportedLimit`
     - ``215``
     - Limit is not supported on this architecture

   * - :term:`hipErrorAssert`
     - ``710``
     - Device-side assert triggered

.. glossary::

  hipErrorECCNotCorrectable

    Uncorrectable ECC error encountered. This hardware-level error occurs when the GPU's
    Error-Correcting Code (ECC) mechanism detects memory corruption that cannot be automatically
    corrected. Common scenarios include:

    * Physical hardware failure or degradation in GPU memory
    * Overheating causing memory bit flips
    * Running at extreme overclocked settings
    * Aging hardware with declining reliability
    * Power supply issues affecting memory integrity

    When this error occurs, the affected memory contents are unreliable and the operation cannot
    continue safely. This error generally requires system intervention, and in persistent cases,
    may indicate hardware that needs replacement.

  hipErrorUnsupportedLimit

    Limit is not supported on this architecture. This error occurs when attempting to query or
    set a device limit that is not supported by the current hardware. Common scenarios include:

    * Using :cpp:func:`hipDeviceSetLimit()` with a limit type not supported by the hardware
    * Requesting advanced features on entry-level or older GPU hardware
    * Setting limits specific to one GPU architecture on a different architecture
    * Using limit types introduced in newer HIP versions with older hardware

    This error indicates a hardware capability limitation rather than an application error.

  hipErrorAssert

    Device-side assert triggered. This error occurs when an assertion inside GPU kernel code
    fails. Common scenarios include:

    * Explicit :cpp:func:`assert()` statement in device code evaluates to false
    * Debug checks added by developers that detect invalid conditions
    * Parameter validation in kernel code that failed
    * Detected algorithmic errors or unexpected conditions

    This error is particularly useful for debugging as it explicitly indicates where a
    programmer-defined condition was violated in device code.
