.. meta::
    :description: HIP environment variables
    :keywords: AMD, HIP, environment variables, environment

HIP GPU isolation variables
--------------------------------------------------------------------------------

The GPU isolation environment variables in HIP are collected in the following table.

.. _hip-env-isolation:
.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``ROCR_VISIBLE_DEVICES``
        | A list of device indices or UUIDs that will be exposed to applications.
      - Example: ``0,GPU-DEADBEEFDEADBEEF``

    * - | ``GPU_DEVICE_ORDINAL``
        | Devices indices exposed to OpenCL and HIP applications.
      - Example: ``0,2``

    * - | ``HIP_VISIBLE_DEVICES`` or ``CUDA_VISIBLE_DEVICES``
        | Device indices exposed to HIP applications.
      - Example: ``0,2``

HIP profiling variables
--------------------------------------------------------------------------------

The profiling environment variables in HIP are collected in the following table.

.. _hip-env-prof:
.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HSA_CU_MASK``
        | Sets the mask on a lower level of queue creation in the driver,
        | this mask will also be set for queues being profiled.
      - Example: ``1:0-8``

    * - | ``ROC_GLOBAL_CU_MASK``
        | Sets the mask on queues created by the HIP or the OpenCL runtimes,
        | this mask will also be set for queues being profiled.
      - Example: ``0xf``, enables only 4 CUs

    * - | ``HIP_FORCE_QUEUE_PROFILING``
        | Used to run the app as if it were run in rocprof. Forces command queue
        | profiling on by default.
      - | 0: Disable
        | 1: Enable

HIP debug variables
--------------------------------------------------------------------------------

The debugging environment variables in HIP are collected in the following table.

.. _hip-env-debug:
.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

    * - | ``AMD_LOG_LEVEL``
        | Enables HIP log on various level.
      - ``0``
      - | 0: Disable log.
        | 1: Enables error logs.
        | 2: Enables warning logs next to lower-level logs.
        | 3: Enables information logs next to lower-level logs.
        | 4: Enables debug logs next to lower-level logs.
        | 5: Enables debug extra logs next to lower-level logs.

    * - | ``AMD_LOG_LEVEL_FILE``
        | Sets output file for ``AMD_LOG_LEVEL``.
      - stderr output
      -

    * - | ``AMD_LOG_MASK``
        | Specifies HIP log filters. Here is the ` complete list of log masks <https://github.com/ROCm/clr/blob/develop/rocclr/utils/debug.hpp#L40>`_.
      - ``0x7FFFFFFF``
      - | 0x1: Log API calls.
        | 0x2: Kernel and copy commands and barriers.
        | 0x4: Synchronization and waiting for commands to finish.
        | 0x8: Decode and display AQL packets.
        | 0x10: Queue commands and queue contents.
        | 0x20: Signal creation, allocation, pool.
        | 0x40: Locks and thread-safety code.
        | 0x80: Kernel creations and arguments, etc.
        | 0x100: Copy debug.
        | 0x200: Detailed copy debug.
        | 0x400: Resource allocation, performance-impacting events.
        | 0x800: Initialization and shutdown.
        | 0x1000: Misc debug, not yet classified.
        | 0x2000: Show raw bytes of AQL packet.
        | 0x4000: Show code creation debug.
        | 0x8000: More detailed command info, including barrier commands.
        | 0x10000: Log message location.
        | 0x20000: Memory allocation.
        | 0x40000: Memory pool allocation, including memory in graphs.
        | 0x80000: Timestamp details.
        | 0xFFFFFFFF: Log always even mask flag is zero.

    * - | ``HIP_LAUNCH_BLOCKING``
        | Used for serialization on kernel execution.
      - ``0``
      - | 0: Disable. Kernel executes normally.
        | 1: Enable. Serializes kernel enqueue, behaves the same as ``AMD_SERIALIZE_KERNEL``.

    * - | ``HIP_VISIBLE_DEVICES`` (or ``CUDA_VISIBLE_DEVICES``)
        | Only devices whose index is present in the sequence are visible to HIP
      - Unset by default.
      - 0,1,2: Depending on the number of devices on the system.

    * - | ``GPU_DUMP_CODE_OBJECT``
        | Dump code object.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``AMD_SERIALIZE_KERNEL``
        | Serialize kernel enqueue.
      - ``0``
      - | 0: Disable
        | 1: Wait for completion before enqueue.
        | 2: Wait for completion after enqueue.
        | 3: Both

    * - | ``AMD_SERIALIZE_COPY``
        | Serialize copies
      - ``0``
      - | 0: Disable
        | 1: Wait for completion before enqueue.
        | 2: Wait for completion after enqueue.
        | 3: Both

    * - | ``AMD_DIRECT_DISPATCH``
        | Enable direct kernel dispatch (Currently for Linux; under development for Windows).
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_MAX_HW_QUEUES``
        | The maximum number of hardware queues allocated per device.
      - ``4``
      - The variable controls how many independent hardware queues HIP runtime can create per process,
        per device. If an application allocates more HIP streams than this number, then HIP runtime reuses
        the same hardware queues for the new streams in a round-robin manner. Note that this maximum
        number does not apply to hardware queues that are created for CU-masked HIP streams, or
        cooperative queues for HIP Cooperative Groups (single queue per device).

HIP memory management related variables
--------------------------------------------------------------------------------

The memory management related environment variables in HIP are collected in the
following table.

.. _hip-env-memory:
.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

    * - | ``HIP_HIDDEN_FREE_MEM``
        | Amount of memory to hide from the free memory reported by hipMemGetInfo.
      - ``0``
      - | 0: Disable
        | Unit: megabyte (MB)

    * - | ``HIP_HOST_COHERENT``
        | Specifies if the memory is coherent between the host and GPU in ``hipHostMalloc``.
      - ``0``
      - | 0: Memory is not coherent.
        | 1: Memory is coherent.
        | Environment variable has effect, if the following conditions are statisfied:
        | - One of the ``hipHostMallocDefault``, ``hipHostMallocPortable``,  ``hipHostMallocWriteCombined`` or ``hipHostMallocNumaUser`` flag set to 1.
        | - ``hipHostMallocCoherent``, ``hipHostMallocNonCoherent`` and ``hipHostMallocMapped`` flags set to 0.

    * - | ``HIP_INITIAL_DM_SIZE``
        | Set initial heap size for device malloc.
      - ``8388608``
      - | Unit: Byte
        | The default value corresponds to 8 MB.

    * - | ``HIP_MEM_POOL_SUPPORT``
        | Enables memory pool support in HIP.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``HIP_MEM_POOL_USE_VM``
        | Enables memory pool support in HIP.
      - | ``0``: other OS
        | ``1``: Windows
      - | 0: Disable
        | 1: Enable

    * - | ``HIP_VMEM_MANAGE_SUPPORT``
        | Virtual Memory Management Support.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``GPU_MAX_HEAP_SIZE``
        | Set maximum size of the GPU heap to % of board memory.
      - ``100``
      - | Unit: Percentage

    * - | ``GPU_MAX_REMOTE_MEM_SIZE``
        | Maximum size that allows device memory substitution with system.
      - ``2``
      - | Unit: kilobyte (KB)

    * - | ``GPU_NUM_MEM_DEPENDENCY``
        | Number of memory objects for dependency tracking.
      - ``256``
      -

    * - | ``GPU_STREAMOPS_CP_WAIT``
        | Force the stream memory operation to wait on CP.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``HSA_LOCAL_MEMORY_ENABLE``
        | Enable HSA device local memory usage.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``PAL_ALWAYS_RESIDENT``
        | Force memory resources to become resident at allocation time.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``PAL_PREPINNED_MEMORY_SIZE``
        | Size of prepinned memory.
      - ``64``
      - | Unit: kilobyte (KB)

    * - | ``REMOTE_ALLOC``
        | Use remote memory for the global heap allocation.
      - ``0``
      - | 0: Disable
        | 1: Enable

HIP miscellaneous variables
--------------------------------------------------------------------------------

The following table lists environment variables that are useful but relate to
different features in HIP.

.. _hip-env-other:
.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

    * - | ``HIPRTC_COMPILE_OPTIONS_APPEND``
        | Sets compile options needed for ``hiprtc`` compilation.
      - None
      - ``--gpu-architecture=gfx906:sramecc+:xnack``, ``-fgpu-rdc``
