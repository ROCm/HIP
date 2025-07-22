The debugging environment variables in HIP are collected in the following table. For
more information, check :doc:`hip:how-to/logging`, :doc:`hip:how-to/debugging`
and :doc:`GPU isolation <rocm:conceptual/gpu-isolation>`.

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
