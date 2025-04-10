The memory management related environment variables in HIP are collected in the
following table. The ``HIP_HOST_COHERENT`` variable linked at the following
pages:

- :ref:`Coherence control <hip:coherence_control>`

- :ref:`Memory allocation flags <hip:memory_allocation_flags>`

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
