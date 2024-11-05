.. meta::
    :description: HIP environment variables reference
    :keywords: AMD, HIP, environment variables, environment, reference

********************************************************************************
HIP environment variables
********************************************************************************

In this section, the reader can find all the important HIP environment variables
on AMD platform, which are grouped by functionality. 

GPU isolation variables
================================================================================

The GPU isolation environment variables in HIP are collected in the next table. 
For more information, check :doc:`GPU isolation page <rocm:conceptual/gpu-isolation>`.

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

Profiling variables
================================================================================

The profiling environment variables in HIP are collected in the next table. For
more information, check :doc:`setting the number of CUs page <rocm:how-to/setting-cus>`.

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

Debug variables
================================================================================

The debugging environment variables in HIP are collected in the next table. For
more information, check :ref:`debugging_with_hip`.

.. include:: ../how-to/debugging_env.rst

Memory management related variables
================================================================================

The memory management related environment variables in HIP are collected in the 
next table.

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

Other useful variables
================================================================================

The following table lists environment variables that are useful but relate to
different features.

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
