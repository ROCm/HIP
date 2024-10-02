.. meta::
  :description: This chapter introduces memory management and shows how to use
                it.
  :keywords: AMD, ROCm, HIP, CUDA, memory management

********************************************************************************
Memory management
********************************************************************************

Memory management is an important part of the HIP runtime API, when creating
high-performance applications. Both allocating and copying memory can result in
bottlenecks, which can significantly impact performance.

The programming model is based on a system with a host and a device, each having
its own distinct memory. Kernels operate on device memory, while host functions
operate on host memory.

The runtime offers functions for allocating, freeing, and copying device memory,
along with transferring data between host and device memory.

The description of these memory type can be located at the following page:

* :ref:`device_memory`
* :ref:`host_memory`

The different memory managements are described in the following pages:

* :ref:`coherence_control`
* :ref:`unified_memory`
* :ref:`virtual_memory`
* :ref:`stream_ordered_memory_allocator_how-to`

Memory allocation
================================================================================

The following API calls with result in these allocations:

.. list-table:: Memory coherence control
    :header-rows: 1
    :align: center

    * - API
      - Data location
      - Allocation
    * - System allocated 
      - Host
      - :ref:`Pageable <pageable_host_memory>`
    * - :cpp:func:`hipMallocManaged`
      - Host
      - :ref:`Managed <unified_memory>`
    * - :cpp:func:`hipHostMalloc`
      - Host
      - :ref:`Pinned <pinned_host_memory>`
    * - :cpp:func:`hipMalloc`
      - Device
      - Pinned
