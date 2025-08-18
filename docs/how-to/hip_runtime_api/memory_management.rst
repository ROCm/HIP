.. meta::
  :description: Memory management and its usage
  :keywords: AMD, ROCm, HIP, CUDA, memory management

.. _memory_management:

********************************************************************************
Memory management
********************************************************************************

Memory management is an important part of the HIP runtime API, when creating
high-performance applications. Both allocating and copying memory can result in
bottlenecks, which can significantly impact performance.

The programming model is based on a system with a host and a device, each having
its own distinct memory. Kernels operate on :ref:`device_memory`, while host functions
operate on :ref:`host_memory`.

The runtime offers functions for allocating, freeing, and copying device memory,
along with transferring data between host and device memory.

Here are the various memory management techniques:

* :ref:`coherence_control`
* :ref:`unified_memory`
* :ref:`virtual_memory`
* :ref:`stream_ordered_memory_allocator_how-to`

Memory allocation
================================================================================

The API calls and the resulting allocations are listed here:

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
