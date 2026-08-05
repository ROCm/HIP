.. meta::
  :description:
  :keywords: stream, memory allocation, SOMA, stream ordered memory allocator

.. _stream_ordered_memory_allocator_how-to:

*******************************************************************************
Stream Ordered Memory Allocator
*******************************************************************************

The Stream Ordered Memory Allocator (SOMA) is part of the HIP runtime API. SOMA provides an asynchronous memory allocation mechanism with stream-ordering semantics. You can use SOMA to allocate and free memory in stream order, which ensures that all asynchronous accesses occur between the stream executions of allocation and deallocation. Compliance with stream order prevents use-before-allocation or use-after-free errors, which helps to avoid an undefined behavior.

Advantages of SOMA:

- Efficient reuse: Enables efficient memory reuse across streams, which reduces unnecessary allocation overhead.
- Fine-grained control: Allows you to set attributes and control caching behavior for memory pools.
- Inter-process sharing: Enables secure sharing of allocations between processes.
- Optimizations: Allows driver to optimize based on its awareness of SOMA and other stream management APIs.

Disadvantages of SOMA:

- Temporal constraints: Requires you to adhere strictly to stream order to avoid errors.
- Complexity: Involves memory management in stream order, which can be intricate.
- Learning curve: Requires you to put additional efforts to understand and utilize SOMA effectively.

Using SOMA
=====================================

You can allocate memory using ``hipMallocAsync()`` with stream-ordered
semantics. This restricts the asynchronous access to the memory between the stream executions of the allocation and deallocation. Accessing
memory if the compliant memory accesses won't overlap
temporally. ``hipFreeAsync()`` frees memory from the pool with stream-ordered
semantics.

Here is how to use stream ordered memory allocation:

.. tab-set::
  .. tab-item:: Stream Ordered Memory Allocation

    .. literalinclude:: ../../../tools/example_codes/stream_ordered_memory_allocation.hip
        :start-after: // [sphinx-start]
        :end-before: // [sphinx-end]
        :language: cpp

  .. tab-item:: Ordinary Allocation

    .. literalinclude:: ../../../tools/example_codes/ordinary_memory_allocation.hip
        :start-after: // [sphinx-start]
        :end-before: // [sphinx-end]
        :language: cpp

For more details, see :ref:`stream_ordered_memory_allocator_reference`.

Memory pools
============

Memory pools provide a way to manage memory with stream-ordered behavior while ensuring proper synchronization and avoiding memory access errors. Division of a single memory system into separate pools facilitates querying the access path properties for each partition. Memory pools are used for host memory, device memory, and unified memory.

Set pools
---------

The ``hipMallocAsync()`` function uses the current memory pool and also provides the opportunity to create and access different pools using ``hipMemPoolCreate()`` and ``hipMallocFromPoolAsync()`` functions respectively.

Unlike NVIDIA CUDA, where stream-ordered memory allocation can be implicit, ROCm HIP is explicit. This requires managing memory allocation for each stream in HIP while ensuring precise control over memory usage and synchronization.

.. literalinclude:: ../../../tools/example_codes/memory_pool.hip
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp

Trim pools
----------

The memory allocator allows you to allocate and free memory in stream order. To control memory usage, set the release threshold attribute using ``hipMemPoolAttrReleaseThreshold``.  This threshold specifies the amount of reserved memory in bytes to hold onto.

.. literalinclude:: ../../../tools/example_codes/memory_pool_threshold.hip
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp

When the amount of memory held in the memory pool exceeds the threshold, the allocator tries to release memory back to the operating system during the next call to stream, event, or context synchronization.

To improve performance, it is a good practice to adjust the memory pool size using ``hipMemPoolTrimTo()``. It helps to reclaim memory from an excessive memory pool, which optimizes memory usage for your application.

.. literalinclude:: ../../../tools/example_codes/memory_pool_trim.cpp
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp

Resource usage statistics
-------------------------

Resource usage statistics help in optimization. Here is the list of pool attributes used to query memory usage:

- ``hipMemPoolAttrReservedMemCurrent``: Returns the total physical GPU memory currently held in the pool.
- ``hipMemPoolAttrUsedMemCurrent``: Returns the total size of all the memory allocated from the pool.
- ``hipMemPoolAttrReservedMemHigh``: Returns the total physical GPU memory held in the pool since the last reset.
- ``hipMemPoolAttrUsedMemHigh``: Returns the total size of all the memory allocated from the pool since the last reset.

To reset these attributes to the current value, use ``hipMemPoolSetAttribute()``.

.. literalinclude:: ../../../tools/example_codes/memory_pool_resource_usage_statistics.cpp
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp

Memory reuse policies
---------------------

The allocator might reallocate memory as long as the compliant memory accesses will not to overlap temporally. To optimize the memory usage, disable or enable the following memory pool reuse policy attribute flags:

- ``hipMemPoolReuseFollowEventDependencies``: Checks event dependencies before allocating additional GPU memory.
- ``hipMemPoolReuseAllowOpportunistic``: Checks freed allocations to determine if the stream order semantic indicated by the free operation has been met.
- ``hipMemPoolReuseAllowInternalDependencies``: Manages reuse based on internal dependencies in runtime. If the driver fails to allocate and map additional physical memory, it searches for memory waiting for another stream's progress and reuses it.

Device accessibility for multi-GPU support
------------------------------------------

Allocations are initially accessible from the device where they reside.

Interprocess memory handling
=============================

.. attention::
    IPC API calls are only supported on systems with an active ``amdgpu-dkms`` driver. Please refer to the
    `AMDGPU documentation <https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/index.html>`__ for information
    on how to install ``amdgpu-dkms``.

Interprocess capable (IPC) memory pools facilitate efficient and secure sharing of GPU memory between processes.

To achieve interprocess memory sharing, you can use either a :ref:`device pointer <device-pointer>` or a :ref:`shareable handle <shareable-handle>`. Both provide allocator (export) and consumer (import) interfaces. Use a device pointer to share a specific allocation; use a shareable handle when the importing process needs to allocate from the shared pool independently.

.. _device-pointer:

Device pointer
--------------

To export a memory pool allocation for access by another process, use
``hipMemPoolExportPointer()``. It produces a serializable ``hipMemPoolPtrExportData``
struct that can be written to any IPC channel, such as a named pipe (FIFO), and then 
read by the importing process.

Here is the exporter:

.. literalinclude:: ../../../tools/example_codes/ipc_memory_pool_device_pointer.hip
    :start-after: // [sphinx-exporter-start]
    :end-before: // [sphinx-exporter-end]
    :language: cpp

To import a memory pool pointer from another process, use ``hipMemPoolImportPointer()``.

Here is the importer:

.. literalinclude:: ../../../tools/example_codes/ipc_memory_pool_device_pointer.hip
    :start-after: // [sphinx-importer-start]
    :end-before: // [sphinx-importer-end]
    :language: cpp

.. _shareable-handle:

Shareable handle
----------------

To export a memory pool as a shareable handle, use ``hipMemPoolExportToShareableHandle()``.
The handle is a POSIX file descriptor, which is process-local and cannot be transferred
by writing its integer value to a pipe. Instead, use ``SCM_RIGHTS`` over a Unix domain
socket to duplicate the file description into the receiving process.

Here is the exporter:

.. literalinclude:: ../../../tools/example_codes/ipc_memory_pool_shareable_handle.hip
    :start-after: // [sphinx-exporter-start]
    :end-before: // [sphinx-exporter-end]
    :language: cpp

To import a memory pool from a shareable handle received via ``SCM_RIGHTS``, use
``hipMemPoolImportFromShareableHandle()``. The importing process can then allocate
GPU memory from the imported pool independently.

Here is the importer:

.. literalinclude:: ../../../tools/example_codes/ipc_memory_pool_shareable_handle.hip
    :start-after: // [sphinx-importer-start]
    :end-before: // [sphinx-importer-end]
    :language: cpp
