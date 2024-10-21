.. meta::
  :description: This chapter describes introduces Virtual Memory (VM) and shows
                how to use it in AMD HIP.
  :keywords: AMD, ROCm, HIP, CUDA, virtual memory, virtual, memory, UM, APU

.. _virtual_memory:

*****************************
Virtual memory management
*****************************

Memory management is important when creating high-performance applications in the HIP ecosystem. Both allocating and copying memory can result in bottlenecks, which can significantly impact performance.

Global memory allocation in HIP uses the C language style allocation function. This works fine for simple cases but can cause problems if your memory needs change. If you need to increase the size of your memory, you must allocate a second larger buffer and copy the data to it before you can free the original buffer. This increases overall memory usage and causes unnecessary ``memcpy`` calls. Another solution is to allocate a larger buffer than you initially need. However, this isn't an efficient way to handle resources and doesn't solve the issue of reallocation when the extra buffer runs out.

Virtual memory management solves these memory management problems. It helps to reduce memory usage and unnecessary ``memcpy`` calls.

.. _memory_allocation_virtual_memory:

Memory allocation
=================

Standard memory allocation uses the ``hipMalloc`` function to allocate a block of memory on the device. However, when using virtual memory, this process is separated into multiple steps using the ``hipMemCreate``, ``hipMemAddressReserve``, ``hipMemMap``, and ``hipMemSetAccess`` functions. This guide explains what these functions do and how you can use them for virtual memory management.

Allocate physical memory
------------------------

The first step is to allocate the physical memory itself with the ``hipMemCreate`` function. This function accepts the size of the buffer, an ``unsigned long long`` variable for the flags, and a ``hipMemAllocationProp`` variable. ``hipMemAllocationProp`` contains the properties of the memory to be allocated, such as where the memory is physically located and what kind of shareable handles are available. If the allocation is successful, the function returns a value of ``hipSuccess``, with ``hipMemGenericAllocationHandle_t`` representing a valid physical memory allocation. The allocated memory size must be aligned with the granularity appropriate for the properties of the allocation. You can use the ``hipMemGetAllocationGranularity`` function to determine the correct granularity.

.. code-block:: cpp

    size_t granularity = 0;
    hipMemGenericAllocationHandle_t allocHandle;
    hipMemAllocationProp prop = {};
    prop.type = HIP_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = HIP_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = currentDev;
    hipMemGetAllocationGranularity(&granularity, &prop, HIP_MEM_ALLOC_GRANULARITY_MINIMUM);
    padded_size = ROUND_UP(size, granularity);
    hipMemCreate(&allocHandle, padded_size, &prop, 0);

Reserve virtual address range
-----------------------------

After you have acquired an allocation of physical memory, you must map it before you can use it. To do so, you need a virtual address to map it to.  Mapping means the physical memory allocation is available from the virtual address range it is mapped to. To reserve a virtual memory range, use the ``hipMemAddressReserve`` function. The size of the virtual memory must match the amount of physical memory previously allocated. You can then map the physical memory allocation to the newly-acquired virtual memory address range using the ``hipMemMap`` function.

.. code-block:: cpp

    hipMemAddressReserve(&ptr, padded_size, 0, 0, 0);
    hipMemMap(ptr, padded_size, 0, allocHandle, 0);

Set memory access
-----------------

Finally, use the ``hipMemSetAccess`` function to enable memory access. It accepts the pointer to the virtual memory, the size, and a ``hipMemAccessDesc`` descriptor as parameters. In a multi-GPU environment, you can map the device memory of one GPU to another. This feature also works with the traditional memory management system, but isn't as scalable as with virtual memory. When memory is allocated with ``hipMalloc``, ``hipDeviceEnablePeerAccess`` is used to enable peer access. This function enables access between two devices, but it means that every call to ``hipMalloc`` takes more time to perform the checks and the mapping between the devices. When using virtual memory management, peer access is enabled by ``hipMemSetAccess``, which provides a finer level of control over what is shared. This has no performance impact on memory allocation and gives you more control over what memory buffers are shared with which devices.

.. code-block:: cpp

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = HIP_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = currentDev;
    accessDesc.flags = HIP_MEM_ACCESS_FLAGS_PROT_READWRITE;
    hipMemSetAccess(ptr, padded_size, &accessDesc, 1);

At this point the memory is allocated, mapped, and ready for use. You can read and write to it, just like you would a C style memory allocation.

Free virtual memory
-------------------

To free the memory allocated in this manner, use the corresponding free functions. To unmap the memory, use ``hipMemUnmap``. To release the virtual address range, use ``hipMemAddressFree``.  Finally, to release the physical memory, use ``hipMemRelease``. A side effect of these functions is the lack of synchronization when memory is released. If you call ``hipFree`` when you have multiple streams running in parallel, it synchronizes the device. This causes worse resource usage and performance.

.. code-block:: cpp

    hipMemUnmap(ptr, size);
    hipMemRelease(allocHandle);
    hipMemAddressFree(ptr, size);

.. _usage_virtual_memory:

Memory usage
============

Dynamically increase allocation size
------------------------------------

The ``hipMemAddressReserve`` function allows you to increase the amount of pre-allocated memory. This function accepts a parameter representing the requested starting address of the virtual memory. This allows you to have a continuous virtual address space without worrying about the underlying physical allocation.

.. code-block:: cpp

    hipMemAddressReserve(&new_ptr, (new_size - padded_size), 0, ptr + padded_size, 0);
    hipMemMap(new_ptr, (new_size - padded_size), 0, newAllocHandle, 0);
    hipMemSetAccess(new_ptr, (new_size - padded_size), &accessDesc, 1);

The code sample above assumes that ``hipMemAddressReserve`` was able to reserve the memory address at the specified location. However, this isn't guaranteed to be true, so you should validate that ``new_ptr`` points to a specific virtual address before using it.
