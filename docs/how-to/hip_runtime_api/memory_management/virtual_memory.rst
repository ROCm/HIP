.. meta::
  :description: This chapter describes introduces Virtual Memory (VM) and shows
                how to use it in AMD HIP.
  :keywords: AMD, ROCm, HIP, CUDA, virtual memory, virtual, memory, UM, APU

.. _virtual_memory:

********************************************************************************
Virtual memory management
********************************************************************************

Memory management is important when creating high-performance applications in
the HIP ecosystem. Both allocating and copying memory can result in bottlenecks,
which can significantly impact performance.

Global memory allocation in HIP uses the C language style allocation function.
This works fine for simple cases but can cause problems if your memory needs
change. If you need to increase the size of your memory, you must allocate a
second larger buffer and copy the data to it before you can free the original
buffer. This increases overall memory usage and causes unnecessary ``memcpy``
calls. Another solution is to allocate a larger buffer than you initially need.
However, this isn't an efficient way to handle resources and doesn't solve the
issue of reallocation when the extra buffer runs out.

Virtual memory management solves these memory management problems. It helps to
reduce memory usage and unnecessary ``memcpy`` calls.

HIP virtual memory management is built on top of HSA, which provides low-level
access to AMD GPU memory. For more details on the underlying HSA runtime,
see :doc:`ROCr documentation <rocr-runtime:index>`

.. _memory_allocation_virtual_memory:

Memory allocation
=================

Standard memory allocation uses the :cpp:func:`hipMalloc` function to allocate a
block of memory on the device. However, when using virtual memory, this process
is separated into multiple steps using the :cpp:func:`hipMemCreate`,
:cpp:func:`hipMemAddressReserve`, :cpp:func:`hipMemMap`, and
:cpp:func:`hipMemSetAccess` functions. This guide explains what these functions
do and how you can use them for virtual memory management.

.. _vmm_support:

Virtual memory management support
---------------------------------

The first step is to check if the targeted device or GPU supports virtual memory management.
Use the :cpp:func:`hipDeviceGetAttribute` function to get the
``hipDeviceAttributeVirtualMemoryManagementSupported`` attribute for a specific GPU, as shown in the following example.

.. code-block:: cpp

    int vmm = 0, currentDev = 0;
    hipDeviceGetAttribute(
        &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, currentDev
    );

    if (vmm == 0) {
        std::cout << "GPU " << currentDev << " doesn't support virtual memory management." << std::endl;
    } else {
        std::cout << "GPU " << currentDev << " support virtual memory management." << std::endl;
    }

.. _allocate_physical_memory:

Allocate physical memory
------------------------

The next step is to allocate the physical memory using the
:cpp:func:`hipMemCreate` function. This function accepts the size of the buffer,
an ``unsigned long long`` variable for the flags, and a
:cpp:struct:`hipMemAllocationProp` variable. :cpp:struct:`hipMemAllocationProp`
contains the properties of the memory to be allocated, such as where the memory
is physically located and what kind of shareable handles are available. If the
allocation is successful, the function returns a value of
:cpp:enumerator:`hipSuccess`, with :cpp:type:`hipMemGenericAllocationHandle_t`
representing a valid physical memory allocation.

The allocated memory must be aligned with the appropriate granularity. The
granularity value can be queried with :cpp:func:`hipMemGetAllocationGranularity`,
and its value depends on the target device hardware and the type of memory
allocation. If the allocation size is not aligned, meaning it is not cleanly
divisible by the minimum granularity value, :cpp:func:`hipMemCreate` will return
an out-of-memory error.

.. code-block:: cpp

    size_t granularity = 0;
    hipMemGenericAllocationHandle_t allocHandle;
    hipMemAllocationProp prop = {};
    // The pinned allocation type cannot be migrated from its current location
    // while the application is actively using it.
    prop.type = hipMemAllocationTypePinned;
    // Set the location type to device, currently there are no other valid option.
    prop.location.type = hipMemLocationTypeDevice;
    // Set the device id, where the memory will be allocated.
    prop.location.id = currentDev;
    hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum);
    padded_size = ROUND_UP(size, granularity);
    hipMemCreate(&allocHandle, padded_size, &prop, 0);

.. _reserve_virtual_address:

Reserve virtual address range
-----------------------------

After you have acquired an allocation of physical memory, you must map it to a
virtual address before you can use it. Mapping means the physical memory
allocation is available from the virtual address range it is mapped to. To
reserve a virtual memory range, use the :cpp:func:`hipMemAddressReserve`
function. The size of the virtual memory must match the amount of physical
memory previously allocated. You can then map the physical memory allocation to
the newly-acquired virtual memory address range using the :cpp:func:`hipMemMap`
function.

.. code-block:: cpp

    hipMemAddressReserve(&ptr, padded_size, 0, 0, 0);
    hipMemMap(ptr, padded_size, 0, allocHandle, 0);

.. _set_memory_access:

Set memory access
-----------------

Finally, use the :cpp:func:`hipMemSetAccess` function to enable memory access.
It accepts the pointer to the virtual memory, the size, and a
:cpp:struct:`hipMemAccessDesc` descriptor as parameters. In a multi-GPU
environment, you can map the device memory of one GPU to another. This feature
also works with the traditional memory management system, but isn't as scalable
as with virtual memory. When memory is allocated with :cpp:func:`hipMalloc`,
:cpp:func:`hipDeviceEnablePeerAccess` is used to enable peer access. This
function enables access between two devices, but it means that every call to
:cpp:func:`hipMalloc` takes more time to perform the checks and the mapping
between the devices. When using virtual memory management, peer access is
enabled by :cpp:func:`hipMemSetAccess`, which provides a finer level of
control over what is shared. This has no performance impact on memory allocation
and gives you more control over what memory buffers are shared with which
devices.

.. code-block:: cpp

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = currentDev;
    accessDesc.flags = hipMemAccessFlagsProtReadwrite;
    hipMemSetAccess(ptr, padded_size, &accessDesc, 1);

At this point the memory is allocated, mapped, and ready for use. You can read
and write to it, just like you would a C style memory allocation.

.. _usage_virtual_memory:

Dynamically increase allocation size
------------------------------------

To increase the amount of pre-allocated memory, use
:cpp:func:`hipMemAddressReserve`, which accepts the starting address, and the
size of the reservation in bytes. This allows you to have a continuous virtual
address space without worrying about the underlying physical allocation.

.. code-block:: cpp

    hipMemAddressReserve(&new_ptr, (new_size - padded_size), 0, ptr + padded_size, 0);
    hipMemMap(new_ptr, (new_size - padded_size), 0, newAllocHandle, 0);
    hipMemSetAccess(new_ptr, (new_size - padded_size), &accessDesc, 1);

The code sample above assumes that :cpp:func:`hipMemAddressReserve` was able to
reserve the memory address at the specified location. However, this isn't
guaranteed to be true, so you should validate that ``new_ptr`` points to a
specific virtual address before using it.

.. _free_virtual_memory:

Free virtual memory
-------------------

To free the memory allocated in this manner, use the corresponding free
functions. To unmap the memory, use :cpp:func:`hipMemUnmap`. To release the
virtual address range, use :cpp:func:`hipMemAddressFree`.  Finally, to release
the physical memory, use :cpp:func:`hipMemRelease`. A side effect of these
functions is the lack of synchronization when memory is released. If you call
:cpp:func:`hipFree` when you have multiple streams running in parallel, it
synchronizes the device. This causes worse resource usage and performance.

.. code-block:: cpp

    hipMemUnmap(ptr, size);
    hipMemRelease(allocHandle);
    hipMemAddressFree(ptr, size);

Example code
============

The virtual memory management example follows these steps:

1. Check virtual memory management :ref:`support <vmm_support>`:
   The :cpp:func:`hipDeviceGetAttribute` function is used to check the virtual
   memory management support of the GPU with ID 0.

2. Physical memory :ref:`allocation <allocate_physical_memory>`: Physical memory
   is allocated using :cpp:func:`hipMemCreate` with pinned memory on the
   device.

3. Virtual memory :ref:`reservation <reserve_virtual_address>`: Virtual address
   range is reserved using :cpp:func:`hipMemAddressReserve`.

4. Mapping virtual address to physical memory: The physical memory is mapped
   to a virtual address (``virtualPointer``) using :cpp:func:`hipMemMap`.

5. Memory :ref:`access permissions<set_memory_access>`: Permission is set for
   pointer to allow read and write access using :cpp:func:`hipMemSetAccess`.

6. Memory operation: Data is written to the memory via ``virtualPointer``.

7. Launch kernels: The ``zeroAddr`` and ``fillAddr`` kernels are
   launched using the virtual memory pointer.

8. :ref:`Cleanup <free_virtual_memory>`: The mappings, physical memory, and
   virtual address are released at the end to avoid memory leaks.

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <iostream>

    #define ROUND_UP(SIZE,GRANULARITY) ((1 + SIZE / GRANULARITY) * GRANULARITY)

    #define HIP_CHECK(expression)              \
    {                                          \
        const hipError_t err = expression;     \
        if(err != hipSuccess){                 \
            std::cerr << "HIP error: "         \
                << hipGetErrorString(err)      \
                << " at " << __LINE__ << "\n"; \
        }                                      \
    }

    __global__ void zeroAddr(int* pointer) {
        *pointer = 0;
    }

    __global__ void fillAddr(int* pointer) {
        *pointer = 42;
    }


    int main() {

        int currentDev = 0;

        // Step 1: Check virtual memory management support on device 0
        int vmm = 0;
        HIP_CHECK(
            hipDeviceGetAttribute(
                &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, currentDev
            )
        );

        std::cout << "Virtual memory management support value: " << vmm << std::endl;

        if (vmm == 0) {
            std::cout << "GPU 0 doesn't support virtual memory management.";
            return 0;
        }

        // Size of memory to allocate
        size_t size = 4 * 1024;

        // Step 2: Allocate physical memory
        hipMemGenericAllocationHandle_t allocHandle;
        hipMemAllocationProp prop = {};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = currentDev;
        size_t granularity = 0;
        HIP_CHECK(
            hipMemGetAllocationGranularity(
                &granularity,
                &prop,
                hipMemAllocationGranularityMinimum));
        size_t padded_size = ROUND_UP(size, granularity);
        HIP_CHECK(hipMemCreate(&allocHandle, padded_size * 2, &prop, 0));

        // Step 3: Reserve a virtual memory address range
        void* virtualPointer = nullptr;
        HIP_CHECK(hipMemAddressReserve(&virtualPointer, padded_size, granularity, nullptr, 0));

        // Step 4: Map the physical memory to the virtual address range
        HIP_CHECK(hipMemMap(virtualPointer, padded_size, 0, allocHandle, 0));

        // Step 5: Set memory access permission for pointer
        hipMemAccessDesc accessDesc = {};
        accessDesc.location.type = hipMemLocationTypeDevice;
        accessDesc.location.id = currentDev;
        accessDesc.flags = hipMemAccessFlagsProtReadWrite;

        HIP_CHECK(hipMemSetAccess(virtualPointer, padded_size, &accessDesc, 1));

        // Step 6: Perform memory operation
        int value = 42;
        HIP_CHECK(hipMemcpy(virtualPointer, &value, sizeof(int), hipMemcpyHostToDevice));

        int result = 1;
        HIP_CHECK(hipMemcpy(&result, virtualPointer, sizeof(int), hipMemcpyDeviceToHost));
        if( result == 42) {
            std::cout << "Success. Value: " << result << std::endl;
        } else {
            std::cout << "Failure. Value: " << result << std::endl;
        }

        // Step 7: Launch kernels
        // Launch zeroAddr kernel
        zeroAddr<<<1, 1>>>((int*)virtualPointer);
        HIP_CHECK(hipDeviceSynchronize());

        // Check zeroAddr kernel result
        result = 1;
        HIP_CHECK(hipMemcpy(&result, virtualPointer, sizeof(int), hipMemcpyDeviceToHost));
        if( result == 0) {
            std::cout << "Success. zeroAddr kernel: " << result << std::endl;
        } else {
            std::cout << "Failure. zeroAddr kernel: " << result << std::endl;
        }

        // Launch fillAddr kernel
        fillAddr<<<1, 1>>>((int*)virtualPointer);
        HIP_CHECK(hipDeviceSynchronize());

        // Check fillAddr kernel result
        result = 1;
        HIP_CHECK(hipMemcpy(&result, virtualPointer, sizeof(int), hipMemcpyDeviceToHost));
        if( result == 42) {
            std::cout << "Success. fillAddr kernel: " << result << std::endl;
        } else {
            std::cout << "Failure. fillAddr kernel: " << result << std::endl;
        }

        // Step 8: Cleanup
        HIP_CHECK(hipMemUnmap(virtualPointer, padded_size));
        HIP_CHECK(hipMemRelease(allocHandle));
        HIP_CHECK(hipMemAddressFree(virtualPointer, padded_size));

        return 0;
    }

Virtual aliases
================================================================================

Virtual aliases are multiple virtual memory addresses mapping to the same
physical memory on the GPU. When this occurs, different threads, processes, or memory
allocations to access shared physical memory through different virtual
addresses on different devices.

Multiple virtual memory mappings can be created using multiple calls to
:cpp:func:`hipMemMap` on the same memory allocation.

.. note::

    RDNA cards may not produce correct results, if users access two different
    virtual addresses that map to the same physical address. In this case, the
    L1 data caches will be incoherent due to the virtual-to-physical aliasing.
    These GPUs will produce correct results if users access virtual-to-physical
    aliases using volatile pointers.

    NVIDIA GPUs require special fences to produce correct results when
    using virtual aliases.

In the following code block, the kernels input device pointers are virtual
aliases of the same memory allocation:

.. code-block:: cpp

    __global__ void updateBoth(int* pointerA, int* pointerB) {
        // May produce incorrect results on RDNA and NVIDIA cards.
        *pointerA = 0;
        *pointerB = 42;
    }

    __global__ void updateBoth_v2(volatile int* pointerA, volatile int* pointerB) {
        // May produce incorrect results on NVIDIA cards.
        *pointerA = 0;
        *pointerB = 42;
    }

