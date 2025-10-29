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

To achieve interprocess memory sharing, you can use either :ref:`device pointer <device-pointer>` or :ref:`shareable handle <shareable-handle>`. Both provide allocator (export) and consumer (import) interfaces.

.. _device-pointer:

Device pointer
--------------

To export data to share a memory pool pointer directly between processes, use ``hipMemPoolExportPointer()``. It allows you to share a memory allocation with another process.

.. code-block:: cpp

    #include <iostream>
    #include <fstream>
    #include <hip/hip_runtime.h>
    #include <sys/stat.h>

    int main() {
        // Allocate memory.
        void* devPtr;
        hipMalloc(&devPtr, sizeof(int));

        // Export the memory pool pointer.
        hipMemPoolPtrExportData exportData;
        hipError_t result = hipMemPoolExportPointer(&exportData, devPtr);
        if (result != hipSuccess) {
            std::cerr << "Error exporting memory pool pointer: " << hipGetErrorString(result) << std::endl;
            return 1;
        }

        // Create a named pipe (FIFO).
        const char* fifoPath = "/tmp/myfifo"; // Change this to a unique path.
        mkfifo(fifoPath, 0666);

        // Write the exported data to the named pipe.
        std::ofstream fifoStream(fifoPath, std::ios::out | std::ios::binary);
        fifoStream.write(reinterpret_cast<char*>(&exportData), sizeof(hipMemPoolPtrExportData));
        fifoStream.close();

        // Clean up.
        hipFree(devPtr);

        return 0;
    }

To import a memory pool pointer directly from another process, use ``hipMemPoolImportPointer()``.

Here is how to read the pool exported in the preceding example:

.. code-block:: cpp

    #include <iostream>
    #include <fstream>
    #include <hip/hip_runtime.h>

    int main() {
        // Considering that you have exported the memory pool pointer already.
        // Now, let's simulate reading the exported data from a named pipe (FIFO).
        const char* fifoPath = "/tmp/myfifo"; // Change this to a unique path.
        std::ifstream fifoStream(fifoPath, std::ios::in | std::ios::binary);

        if (!fifoStream.is_open()) {
            std::cerr << "Error opening FIFO file: " << fifoPath << std::endl;
            return 1;
        }

        // Read the exported data.
        hipMemPoolPtrExportData importData;
        fifoStream.read(reinterpret_cast<char*>(&importData), sizeof(hipMemPoolPtrExportData));
        fifoStream.close();

        if (fifoStream.fail()) {
            std::cerr << "Error reading from FIFO file." << std::endl;
            return 1;
        }

        // Create a memory pool with default properties.
        hipMemPoolProps poolProps = {};
        poolProps.allocType = hipMemAllocationTypePinned;
        poolProps.handleTypes = hipMemHandleTypePosixFileDescriptor;
        poolProps.location.type = hipMemLocationTypeDevice;
        poolProps.location.id = 0; // Assuming device 0.

        hipMemPool_t memPool;
        hipMemPoolCreate(&memPool, &poolProps);

        // Import the memory pool pointer.
        void* importedDevPtr;
        hipError_t result = hipMemPoolImportPointer(&importedDevPtr, memPool, &importData);
        if (result != hipSuccess) {
            std::cerr << "Error imported memory pool pointer: " << hipGetErrorString(result) << std::endl;
            return 1;
        }

        // Now you can use the importedDevPtr for your computations.

        // Clean up (free the memory).
        hipFree(importedDevPtr);

        return 0;
    }

.. _shareable-handle:

Shareable handle
----------------

To export a memory pool pointer to a shareable handle, use ``hipMemPoolExportToSharedHandle()``. This handle could be a file descriptor or a handle obtained from another process. The exported handle contains information about the memory pool, such as size, location, and other relevant details.

.. code-block:: cpp

    #include <iostream>
    #include <fstream>
    #include <hip/hip_runtime.h>
    #include <sys/stat.h>

    int main() {
        // Create a memory pool with default properties.
        hipMemPoolProps poolProps = {};
        poolProps.allocType = hipMemAllocationTypePinned;
        poolProps.handleTypes = hipMemHandleTypePosixFileDescriptor;
        poolProps.location.type = hipMemLocationTypeDevice;
        poolProps.location.id = 0; // Assuming device 0.

        hipMemPool_t memPool;
        hipError_t poolResult = hipMemPoolCreate(&memPool, &poolProps);
        if (poolResult != hipSuccess) {
            std::cerr << "Error creating memory pool: " << hipGetErrorString(poolResult) << std::endl;
            return 1;
        }

        // Allocate memory from the memory pool.
        void* devPtr;
        hipMallocFromPoolAsync(&devPtr, sizeof(int), memPool, 0);

        // Export the memory pool pointer.
        int descriptor;
        hipError_t result = hipMemPoolExportToShareableHandle(&descriptor, memPool, hipMemHandleTypePosixFileDescriptor, 0);
        if (result != hipSuccess) {
            std::cerr << "Error exporting memory pool pointer: " << hipGetErrorString(result) << std::endl;
            return 1;
        }

        // Create a named pipe (FIFO).
        const char* fifoPath = "/tmp/myfifo"; // Change this to a unique path.
        mkfifo(fifoPath, 0666);

        // Write the exported data to the named pipe.
        std::ofstream fifoStream(fifoPath, std::ios::out | std::ios::binary);
        fifoStream.write(reinterpret_cast<char*>(&descriptor), sizeof(int));
        fifoStream.close();

        // Clean up.
        hipFree(devPtr);
        hipMemPoolDestroy(memPool);

        return 0;
    }

To import and restore a memory pool pointer from a shareable handle, which could be a file descriptor or a handle obtained from another process, use ``hipMemPoolImportFromShareableHandle()``. The exported shareable handle data contains information about the memory pool, including its size, location, and other relevant details. Importing the handle provides a valid memory pointer to the same memory, which allows you to share memory across different contexts.

.. code-block:: cpp

    #include <iostream>
    #include <fstream>
    #include <hip/hip_runtime.h>

    int main() {
        // Considering that you have exported the memory pool pointer already.
        // Now, let's simulate reading the exported data from a named pipe (FIFO).
        const char* fifoPath = "/tmp/myfifo"; // Change this to a unique path
        std::ifstream fifoStream(fifoPath, std::ios::in | std::ios::binary);

        if (!fifoStream.is_open()) {
            std::cerr << "Error opening FIFO file: " << fifoPath << std::endl;
            return 1;
        }

        // Read the exported data.
        int descriptor;
        fifoStream.read(reinterpret_cast<char*>(&descriptor), sizeof(int));
        fifoStream.close();

        if (fifoStream.fail()) {
            std::cerr << "Error reading from FIFO file." << std::endl;
            return 1;
        }

        // Import the memory pool.
        hipMemPool_t memPool;
        hipError_t result = hipMemPoolImportFromShareableHandle(&memPool, &descriptor, hipMemHandleTypePosixFileDescriptor, 0);
        if (result != hipSuccess) {
            std::cerr << "Error importing memory pool: " << hipGetErrorString(result) << std::endl;
            return 1;
        }

        // Allocate memory from the imported memory pool.
        void* importedDevPtr;
        hipMallocFromPoolAsync(&importedDevPtr, sizeof(int), memPool, 0);

        // Now you can use the importedDevPtr for your computations.

        // Clean up (free the memory).
        hipFree(importedDevPtr);
        hipMemPoolDestroy(memPool);

        return 0;
    }
