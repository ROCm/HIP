.. meta::
  :description:
  :keywords: stream, memory allocation, SOMA, stream ordered memory allocator

*******************************************************************************
Stream Ordered Memory Allocator
*******************************************************************************

The Stream Ordered Memory Allocator (SOMA) is part of the HIP runtime API. It
provides an asynchronous memory allocation mechanism with stream-ordering
semantics. With SOMA, you can allocate and free memory in stream order,
ensuring that all asynchronous accesses occur between the stream executions of
allocation and de-allocation. Compliance with stream order prevents
use-before-allocation or use-after-free errors, which would otherwise lead to
undefined behavior.

Advantages of SOMA:

- Efficient Reuse: SOMA enables efficient memory reuse across streams, reducing
  unnecessary allocation overhead.
- Fine-Grained Control: You can set attributes and control caching behavior for
  memory pools.
- Inter-Process Sharing: Secure sharing of allocations between processes is
  possible.
- Optimizations: The driver can optimize based on its awareness of SOMA and
  other stream management APIs.

Disadvantages of SOMA:
- Temporal Constraints: Developers must adhere strictly to stream order to
  avoid errors.
- Complexity: Properly managing memory in stream order can be intricate.
- Learning Curve: Understanding and utilizing SOMA effectively may require
  additional effort.

How is Stream Ordered Memory Allocator Used?
============================================

Users can allocate memory using ``hipMallocAsync()`` with stream-ordered
semantics. This means that all asynchronous accesses to the allocation must
occur between the stream executions of the allocation and the free.
If memory is accessed outside of this promised stream order, it can lead to
undefined behavior (e.g., use before allocation or use after free errors).
The allocator may reallocate memory as long as it guarantees compliant memory
accesses will not overlap temporally. ``hipFreeAsync()`` frees memory from the
pool with stream-ordered semantics.

The following example explains how to use stream ordered memory allocation.

.. tab-set::

    .. tab-item::  Stream Ordered Memory Allocation

        .. code-block::cpp

            #include <iostream>
            #include <hip/hip_runtime.h>

            // Kernel to perform some computation on allocated memory.
            __global__ void myKernel(int* data, size_t numElements) {
                int tid = threadIdx.x + blockIdx.x * blockDim.x;
                if (tid < numElements) {
                    data[tid] = tid * 2;
                }
            }

            int main() {
                // Initialize HIP.
                hipInit(0);

                // Stream 0.
                constexpr hipStream_t streamId = 0;

                // Allocate memory with stream ordered semantics.
                constexpr size_t numElements = 1024;
                int* devData;
                hipMallocAsync(&devData, numElements * sizeof(*devData), streamId);

                // Launch the kernel to perform computation.
                dim3 blockSize(256);
                dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);
                myKernel<<<gridSize, blockSize>>>(devData, numElements);

                // Free memory with stream ordered semantics.
                hipFreeAsync(devData, streamId);

                // Synchronize to ensure completion.
                hipDeviceSynchronize();

                return 0;
            }

    .. tab-item::  Ordinary Allocation

        .. code-block::cpp

            #include <iostream>
            #include <hip/hip_runtime.h>

            // Kernel to perform some computation on allocated memory.
            __global__ void myKernel(int* data, size_t numElements) {
                int tid = threadIdx.x + blockIdx.x * blockDim.x;
                if (tid < numElements) {
                    data[tid] = tid * 2;
                }
            }

            int main() {
                // Initialize HIP.
                hipInit(0);

                // Allocate memory.
                constexpr size_t numElements = 1024;
                int* devData;
                hipMalloc(&devData, numElements * sizeof(*devData));

                // Launch the kernel to perform computation.
                dim3 blockSize(256);
                dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x);
                myKernel<<<gridSize, blockSize>>>(devData, numElements);

                // Free memory.
                hipFree(devData);

                // Synchronize to ensure completion.
                hipDeviceSynchronize();

                return 0;
            }

Memory Pools
============

Memory pools provide a way to manage memory with stream-ordered behavior,
ensuring proper synchronization and avoiding memory access errors. Division of
a single memory system into separate pools allows querying each partition's
access path properties. Memory pools are used for host memory, device memory,
and unified memory.

Set Pools
---------

The ``hipMallocAsync()`` function uses the current memory pool, while also
providing the opportunity to create and use different pools with the
``hipMemPoolCreate()`` and ``hipMallocFromPoolAsync()`` functions
respectively.

Unlike CUDA, where stream-ordered memory allocation can be implicit, in AMD
HIP, it's always explicit. This means that you need to manage memory allocation
for each stream, ensuring precise control over memory usage and
synchronization.

.. code-block::cpp

    #include <hip/hip_runtime.h>

    // Kernel to perform some computation on allocated memory.
    __global__ void myKernel(int* data, size_t numElements) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < numElements) {
            data[tid] = tid * 2;
        }
    }

    int main() {
        // Initialize HIP.
        hipInit(0);

        // Create a stream.
        hipStream_t stream;
        hipStreamCreate(&stream);

        // Allocate memory pool.
        hipDeviceptr_t pool;
        hipMalloc(&pool, 1024 * sizeof(int));

        // Allocate memory from the pool asynchronously.
        int* devData;
        hipMallocFromPoolAsync(&devData, 256 * sizeof(int), pool, stream);

        // Launch the kernel to perform computation.
        dim3 blockSize(256);
        dim3 gridSize(1);
        myKernel<<<gridSize, blockSize>>>(devData, 256);

        // Free the allocated memory.
        hipFreeAsync(devData, stream);

        // Destroy the stream and release the pool.
        hipStreamDestroy(stream);
        hipFree(pool);

        return 0;
    }

Trim Pools
----------

The memory allocator allows you to allocate and free memory in stream order.
To control memory usage, the release threshold attribute can be set by
``hipMemPoolAttrReleaseThreshold``. This threshold specifies the amount of
reserved memory in bytes that a pool should hold onto before attempting to
release memory back to the operating system.

.. code-block::cpp
    uint64_t threshold = UINT64_MAX;
    hipMemPoolSetAttribute(memPool, hipMemPoolAttrReleaseThreshold, &threshold);

When more than the specified threshold bytes of memory are held by the
memory pool, the allocator will try to release memory back to the operating
system during the next call to stream, event, or context synchronization.

Sometimes for a better performance it is a good practice to adjust the memory
pool size with ``hipMemPoolTrimTo()``. It can be useful to reclaim memory from
a memory pool that is larger than necessary, optimizing memory usage for your
application.

.. code-block::cpp

    #include <hip/hip_runtime.h>
    #include <iostream>

    int main() {
        hipMemPool_t memPool;
        hipDevice_t device = 0; // Specify the device index

        // Create a memory pool.
        hipMemPoolCreate(&memPool, 0, 0);

        // Allocate memory from the pool (e.g., 1 MB).
        size_t allocSize = 1 * 1024 * 1024;
        void* ptr;
        hipMalloc(&ptr, allocSize);

        // Free the allocated memory.
        hipFree(ptr);

        // Trim the memory pool to a specific size (e.g., 512 KB).
        size_t newSize = 512 * 1024;
        hipMemPoolTrimTo(memPool, newSize);

        // Clean up.
        hipMemPoolDestroy(memPool);

        std::cout << "Memory pool trimmed to " << newSize << " bytes." << std::endl;
        return 0;
    }


Resource Usage Statistics
-------------------------
Resource usage statistics can help in optimization. The following pool
attributes to query memory usage:

    - ``hipMemPoolAttrReservedMemCurrent`` returns the current total physical
      GPU memory consumed by the pool.
    - ``hipMemPoolAttrUsedMemCurrent`` returns the total size of all memory
      allocated from the pool.
    - ``hipMemPoolAttrReservedMemHigh`` returns the total physical GPU memory
      consumed by the pool since the last reset.
    - ``hipMemPoolAttrUsedMemHigh`` returns the all memory allocated from the
      pool since the last reset.

You can reset them to the current value using the ``hipMemPoolSetAttribute()``.

.. code-block::cpp

    #include <hip/hip_runtime.h>

    // sample helper functions for getting the usage statistics in bulk
    struct usageStatistics {
        uint64_t reservedMemCurrent;
        uint64_t reservedMemHigh;
        uint64_t usedMemCurrent;
        uint64_t usedMemHigh;
    };

    void getUsageStatistics(hipMemoryPool_t memPool, struct usageStatistics *statistics)
    {
        hipMemPoolGetAttribute(memPool, hipMemPoolAttrReservedMemCurrent, &statistics->reservedMemCurrent);
        hipMemPoolGetAttribute(memPool, hipMemPoolAttrReservedMemHigh, &statistics->reservedMemHigh);
        hipMemPoolGetAttribute(memPool, hipMemPoolAttrUsedMemCurrent, &statistics->usedMemCurrent);
        hipMemPoolGetAttribute(memPool, hipMemPoolAttrUsedMemHigh, &statistics->usedMemHigh);
    }

    // resetting the watermarks will make them take on the current value.
    void resetStatistics(hipMemoryPool_t memPool)
    {
        uint64_t value = 0;
        hipMemPoolSetAttribute(memPool, hipMemPoolAttrReservedMemHigh, &value);
        hipMemPoolSetAttribute(memPool, hipMemPoolAttrUsedMemHigh, &value);
}


Memory Reuse Policies
---------------------

The allocator may reallocate memory as long as it guarantees that compliant
memory accesses won't overlap temporally. Turning on and of the following
memory pool reuse policy attribute flags can optimize the memory use:

    - ``hipMemPoolReuseFollowEventDependencies`` checks event
      dependencies before allocating additional GPU memory.
    - ``hipMemPoolReuseAllowOpportunistic`` checks freed allocations to
      determine if the stream order semantic indicated by the free operation
      has been met.
    - ``hipMemPoolReuseAllowInternalDependencies`` manages reuse based on
      internal dependencies in runtime. If the driver fails to allocate and map
      additional physical memory, it will search for memory that relies on
      another stream's pending progress and reuse it.

Device Accessibility for Multi-GPU Support
------------------------------------------

Allocations are initially accessible only from the device where they reside.

Inter-process Memory Handling
=============================

Inter-process capable (IPC) memory pools facilitate efficient and secure
sharing of GPU memory between processes.

There are two ways for inter-process memory sharing: pointer sharing or
shareable handles. Both have allocator (export) and consumer (import)
interface.

Device Pointer
--------------

The ``hipMemPoolExportPointer()`` function allows to export data to share a
memory pool pointer directly between processes. It is useful to share a memory
allocation with another process.

.. code-block::cpp

    #include <iostream>
    #include <fstream>
    #include <hip/hip_runtime.h>

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

The ``hipMemPoolImportPointer()`` function allows to import a memory pool
pointer directly from another process.

Here is the example code to read the exported
pool from the previous example.

.. code-block::cpp

    #include <iostream>
    #include <fstream>
    #include <hip/hip_runtime.h>

    int main() {

        // Assume you previously exported the memory pool pointer.
        // Now, let's simulate reading the exported data from a named pipe (FIFO).
        const char* fifoPath = "/tmp/myfifo"; // Change this to a unique path.
        std::ifstream fifoStream(fifoPath, std::ios::in | std::ios::binary);

        // Read the exported data.
        hipMemPoolPtrExportData importData;
        fifoStream.read(reinterpret_cast<char*>(&importData), sizeof(hipMemPoolPtrExportData));
        fifoStream.close();

        // Import the memory pool pointer.
        void* importedDevPtr;
        hipError_t result = hipMemPoolImportPointer(importData, &importedDevPtr);
        if (result != hipSuccess) {
            std::cerr << "Error imported memory pool pointer: " << hipGetErrorString(result) << std::endl;
            return 1;
        }

        // Now you can use the importedDevPtr for your computations.

        // Clean up (free the memory).
        hipFree(importedDevPtr);

        return 0;
    }

Shareable Handle
----------------

The ``hipMemPoolExportToSharedHandle()`` is used to export a memory pool
pointer to a shareable handle. This handle can be a file descriptor or a handle
obtained from another process. The exported handle contains information about
the memory pool, including its size, location, and other relevant details.

.. code-block::cpp

    #include <iostream>
    #include <fstream>
    #include <hip/hip_runtime.h>

    int main() {
        // Allocate memory.
        void* devPtr;
        hipMalloc(&devPtr, sizeof(int));

        // Export the memory pool pointer.
        hipMemPoolPtrExportData exportData;
        hipError_t result = hipMemPoolExportToShareableHandle(&exportData, devPtr);
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

The ``hipMemPoolImportFromShareableHandle()`` function is used to import a
memory pool pointer from a shareable handle -- such as a file descriptor or a
handle obtained from another process. It allows to restore a memory pool
pointer that was previously exported using ``hipMemPoolExportPointer()`` or a
similar mechanism. The exported shareable handle data contains information
about the memory pool, including its size, location, and other relevant
details. After importing, valid memory pointer is received that points to the
same memory area. Useful for inter-process communication or sharing memory
across different contexts.

.. code-block::cpp

    #include <iostream>
    #include <fstream>
    #include <hip/hip_runtime.h>

    int main() {
        // Assume you previously exported the memory pool pointer.
        // Now, let's simulate reading the exported data from a named pipe (FIFO).
        const char* fifoPath = "/tmp/myfifo"; // Change this to a unique path
        std::ifstream fifoStream(fifoPath, std::ios::in | std::ios::binary);

        // Read the exported data.
        hipMemPoolPtrExportData importData;
        fifoStream.read(reinterpret_cast<char*>(&importData), sizeof(hipMemPoolPtrExportData));
        fifoStream.close();

        // Import the memory pool pointer.
        void* importedDevPtr;
        hipError_t result = hipMemPoolImportFromShareableHandle(importData, &importedDevPtr);
        if (result != hipSuccess) {
            std::cerr << "Error importing memory pool pointer: " << hipGetErrorString(result) << std::endl;
            return 1;
        }

        // Now you can use the importedDevPtr for your computations.

        // Clean up (free the memory).
        hipFree(importedDevPtr);

        return 0;
    }
