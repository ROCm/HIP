.. meta::
  :description: This chapter describes introduces Unified Memory (UM) and shows
                how to use it in AMD HIP.
  :keywords: AMD, ROCm, HIP, CUDA, unified memory, unified, memory, UM, APU

*******************************************************************************
Unified Memory
*******************************************************************************

Introduction
============
In conventional architectures, CPUs and GPUs have dedicated memory like Random
Access Memory (RAM) and Video Random Access Memory (VRAM). This architectural
design, while effective, can be limiting in terms of memory capacity and
bandwidth, as continuous memory copying is required to allow the processors to
access the appropriate data. New architectural features like Heterogeneous
System Architectures (HSA) and Unified Memory (UM) help avoid these limitations
and promise increased efficiency and innovation.

Unified memory
==============
Unified Memory is a single memory address space accessible from any processor
within a system. This setup simplifies memory management processes and enables
applications to allocate data that can be read or written by code running on
either CPUs or GPUs. The Unified memory model is shown in the following figure.

.. figure:: ../data/unified_memory/um.svg

AMD Accelerated Processing Unit (APU) is a typical example of a Unified Memory
Architecture. On a single die, a central processing unit (CPU) is combined
with an integrated graphics processing unit (iGPU), and both have access to a
high-bandwidth memory (HBM) module named Unified Memory. The CPU enables
high-performance, low-latency operations, while the GPU is optimized for high
throughput (data processed by unit time).

.. _unified memory system requirements:

System Requirements
===================
Unified memory is supported on Linux by all modern AMD GPUs from the Vega
series onward. Unified memory management can be achieved with managed memory
allocation and, for the latest GPUs, with a system allocator.

The table below lists the supported allocators. The allocators are described in
the next section.

.. list-table:: Supported Unified Memory Allocators
    :widths: 40, 25, 25, 25
    :header-rows: 1
    :align: center

    * - Architecture
      - ``hipMallocManaged()``
      - ``__managed__``
      - ``malloc()``
    * - MI200, MI300 Series
      - ✅
      - ✅
      - ✅ :sup:`1`
    * - MI100
      - ✅
      - ✅
      - ❌
    * - RDNA (Navi) Series
      - ✅
      - ✅
      - ❌
    * - GCN5 (Vega) Series
      - ✅
      - ✅
      - ❌

✅: **Supported**

❌: **Unsupported**

:sup:`1` Works only with ``XNACK=1``. First GPU access causes recoverable
page-fault.

.. _unified memory programming models:

Unified Memory Programming Models
=================================

Showcasing various unified memory programming models, their availability
depends on your architecture. For further details, visit :ref:`unified memory
system requirements` and :ref:`checking unified memory management support`.

- **HIP Managed Memory Allocation API**:
The ``hipMallocManaged()`` is a dynamic memory allocator that is available on
all GPUs with unified memory support. For more details, visit :doc:`reference
page <reference/unified_memory_reference>`.

- **HIP Managed Variables**:
The ``__managed__`` declaration specifier, which serves as its counterpart, is
supported on all modern AMD cards and can be utilized for static allocation.

- **System Allocation API**:
Starting with the MI300 series, the ``malloc()`` system allocator allows you
to reserve unified memory. The system allocator is more versatile, and it
offers an easy transition from a CPU written C++ code to a HIP code as the same
system allocation API is used.

.. _checking unified memory management support:

Checking Unified Memory Management Support
------------------------------------------
Some device attribute can offer information about which :ref:`unified memory
programming models` are supported. The attribute value is an integer 1 if the
functionality is supported, and 0 if it is not supported.

.. list-table:: Device attributes for unified memory management
    :widths: 40, 60
    :header-rows: 1
    :align: center

    * - attribute
      - description
    * - ``hipDeviceAttributeManagedMemory``
      - unified addressing is supported
    * - ``hipDeviceAttributeConcurrentManagedAccess``
      - full managed memory support, concurrent access is supported
    * - ``hipDeviceAttributePageableMemoryAccess``
      - both managed and system memory allocation API is supported

The following examples show how to use device attributes:

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <iostream>

    int main() {
        int d;
        hipGetDevice(&d);

        int is_cma = 0;
        hipDeviceGetAttribute(&is_cma, hipDeviceAttributeConcurrentManagedAccess, d);
        std::cout << "HIP Managed Memory: "
                  << (is_cma == 1 ? "is" : "NOT")
                  << " supported" << std::endl;
        return 0;
    }

Example for Unified Memory Management
-------------------------------------

The following example shows how to use unified memory management with
``hipMallocManaged()``, function, with ``__managed__`` attribute for static
allocation and standard  ``malloc()`` allocation. For comparison, the Explicit
Memory Management example is presented in the last tab.

.. tab-set::

    .. tab-item:: hipMallocManaged()

        .. code-block:: cpp
            :emphasize-lines: 12-15

            #include <hip/hip_runtime.h>
            #include <iostream>

            // Addition of two values.
            __global__ void add(int *a, int *b, int *c) {
                *c = *a + *b;
            }

            int main() {
                int *a, *b, *c;

                // Allocate memory for a, b and c that is accessible to both device and host codes.
                hipMallocManaged(&a, sizeof(*a));
                hipMallocManaged(&b, sizeof(*b));
                hipMallocManaged(&c, sizeof(*c));

                // Setup input values.
                *a = 1;
                *b = 2;

                // Launch add() kernel on GPU.
                hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, a, b, c);

                // Wait for GPU to finish before accessing on host.
                hipDeviceSynchronize();

                // Prints the result.
                std::cout << *a << " + " << *b << " = " << *c << std::endl;

                // Cleanup allocated memory.
                hipFree(a);
                hipFree(b);
                hipFree(c);

                return 0;
            }


    .. tab-item:: __managed__

        .. code-block:: cpp
            :emphasize-lines: 9-10

            #include <hip/hip_runtime.h>
            #include <iostream>

            // Addition of two values.
            __global__ void add(int *a, int *b, int *c) {
                *c = *a + *b;
            }

            // Declare a, b and c as static variables.
            __managed__ int a, b, c;

            int main() {
                // Setup input values.
                a = 1;
                b = 2;

                // Launch add() kernel on GPU.
                hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, &a, &b, &c);

                // Wait for GPU to finish before accessing on host.
                hipDeviceSynchronize();

                // Prints the result.
                std::cout << a << " + " << b << " = " << c << std::endl;

                return 0;
            }


    .. tab-item:: malloc()

        .. code-block:: cpp
            :emphasize-lines: 12-15

            #include <hip/hip_runtime.h>
            #include <iostream>

            // Addition of two values.
            __global__ void add(int* a, int* b, int* c) {
                *c = *a + *b;
            }

            int main() {
                int* a, * b, * c;

                // Allocate memory for a, b, and c.
                a = (int*)malloc(sizeof(*a));
                b = (int*)malloc(sizeof(*b));
                c = (int*)malloc(sizeof(*c));

                // Setup input values.
                *a = 1;
                *b = 2;

                // Launch add() kernel on GPU.
                hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, a, b, c);

                // Wait for GPU to finish before accessing on host.
                hipDeviceSynchronize();

                // Prints the result.
                std::cout << *a << " + " << *b << " = " << *c << std::endl;

                // Cleanup allocated memory.
                free(a);
                free(b);
                free(c);

                return 0;
            }


    .. tab-item:: Explicit Memory Management

        .. code-block:: cpp
            :emphasize-lines: 17-24, 29-30

            #include <hip/hip_runtime.h>
            #include <iostream>

            // Addition of two values.
            __global__ void add(int *a, int *b, int *c) {
                *c = *a + *b;
            }

            int main() {
                int a, b, c;
                int *d_a, *d_b, *d_c;

                // Setup input values.
                a = 1;
                b = 2;

                // Allocate device copies of a, b and c.
                hipMalloc(&d_a, sizeof(*d_a));
                hipMalloc(&d_b, sizeof(*d_b));
                hipMalloc(&d_c, sizeof(*d_c));

                // Copy input values to device.
                hipMemcpy(d_a, &a, sizeof(*d_a), hipMemcpyHostToDevice);
                hipMemcpy(d_b, &b, sizeof(*d_b), hipMemcpyHostToDevice);

                // Launch add() kernel on GPU.
                hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, d_a, d_b, d_c);

                // Copy the result back to the host.
                hipMemcpy(&c, d_c, sizeof(*d_c), hipMemcpyDeviceToHost);

                // Cleanup allocated memory.
                hipFree(d_a);
                hipFree(d_b);
                hipFree(d_c);

                // Prints the result.
                std::cout << a << " + " << b << " = " << c << std::endl;

                return 0;
            }

.. _using unified memory management:

Using Unified Memory Management (UMM)
=====================================
Unified Memory Management (UMM) is a feature that can simplify the complexities
of memory management in GPU computing. It is particularly useful in
heterogeneous computing environments with heavy memory usage with both a CPU
and a GPU, which would require large memory transfers. Here are some areas
where UMM can be beneficial:

- **Simplification of Memory Management**:
UMM can help to simplify the complexities of memory management. This can make
it easier for developers to write code without worrying about memory allocation
and deallocation details.

- **Data Migration**:
UMM allows for efficient data migration between the host (CPU) and the device
(GPU). This can be particularly useful for applications that need to move data
back and forth between the device and host.

- **Improved Programming Productivity**:
As a positive side effect, the use of UMM can reduce the lines of code,
thereby improving programming productivity.

In HIP, pinned memory allocations are coherent by default. Pinned memory is
host memory mapped into the address space of all GPUs, meaning that the pointer
can be used on both host and device. Using pinned memory instead of pageable
memory on the host can improve bandwidth.

While UMM can provide numerous benefits, it is important to be aware of the
potential performance overhead associated with UMM. You must thoroughly test
and profile your code to ensure it is the most suitable choice for your use
case.

.. _unified memory compiler hints:

Unified Memory Compiler Hints for the Better Performance
========================================================

Unified memory compiler hints can help to improve the performance of your code,
if you know the ability of your code and the infrastructure that you use. Some
hint techniques are presented in this section.

The hint functions can set actions on a selected device, which can be
identified by ``hipGetDeviceProperties(&prop, device_id)``. There are two
special ``device_id`` values:

- ``hipCpuDeviceId`` = -1 means that the advised device is the CPU.
- ``hipInvalidDeviceId`` = -2 means that the device is invalid.

For the best performance you can profile your application to optimize the
utilization of compiler hits.

Data Prefetching
----------------
Data prefetching is a technique used to improve the performance of your
application by moving data closer to the processing unit before it is actually
needed.

.. code-block:: cpp
    :emphasize-lines: 20-23,31-32

    // Addition of two values.
    __global__ void add(int *a, int *b, int *c) {
        *c = *a + *b;
    }

    int main() {
        int *a, *b, *c;
        int deviceId;
        hipGetDevice(&deviceId); // Get the current device ID

        // Allocate memory for a, b and c that is accessible to both device and host codes.
        hipMallocManaged(&a, sizeof(*a));
        hipMallocManaged(&b, sizeof(*b));
        hipMallocManaged(&c, sizeof(*c));

        // Setup input values.
        *a = 1;
        *b = 2;

        // Prefetch the data to the GPU device.
        hipMemPrefetchAsync(a, sizeof(*a), deviceId, 0);
        hipMemPrefetchAsync(b, sizeof(*b), deviceId, 0);
        hipMemPrefetchAsync(c, sizeof(*c), deviceId, 0);

        // Launch add() kernel on GPU.
        hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, a, b, c);

        // Wait for GPU to finish before accessing on host.
        hipDeviceSynchronize();

        // Prefetch the result back to the CPU.
        hipMemPrefetchAsync(c, sizeof(*c), hipCpuDeviceId, 0);

        // Wait for the prefetch operations to complete.
        hipDeviceSynchronize();

        // Prints the result.
        std::cout << *a << " + " << *b << " = " << *c << std::endl;

        // Cleanup allocated memory.
        hipFree(a);
        hipFree(b);
        hipFree(c);

        return 0;
    }

Remember to check the return status of ``hipMemPrefetchAsync()`` to ensure that
the prefetch operations complete successfully!

Memory Advise
-------------
The effectiveness of ``hipMemAdvise()`` comes from its ability to inform the
runtime system of the developer's intentions regarding memory usage. When the
runtime system has knowledge of the expected memory access patterns, it can
make better decisions about data placement and caching, leading to more
efficient execution of the application. However, the actual impact on
performance can vary based on the specific use case and the hardware
architecture.

For the description of ``hipMemAdvise()`` and the detailed list of advises,
visit the :doc:`reference page <reference/unified_memory_reference>`.

Here is the updated version of the example above with memory advises.

.. code-block:: cpp
    :emphasize-lines: 17-26

    #include <hip/hip_runtime.h>
    #include <iostream>

    // Addition of two values.
    __global__ void add(int *a, int *b, int *c) {
        *c = *a + *b;
    }

    int main() {
        int *a, *b, *c;

        // Allocate memory for a, b and c that is accessible to both device and host codes.
        hipMallocManaged(&a, sizeof(*a));
        hipMallocManaged(&b, sizeof(*b));
        hipMallocManaged(&c, sizeof(*c));

        // Set memory advise for a, b, and c to be accessed by the CPU.
        hipMemAdvise(a, sizeof(*a), hipMemAdviseSetPreferredLocation, hipCpuDeviceId);
        hipMemAdvise(b, sizeof(*b), hipMemAdviseSetPreferredLocation, hipCpuDeviceId);
        hipMemAdvise(c, sizeof(*c), hipMemAdviseSetPreferredLocation, hipCpuDeviceId);

        // Additionally, set memory advise for a, b, and c to be read mostly from the device 0.
        constexpr int device = 0;
        hipMemAdvise(a, sizeof(*a), hipMemAdviseSetReadMostly, device);
        hipMemAdvise(b, sizeof(*b), hipMemAdviseSetReadMostly, device);
        hipMemAdvise(c, sizeof(*c), hipMemAdviseSetReadMostly, device);

        // Setup input values.
        *a = 1;
        *b = 2;

        // Launch add() kernel on GPU.
        hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, a, b, c);

        // Wait for GPU to finish before accessing on host.
        hipDeviceSynchronize();

        // Prints the result.
        std::cout << *a << " + " << *b << " = " << *c << std::endl;

        // Cleanup allocated memory.
        hipFree(a);
        hipFree(b);
        hipFree(c);

        return 0;
    }


Memory Range attributes
-----------------------
Memory Range attributes allow you to query attributes of a given memory range.

The ``hipMemRangeGetAttribute()`` is added to the example to query the
``hipMemRangeAttributeReadMostly`` attribute of the memory range pointed to by
``a``. The result is stored in ``attributeValue`` and then printed out.

For more details, visit the
:doc:`reference page <reference/unified_memory_reference>`.

.. code-block:: cpp
    :emphasize-lines: 29-34

    #include <hip/hip_runtime.h>
    #include <iostream>

    // Addition of two values.
    __global__ void add(int *a, int *b, int *c) {
        *c = *a + *b;
    }

    int main() {
        int *a, *b, *c;
        unsigned int attributeValue;
        constexpr size_t attributeSize = sizeof(attributeValue);

        // Allocate memory for a, b and c that is accessible to both device and host codes.
        hipMallocManaged(&a, sizeof(*a));
        hipMallocManaged(&b, sizeof(*b));
        hipMallocManaged(&c, sizeof(*c));

        // Setup input values.
        *a = 1;
        *b = 2;

        // Launch add() kernel on GPU.
        hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, a, b, c);

        // Wait for GPU to finish before accessing on host.
        hipDeviceSynchronize();

        // Query an attribute of the memory range.
        hipMemRangeGetAttribute(&attributeValue,
                                attributeSize,
                                hipMemRangeAttributeReadMostly,
                                a,
                                sizeof(*a));

        // Prints the result.
        std::cout << *a << " + " << *b << " = " << *c << std::endl;
        std::cout << "The queried attribute value is: " << attributeValue << std::endl;

        // Cleanup allocated memory.
        hipFree(a);
        hipFree(b);
        hipFree(c);

        return 0;
    }

Asynchronously Attach Memory to a Stream
----------------------------------------

The ``hipStreamAttachMemAsync`` function is used to asynchronously attach
memory to a stream, which can help with concurrent execution when using
streams.

In the example, a stream is created by using ``hipStreamCreate()`` and then
the managed memory is attached to the stream using
``hipStreamAttachMemAsync()``. The ``hipMemAttachGlobal`` flag is used to
indicate that the memory can be accessed from any stream on any device.
The kernel launch and synchronization are now done on this stream.
Using streams and attaching memory to them can help with overlapping data
transfers and computation.

For more details and description of flags, visit
:doc:`reference page <reference/unified_memory_reference>`.

.. code-block:: cpp
    :emphasize-lines: 21-24

    #include <hip/hip_runtime.h>
    #include <iostream>

    // Addition of two values.
    __global__ void add(int *a, int *b, int *c) {
        *c = *a + *b;
    }

    int main() {
        int *a, *b, *c;
        hipStream_t stream;

        // Create a stream.
        hipStreamCreate(&stream);

        // Allocate memory for a, b and c that is accessible to both device and host codes.
        hipMallocManaged(&a, sizeof(*a));
        hipMallocManaged(&b, sizeof(*b));
        hipMallocManaged(&c, sizeof(*c));

        // Attach memory to the stream asynchronously.
        hipStreamAttachMemAsync(stream, a, sizeof(*a), hipMemAttachGlobal);
        hipStreamAttachMemAsync(stream, b, sizeof(*b), hipMemAttachGlobal);
        hipStreamAttachMemAsync(stream, c, sizeof(*c), hipMemAttachGlobal);

        // Setup input values.
        *a = 1;
        *b = 2;

        // Launch add() kernel on GPU on the created stream.
        hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, stream, a, b, c);

        // Wait for stream to finish before accessing on host.
        hipStreamSynchronize(stream);

        // Prints the result.
        std::cout << *a << " + " << *b << " = " << *c << std::endl;

        // Cleanup allocated memory.
        hipFree(a);
        hipFree(b);
        hipFree(c);

        // Destroy the stream.
        hipStreamDestroy(stream);

        return 0;
    }
