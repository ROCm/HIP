.. meta::
  :description: This chapter describes introduces Unified Memory (UM) and shows
                how to use it in AMD HIP.
  :keywords: AMD, ROCm, HIP, CUDA, unified memory, unified, memory, UM, APU

.. _unified_memory:

*******************************************************************************
Unified memory management
*******************************************************************************

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

.. figure:: ../../../data/how-to/hip_runtime_api/memory_management/unified_memory/um.svg

.. _unified memory system requirements:

System requirements
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
page-fault. For more details, visit
`GPU memory <https://rocm.docs.amd.com/en/latest/conceptual/gpu-memory.html#xnack>`_.

.. _unified memory programming models:

Unified memory programming models
=================================

Showcasing various unified memory programming models, the model availability
depends on your architecture. For more information, see :ref:`unified memory
system requirements` and :ref:`checking unified memory management support`.

- **HIP managed memory allocation API**:

  The ``hipMallocManaged()`` is a dynamic memory allocator available on
  all GPUs with unified memory support. For more details, visit
  :ref:`unified_memory_reference`.

- **HIP managed variables**:

  The ``__managed__`` declaration specifier, which serves as its counterpart,
  is supported on all modern AMD cards and can be utilized for static
  allocation.

- **System allocation API**:

  Starting with the AMD MI300 series, the ``malloc()`` system allocator allows
  you to reserve unified memory. The system allocator is more versatile and
  offers an easy transition from a CPU written C++ code to a HIP code as the
  same system allocation API is used.

.. _checking unified memory management support:

Checking unified memory management support
------------------------------------------

Some device attributes can offer information about which :ref:`unified memory
programming models` are supported. The attribute value is 1 if the
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

Example for unified memory management
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

Using unified memory management (UMM)
=====================================

Unified memory management (UMM) is a feature that can simplify the complexities
of memory management in GPU computing. It is particularly useful in
heterogeneous computing environments with heavy memory usage with both a CPU
and a GPU, which would require large memory transfers. Here are some areas
where UMM can be beneficial:

- **Simplification of Memory Management**:

  UMM can help to simplify the complexities of memory management. This can make
  it easier for developers to write code without worrying about memory
  allocation and deallocation details.

- **Data Migration**:

  UMM allows for efficient data migration between the host (CPU) and the device
  (GPU). This can be particularly useful for applications that need to move
  data back and forth between the device and host.

- **Improved Programming Productivity**:

  As a positive side effect, UMM can reduce the lines of code, thereby
  improving programming productivity.

In HIP, pinned memory allocations are coherent by default. Pinned memory is
host memory mapped into the address space of all GPUs, meaning that the pointer
can be used on both host and device. Using pinned memory instead of pageable
memory on the host can improve bandwidth.

While UMM can provide numerous benefits, it's important to be aware of the
potential performance overhead associated with UMM. You must thoroughly test
and profile your code to ensure it's the most suitable choice for your use
case.

.. _unified memory runtime hints:

Unified memory HIP runtime hints for the better performance
===========================================================

Unified memory HIP runtime hints can help improve the performance of your code if
you know your code's ability and infrastructure. Some hint techniques are
presented in this section.

The hint functions can set actions on a selected device, which can be
identified by ``hipGetDeviceProperties(&prop, device_id)``. There are two
special ``device_id`` values:

- ``hipCpuDeviceId`` = -1 means that the advised device is the CPU.
- ``hipInvalidDeviceId`` = -2 means that the device is invalid.

For the best performance, profile your application to optimize the
utilization of HIP runtime hints.

Data prefetching
----------------

Data prefetching is a technique used to improve the performance of your
application by moving data closer to the processing unit before it's actually
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
the prefetch operations are completed successfully.

Memory advice
-------------

The effectiveness of ``hipMemAdvise()`` comes from its ability to inform the
runtime system of the developer's intentions regarding memory usage. When the
runtime system has knowledge of the expected memory access patterns, it can
make better decisions about data placement and caching, leading to more
efficient execution of the application. However, the actual impact on
performance can vary based on the specific use case and the hardware
architecture.

For the description of ``hipMemAdvise()`` and the detailed list of advice,
visit the :ref:`unified_memory_reference`.

Here is the updated version of the example above with memory advice.

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

        // Allocate memory for a, b, and c accessible to both device and host codes.
        hipMallocManaged(&a, sizeof(*a));
        hipMallocManaged(&b, sizeof(*b));
        hipMallocManaged(&c, sizeof(*c));

        // Set memory advice for a, b, and c to be accessed by the CPU.
        hipMemAdvise(a, sizeof(*a), hipMemAdviseSetPreferredLocation, hipCpuDeviceId);
        hipMemAdvise(b, sizeof(*b), hipMemAdviseSetPreferredLocation, hipCpuDeviceId);
        hipMemAdvise(c, sizeof(*c), hipMemAdviseSetPreferredLocation, hipCpuDeviceId);

        // Additionally, set memory advice for a, b, and c to be read mostly from the device 0.
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


Memory range attributes
-----------------------

Memory Range attributes allow you to query attributes of a given memory range.

The ``hipMemRangeGetAttribute()`` is added to the example to query the
``hipMemRangeAttributeReadMostly`` attribute of the memory range pointed to by
``a``. The result is stored in ``attributeValue`` and then printed out.

For more details, visit the
:ref:`unified_memory_reference`.

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

Asynchronously attach memory to a stream
----------------------------------------

The ``hipStreamAttachMemAsync`` function would be able to asynchronously attach memory to a stream, which can help concurrent execution when using streams.

Currently, this function is a no-operation (NOP) function on AMD GPUs. It simply returns success after the runtime memory validation passed. This function is necessary on Microsoft Windows, and UMM is not supported on this operating system with AMD GPUs at the moment.
