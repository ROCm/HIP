.. meta::
  :description: This chapter describes the host memory of the HIP ecosystem
                ROCm software.
  :keywords: AMD, ROCm, HIP, host memory

.. _host_memory:

********************************************************************************
Host memory
********************************************************************************

Host memory is "normal" memory as allocated by C or C++, and resides in the RAM of the host.
Host memory can be allocated in two different ways:

* Pageable memory
* Pinned memory

These different types of memory should be used at different use cases. The
pageable and pinned memory using explicit memory management, where the
developers have direct control over memory operations, while at the unified
memory case the developer gets a simplified memory model with less control over
low level memory operations.

The data transfer differences between the pageable or pinned memory usage
represented in the next figure.

.. figure:: ../../../data/how-to/hip_runtime_api/memory_management/pageable_pinned.svg

The difference between memory transfers of explicit memory management and unified memory management are highlighted in the following figure.

.. figure:: ../../../data/how-to/hip_runtime_api/memory_management/unified_memory/um.svg

Unified memory management is described further in :doc:`/how-to/hip_runtime_api/memory_management/unified_memory`.

.. _pageable_host_memory:

Pageable memory
================================================================================

Pageable memory is exists on "pages" (blocks of memory), which can be
migrated to other memory storage. For example, migrating memory between CPU
sockets on a motherboard, or a system that runs out of space in RAM and starts
dumping pages of RAM into the swap partition of your hard drive.

Pageable memory is usually gotten when calling ``malloc`` or ``new`` in a C++
application. The following example shows the pageable host memory usage in HIP.

.. code-block:: cpp

  #include <hip/hip_runtime.h>
  #include <iostream>

  #define HIP_CHECK(expression)                  \
  {                                              \
      const hipError_t status = expression;      \
      if(status != hipSuccess){                  \
          std::cerr << "HIP error "              \
                    << status << ": "            \
                    << hipGetErrorString(status) \
                    << " at " << __FILE__ << ":" \
                    << __LINE__ << std::endl;    \
      }                                          \
  }

  int main()
  {
      const int element_number = 100;

      int *host_input, *host_output;
      // Host allocation
      host_input  = new int[element_number];
      host_output = new int[element_number];

      // Host data preparation
      for (int i = 0; i < element_number; i++) {
          host_input[i] = i;
      }
      memset(host_output, 0, element_number * sizeof(int));

      int *device_input, *device_output;

      // Device allocation
      HIP_CHECK(hipMalloc((int **)&device_input,  element_number * sizeof(int)));
      HIP_CHECK(hipMalloc((int **)&device_output, element_number * sizeof(int)));

      // Device data preparation
      HIP_CHECK(hipMemcpy(device_input, host_input, element_number * sizeof(int), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(device_output, 0, element_number * sizeof(int)));

      // Run the kernel
      // ...

      HIP_CHECK(hipMemcpy(device_input, host_input, element_number * sizeof(int), hipMemcpyHostToDevice));

      // Free host memory
      delete[] host_input;
      delete[] host_output;

      // Free device memory
      HIP_CHECK(hipFree(device_input));
      HIP_CHECK(hipFree(device_output));
  }

.. note::

  :cpp:func:`hipMalloc` and :cpp:func:`hipFree` are blocking calls, however, HIP
  also provides non-blocking versions :cpp:func:`hipMallocAsync` and
  :cpp:func:`hipFreeAsync` which take in a stream as an additional argument.

.. _pinned_host_memory:

Pinned memory
================================================================================

Pinned memory (or page-locked memory) is stored in pages that are locked to
specific sectors in RAM and cannot be migrated. The pointer can be used on both
host and device. Accessing host-resident pinned memory in device kernels is
generally not recommended for performance, as it can force the data to traverse
the host-device interconnect (e.g. PCIe), which is much slower than
the on-device bandwidth.

Advantage of pinned memory is the improved transfer times between host and
device. For transfer operations, such as :cpp:func:`hipMemcpy` or :cpp:func:`hipMemcpyAsync`,
using pinned memory instead of pageable memory on host can lead to a ~3x
improvement in bandwidth.

The disadvantage of pinned memory is, that it reduces the available RAM for other
processes, which can negatively impact the overall performance of the host.

The example code how to use pinned memory in HIP showed at the following example.

.. code-block:: cpp

  #include <hip/hip_runtime.h>
  #include <iostream>

  #define HIP_CHECK(expression)                  \
  {                                              \
      const hipError_t status = expression;      \
      if(status != hipSuccess){                  \
          std::cerr << "HIP error "              \
                    << status << ": "            \
                    << hipGetErrorString(status) \
                    << " at " << __FILE__ << ":" \
                    << __LINE__ << std::endl;    \
      }                                          \
  }

  int main()
  {
      const int element_number = 100;
      
      int *host_input, *host_output;
      // Host allocation
      HIP_CHECK(hipHostMalloc((int **)&host_input, element_number * sizeof(int)));
      HIP_CHECK(hipHostMalloc((int **)&host_output, element_number * sizeof(int)));

      // Host data preparation
      for (int i = 0; i < element_number; i++) {
          host_input[i] = i;
      }
      memset(host_output, 0, element_number * sizeof(int));

      int *device_input, *device_output;

      // Device allocation
      HIP_CHECK(hipMalloc((int **)&device_input,  element_number * sizeof(int)));
      HIP_CHECK(hipMalloc((int **)&device_output, element_number * sizeof(int)));

      // Device data preparation
      HIP_CHECK(hipMemcpy(device_input, host_input, element_number * sizeof(int), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(device_output, 0, element_number * sizeof(int)));

      // Run the kernel
      // ...

      HIP_CHECK(hipMemcpy(device_input, host_input, element_number * sizeof(int), hipMemcpyHostToDevice));

      // Free host memory
      delete[] host_input;
      delete[] host_output;

      // Free device memory
      HIP_CHECK(hipFree(device_input));
      HIP_CHECK(hipFree(device_output));
  }

The pinned memory allocation is effected with different flags, which details
described at :ref:`memory_allocation_flags`.

.. _memory_allocation_flags:

Memory allocation flags of pinned memory
--------------------------------------------------------------------------------

The ``hipHostMalloc`` flags specify different memory allocation types for pinned
host memory:

* ``hipHostMallocPortable``: The memory is considered allocated by all contexts,
  not just the one on which the allocation is made.
* ``hipHostMallocMapped``: Map the allocation into the address space for
  the current device, and the device pointer can be obtained with
  :cpp:func:`hipHostGetDevicePointer`.
* ``hipHostMallocNumaUser``: The flag to allow host memory allocation to
  follow Numa policy by user. Target of Numa policy is to select a CPU that is
  closest to each GPU. Numa distance is the measurement of how far between GPU
  and CPU devices.
* ``hipHostMallocWriteCombined``: Allocates the memory as write-combined. On
  some system configurations, write-combined allocation may be transferred
  faster across the PCI Express bus, however, could have low read efficiency by
  most CPUs. It's a good option for data transfer from host to device via mapped
  pinned memory.
* ``hipHostMallocCoherent``: Allocate fine-grained memory. Overrides
  ``HIP_HOST_COHERENT`` environment variable for specific allocation. For
  further details, check :ref:`coherence_control`.
* ``hipHostMallocNonCoherent``: Allocate coarse-grained memory. Overrides
  ``HIP_HOST_COHERENT`` environment variable for specific allocation. For
  further details, check :ref:`coherence_control`.

All allocation flags are independent and can be used in any of the combinations
with on exception. For example, :cpp:func:`hipHostMalloc` can be called with both
``hipHostMallocPortable`` and ``hipHostMallocMapped`` flags set. Both usage
models described above use the same allocation flags, and the difference is in
how the surrounding code uses the host memory.

The one exception is when the ``hipHostMallocCoherent`` and
``hipHostMallocNonCoherent`` flags are set, what is an illegal state.

.. note:: 
  
  By default, each GPU selects a Numa CPU node that has the least Numa distance
  between them, that is, host memory will be automatically allocated closest on
  the memory pool of Numa node of the current GPU device. Using
  :cpp:func:`hipSetDevice` API to a different GPU will still be able to access
  the host allocation, but can have longer Numa distance. 

  Numa policy is implemented on Linux and is under development on Microsoft
  Windows.