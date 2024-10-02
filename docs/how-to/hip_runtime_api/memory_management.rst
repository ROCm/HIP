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
its own distinct memory. Kernels operate on device memory, while host functions operate on host memory.
The runtime
offers functions for allocating, freeing, and copying device memory, along
with transferring data between host and device memory.

How to manage the different memory types is described in the following chapters:

* :ref:`device_memory`
* :ref:`host_memory`
* :ref:`coherence_control`
* :ref:`unified_memory`
* :ref:`virtual_memory`
