.. meta::
    :description: HIP programming guide introduction
    :keywords: HIP programming guide introduction, HIP programming guide

.. _hip-programming-guide:

********************************************************************************
HIP programming guide introduction
********************************************************************************

This topic provides key HIP programming concepts and links to more detailed
information. 

Write GPU Kernels for Parallel Execution
================================================================================

To make the most of the parallelism inherent to GPUs, a thorough understanding
of the :ref:`programming model <programming_model>` is helpful. The HIP
programming model is designed to make it easy to map data-parallel algorithms to
architecture of the GPUs. HIP employs the SIMT-model (Single
Instruction Multiple Threads) with a multi-layered thread hierarchy for
efficient execution.

Understand the Target Architecture (CPU and GPU)
================================================================================

The :ref:`hardware implementation <hardware_implementation>` topic outlines the
GPUs supported by HIP. In general, GPUs are made up of Compute Units that excel
at executing parallelizable, computationally intensive workloads without complex
control-flow.

Increase parallelism on multiple level
================================================================================

To maximize performance and keep all system components fully utilized, the
application should expose and efficiently manage as much parallelism as possible.
:ref:`Parallel execution <parallel execution>` can be achieved at the
application, device, and multiprocessor levels.

The application’s host and device operations can achieve parallel execution
through asynchronous calls, streams, or HIP graphs. On the device level,
multiple kernels can execute concurrently when resources are available, and at
the multiprocessor level, developers can overlap data transfers with
computations to further optimize performance.

Memory management
================================================================================

GPUs generally have their own distinct memory, also called :ref:`device
memory <device_memory>`, separate from the :ref:`host memory <host_memory>`.
Device memory needs to be managed separately from the host memory. This includes
allocating the memory and transfering it between the host and the device. These
operations can be performance critical, so it's important to know how to use
them effectively. For more information, see :ref:`Memory management <memory_management>`.

Synchronize CPU and GPU Workloads
================================================================================

Tasks on the host and devices run asynchronously, so proper synchronization is
needed when dependencies between those tasks exist. The asynchronous execution
of tasks is useful for fully utilizing the available resources. Even when only a
single device is available, memory transfers and the execution of tasks can be
overlapped with asynchronous execution.

Error Handling
================================================================================

All functions in the HIP runtime API return an error value of type
:cpp:enum:`hipError_t` that can be used to verify whether the function was
successfully executed. It's important to confirm these
returned values, in order to catch and handle those errors, if possible.
An exception is kernel launches, which don't return any value. These
errors can be caught with specific functions like :cpp:func:`hipGetLastError()`.

For more information, see :ref:`error_handling` .

Multi-GPU and Load Balancing
================================================================================

Large-scale applications that need more compute power can use multiple GPUs in
the system. This requires distributing workloads across multiple GPUs to balance
the load to prevent GPUs from being overutilized while others are idle.

For more information, see :ref:`multi-device` .