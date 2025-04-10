.. meta::
  :description: This chapter describes a set of best practices designed to help
   developers optimize the performance of HIP-capable GPU architectures.
  :keywords: AMD, ROCm, HIP, CUDA, performance, guidelines

.. _how_to_performance_guidelines:

*******************************************************************************
Performance guidelines
*******************************************************************************

The AMD HIP performance guidelines are a set of best practices designed to help
you optimize the application performance on AMDGPUs. The guidelines discuss
established parallelization and optimization techniques to improve the
application performance on HIP-capable GPU architectures.

Here are the four main cornerstones to help you exploit HIP's performance
optimization potential:

- Parallel execution
- Memory bandwidth usage optimization
- Maximum throughput optimization
- Memory thrashing minimization

This document discusses the usage and benefits of these cornerstones in detail.

.. _parallel execution:

Parallel execution
================================================================================

For optimal use and to keep all system components busy, the application must
reveal and efficiently provide as much parallelism as possible. The parallelism
can be performed at the application level, device level, and multiprocessor
level.

.. _application_parallel_execution:

Application level
--------------------------------------------------------------------------------

To enable parallel execution of the application across the host and devices, use
:ref:`asynchronous calls and streams <asynchronous_how-to>`. Assign workloads
based on efficiency: serial to the host or parallel to the devices.

For parallel workloads, when threads belonging to the same block need to
synchronize to share data, use :cpp:func:`__syncthreads()` (see:
:ref:`synchronization_functions`) within the same kernel invocation. For threads
belonging to different blocks, use global memory with two separate
kernel invocations. It is recommended to avoid the latter approach as it adds
overhead.

Device level
--------------------------------------------------------------------------------

Device level optimization primarily involves maximizing parallel execution
across the multiprocessors on the device. You can achieve device level
optimization by executing multiple kernels concurrently on a device. To enhance
performance, the management of these kernels is facilitated by streams, which
allows overlapping of computation and data transfers. This approach aims at
keeping all multiprocessors busy by executing enough kernels concurrently.
However, launching too many kernels can lead to resource contention, hence a
balance must be found for optimal performance. The device level optimization
helps in achieving maximum utilization of the device resources.

Multiprocessor level
--------------------------------------------------------------------------------

Multiprocessor level optimization involves maximizing parallel execution within
each multiprocessor on a device. The key to multiprocessor level optimization
is to efficiently utilize the various functional units within a multiprocessor.
For example, ensuring a sufficient number of resident warps, so that every clock
cycle has an instruction from a warp is ready for execution. This instruction
could either be another independent instruction of the same warp, which exploits
:ref:`instruction level optimization <instruction optimization>`, or more
commonly an instruction of another warp, which exploits thread-level parallelism.

On the other hand, device level optimization focuses on the device as a whole,
aiming at keeping all multiprocessors busy by executing enough kernels
concurrently. Both multiprocessor and device levels of optimization are crucial
for achieving maximum performance. They work together to ensure efficient
utilization of the GPU resources, ranging from individual multiprocessors to the
device as a whole.

.. _memory optimization:

Memory throughput optimization
================================================================================

The first step in maximizing memory throughput is to minimize low-bandwidth
data transfers between the host and the device.

Additionally, maximize the use of on-chip memory, that is, shared memory and
caches, and minimize transfers with global memory. Shared memory acts as a
user-managed cache explicitly allocated and accessed by the application. A
common programming pattern is to stage data from device memory into shared
memory. The staging of data from the device to shared memory involves the
following steps:

1. Each thread of a block loading data from device memory to shared memory.
2. Synchronizing with all other threads of the block.
3. Processing the data stored in shared memory.
4. Synchronizing again if necessary.
5. Writing the results back to the device global memory.

For some applications, a traditional hardware-managed cache is more appropriate
for exploiting data locality.

In conclusion, the throughput of memory accesses by a kernel can vary
significantly depending on the access pattern. Therefore, the next step in
maximizing memory throughput is to organize memory accesses as optimally as
possible. This is especially important for global memory accesses, as global
memory bandwidth is low compared to available on-chip bandwidths and arithmetic
instruction throughput. Thus, non-optimal global memory accesses generally have
a high impact on performance.
The memory throughput optimization techniques are further discussed in detail in
the following sections.

.. _data transfer:

Data transfer
--------------------------------------------------------------------------------

To minimize data transfers between the host and the device, applications should
move more computations from the host to the device, even at the cost of running
kernels that don't fully utilize parallelism for the device. Intermediate data
structures should be created, used, and discarded in device memory without being
mapped or copied to host memory.

Batching small transfers into a single large transfer can improve performance
due to the overhead associated with each transfer. On systems with a front-side
bus, using page-locked host memory can enhance data transfer performance.

When using mapped page-locked memory, there is no need to allocate device
memory or explicitly copy data between device and host memory. Data transfers
occur implicitly each time the kernel accesses the mapped memory. For optimal
performance, these memory accesses should be coalesced, similar to global
memory accesses. The process where threads in a warp access sequential memory
locations is known as coalesced memory access, which can enhance memory data
transfer efficiency.

On integrated systems where device and host memory are physically the same, no
copy operation between host and device memory is required and hence mapped
page-locked memory should be used instead. To check if the device is integrated,
applications can query the integrated device property.

.. _device memory access:

Device memory access
---------------------

Memory access instructions might be repeated due to the spread of memory
addresses across warp threads. The impact on throughput varies with memory type
and is generally reduced when addresses are more scattered, especially in
global memory.

Device memory is accessed via 32-, 64-, or 128-byte transactions that must be
naturally aligned.
Maximizing memory throughput involves:

- Coalescing memory accesses of threads within a warp into minimal transactions.
- Following optimal access patterns.
- Using properly sized and aligned data types.
- Padding data when necessary.

Global memory instructions support reading or writing data of specific sizes (1,
2, 4, 8, or 16 bytes) that are naturally aligned. Not meeting the size and
alignment requirements leads to multiple instructions, which reduces
performance. Therefore, for correct results and optimal performance:

- Use data types that meet these requirements
- Ensure alignment for structures
- Maintain alignment for all values or arrays.

Threads often access 2D arrays at an address calculated as
``BaseAddress + xIndex + width * yIndex``. For efficient memory access, the
array and thread block widths should be multiples of the warp size. If the
array width is not a multiple of the warp size, it is usually more efficient to
allocate the array with a width rounded up to the nearest multiple and pad the
rows accordingly.

Local memory is used for certain automatic variables, such as arrays with
non-constant indices, large structures of arrays, and any variable where the
kernel uses more registers than available. Local memory resides in device
memory, which leads to high latency and low bandwidth, similar to global memory
accesses. However, the local memory is organized for consecutive 32-bit words to
be accessed by consecutive thread IDs, which allows full coalescing when all
threads in a warp access the same relative address.

Shared memory is located on-chip and provides higher bandwidth and lower latency
than local or global memory. It is divided into banks that can be simultaneously
accessed, which boosts bandwidth. However, bank conflicts, where two addresses
fall in the same bank, lead to serialized access and decreased throughput.
Therefore, understanding how memory addresses map to banks and scheduling
requests to minimize conflicts is crucial for optimal performance.

Constant memory is in the device memory and cached in the constant cache.
Requests are split based on different memory addresses and are serviced based
either on the throughput of the constant cache for cache hits or on the
throughput of the device memory otherwise. This splitting of requests affects
throughput.

Texture and surface memory are stored in the device memory and cached in the
texture cache. This setup optimizes 2D spatial locality, which leads to better
performance for threads reading close 2D addresses.
Reading device memory through texture or surface fetching provides the following
advantages:

- Higher bandwidth for local texture fetches or surface reads.
- Offloading addressing calculation.
- Data broadcasting.
- Optional conversion of 8-bit and 16-bit integer input data to 32-bit
  floating-point values on the fly.

.. _instruction optimization:

Optimization for maximum instruction throughput
================================================================================

To maximize instruction throughput:

- Minimize low throughput arithmetic instructions.
- Minimize divergent warps inflicted by flow control instructions.
- Maximize instruction parallelism.

These techniques are discussed in detail in the following sections.

Arithmetic instructions
--------------------------------------------------------------------------------

The type and complexity of arithmetic operations can significantly impact the
performance of your application. We are highlighting some hints how to maximize
it.

Use efficient operations: Some arithmetic operations are costlier than others.
For example, multiplication is typically faster than division, and integer
operations are usually faster than floating-point operations, especially with
double precision.

Minimize low-throughput instructions: This might involve trading precision for
speed when it does not affect the final result. For instance, consider using
single-precision arithmetic instead of double-precision.

Leverage intrinsic functions: Intrinsic functions are predefined functions
available in HIP that can often be executed faster than equivalent arithmetic
operations (subject to some input or accuracy restrictions). They can help
optimize performance by replacing more complex arithmetic operations.

Optimize memory access: The memory access efficiency can impact the speed of
arithmetic operations. See: :ref:`device memory access`.

.. _control flow instructions:

Control flow instructions
--------------------------------------------------------------------------------

Control flow instructions (``if``, ``else``, ``for``, ``do``, ``while``,
``break``, ``continue``, ``switch``) can impact instruction throughput by
causing threads within a warp to diverge and follow different execution paths.
To optimize performance, write control conditions to minimize divergent warps.
For example, when the control condition depends on ``threadIdx`` or ``warpSize``,
warp doesn't diverge. The compiler might optimize loops, short ifs, or switch
blocks using branch predication, which prevents warp divergence. With branch
predication, instructions associated with a false predicate are scheduled but
not executed, which avoids unnecessary operations. For control conditions where
one outcome is significantly more likely than the other, use `__builtin_expect <https://clang.llvm.org/docs/LanguageExtensions.html#builtin-expect>`_
or ``[[likely]]`` to indicate the likely condition result.

Avoiding divergent warps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Warps diverge when threads within the same warp follow different execution paths.
This is caused by conditional statements that lead to different arithmetic
operations being performed by different threads. Divergent warps can
significantly reduce instruction throughput, so it is advisable to structure
your code to minimize divergence.

Synchronization
--------------------------------------------------------------------------------

Synchronization ensures that all threads within a block complete their
computations and memory accesses before moving forward, which is critical when
threads depend on other thread results. However, synchronization can also cause
performance overhead, as it needs the threads to wait, which might lead to idle
GPU resources.

To synchronize all threads in a block, use :cpp:func:`__syncthreads()`.
:cpp:func:`__syncthreads()` ensures that, all threads reach the same point in
the code and can access shared memory after reaching that point.

An alternative way to synchronize is to use streams. Different streams can
execute commands either without following a specific order or concurrently. This
is why streams allow more fine-grained control over the execution order of
commands, which can be beneficial in certain scenarios.

Minimizing memory thrashing
================================================================================

Applications frequently allocating and freeing memory might experience slower
allocation calls over time as memory is released back to the operating system.
To optimize performance in such scenarios, follow these guidelines:

- Avoid allocating all available memory with :cpp:func:`hipMalloc` or
  :cpp:func:`hipHostMalloc`, as this immediately reserves memory and might
  prevent other applications from using it. This behavior could strain the
  operating system schedulers or prevent other applications from running on the
  same GPU.
- Try to allocate memory in suitably sized blocks early in the application's
  lifecycle and deallocate only when the application no longer needs it.
  Minimize the number of :cpp:func:`hipMalloc` and :cpp:func:`hipFree` calls in
  your application, particularly in performance-critical areas.
- Consider resorting to other memory types such as :cpp:func:`hipHostMalloc` or
  :cpp:func:`hipMallocManaged`, if an application can't allocate sufficient
  device memory. While the other memory types might not offer similar
  performance, they allow the application to continue running.
- For supported platforms, use :cpp:func:`hipMallocManaged`, as it allows
  oversubscription. With the right policies, :cpp:func:`hipMallocManaged` can
  maintain most, if not all, :cpp:func:`hipMalloc` performance.
  :cpp:func:`hipMallocManaged` doesn't require an allocation to be resident
  until it is needed or prefetched, which eases the load on the operating
  system's schedulers and facilitates multitenant scenarios.
