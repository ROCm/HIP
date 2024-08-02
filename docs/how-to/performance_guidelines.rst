.. meta::
  :description: This chapter describes a set of best practices designed to help developers optimize the performance of HIP-capable GPU architectures.
  :keywords: AMD, ROCm, HIP, CUDA, performance, guidelines

*******************************************************************************
Performance guidelines
*******************************************************************************

The AMD HIP Performance Guidelines are a set of best practices designed to help
developers optimize the performance of AMD GPUs. They cover established
parallelization and optimization techniques, coding metaphors, and idioms that
can greatly simplify programming for HIP-capable GPU architectures.

By following four main cornerstones, we can exploit the performance
optimization potential of HIP.

- parallel execution
- memory usage optimization
- optimization for maximum throughput
- minimizing memory thrashing

In the following chapters, we will show you their benefits and how to use them
effectively.

.. _parallel execution:

Parallel execution
==================

For optimal use, the application should reveal and efficiently imply as much
parallelism as possible to keep all system components active.

Application level
-----------------

The application should optimize parallel execution across the host and devices
using asynchronous calls and streams. Workloads should be assigned based on
efficiency: serial to the host, parallel to the devices.

For parallel workloads, when threads need to synchronize to share data, if they
belong to the same block, they should use ``__syncthreads()`` (see:
:ref:`synchronization functions`) within the same kernel invocation. If they
belong to different blocks, they must use global memory with two separate
kernel invocations. The latter should be minimized as it adds overhead.

Device level
------------

Device-level optimization primarily involves maximizing parallel execution
across the multiprocessors of the device. This can be achieved by executing
multiple kernels concurrently on a device. The management of these kernels is
facilitated by streams, which allow for the overlapping of computation and data
transfers, enhancing performance. The aim is to keep all multiprocessors busy
by executing enough kernels concurrently. However, launching too many kernels
can lead to resource contention, so a balance must be found for optimal
performance. This approach helps in achieving maximum utilization of the
resources of the device.

Multiprocessor level
--------------------

Multiprocessor-level optimization involves maximizing parallel execution within
each multiprocessor on a device. Each multiprocessor can execute a number of
threads concurrently, and the total number of threads that can run in parallel
is determined by the number of concurrent threads each multiprocessor can
handle.

The key to multiprocessor-level optimization is to efficiently utilize the
various functional units within a multiprocessor. This can be achieved by
ensuring a sufficient number of resident warps, as at every instruction issue
time, a warp scheduler selects an instruction that is ready to execute. This
instruction can be another independent instruction of the same warp, exploiting
:ref:`instruction optimization`, or more commonly an instruction of another warp, 
exploiting thread-level parallelism.

In comparison, device-level optimization focuses on the device as a whole,
aiming to keep all multiprocessors busy by executing enough kernels
concurrently. Both levels of optimization are crucial for achieving maximum
performance. They work together to ensure efficient utilization of the
resources of the GPU, from the individual multiprocessors to the device as a
whole.

.. _memory optimization:

Memory optimization
===================

The first step in maximizing memory throughput is to minimize low-bandwidth
data transfers. This involves reducing data transfers between the host and the
device, as these have lower bandwidth than transfers between global memory and
the device.

Additionally, data transfers between global memory and the device should be
minimized by maximizing the use of on-chip memory: shared memory and caches.
Shared memory acts as a user-managed cache, where the application explicitly
allocates and accesses it. A common programming pattern is to stage data from
device memory into shared memory. This involves each thread of a block loading
data from device memory to shared memory, synchronizing with all other threads
of the block, processing the data in shared memory, synchronizing again if
necessary, and writing the results back to device global memory.

For some applications, a traditional hardware-managed cache is more appropriate
to exploit data locality. On devices of certain compute capabilities, the same
on-chip memory is used for both L1 and shared memory, and the amount dedicated
to each is configurable for each kernel call.

Finally, the throughput of memory accesses by a kernel can vary significantly
depending on the access pattern for each type of memory. Therefore, the next
step in maximizing memory throughput is to organize memory accesses as
optimally as possible. This is especially important for global memory accesses,
as global memory bandwidth is low compared to available on-chip bandwidths and
arithmetic instruction throughput. Thus, non-optimal global memory accesses
generally have a high impact on performance.

Data Transfer
-------------

Applications should aim to minimize data transfers between the host and the
device. This can be achieved by moving more computations from the host to the
device, even if it means running kernels that do not fully utilize the
parallelism for device. Intermediate data structures can be created, used,
and discarded in device memory without being mapped or copied to host memory.

Batching small transfers into a single large transfer can improve performance
due to the overhead associated with each transfer. On systems with a front-side
bus, using page-locked host memory can enhance data transfer performance.

When using mapped page-locked memory, there is no need to allocate device
memory or explicitly copy data between device and host memory. Data transfers
occur implicitly each time the kernel accesses the mapped memory. For optimal
performance, these memory accesses should be coalesced, similar to global
memory accesses.

On integrated systems where device and host memory are physically the same,
any copy operation between host and device memory is unnecessary, and mapped
page-locked memory should be used instead. Applications can check if a device
is integrated by querying the integrated device property.


Device Memory Access
--------------------

Memory access instructions may be repeated due to the spread of memory
addresses across warp threads. The impact on throughput varies with memory type
and is generally reduced when addresses are more scattered, especially in
global memory.

Device memory is accessed via 32-, 64-, or 128-byte transactions that must be
naturally aligned. Maximizing memory throughput involves coalescing memory
accesses of threads within a warp into minimal transactions, following optimal
access patterns, using properly sized and aligned data types, and padding data
when necessary.

Global memory instructions support reading or writing data of specific sizes
(1, 2, 4, 8, or 16 bytes) that are naturally aligned. If the size and alignment
requirements are not met, it leads to multiple instructions, reducing
performance. Therefore, using data types that meet these requirements, ensuring
alignment for structures, and maintaining alignment for all values or arrays is
crucial for correct results and optimal performance.

Threads often access 2D arrays at an address calculated as
``BaseAddress + xIndex + width * yIndex``. For efficient memory access, the
array and thread block widths should be multiples of the warp size. If the
array width is not a multiple of the warp size, it is usually more efficient to
allocate it with a width rounded up to the nearest multiple and pad the rows
accordingly.

Local memory is used for certain automatic variables, such as arrays with
non-constant indices, large structures or arrays, and any variable when the
kernel uses more registers than available. Local memory resides in device
memory, leading to high latency and low bandwidth similar to global memory
accesses. However, it is organized for consecutive 32-bit words to be accessed
by consecutive thread IDs, allowing full coalescing when all threads in a warp
access the same relative address.

Shared memory, located on-chip, provides higher bandwidth and lower latency
than local or global memory. It is divided into banks that can be
simultaneously accessed, boosting bandwidth. However, bank conflicts, where two
addresses fall in the same bank, lead to serialized access and decreased
throughput. Therefore, understanding how memory addresses map to banks and
scheduling requests to minimize conflicts is crucial for optimal performance.

Constant memory is in device memory and cached in the constant cache. Requests
are split based on different memory addresses, affecting throughput, and are
serviced at the throughput of the constant cache for cache hits, or the
throughput of the device memory otherwise.

Texture and surface memory are stored in device memory and cached in texture
cache. This setup optimizes 2D spatial locality, leading to better performance
for threads reading close 2D addresses. Reading device memory through texture
or surface fetching can be advantageous, offering higher bandwidth for local
texture fetches or surface reads, offloading addressing calculations,
allowing data broadcasting, and optional conversion of 8-bit and 16-bit integer
input data to 32-bit floating-point values on-the-fly.

.. _instruction optimization:

Optimization for maximum instruction throughput
===============================================

To maximize instruction throughput:

- minimize low throughput arithmetic instructions
- minimize divergent warps inflicted by control flow instructions
- minimize the number of instruction as possible
- maximize instruction parallelism

Arithmetic instructions
-----------------------

The type and complexity of arithmetic operations can significantly impact the
performance of your application. We are highlighting some hints how to maximize
it.

Using efficient operations: Some arithmetic operations are more costly than
others. For example, multiplication is typically faster than division, and
integer operations are usually faster than floating-point operations,
especially with double-precision.

Minimizing low-throughput instructions: This might involve trading precision
for speed when it does not affect the final result. For instance, consider
using single-precision arithmetic instead of double-precision.

Leverage intrinsic functions: Intrinsic functions are pre-defined functions
available in HIP that can often be executed faster than equivalent arithmetic
operations (subject to some input or accuracy restrictions). They can help
optimize performance by replacing more complex arithmetic operations.

Avoiding divergent warps: Divergent warps occur when threads within the same
warp follow different execution paths. This can happen due to conditional
statements that lead to different arithmetic operations being performed by
different threads. Divergent warps can significantly reduce instruction
throughput, so try to structure your code to minimize divergence.

Optimizing memory access: The efficiency of memory access can impact the speed
of arithmetic operations. Coalesced memory access, where threads in a warp
access consecutive memory locations, can improve memory throughput and thus
the speed of arithmetic operations.

Maximizing instruction parallelism: Some GPU architectures could issue parallel
independent instructions simultaneously, for example integer and floating
point, or two operations with independent inputs and outputs. Mostly this is a
work for compiler, but expressing parallelism in the code explicitly can
improve instructions throughput.

Control flow instructions
-------------------------

Flow control instructions (``if``, ``else``, ``for``, ``do``, ``while``,
``break``, ``continue``, ``switch``) can impact instruction throughput by
causing threads within a warp to diverge and follow different execution paths.
To optimize performance, control conditions should be written to minimize
divergent warps. For example, when the control condition depends on
(``threadIdx`` / ``warpSize``), no warp diverges. The compiler may optimize
loops or short if or switch blocks using branch predication, preventing warp
divergence. With branch predication, instructions associated with a false
predicate are scheduled but not executed, avoiding unnecessary operations.

Synchronization
---------------

Synchronization ensures that all threads within a block have completed their
computations and memory accesses before moving forward, which is critical when
threads are dependent on the results of other threads. However,
synchronization can also lead to performance overhead, as it requires threads
to wait, potentially leading to idle GPU resources.

``__syncthreads()`` is used to synchronize all threads in a block, ensuring
that all threads have reached the same point in the code and that shared memory
is visible to all threads after the point of synchronization.

An alternative way to synchronize is using streams. Different streams can
execute commands out of order with respect to one another or concurrently. This
allows for more fine-grained control over the execution order of commands,
which can be beneficial in certain scenarios.

Minimizing memory thrashing
===========================

Applications frequently allocating and freeing memory may experience slower
allocation calls over time. This is expected as memory is released back to the
operating system. To optimize performance in such scenarios, consider some
recommendations:

- avoid allocating all available memory with ``hipMalloc`` / ``hipHostMalloc``,
  as this immediately reserves memory and can block other applications from
  using it. This could strain the operating system schedulers or even prevent
  other applications from running on the same GPU.
- aim to allocate memory in suitably sized blocks early in the lifecycle of the
  application and deallocate only when the application no longer needs it.
  Minimize the number of ``hipMalloc`` and ``hipFree`` calls in your
  application, particularly in areas critical to performance.
- if an application is unable to allocate sufficient device memory, consider
  resorting to other memory types such as ``hipHostMalloc`` or
  ``hipMallocManaged``. While these may not offer the same performance, they
  can allow the application to continue running.
- For supported platforms, ``hipMallocManaged`` allows for oversubscription.
  With the right memory advise policies, it can maintain most, if not all, of
  the performance of ``hipMalloc``. ``hipMallocManaged`` does not require an
  allocation to be resident until it is needed or prefetched, easing the load
  on the operating system schedulers and facilitating multi-tenant scenarios.
