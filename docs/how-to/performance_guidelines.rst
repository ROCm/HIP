.. meta::
  :description: This chapter describes a set of best practices designed to help
   developers optimize the performance of HIP-capable GPU architectures.
  :keywords: AMD, ROCm, HIP, CUDA, performance, guidelines, optimization, how-to

.. _how_to_performance_guidelines:

*******************************************************************************
Performance guidelines
*******************************************************************************

The AMD HIP performance guidelines provide practical, actionable techniques for
optimizing application performance on AMD GPUs. This guide focuses on
step-by-step instructions and best practices for improving performance.

For theoretical foundations and performance concepts, see
:doc:`../understand/performance_optimization`.

Optimization workflow
=====================

Follow this systematic approach to optimize GPU performance:

1. **Profile and measure baseline**

   Use ``rocprofv3`` to identify bottlenecks:

   .. code-block:: bash

      rocprofv3 --stats --<tracing_option> -- <application_path>

   Collect metrics on kernel execution time, memory bandwidth, occupancy, and
   CU utilization. For more details on using ``rocprofv3`` for application tracing and profiling, see :doc:`rocprofv3 documentation <rocprofiler-sdk:how-to/using-rocprofv3>`.

2. **Analyze metrics to identify bottlenecks**

   Determine if kernels are compute-bound or memory-bound. Check arithmetic
   intensity, memory bandwidth achieved vs peak, and compute throughput.

   For understanding the roofline model, see :ref:`roofline_model`.

3. **Apply targeted optimizations**

   Based on identified bottlenecks, apply techniques from this guide.

4. **Verify improvements**

   Re-profile to confirm performance gains.

5. **Iterate**

   Repeat until performance goals are met.

.. _parallel execution:

Parallel execution
==================

For optimal use and to keep all system components busy, the application must
reveal and efficiently provide as much parallelism as possible.

Application level
-----------------

To enable parallel execution across the host and devices:

* Use :ref:`asynchronous calls and streams <asynchronous_how-to>`
* Assign serial workloads to the host
* Assign parallel workloads to the devices

For parallel workloads:

* Use :cpp:func:`__syncthreads()` (see :ref:`synchronization_functions`) for
  intra-block synchronization
* Use global memory with separate kernel invocations for inter-block
  synchronization (has overhead, minimize when possible)

Device level
------------

Maximize parallel execution across multiprocessors:

* Execute multiple kernels concurrently on a device
* Use streams to overlap computation and data transfers
* Keep all multiprocessors busy with enough concurrent kernels
* Avoid launching too many kernels (causes resource contention)

Multiprocessor level
--------------------

Maximize parallel execution within each multiprocessor:

* Ensure sufficient resident warps for every clock cycle
* Exploit instruction-level parallelism within warps
* Exploit thread-level parallelism across warps
* Balance resource usage for optimal occupancy

.. _memory optimization:

Memory throughput optimization
==============================

The first step in maximizing memory throughput is to minimize low-bandwidth
data transfers between the host and the device.

Additionally, maximize the use of on-chip memory (shared memory and caches) and
minimize transfers with global memory.

.. _data transfer:

Data transfer optimization
--------------------------

**Minimize host-device transfers**

* Move computations from host to device when possible
* Create, use, and discard intermediate data structures on device
* Avoid unnecessary copies to host memory

**Batch small transfers**

Each memory transfer incurs a fixed overhead from driver calls and PCIe
transaction setup. Consolidating many small transfers into a single large
transfer amortizes this overhead across more data, resulting in much higher
effective bandwidth.

.. code-block:: cuda

   // Instead of many small transfers
   for (int i = 0; i < n; i++) {
       hipMemcpy(&d_data[i], &h_data[i], sizeof(float), ...);
   }

   // Use a single large transfer
   hipMemcpy(d_data, h_data, n * sizeof(float), ...);

**Use page-locked memory for transfers**

Page-locked (pinned) memory cannot be swapped to disk by the operating system,
allowing the GPU to access it directly via DMA without CPU involvement. This
eliminates an extra copy through a staging buffer and achieves higher bandwidth.

.. code-block:: cuda

   float* h_pinned;
   hipHostMalloc(&h_pinned, size);
   // Faster transfers than pageable memory
   hipMemcpy(d_data, h_pinned, size, hipMemcpyHostToDevice);

**Use mapped memory on integrated systems**

On integrated GPUs (APUs), the CPU and GPU share the same physical memory.
Mapped page-locked memory allows zero-copy access, where the GPU reads directly
from host memory without requiring an explicit transfer, eliminating redundant copies.

.. code-block:: cuda

   int integrated;
   hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, device);
   if (integrated) {
       // Use mapped page-locked memory - no explicit copy needed
       hipHostMalloc(&ptr, size, hipHostMallocMapped);
   }

.. _device memory access:

Device memory access
--------------------

**Ensure proper alignment**

Memory hardware loads data in aligned chunks (typically 128 bytes). Using
naturally aligned data types ensures each access maps to a single memory
transaction, maximizing bandwidth and avoiding split transactions.

.. code-block:: cuda

   // Use naturally aligned types
   float4 data;  // 16-byte aligned
   float2 data;  // 8-byte aligned

   // Ensure structure alignment
   struct __align__(16) MyStruct {
       float4 data;
   };

**Optimize 2D array access**

Padding 2D arrays to multiples of the wavefront size ensures each row starts
at an aligned memory boundary. This allows consecutive threads accessing the
same row to generate coalesced memory transactions, thereby maximizing
bandwidth.

.. code-block:: cuda

   // Ensure array width is multiple of warp size
   int width = ((actual_width + warpSize - 1) / warpSize) * warpSize;
   hipMalloc(&array, width * height * sizeof(float));

   // Access pattern
   int idx = x + width * y;  // width should be warp-aligned

**Coalesce memory accesses**

When consecutive threads in a wavefront access consecutive memory addresses,
the hardware combines these into a single wide transaction. Non-coalesced
patterns require multiple transactions, reducing effective bandwidth.

.. code-block:: cuda

   // Good: consecutive threads access consecutive addresses
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   data[idx] = value;

   // Bad: strided access
   int idx = threadIdx.x * stride;  // Non-coalesced if stride > 1
   data[idx] = value;

For understanding memory coalescing theory, see :ref:`memory_hierarchy_theory`.

**Use shared memory for data reuse**

Shared memory (LDS) provides low-latency on-chip storage shared across threads
in a block. Loading data into shared memory once and reusing it many times
reduces global memory traffic, particularly effective for tiled algorithms such
as matrix multiplication.

.. code-block:: cuda

   __global__ void optimized_kernel(float* input, float* output) {
       __shared__ float tile[TILE_SIZE][TILE_SIZE];

       // Load data into shared memory
       tile[threadIdx.y][threadIdx.x] = input[...];
       __syncthreads();

       // Reuse data from fast shared memory
       float result = 0;
       for (int i = 0; i < TILE_SIZE; i++) {
           result += tile[threadIdx.y][i] * tile[i][threadIdx.x];
       }
       __syncthreads();

       output[...] = result;
   }

**Avoid bank conflicts in shared memory**

Shared memory is organized into banks, each capable of servicing one request per
cycle. When multiple threads in a warp access the same bank simultaneously, the
requests are serialized, reducing throughput. Padding arrays by one element
shifts addresses to avoid systematic conflicts.

.. code-block:: cuda

   // Bad: power-of-2 stride causes conflicts
   __shared__ float data[32][32];
   float value = data[threadIdx.x][threadIdx.y];

   // Good: padding avoids conflicts
   __shared__ float data[32][33];  // Extra column
   float value = data[threadIdx.x][threadIdx.y];

For bank conflict theory, see :ref:`bank_conflicts_theory`.

**Use texture memory for 2D spatial access**

Texture memory provides hardware-accelerated 2D filtering and caching optimized
for spatial locality. It automatically handles boundary conditions and can
interpolate values, making it ideal for image processing and nearby-neighbor access patterns.

.. code-block:: cuda

   // Create texture object
   hipTextureObject_t texObj;
   hipCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

   // Access in kernel
   float value = tex2D<float>(texObj, x, y);

.. _instruction optimization:

Instruction throughput optimization
====================================

Arithmetic instructions
-----------------------

**Use efficient operations**

Division requires many more hardware cycles than multiplication. Similarly,
bitwise operations (shifts, AND, OR) are single-cycle instructions on integer
units, making them far more efficient than equivalent arithmetic for power-of-two calculations.

.. code-block:: cuda

   // Prefer multiplication over division
   float result = value * 0.5f;     // Fast
   float result = value / 2.0f;     // Slower

   // Use bitwise operations for powers of 2
   int index = threadIdx.x << 2;    // Multiply by 4
   int mask = (1 << n) - 1;         // Create bit mask

**Use single-precision when possible**

AMD GPUs have significantly higher throughput for single-precision (FP32)
operations compared to double-precision (FP64). Using single-precision math
functions can deliver substantial performance gains when FP64 accuracy is not required.

.. code-block:: cuda

   // Single-precision (faster)
   float result = sinf(x);
   float result = expf(x);

   // Double-precision (slower, use only when necessary)
   double result = sin(x);
   double result = exp(x);

**Leverage fast math intrinsics**

Hardware-specific intrinsics bypass certain accuracy checks and use lookup
tables or polynomial approximations, trading slight precision loss for
significantly higher throughput. These should be used when the application can
tolerate reduced precision.

.. code-block:: cuda

   // Fast intrinsic versions
   float ex = __expf(x);            // Fast exponential
   float lg = __logf(x);            // Fast logarithm
   float sq = __fsqrt_rn(x);        // Fast square root
   float rc = __frcp_rn(x);         // Fast reciprocal

.. _control flow instructions:

Control flow optimization
-------------------------

**Minimize divergence**

When threads in a wavefront take different execution paths, the hardware
serializes both branches, executing each path with only the relevant threads
active. This reduces effective parallelism and wastes cycles on inactive threads.

.. code-block:: cuda

   // Good: no divergence (condition depends on threadIdx)
   if (threadIdx.x < 32) {
       // All threads in first half-warp execute
   }

   // Bad: divergence within warp
   if (data[threadIdx.x] > threshold) {
       // Some threads execute, others don't
   }

**Use branch hints for predictable conditions**

Providing hints about branch likelihood helps the compiler generate better
instruction ordering and can improve the branch predictor's accuracy, reducing
pipeline stalls when the prediction proves correct.

.. code-block:: cuda

   if (__builtin_expect(rare_condition, 0)) {
       // Unlikely branch
   }

   // C++20 attribute
   if (common_condition) [[likely]] {
       // Likely branch
   }

**Avoid divergent warps**

When divergence is unavoidable, restructure the code to separate divergent paths
into different kernel launches or use predication (branchless programming) to
keep all threads active, though computing unnecessary values may be acceptable
if it avoids the serialization penalty.

.. code-block:: cuda

   // Instead of:
   if (threadIdx.x % 2 == 0) {
       result = compute_even();
   } else {
       result = compute_odd();
   }

   // Consider separating into different kernels or using predication

Synchronization
---------------

**Use minimal synchronization**

Each synchronization point stalls all threads in a block until the slowest one
reaches the barrier. Minimize synchronizations by carefully analyzing data
dependencies—only synchronize when threads genuinely need to exchange data
through shared memory.

.. code-block:: cuda

   __global__ void kernel() {
       __shared__ float data[256];

       // Load phase
       data[threadIdx.x] = input[...];
       __syncthreads();  // Necessary sync

       // Compute phase - no sync needed if threads are independent
       float result = compute(data[...]);

       // Store phase - sync only if needed
       output[...] = result;
   }

**Use streams for async execution**

Streams enable concurrent execution of independent operations. Commands in
different streams can overlap in time, allowing kernel execution and memory
transfers to run simultaneously. This maximizes GPU utilization by keeping
multiple execution engines busy concurrently.

.. code-block:: cuda

   hipStream_t stream1, stream2;
   hipStreamCreate(&stream1);
   hipStreamCreate(&stream2);

   // Overlap independent operations
   kernel1<<<grid, block, 0, stream1>>>(...);
   kernel2<<<grid, block, 0, stream2>>>(...);

   hipStreamSynchronize(stream1);
   hipStreamSynchronize(stream2);

Managing register pressure
==========================

High register usage can limit occupancy. Follow these steps:

**Minimize live variables**

The compiler allocates registers for every variable that must remain accessible.
Reducing the number of simultaneously live variables frees registers, allowing
more wavefronts to fit on each CU. Chaining function calls trades some redundant
computation for lower register usage.

.. code-block:: cuda

   // Instead of storing all intermediate results
   float a = compute_a();
   float b = compute_b();
   float c = compute_c();
   float result = combine(a, b, c);

   // Recompute or chain operations
   float result = combine(compute_a(), compute_b(), compute_c());

**Use shared memory for temporary storage**

Per-thread arrays stored in registers consume valuable register space, limiting
occupancy. Moving temporary storage to shared memory trades register usage for
shared memory usage, often allowing higher occupancy since shared memory limits
are typically less restrictive.

.. code-block:: cuda

   // Instead of per-thread arrays (uses registers)
   float temp[100];

   // Use shared memory
   __shared__ float temp[blockDim.x][100];
   float* my_temp = temp[threadIdx.x];

**Adjust launch bounds**

The ``__launch_bounds__`` attribute provides hints to the compiler about expected
thread block size and minimum blocks per CU. This guides register allocation
decisions, potentially trading per-thread register count for higher occupancy.

.. code-block:: cuda

   __global__ void
   __launch_bounds__(256, 4)  // 256 threads, 4 blocks per CU
   my_kernel() {
       // Kernel code
   }

**Check register usage during compilation**

The compiler can report per-kernel register usage statistics. Monitoring this
output helps identify kernels consuming excessive registers, guiding optimization
efforts toward reducing register pressure in the most impactful areas.

.. code-block:: bash

   hipcc --resource-usage kernel.hip

For register pressure theory, see :ref:`register_pressure_theory`.

Improving occupancy
===================

Higher occupancy helps hide latency. Follow these steps:

**Reduce register usage per thread**

Use techniques from "Managing register pressure" above.

**Reduce shared memory usage per block**

Each CU has limited shared memory that must be divided among resident blocks.
Reducing per-block shared memory usage allows more blocks to reside simultaneously,
increasing occupancy and improving latency hiding through greater thread-level parallelism.

.. code-block:: cuda

   // Allocate only what's needed
   __shared__ float tile[TILE_SIZE][TILE_SIZE];

   // Or use dynamic allocation
   extern __shared__ float dynamic_shared[];

**Optimize block size**

AMD GPUs execute threads in wavefronts of 64. Choosing block sizes as multiples
of 64 prevents partial wavefronts that waste execution slots. Larger blocks
(128-256 threads) typically achieve better occupancy and resource utilization.

.. code-block:: cuda

   // Use multiples of wavefront size
   dim3 block(64);    // Good for AMD GPUs (wavefront=64)
   dim3 block(128);   // Common choice
   dim3 block(256);   // Good for high-occupancy kernels

   // Avoid very small blocks
   dim3 block(32);    // May waste resources

**Profile occupancy**

Profiling tools report the ratio of active wavefronts to maximum possible
wavefronts per CU. Low occupancy suggests resource constraints (registers or
shared memory) are limiting parallelism and may indicate opportunities for optimization.

.. code-block:: bash

   rocprofv3 --occupancy ./your_application

For occupancy theory, see :ref:`occupancy`.

Minimizing memory thrashing
============================

Applications frequently allocating and freeing memory might experience slower
allocation calls over time. To optimize:

**Allocate early, deallocate late**

Frequent allocation and deallocation causes memory fragmentation and increases
allocator overhead. Reusing allocations across iterations amortizes the cost
of memory management and maintains better memory locality.

.. code-block:: cuda

   // Bad: frequent allocation in loop
   for (int i = 0; i < iterations; i++) {
       float* temp;
       hipMalloc(&temp, size);
       // Use temp
       hipFree(temp);
   }

   // Good: allocate once
   float* temp;
   hipMalloc(&temp, size);
   for (int i = 0; i < iterations; i++) {
       // Reuse temp
   }
   hipFree(temp);

**Avoid allocating all available memory**

Reserving some memory headroom prevents allocation failures and system instability.
The driver and runtime need workspace for internal operations, and leaving a
safety margin ensures stable operation without unexpected out-of-memory errors.

.. code-block:: cuda

   size_t free, total;
   hipMemGetInfo(&free, &total);

   // Don't allocate all free memory
   size_t safe_size = free * 0.9;  // Leave some margin

**Use managed memory for oversubscription**

Managed memory automatically migrates data between host and device on demand,
allowing allocations larger than physical GPU memory. Prefetching hints help
the runtime optimize page placement, reducing migration overhead during kernel execution.

.. code-block:: cuda

   // Allows exceeding physical memory
   float* data;
   hipMallocManaged(&data, large_size);

   // Optionally prefetch to device
   hipMemPrefetchAsync(data, size, device, stream);

Summary
=======

Key optimization techniques:

* **Profile first**: Use ``rocprofv3`` to identify actual bottlenecks
* **Parallelize effectively**: Maximize work at all levels (application, device, CU)
* **Optimize memory**: Minimize transfers, maximize coalescing, use LDS
* **Manage resources**: Balance registers, shared memory, and occupancy
* **Minimize divergence**: Structure control flow to keep warps coherent

For understanding the theory behind these techniques, refer to
:doc:`../understand/performance_optimization` and :doc:`../understand/hardware_implementation`.
