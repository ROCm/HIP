.. meta::
  :description: HIP atomic operations histogram tutorial
  :keywords: AMD, ROCm, HIP, atomic operations, GPU programming, histogram, synchronization primitives

*******************************************************************************
Atomic operations: Histogram tutorial
*******************************************************************************

In GPU programming, a core design principle is to **avoid simultaneous writes to
the same memory address by multiple threads**. When multiple threads write to
the same location without proper synchronization, this creates a
**race condition**, where the final result depends on unpredictable thread
execution order.

Unlike CPUs, GPUs are designed for high-throughput parallel execution with
relaxed memory consistency models and limited cache coherence mechanisms. This
architectural choice maximizes bandwidth and scalability but introduces
challenges when multiple threads need to safely update shared state.

This tutorial demonstrates how to safely handle **concurrent memory updates**
using **atomic operations**, illustrated through the practical example of
computing an image brightness histogram on the GPU.

.. include:: ../prerequisites.rst

Race condition
==============

A **race condition** occurs when two or more threads attempt to
read-modify-write the same memory location concurrently without proper
synchronization. Because GPU threads execute asynchronously across multiple
cores (compute units), concurrent writes can interleave unpredictably,
leading to incorrect results.

For example, if two threads simultaneously attempt:

.. code-block:: c++

   histogram[bin] = histogram[bin] + 1;

both may read the same old value before either writes back,
resulting in only one increment being reflected. This results in **lost updates**
and **nondeterministic output**, which must be avoided.

Histogram
=========

A **histogram** partitions continuous data into discrete intervals called
**bins** and counts how many data points fall into each bin. In image processing,
a histogram typically represents the **distribution of pixel intensities** for
example brightness or color channel values.

The histogram algorithm can be expressed as:

.. math::

   H[b] = \sum_{i=1}^{N} \delta(b - \lfloor f(x_i) \rfloor)

where :math:`f(x_i)` maps each data value to its corresponding bin index
:math:`b`, and :math:`\delta()` is 1 when the value belongs to bin :math:`b` and
0 otherwise.

The basic computational steps are:

1. Iterate through all pixels (or data points).
2. Determine the appropriate bin for each value.
3. Increment that bin’s count.

In a serial CPU program, this is straightforward. On a GPU, thousands of threads
may attempt to increment the same bin concurrently, leading to **race
conditions** unless atomic synchronization is used.

The Challenge in parallel context
---------------------------------

When multiple threads attempt to increment the same bin:

* One thread’s update can overwrite another’s pending increment.

* Memory coherence cannot guarantee ordered visibility across thread blocks.

* The final result may be inconsistent or incorrect.

This necessitates synchronization mechanisms to ensure that updates occur in a
**mutually exclusive** manner without introducing high overhead.

Atomic operations
=================

An **atomic operation** ensures that a compound operation — typically a
read-modify-write sequence — executes as an **indivisible unit**. From the
programmer’s perspective, atomicity guarantees that no other thread can observe
a partially completed operation.

Formally, an operation :math:`O(x)` on shared variable :math:`x` is **atomic**
if its execution satisfies:

.. math::

   \forall T_i, T_j, \text{ the effects of } O(x) \text{ appear serializable.}

That is, all threads observe results as if operations occurred in a single,
sequential order.

Mechanics
---------

Atomic operations on GPUs are implemented in hardware through a **memory
arbitration unit** that locks a cache line, performs the modification, and
releases the lock. This ensures correctness even under massive parallelism.

When a thread performs an atomic operation:

1. The target memory location is temporarily locked.
2. The value is fetched and updated.
3. The update is written back, and the lock is released.

No other thread can modify the same memory location during this sequence.

Atomic functions
----------------

HIP provides a wide set of atomic primitives to synchronize updates to shared
memory or global memory locations:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Operation
     - Description
   * - ``atomicAdd``
     - Atomically adds a value to a memory location and returns the old value.
   * - ``atomicSub``
     - Atomically subtracts a value.
   * - ``atomicExch``
     - Atomically exchanges values between a register and memory.
   * - ``atomicCAS``
     - Performs an atomic compare-and-swap; fundamental for implementing locks.
   * - ``atomicMax`` / ``atomicMin``
     - Updates to the maximum or minimum of two values.
   * - ``atomicInc`` / ``atomicDec``
     - Atomically increments or decrements a counter, wrapping at a boundary.

Atomic operations in kernels can operate on block scope (shared memory),
device scope (global memory), or system scope (system memory), depending on
:doc:`hardware support <rocm:reference/gpu-atomics-operation>`.

For more information, please check :ref:`atomic functions <atomic functions>`.

Image brightness histogram
==========================

We will compute a histogram that captures the **distribution of pixel
brightness** in an RGB image. The algorithm:

1. Reads image data in **channel-height-width** format.
2. Converts RGB values to grayscale brightness.
3. Maps brightness to a histogram bin.
4. Atomically increments the corresponding bin counter.

Kernel implementation
---------------------

.. code-block:: c++

    __global__ void calculateHistogram(float* imageData, int* histogram,
                                       int width, int height,
                                       int channels, int numBins)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int idx = (y * width + x) * channels;
        float brightness = 0.0f;

        for (int c = 0; c < channels; ++c)
            brightness += imageData[idx + c];

        brightness /= channels; // Normalize to [0, 1]
        int bin = static_cast<int>(brightness * numBins);

        // Atomic increment to avoid race conditions
        atomicAdd(&histogram[bin], 1);
    }

Thread identification
~~~~~~~~~~~~~~~~~~~~~

Each thread computes one pixel’s contribution using its 2D thread and block
indices:

.. code-block:: c++

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

This mapping provides a 1:1 correspondence between threads and pixels, making
the computation naturally parallel.

Brightness computation
~~~~~~~~~~~~~~~~~~~~~~

Each pixel’s brightness is computed as the arithmetic mean of its RGB channels:

.. math::
   :nowrap:

   \[
   I'(x, y) = \frac{R + G + B}{3}
   \]

This value is then normalized to [0, 1] and mapped to one of `numBins`
histogram intervals.

Safe histogram update
~~~~~~~~~~~~~~~~~~~~~

The key step is:

.. code-block:: c++

    atomicAdd(&histogram[bin], 1);

This ensures that even if thousands of threads map to the same bin, each
increment is serialized correctly, maintaining an accurate bin count.

Performance characteristics
===========================

Benefits
--------

* **Correctness under parallel updates:** Ensures race-free accumulation.

* **Simplified synchronization:** No explicit locks or barriers needed.

* **Hardware-level efficiency:** Implemented directly in the GPU memory
  subsystem.

Limitations
-----------

While atomic operations guarantee correctness, they can **serialize execution**
when multiple threads target the same memory address. This causes contention and
reduces effective parallelism.

Typical performance degradation sources include:

* **Hot bins:** When many pixels fall into a small subset of bins.

* **Global memory atomics:** Global memory atomics are slower than shared memory
  atomics due to higher access latency.

* **Warp serialization:** Threads within a warp waiting for the same atomic
  target serialize.

Best practices
==============

1. **Apply atomic operations only where necessary** 

   Atomic instructions serialize access to a memory location and use can
   diminish SIMT parallel efficiency and increase warp stalls. Restrict atomic
   usage to code paths where data races cannot be eliminated through algorithmic
   restructuring.

2. **Minimize contention**

   High contention on a single address or a small set of addresses leads to
   serialization. Distribute writes across independent memory locations.

3. **Leverage shared memory**

   Use fast, low-latency shared memory to aggregate partial results within a
   block before issuing a single atomic update to global memory.

4. **Validate correctness** 

   Validate the numerical and logical correctness of GPU kernels by comparing
   against single-threaded or deterministic multi-threaded CPU baselines.

5. **Profile regularly**

   GPU performance is highly sensitive to thread divergence, memory-access
   patterns, and workload distribution. Regularly use profiling tools such as 
   :doc:`rocprofv3<rocprofiler-sdk:how-to/using-rocprofv3>` or
   :doc:`ROCm compute profiler<rocprofiler-compute:how-to/profile/mode>` to
   examine warp-level execution efficiency, memory-coalescing behavior,
   occupancy, and atomic throughput bottlenecks.

Conclusion
==========

Atomic operations provide a low-level synchronization mechanism that allows
correct and deterministic parallel updates to shared data structures. In the
histogram example, :cpp:func:`atomicAdd` ensures that all threads safely
contribute to their corresponding bins, preventing race conditions.

While atomics incur some serialization overhead, they are indispensable for
algorithms that require concurrent accumulation or counting. By applying
techniques like privatization and reduction, developers can achieve both
**correctness** and **high performance** on modern GPUs.

Atomic operations form the foundation for more advanced synchronization
patterns, including parallel reductions, prefix sums, and graph traversal, and
are essential for developing scalable, data-parallel GPU algorithms.
