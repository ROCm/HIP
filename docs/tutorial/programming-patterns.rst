.. meta::
  :description: GPU programming patterns and tutorials
  :keywords: AMD, ROCm, HIP, GPU, programming patterns, parallel computing, tutorial

.. _gpu_programming-patterns:

********************************************************************************
GPU programming patterns
********************************************************************************

GPU programming patterns are fundamental algorithmic structures that enable
efficient parallel computation on GPUs. Understanding these
patterns is essential for developers looking to effectively harness the massive parallel
processing capabilities of modern GPUs for scientific computing, machine learning,
image processing, and other computationally intensive applications.

These tutorials describe core programming patterns demonstrating how to
efficiently implement common parallel algorithms using the HIP runtime API and
kernel extensions. Each pattern addresses a specific computational challenge and
provides practical implementations with detailed explanations.

Common GPU programming challenges
==================================

GPU programming introduces unique challenges not present in traditional CPU
programming:

* **Memory coherence**: GPUs lack robust cache coherence mechanisms, requiring
  careful coordination when multiple threads access shared memory.

* **Race conditions**: Concurrent memory access requires atomic operations or
  careful algorithm design.

* **Irregular parallelism**: Real-world algorithms often have varying amounts of
  parallel work across iterations.

* **CPU-GPU communication**: Data transfer overhead between host and device must
  be minimized.

Tutorial overview
=================

This collection provides comprehensive tutorials on essential GPU programming
patterns:

* :doc:`Two-dimensional kernels <./programming-patterns/matrix_multiplication>`:
  Processing grid-structured data such as matrices and images.

* :doc:`Stencil operations  <./programming-patterns/stencil_operations>`:
  Updating array elements based on neighboring values.

* :doc:`Atomic operations <./programming-patterns/atomic_operations_histogram>`:
  Ensuring data integrity during concurrent memory access.

* :doc:`Multi-kernel applications <./programming-patterns/multikernel_bfs>`:
  Coordinating multiple GPU kernels to solve complex problems.

* :doc:`CPU-GPU cooperation <./programming-patterns/cpu_gpu_kmeans>`: Strategic
  work distribution between CPU and GPU.

Prerequisites
-------------

To get the most from these tutorials, you should have:

* Basic understanding of C/C++ programming.

* Familiarity with parallel programming concepts.

* HIP runtime environment installed (see :doc:`../install/install`).

* Basic knowledge of GPU architecture (recommended).

Getting started
---------------

Each tutorial is self-contained and can be studied independently, though we
recommend following the order presented for a comprehensive understanding:

1. **Start with Two-dimensional kernels** to understand basic GPU thread
   organization and memory access patterns.
2. **Progress to stencil operations** to learn about neighborhood dependencies.
3. **Study atomic operations** to understand concurrent memory access.
4. **Explore multi-kernel programming** for complex algorithmic patterns.
5. **Check CPU-GPU cooperation** to handle mixed-parallelism workloads.
