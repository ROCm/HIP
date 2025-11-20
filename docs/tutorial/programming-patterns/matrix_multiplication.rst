.. meta::
  :description: Matrix multiplication tutorial
  :keywords: AMD, ROCm, HIP, two-dimensional kernels, matrix multiplication tutorial

*******************************************************************************
Two-dimensional kernels: Matrix multiplication tutorial
*******************************************************************************

GPUs provide a massively parallel architecture consisting of thousands of cores,
making them exceptionally well-suited for data-parallel computations.
Two-dimensional kernel patterns are commonly data-parallel, enabling us to
leverage GPU capabilities to exploit this inherent parallelism.

Tasks involving large matrices, which are common in image processing and machine
learning applications, can be significantly accelerated by distributing
computations across GPU cores. This tutorial explores how to implement matrix
multiplication using two-dimensional GPU kernels with :doc:`HIP <hip:index>`.

.. include:: ../prerequisites.rst

Characteristics of 2D computational problems
============================================

* Spatial locality and data dependencies: Adjacent elements in the grid often
  exhibit strong spatial correlations, making memory access patterns and cache
  utilization critical for performance.

* Natural 2D data representation: Many datasets—such as images, numerical
  matrices, and discretized physical fields—map directly onto a two-dimensional
  coordinate space.

* Prevalence in simulation and modeling: Numerous scientific and engineering
  workloads such as finite difference methods, fluid dynamics, heat transfer,
  and image processing, are inherently two-dimensional.

Modern GPU architectures are engineered to exploit the parallelism of 2D
computational grids. By designing kernels that operate on two-dimensional thread
blocks and memory layouts, developers can optimize global memory access,
minimize latency, and maximize throughput. Leveraging 2D kernel configurations
not only aligns the computation with the GPU’s hardware topology but also
enables substantial performance improvements for domain-specific applications.

Matrix multiplication
=====================

Let **A** and **B** be two matrices defined as follows,
:math:`A \in \mathbb{R}^{m \times n}` and :math:`B \in \mathbb{R}^{n \times k}`.
The matrix product :math:`C = A \cdot B` is defined only when the number of
columns of :math:`A` equals the number of rows of :math:`B`. The resulting
matrix :math:`C \in \mathbb{R}^{m \times k}` has elements given by:

.. math::

   C_{ij} = \sum_{r=1}^{n} A_{ir} B_{rj}

for all :math:`i = 1, \dots, m` and :math:`j = 1, \dots, k`.

In other words, each element :math:`C_{ij}` is computed as the dot product of
the :math:`i`-th row vector of :math:`A` and the *j*-th column vector of
:math:`B`. This operation is repeated for all valid pairs of :math:`(i, j)` to
construct the complete matrix :math:`C`.

For two square matrices of size :math:`N \times N`, the computational cost of
classical matrix multiplication is :math:`O(N^3)`. As an example, consider
:math:`N = 32`:

* **Multiplication operations**: :math:`32^3 = 32{,}768`
* **Addition operations**: :math:`32^2 \times (32 - 1) = 31{,}744`

Each element in the resulting matrix :math:`C` is computed independently of the
others, since it depends only on a single row of :math:`A` and a single column
of :math:`B`. This property makes matrix multiplication
**highly parallelizable** and well-suited for execution on GPUs, multi-core
CPUs, or distributed computing architectures.

CPU implementation
==================

A baseline CPU implementation provides a clear understanding of the classical
matrix multiplication algorithm before exploring parallel GPU execution.

.. code-block:: c++

    #include <iostream>
    #include <cstdlib>
    #define N 32

    void cpu_matrix_multiplication(float *a, float *b, float *c, int n) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k) {
                    sum += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    int main() {
        float *a, *b, *c;
        a = (float*)malloc(sizeof(float) * N * N);
        b = (float*)malloc(sizeof(float) * N * N);
        c = (float*)malloc(sizeof(float) * N * N);

        // Initialize matrix A
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                a[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
            }
        }

        // Initialize matrix B
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                b[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
            }
        }

        cpu_matrix_multiplication(a, b, c, N);
        free(a);
        free(b);
        free(c);
        return 0;
    }

The ``cpu_matrix_multiplication`` function performs the classical :math:`O(N^3)`
matrix multiplication algorithm using three nested loops. The implementation
proceeds as follows:

* **Input parameters:** Three pointers to contiguous memory blocks representing
  matrices :math:`A`, :math:`B`, and the output matrix :math:`C`.

* **Outer and middle loops:** The indices :math:`i` and :math:`j` iterate over
  the rows and columns of the output matrix :math:`C`, respectively.

* **Innermost loop:** For each element :math:`C_{ij}`, the loop over :math:`k`
  performs a dot product between the *i*-th row of :math:`A` and the :math:`j`-th
  column of :math:`B`:

  .. math::

     C_{ij} = \sum_{k=0}^{n-1} A_{ik} \cdot B_{kj}

* **Temporary accumulation:** A local scalar :code:`sum` accumulates the
  intermediate sum before being written to :code:`c[i * n + j]`.

This implementation has a computational complexity of :math:`O(N^3)` and poor
cache locality for large matrices, but it serves as a reference for
understanding sequential computation before introducing GPU parallelization.

GPU implementation
==================

The following example demonstrates a complete HIP implementation of matrix
multiplication, including host and device memory management, kernel
implementation, configuration, and synchronization.

GPU kernel
----------

The core computation is performed by the GPU kernel, where each thread computes
one element of the output matrix. For comparison, the CPU implementation is
also provided.

.. list-table::
   :header-rows: 1

   * - **GPU version**

   * - .. code-block:: c++
         
          __global__ void gpu_matrix_multiplication(float *a, float *b, float *c, int n) {

              int row = blockIdx.y * blockDim.y + threadIdx.y;
              int col = blockIdx.x * blockDim.x + threadIdx.x;
              float sum = 0.0f;

              if (row < n && col < n) {
                  for (int k = 0; k < n; ++k) {
                      sum += a[row * n + k] * b[k * n + col];
                  }
                  c[row * n + col] = sum;
              }
          }

.. list-table::
   :header-rows: 1

   * - **CPU version**

   * - .. code-block:: c++
         
          void cpu_matrix_multiplication(float *a, float *b, float *c, int n) {

              for (int i = 0; i < n; ++i) {
                  for (int j = 0; j < n; ++j) {
                      float sum = 0.0f;
                      for (int k = 0; k < n; ++k) {
                          sum += a[i * n + k] * b[k * n + j];
                      }
                      c[i * n + j] = sum;
                  }
              }
          }

The outer and middle loops of the CPU implementation are replaced by
the parallel execution of the GPU implementation. Each GPU thread computes the
**single element** :math:`C_{ij}` of the output matrix corresponding to one dot
product between a row of :math:`A` and a column of :math:`B`. This decomposition
exposes massive parallelism, as all elements of :math:`C` can be computed
independently and concurrently.

Thread and block identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each thread’s global position within the grid is determined by:

.. code-block:: c++

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

Here:

* ``threadIdx.(x|y)``: Local thread indices within a block.
* ``blockIdx.(x|y)``: Block indices within the grid.
* ``blockDim.(x|y)``: Dimensions of each block (used to scale offsets).

Boundary checking
~~~~~~~~~~~~~~~~~

Since the total number of threads launched may exceed :math:`N^2`, boundary
checking ensures that threads outside the matrix domain do not perform invalid
memory accesses:

.. code-block:: c++

    if (row < n && col < n) {
        // Safe computation region
    }

Dot product computation
~~~~~~~~~~~~~~~~~~~~~~~

Within the valid region, each thread executes a dot product over :math:`k`:

* Loads one element from row :code:`row` of matrix :math:`A`.

* Loads one element from column :code:`col` of matrix :math:`B`.

* Multiplies and accumulates these values into :code:`sum`.

* Writes the final scalar result to :code:`c[row * n + col]`.

This kernel performs the same :math:`O(N^3)` arithmetic operations as the CPU
version but distributes them across thousands of concurrent GPU threads,
achieving significant acceleration through parallel execution and memory
throughput optimization.

Step 1: Host memory allocation and initialization
--------------------------------------------------

Host memory is allocated for matrices and initialized with random floating-point
values.

.. code-block:: c++

    int main() {
        float *h_a, *h_b, *h_c;
        h_a = (float*)malloc(sizeof(float) * N * N);
        h_b = (float*)malloc(sizeof(float) * N * N);
        h_c = (float*)malloc(sizeof(float) * N * N);

        // Initialize matrix A
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                h_a[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
            }
        }

        // Initialize matrix B
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                h_b[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
            }
        }

**Description:**

* Declare host (CPU) pointers ``h_a``, ``h_b``, and ``h_c``.

* Allocate contiguous memory for each :math:`N \times N` matrix.

* Initialize input matrices :math:`A` and :math:`B` with pseudo-random
  floating-point values in the range [0, 1).

Step 2: Device memory allocation and data transfer
---------------------------------------------------

Memory is allocated on the GPU, and input matrices are transferred from host to
device.

.. code-block:: c++

    float *d_a, *d_b, *d_c;
    hipMalloc((void**)&d_a, sizeof(float) * N * N);
    hipMalloc((void**)&d_b, sizeof(float) * N * N);
    hipMalloc((void**)&d_c, sizeof(float) * N * N);

    hipMemcpy(d_a, h_a, sizeof(float) * N * N, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, sizeof(float) * N * N, hipMemcpyHostToDevice);

**Operations:**

1. Allocate GPU (device) memory for matrices ``d_a``, ``d_b``, and ``d_c``.
2. Transfer data from host to device using :cpp:func:`hipMemcpy` with direction
   :cpp:enum:`hipMemcpyHostToDevice`.

Step 3: Configure and launch kernel
------------------------------------

Kernel launch parameters define how threads are organized across blocks and the
grid.

.. code-block:: c++

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    int n_blocks = static_cast<int>(ceil(static_cast<float>(N) / BLOCK_SIZE));
    dim3 blocksPerGrid(n_blocks, n_blocks);

    hipLaunchKernelGGL(gpu_matrix_multiplication,
                       blocksPerGrid,
                       threadsPerBlock,
                       0, 0,
                       d_a, d_b, d_c, N);

    hipDeviceSynchronize();

Configuration details
~~~~~~~~~~~~~~~~~~~~~

The :code:`dim3` type defines thread and block dimensions:

* :code:`threadsPerBlock`: Number of threads per block. A 16 × 16 block
  (256 threads total).

* :code:`n_blocks`: Number of blocks per dimension. Computed as
  :math:`\lceil N / \mathrm{BLOCK\_SIZE} \rceil`.

* :code:`blocksPerGrid`: A grid of blocks covering the entire :math:`N \times N`
  matrix.

For example :math:`N = 256` and ``BLOCK_SIZE = 16``:

* :code:`n_blocks`: :math:`\lceil 256 / 16 \rceil = 16`

* :code:`blocksPerGrid`: :math:`16 \times 16 = 256`

* Total threads: :math:`256 \text{ blocks} \times 256 \text{ threads/block} = 65{,}536`
  threads.

Rounding up ensures full coverage of the matrix even when :math:`N` is not an
exact multiple of ``BLOCK_SIZE``. The boundary check in the kernel

.. code-block:: c++

    if (row < n && col < n)

prevents out-of-bounds memory access for extra threads.

Synchronization
~~~~~~~~~~~~~~~~

The call to :cpp:func:`hipDeviceSynchronize()` ensures that all GPU computations
complete before the CPU accesses results or proceeds to subsequent operations.
This is essential for correctness and debugging.

Step 4: Copy results back and cleanup
--------------------------------------

After the kernel execution, results are transferred back to host memory, and
all allocated resources are released.

.. code-block:: c++

    hipMemcpy(h_c, d_c, sizeof(float) * N * N, hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
    }

**Summary of final steps:**

1. Copy the result matrix from device to host using ``hipMemcpy``.
2. Free all device memory allocations with ``hipFree``.
3. Release host memory with ``free``.
4. Return control to the operating system, indicating successful program termination.

Parallelization benefits
========================

For a 256 × 256 matrix multiplication:

* **Sequential CPU version:** Computes 65,536 output elements serially.

* **Parallel GPU version:** Executes up to 65,536 independent threads concurrently.

This results in a theoretical performance gain proportional to the number of
active GPU threads and the device’s compute throughput.

Each output element :math:`C_{ij}` is computed independently from others, since
it depends solely on row :math:`i` of matrix :math:`A` and column :math:`j` of
matrix :math:`B`. This independence allows full utilization of GPU streaming
multiprocessors and makes the algorithm highly scalable.

Best practices
==============

1. **Choose optimal block sizes** 

   Powers of two (e.g., 16 or 32) often yield better occupancy and memory
   alignment.

2. **Handle boundary conditions** 

   Always include thread boundary checks.

3. **Synchronize appropriately**

   Use :cpp:func:`hipDeviceSynchronize()` after kernel launches to ensure data
   consistency.

4. **Memory coalescing** 

   Arrange data access patterns so consecutive threads access contiguous memory
   locations, maximizing bandwidth utilization.

5. **Use shared memory** 

   Use shared memory to cache sub-blocks of matrices, significantly 
   reducing global memory latency.

6. **Profile and tune** 

   Use tools such as :doc:`rocprofv3<rocprofiler-sdk:how-to/using-rocprofv3>`
   or :doc:`ROCm compute profiler<rocprofiler-compute:how-to/profile/mode>`
   to identify bottlenecks and fine-tune kernel launch configurations.

Conclusion
==========

Two-dimensional GPU kernels provide an efficient mechanism to accelerate dense
linear algebra computations such as matrix multiplication by exploiting
fine-grained data parallelism. This example demonstrates:

* Structuring GPU kernels for 2D problems.

* Managing memory transfers between host and device.

* Configuring thread and block hierarchies.

* Achieving substantial speedups via massive parallel execution.

Understanding these concepts enables developers to implement optimized GPU
solutions for computationally intensive workloads, including scientific
simulations, numerical linear algebra, and machine learning. Because each output
element is computed independently, matrix multiplication serves as an ideal
introductory example for mastering GPU programming paradigms applicable to a
wide range of data-parallel applications.
