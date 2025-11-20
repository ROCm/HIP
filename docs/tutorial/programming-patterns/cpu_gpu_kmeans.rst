.. meta::
  :description: CPU-GPU cooperative computing at K-means clustering
  :keywords: AMD, ROCm, HIP, CPU-GPU cooperative computing, K-means, clustering, K-means clustering

*******************************************************************************
CPU-GPU cooperative computing: K-means clustering tutorial
*******************************************************************************

Modern heterogeneous systems combine CPUs and GPUs to maximize computational
throughput. GPUs provide massive data-parallel performance, whereas CPUs excel
at complex control logic, and latency-sensitive serial tasks. Many real-world
algorithms—including unsupervised clustering—contain both parallelizable and
inherently sequential components.

This tutorial demonstrates a hybrid CPU–GPU cooperative execution model using
the K-means clustering algorithm, showcasing partitioned workload distribution,
memory management, and performance optimization strategies in heterogeneous
architectures.

.. include:: ../prerequisites.rst

CPU–GPU cooperative computing
=============================

Modern computing platforms integrate **central processing units (CPUs)** and
**graphics processing units (GPUs)** into unified heterogeneous systems.
Each processor type offers distinct architectural strengths:

* **CPUs** feature a small number of complex cores optimized for sequential
  execution, branching, and latency-sensitive control flow.

* **GPUs** consist of thousands of simpler cores optimized for throughput and
  massive data-level parallelism.

**Cooperative computing** refers to a programming paradigm that combines these
capabilities in a coordinated execution model, where both CPU and GPU perform
complementary portions of a workload. Rather than treating the GPU as a passive
accelerator, this approach distributes computation dynamically between devices
to maximize total system utilization.

K-means clustering
==================

K-means is an unsupervised machine learning algorithm that partitions a dataset
into :math:`k` clusters (groups of similar data points) by minimizing
intra-cluster variance. It iteratively refines cluster assignments and centroid
(center point of a cluster) locations until convergence.

The optimization objective is defined as:

.. math::

    \min_{C_1, \dots, C_k} \sum_{i=1}^{k} \sum_{x_j \in C_i} \|x_j - \mu_i\|^2

where :math:`\mu_i` is the centroid of cluster :math:`C_i`.

K-means is frequently used in:

* **Customer segmentation**: Grouping customers by behavior patterns

* **Image compression**: Reducing color palette by clustering similar colors

* **Anomaly detection**: Identifying outliers that don't fit clusters

* **Document clustering**: Organizing similar documents together

* **Feature engineering**: Creating new features based on cluster membership

Algorithm 
=========

The K-means algorithm iteratively refines a partition of a dataset into
:math:`k` clusters by alternating between two primary computational phases:
**assignment** and **update**. Each iteration minimizes the total within-cluster
variance, driving the system toward convergence where centroid movement or
membership changes fall below a defined threshold.

Iterative procedure
-------------------

1. **Initialization**: Select initial centroids (random or via K-means++).

2. **Assignment**: Assign each data point to the nearest centroid.

3. **Update**: Recalculate each centroid as the mean of its assigned members.

4. **Convergence**: Repeat until centroids stabilize or a maximum iteration
   limit is reached.


These phases alternate until the solution converges or a maximum iteration count
is reached.

Initialization
~~~~~~~~~~~~~~

The algorithm begins by selecting initial centroid positions. This step
significantly influences both convergence rate and clustering quality.

* **Random initialization**: centroids are randomly chosen from the dataset.
* **K-means++**: probabilistic seeding to spread centroids across data space.
* **Domain-specific heuristics**: custom initializations for structured data.

Initial centroid diversity reduces the likelihood of poor local minima and
improves overall stability.

Assignment
~~~~~~~~~~

For each data point :math:`x_i`, the algorithm computes the Euclidean distance
to each centroid :math:`\mu_j` and assigns the point to the nearest cluster:

.. math::

   C_i = \arg \min_j \|x_i - \mu_j\|^2

* Each point–centroid distance calculation is independent.

* Ideal for SIMD/SIMT architectures (GPU execution).

* Dominated by dense floating-point arithmetic and memory bandwidth utilization.

This phase represents the **embarrassingly parallel** portion of the algorithm
and provides the largest opportunity for GPU acceleration.

Update
~~~~~~

Once all data points are assigned, new centroid positions are computed as the
mean of their corresponding cluster members:

.. math::

   \mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i

* Requires aggregation and division per cluster.

* Involves variable membership counts across clusters.

* Reduction-heavy and branch-divergent.

Although reduction can be implemented on GPUs, small :math:`k` values and
irregular membership sizes typically make the CPU more efficient for this phase,
given its superior cache hierarchy and flexible control flow.

Convergence
~~~~~~~~~~~

After the update phase, convergence is evaluated using one of the following
criteria:

* No change in point memberships.

* Centroid displacement below a user-defined threshold.

* Iteration count exceeds maximum limit.

If the convergence condition is not met, the assignment and update phases
repeat.

Summary of cooperative execution
--------------------------------

1. **GPU:** performs parallel distance evaluations and membership assignments.
2. **CPU:** executes centroid averaging and convergence checks.
3. **Synchronization:** only essential data (centroids and membership
   arrays) is exchanged per iteration.

This division of labor minimizes data movement and maximizes hardware
utilization. The resulting hybrid implementation combines GPU throughput for
massive data-parallel operations with CPU efficiency for aggregation and control
tasks, enabling scalable clustering performance on heterogeneous systems.

Implementation
==============

This implementation follows a hybrid CPU/GPU design to accelerate the most
computationally expensive phase of K-means: assigning each data point to its
nearest centroid. The GPU performs distance calculations in parallel, while the
CPU handles the centroid recomputation, where sequential reductions are
efficient and data sizes are small. The algorithm iterates between GPU-based
membership updates and CPU-based centroid averaging until convergence or a
maximum iteration count is reached.

Data Structures
---------------

The implementation stores all data in simple, contiguous arrays for efficient
memory access on both CPU and GPU. Data points and centroids are represented as
flattened ``std::vector<float>`` arrays, while cluster assignments are stored as
``std::vector<int>``.

.. code-block:: c++

    // length: Number of data points to cluster
    // dimension: Number of features per data point
    // k: Number of clusters
    
    std::vector<float> data;           // Data points (length * dimension)
    std::vector<float> centroids;      // Centroid positions (k * dimension)
    std::vector<int> memberships;      // Cluster assignments (length)

Main Loop
---------

The core K-means iteration:

.. code-block:: c++

    // length is an integer for the number of entries to be clustered.
    // dimension is an integer for the number of properties of each entry.
    // k is an integer that determines the number of clusters.
    
    std::vector<float> centroids = initializeCentroids(length * dimension, k);
    std::vector<int> memberships(length, 0);
    
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        // Determine the cluster that each entry belongs to. 
        // The function returns how many entries changed membership.
        int membershipChanges = updateMembership(data, centroids, memberships);
        
        // Converge checking.
        if (membershipChanges == 0) {
            break;
        }
        
        // Calculate new centroids.
        std::vector<Point> newCentroids = centroids;
        updateCentroid(data, newCentroids, memberships);
        centroids = newCentroids;
    }

**Loop structure:**

1. Update memberships using GPU
2. Check for convergence (no changes)
3. Update centroids using CPU
4. Repeat until convergence or max iterations

GPU membership update function
------------------------------

The ``updateMembership`` function serves as the interface between CPU and GPU:

.. code-block:: c++

    int updateMembership(float* data, float* centroids, int* membership, 
                        int dataSize, int dimension, int k) {
        // gpuData is allocated and copied to the GPU earlier.
        
        float *gpuCentroids, *gpuMembership;
        
        // Allocate GPU memory
        hipMalloc(&gpuCentroids, k * dimension * sizeof(float));
        hipMalloc(&gpuMembership, dataSize * sizeof(int));
        
        // Copy data from CPU to GPU
        hipMemcpy(gpuCentroids, centroids, k * dimension * sizeof(float),
                  hipMemcpyHostToDevice);
        
        // Calculate the sizes for the kernel launch
        int localSize = 256;
        int globalSize = (dataSize + localSize - 1) / localSize;
        
        // Launch the kernel
        updateMembershipGPU<<<globalSize, localSize>>>(
            gpuData, gpuCentroids, gpuMembership, dataSize, dimension, k);
        hipDeviceSynchronize();
        
        // Create CPU Data to hold the results
        std::vector<int> cpuNewMembership(dataSize);
        
        // Copy GPU data back to CPU
        hipMemcpy(cpuNewMembership.data(), gpuMembership, 
                  dataSize * sizeof(int), hipMemcpyDeviceToHost);
        
        // Count membership updates
        int membershipUpdate = 0;
        for (int i = 0; i < dataSize; ++i) {
            if (membership[i] != cpuNewMembership[i]) {
                membershipUpdate++;
                membership[i] = cpuNewMembership[i]; // Update the original
            }
        }
        
        // Free GPU memory
        hipFree(gpuCentroids);
        hipFree(gpuMembership);
        
        return membershipUpdate;
    }

Step 1: GPU memory allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

    float *gpuCentroids, *gpuMembership;
    hipMalloc(&gpuCentroids, k * dimension * sizeof(float));
    hipMalloc(&gpuMembership, dataSize * sizeof(int));

Allocate space on the GPU for:

* **Centroids**: :math:`k` centroids, each with dimension features
* **Membership assignments**: One cluster ID per data point

Step 2: Data transfer to GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

    hipMemcpy(gpuCentroids, centroids, k * dimension * sizeof(float),
              hipMemcpyHostToDevice);

Transfer the current centroid positions from CPU to GPU. Note that the data
points (``gpuData``) are already on the GPU from earlier initialization.

Step 3: Kernel configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

    int localSize = 256;
    int globalSize = (dataSize + localSize - 1) / localSize;

* ``localSize``: 256 threads per block (common choice)
* ``globalSize``: Number of blocks needed to cover all data points
* Rounding up ensures we process all data points

Step 4: Kernel launch
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

    updateMembershipGPU<<<globalSize, localSize>>>(
        gpuData, gpuCentroids, gpuMembership, dataSize, dimension, k);
    hipDeviceSynchronize();

Launch the GPU kernel to compute cluster assignments in parallel, then wait for
completion.

Step 5: Retrieve results
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

    std::vector<int> cpuNewMembership(dataSize);
    hipMemcpy(cpuNewMembership.data(), gpuMembership, 
              dataSize * sizeof(int), hipMemcpyDeviceToHost);

Copy the new membership assignments back from GPU to CPU.

Step 6: Count changes
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

    int membershipUpdate = 0;
    for (int i = 0; i < dataSize; ++i) {
        if (membership[i] != cpuNewMembership[i]) {
            membershipUpdate++;
            membership[i] = cpuNewMembership[i];
        }
    }

Count how many data points changed clusters. This value determines if the algorithm has converged.

Step 7: Cleanup
~~~~~~~~~~~~~~~

.. code-block:: c++

    hipFree(gpuCentroids);
    hipFree(gpuMembership);
    return membershipUpdate;

Free temporary GPU memory and return the change count.

Kernel implementation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c++

    __global__ void updateMembershipGPU(
        float* data, float* centroids, int* membership,
        int dataSize, int dimension, int k)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (tid < dataSize) {
            float minDistance = INFINITY;
            int bestCluster = 0;
            
            // Find nearest centroid
            for (int cluster = 0; cluster < k; ++cluster) {
                float distance = 0.0f;
                
                // Calculate Euclidean distance
                for (int d = 0; d < dimension; ++d) {
                    float diff = data[tid * dimension + d] - 
                                centroids[cluster * dimension + d];
                    distance += diff * diff;
                }
                
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = cluster;
                }
            }
            
            membership[tid] = bestCluster;
        }
    }

**Kernel operation:**

1. Each thread processes one data point
2. Calculates distance to all :math:`k` centroids
3. Assigns point to nearest centroid
4. Stores result in membership array

CPU centroid update
-------------------

The CPU handles the averaging operation:

.. code-block:: c++

    void updateCentroid(float* data, float* centroids, int* membership,
                       int dataSize, int dimension, int k)
    {
        std::vector<int> counts(k, 0);
        std::vector<float> sums(k * dimension, 0.0f);
        
        // Accumulate sums for each cluster
        for (int i = 0; i < dataSize; ++i) {
            int cluster = membership[i];
            counts[cluster]++;
            
            for (int d = 0; d < dimension; ++d) {
                sums[cluster * dimension + d] += data[i * dimension + d];
            }
        }
        
        // Calculate averages (new centroids)
        for (int cluster = 0; cluster < k; ++cluster) {
            if (counts[cluster] > 0) {
                for (int d = 0; d < dimension; ++d) {
                    centroids[cluster * dimension + d] = 
                        sums[cluster * dimension + d] / counts[cluster];
                }
            }
        }
    }

The centroid update requires:

1. **Reduction operation**: Sum all points in each cluster.
2. **Variable-sized groups**: Clusters have different numbers of points.
3. **Division**: Calculate average (not easily parallelized).

While GPUs can perform reductions, for this workload:

* The number of clusters (:math:`k`) is typically small (< 100).
* The CPU can efficiently handle this sequential aggregation.
* Avoiding GPU complexity keeps code simpler.

Data transfer considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data already on CPU:**

* Membership array was just copied back.
* Data points can be kept on CPU for small datasets.
* Centroids are small (k × dimension values).

**Transfer costs:**

* Only centroids need to copy back to GPU.
* Much smaller than the full dataset.
* Overhead justified by parallel speedup in assignment phase.

Best practices
==============

This section outlines recommended practices for implementing an efficient
GPU-accelerated Breadth-First Search (BFS). It highlights design principles,
memory-management strategies, and debugging techniques that help ensure
correctness, maintainability, and high performance when mapping BFS onto modern
GPU architectures.

Design principles
-----------------

1. **Identify parallelism**

   Decompose the K-means workflow into independent, data-parallel kernels such
   as distance computation. Minimize sequential dependencies in assignment and
   reduction steps.

2. **Minimize host–device data movement**

   Maintain dataset and intermediate buffers resident on the GPU across
   iterations. Transfer only convergence metrics or centroids when absolutely
   necessary.

3. **Batch and fuse operations**

   Combine smaller kernels such as distance computation and assignment, or
   process multiple mini-batches per kernel launch to reduce kernel invocation
   and PCIe transfer overhead.

4. **Overlap communication with computation**

   Utilize asynchronous HIP streams to overlap host–device memory transfers with
   GPU kernel execution. Employ pinned (page-locked) memory for faster DMA
   transfers.

Memory strategy
---------------

1. **Persistent device allocations**

   Allocate GPU memory once before iterative processing. Reuse buffers such as
   those for centroids, assignments, and temporary reductions to avoid the
   overhead of repeated :cpp:func:`hipMalloc` and :cpp:func:`hipFree` calls.

2. **Data locality optimization**

   Co-locate data structures according to access frequency:

   * Store high-throughput arrays (points, centroids) in global memory with
     coalesced access.

   * Cache small, frequently reused values in shared memory or registers.

3. **Transfer scheduling**

   Schedule host–device transfers asynchronously while the GPU executes kernels.
   Use HIP events or streams to synchronize only when necessary.

4. **Memory pooling and reuse**

   Use memory pools such as :cpp:func:`hipMallocAsync`,
   :cpp:func:`hipMallocFromPoolAsync`, or custom allocators to mitigate
   fragmentation and reduce allocation latency, especially in iterative or
   batched workloads.

Performance Considerations
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Aspect

     - Recommendation

   * - **Large datasets**

     - GPU acceleration scales linearly with dataset size. Prioritize keeping
       data on GPU to avoid PCIe bottlenecks.

   * - **Many iterations**

     - Use persistent buffers and asynchronous reduction to minimize 
       per-iteration synchronization overhead.

   * - **Small number of clusters (k)**

     - Kernel execution may be underutilized. CPU execution or hybrid scheduling
       may be more efficient.

   * - **High-dimensional data**

     - Distance computation dominates cost. Leverage GPU shared memory for
       centroid caching and ensure memory coalescing for point vectors.

When to use CPU-GPU cooperation
===============================

* **Hybrid computational structure**

  The algorithm exhibits a clear division between compute-intensive,
  data-parallel kernels such as distance computation in K-means and lightweight,
  control-heavy serial stages such as centroid updates or convergence checks.

* **High arithmetic intensity in parallel regions**

  The GPU-executed portion performs substantial arithmetic operations per memory
  access, ensuring efficient utilization of GPU cores and minimizing the impact
  of memory latency.

* **Low-complexity reduction or synchronization on CPU**

  The CPU phase primarily aggregates results or performs decision logic that
  does not justify GPU kernel execution overhead.

* **Large-scale data processing**

  Dataset size exceeds the threshold where PCIe transfer costs can be amortized,
  enabling sustained GPU throughput and effective memory reuse.

* **Iterative or streaming workloads**

  Algorithms involving multiple iterations benefit from persistent GPU memory
  allocations and overlapped data transfer, significantly reducing
  per-iteration latency.

When to avoid CPU-GPU cooperation
=================================

* **Fully parallelizable workloads**

  Algorithms with uniform, independent computations across all data points are
  better suited for GPU-only execution without CPU coordination overhead.

* **Low computational density per element**

  If each data element requires few operations relative to transfer cost,
  CPU–GPU communication latency will dominate, leading to suboptimal
  performance.

* **High inter-phase data dependency**

  Frequent synchronization or data exchange between CPU and GPU phases prevents
  effective pipeline overlap and leads to idle compute units.

* **Small problem sizes**

  When datasets fit comfortably in CPU cache or system memory, the GPU launch
  overhead and transfer latency outweigh any computational gains from
  offloading.

Conclusion
==========

The K-means implementation illustrates **heterogeneous workload partitioning**
between CPU and GPU. By mapping compute-intensive, data-parallel operations to
the GPU and reduction-heavy serial logic to the CPU, total runtime is minimized
while code complexity remains manageable.

This CPU–GPU cooperative execution paradigm generalizes to many algorithms
combining reduction, aggregation, and distance-based computation—enabling
scalable, efficient utilization of modern heterogeneous hardware.
