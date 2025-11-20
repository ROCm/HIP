.. meta::
  :description: Multi-kernel programming with breadth-first search (BFS)
  :keywords: AMD, ROCm, HIP, GPU programming, multi-kernel, BFS, breadth-first search

*******************************************************************************
Multi-kernel programming: breadth-first search tutorial
*******************************************************************************

Many real-world GPU workloads involve multiple kernels cooperating to solve a
single problem. This tutorial explores **multi-kernel GPU programming** using
the breadth-first search (BFS) algorithm, a foundational graph traversal
method widely used in networking, path-finding, and social network analysis.

The implementation is adapted from the **Rodinia benchmark suite**, a
well-known collection of heterogeneous computing workloads that demonstrate
different parallel programming strategies.

.. include:: ../prerequisites.rst

Multi-kernel GPU programming
============================

In GPU computing, some algorithms cannot be efficiently expressed using a
single kernel due to synchronization or dependency constraints. Instead, they
are decomposed into multiple kernels that execute sequentially, with each
kernel responsible for a specific computation phase.

This approach, called **multi-kernel programming**, is essential when:

* Results from one kernel determine the input for the next.

* Global synchronization between thread blocks is required.

* Control flow depends on runtime conditions.

* The algorithm involves iterative or level-wise processing.

Breadth-first search (BFS)
==========================

Breadth-first search (BFS) is a **layered graph traversal algorithm** that
explores nodes level by level, starting from a root node. It guarantees finding
the shortest path (in edge count) to all reachable nodes in an unweighted
graph.

Applications of BFS include:

* **Path-finding**: Finding shortest paths between nodes.
* **Peer-to-peer networking**: Network topology discovery.
* **GPS navigation**: Route planning and optimization.
* **Social networks**: Friend recommendations and connection analysis.
* **Web crawling**: Systematic website exploration.

Algorithm characteristics
-------------------------

BFS is structured as a level-synchronous algorithm:

* Nodes in the same graph level are processed concurrently.
* A queue (or "frontier") tracks which nodes to explore next.
* Each node is visited once to prevent redundant processing.

Sequential BFS is straightforward but inherently serial due to the
level-by-level dependency between nodes. GPU parallelization requires
restructuring the traversal to exploit data parallelism across nodes
within the same frontier.

Sequential BFS algorithm
=========================

Let's first understand how BFS works sequentially before parallelizing it.

Example Graph
~~~~~~~~~~~~~

Consider a simple graph with four nodes:

.. code-block:: text

           R (root)
          / \
         A   B
          \ /
           C

Step-by-step execution
~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Start at the root node ``R``

* Mark ``R`` as visited  
* Enqueue ``R``  
* Queue: [R]

**Step 2**: Process ``R``

* Dequeue ``R``  
* Discover neighbors: ``A`` and ``B``  
* Enqueue both, mark as visited  
* Queue: [A, B]

**Step 3**: Process ``A``

* Dequeue ``A``  
* Neighbors: ``R`` (visited) and ``C`` (new)  
* Enqueue ``C``  
* Queue: [B, C]

**Step 4**: Process ``B``

* Dequeue ``B``  
* Neighbors: ``R`` (visited) and ``C`` (visited)  
* Queue: [C]

**Step 5**: Process ``C``

* Dequeue ``C``  
* All neighbors visited  
* Queue becomes empty — traversal complete

Parallel BFS on GPU
===================

Unlike dense linear algebra, BFS is an **irregular** algorithm. The amount of
work per node varies, and the connectivity pattern of the graph drives
execution. The main challenges are:

1. **Data dependencies**: nodes in the next level depend on the previous level.

2. **Irregular parallelism**: each frontier may contain a very different number of nodes.

3. **Dynamic workload**: the size of the next frontier is unknown at runtime.

4. **Synchronization**: all nodes in one frontier must complete before the next begins.

The **frontier** is the set of nodes being processed at a given BFS level.
Parallel BFS executes all frontier nodes simultaneously, using one thread per
node to discover new neighbors and mark them for the next iteration.

Implementation strategy
-----------------------

The GPU implementation performs BFS using **two cooperating kernels**:

1. **Kernel 1**: processes all nodes in the current frontier.
2. **Kernel 2**: updates the next frontier and checks if work remains.

This design provides **implicit synchronization** between levels while avoiding
race conditions. The host (CPU) manages the iterative control loop, launching
kernels repeatedly until no more frontier nodes exist.

Data structures
===============

The graph is represented using adjacency lists stored in arrays:

.. code-block:: c++

    struct Node {
        int starting;       // starting index in the edge list
        int no_of_edges;    // number of outgoing edges
    };

**Main arrays:**

* ``g_graph_nodes``: node array storing offsets into the edge list.
* ``g_graph_edges``: flattened list of edge destinations.
* ``g_graph_mask``: boolean array indicating active frontier nodes.
* ``g_updating_graph_mask``: marks nodes to be added to the next frontier.
* ``g_graph_visited``: tracks which nodes were visited.
* ``g_graph_cost``: stores the distance (edge count) from the source node.

**Control flow flags:**

* ``g_over``: device-side flag indicating whether another iteration is needed.
* The host resets this flag each iteration and checks it after kernel execution.

The two-kernel approach
=======================

The two-kernel structure ensures correctness and efficient synchronization:

* **Exploration kernel (Kernel 1)** discovers new nodes.
* **Update kernel (Kernel 2)** finalizes state for the next iteration.

This separation:

* Avoids race conditions between threads of different levels.

* Provides synchronization between BFS levels.

* Keeps control logic simple on the host side.

Kernel 1: process current frontier
----------------------------------

Each thread processes one node from the current frontier, examining all of its
outgoing edges:

.. code-block:: c++

    __global__ void Kernel1(
        Node* g_graph_nodes,
        int* g_graph_edges,
        bool* g_graph_mask,
        bool* g_updating_graph_mask,
        bool* g_graph_visited,
        int* g_graph_cost,
        int no_of_nodes)
    {
        int tid = hipBlockIdx_x * MAX_THREADS_PER_BLOCK + hipThreadIdx_x;

        if (tid < no_of_nodes && g_graph_mask[tid]) {
            g_graph_mask[tid] = false;

            for (int i = g_graph_nodes[tid].starting;
                 i < g_graph_nodes[tid].starting + g_graph_nodes[tid].no_of_edges;
                 i++) {
                int id = g_graph_edges[i];
                if (!g_graph_visited[id]) {
                    g_graph_cost[id] = g_graph_cost[tid] + 1;
                    g_updating_graph_mask[id] = true;
                }
            }
        }
    }

**Kernel 1 responsibilities:**

* Clear the node’s mask (mark processed).

* Explore all edges.

* For each unvisited neighbor:

  * Compute cost (distance).

  * Add to the next frontier.

Kernel 2: update frontier
-------------------------

This kernel finalizes the next frontier:

.. code-block:: c++

    __global__ void Kernel2(
        bool* g_graph_mask,
        bool* g_updating_graph_mask,
        bool* g_graph_visited,
        bool* g_over,
        int no_of_nodes)
    {
        int tid = hipBlockIdx_x * MAX_THREADS_PER_BLOCK + hipThreadIdx_x;

        if (tid < no_of_nodes && g_updating_graph_mask[tid]) {
            g_graph_mask[tid] = true;
            g_graph_visited[tid] = true;
            *g_over = true;
            g_updating_graph_mask[tid] = false;
        }
    }

**Kernel 2 responsibilities:**

* Move newly discovered nodes into the active frontier.

* Mark them as visited.

* Signal continuation via ``*g_over``.

Host-side control loop
======================

.. code-block:: c++

    do {
        h_over = false;
        hipMemcpy(d_over, &h_over, sizeof(bool), hipMemcpyHostToDevice);

        Kernel1<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(
            d_graph_nodes, d_graph_edges, d_graph_mask,
            d_graph_updating_graph_mask, d_graph_visited,
            d_graph_cost, no_of_nodes);
        hipDeviceSynchronize();

        Kernel2<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(
            d_graph_mask, d_graph_updating_graph_mask,
            d_graph_visited, d_over, no_of_nodes);
        hipDeviceSynchronize();

        hipMemcpy(&h_over, d_over, sizeof(bool), hipMemcpyDeviceToHost);
    } while (h_over);


The loop exits when no new nodes are discovered. ``g_over`` or ``h_over`` on
host side remains ``false`` after one full iteration.

Performance Characteristics
===========================

Parallelism Patterns
--------------------

**Within each iteration:**

- High parallelism: All frontier nodes processed simultaneously
- Work distribution: One thread per node

**Across iterations:**

- Sequential: Must complete one level before starting the next
- Variable parallelism: Different levels may have different numbers of nodes

Workload Characteristics
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Characteristic
     - Description
   * - **Irregular**
     - Frontier size varies dramatically across levels
   * - **Data-dependent**
     - Graph structure determines parallel work available
   * - **Dynamic**
     - Cannot predict workload statically
   * - **Memory-bound**
     - Many memory accesses per computation

Best practices
==============

This section outlines recommended practices for implementing an efficient
GPU-accelerated Breadth-First Search (BFS). It highlights design principles,
memory-management strategies, and debugging techniques that help ensure
correctness, maintainability, and high performance when mapping BFS onto modern
GPU architectures.

Design principles
-----------------

1. **Define clear kernel roles**

   Decompose BFS into well-defined GPU kernels, each responsible for a specific
   phase of computation. For example:

   * **Kernel 1**: frontier expansion (discovering new nodes)
   * **Kernel 2**: frontier update (marking next-level nodes)

   This separation simplifies synchronization and ensures that each kernel
   operates on independent data regions.

2. **Minimize host–device communication**

   Keep graph data structures (nodes, edges, masks) resident on the GPU across
   iterations. Only transfer lightweight control flags such as ``g_over`` to the
   host each loop iteration to check termination conditions.

3. **Kernel boundaries as synchronization points**

   Kernel launch boundaries on the same stream naturally enforce global
   synchronization across all threads on the GPU. Each kernel invocation
   completes before the next begins, ensuring that:

   * All nodes in the current frontier are fully processed before updating the
     next frontier.

   * Memory updates to arrays like ``g_graph_cost`` or ``g_graph_mask`` are
     visible to all threads in subsequent kernels.

   This avoids the need for costly device-wide barriers or explicit
   synchronization primitives within a single kernel. Leverage kernel sequencing
   to structure iterative algorithms cleanly—each kernel represents one
   computation phase per BFS level.

4. **Flag-based control**

   Use device-side flags for dynamic termination and conditional control flow.
   In BFS, the Boolean flag ``g_over`` serves as a device-to-host signal
   indicating whether new nodes were discovered during the current iteration.

   * Initialize ``g_over`` to ``false`` on the host at the start of each
     iteration.

   * Allow GPU threads in **Kernel 2** to set ``*g_over = true`` when adding new
     nodes to the next frontier.

   * After kernel completion, copy the flag back to the host using
     :cpp:func:`hipMemcpy`. If ``g_over`` remains false, the traversal is
     complete.

   This mechanism avoids repeated host intervention and enables a tight
   CPU–GPU control loop that dynamically adapts to workload size without
   transferring large data structures.

Memory strategy
---------------

1. **Persistent device allocations**

   Allocate all required device buffers once prior to traversal. Reuse these
   allocations across multiple BFS runs or multiple source nodes to minimize
   the overhead of repeated :cpp:func:`hipMalloc` and :cpp:func:`hipFree`
   calls.

2. **Minimize host–device communication**

   Keep graph data structures (nodes, edges, masks) resident on the GPU across
   iterations. Only transfer lightweight control flags such as ``g_over`` to the
   host each loop iteration to check termination conditions.

3. **Use pinned host memory for control flags**

   When copying ``g_over`` or other control signals between host and device,
   allocate host memory using pinned (page-locked) buffers to accelerate DMA
   transfers.

Debugging and validation
------------------------

1. **Frontier validation**

   After each iteration, verify the number of nodes marked in
   ``g_graph_mask``. Unexpected empty or overfull frontiers often indicate
   incorrect synchronization or uninitialized masks.

2. **Termination condition check**

   Confirm that the host-side loop terminates when ``g_over`` remains false
   for one iteration. If the loop never ends, ensure ``g_over`` is reset on the
   host before each kernel launch.

3. **Result verification**

   Compare computed distances in ``g_graph_cost`` against a CPU reference
   implementation for small graphs to validate correctness.

4. **Profiling and bottleneck detection**

   Use tools such as :doc:`rocprofv3<rocprofiler-sdk:how-to/using-rocprofv3>`
   or :doc:`ROCm compute profiler<rocprofiler-compute:how-to/profile/mode>`
   to measure per-kernel execution times, memory throughput, and
   synchronization overhead.

5. **Logging and debug builds**

   Enable optional logging for iteration counts, frontier sizes, and
   synchronization states during development. Disable logging in production
   builds to avoid performance impact.

Conclusion
==========

Multi-kernel GPU programming is essential for complex algorithms that require:

* Multiple phases of computation.

* Data dependencies between phases.

* Dynamic control flow based on intermediate results.

The BFS example demonstrates:

* How to decompose algorithms into multiple cooperating kernels.

* Techniques for managing frontiers and iterative processing.

* Strategies for handling irregular and dynamic parallelism.

* Proper synchronization between kernel launches.

Key takeaways:

1. **Kernel boundaries provide synchronization**: Use them strategically to ensure correctness.
2. **Separate exploration from update**: Prevents race conditions in level-based algorithms.
3. **Host controls iteration**: CPU manages the overall loop while GPU does heavy lifting.
4. **Flags enable dynamic control**: Device-side flags allow work-dependent termination.

Understanding multi-kernel patterns enables developers to implement 
sophisticated algorithms like graph processing, dynamic programming, and 
iterative refinement methods efficiently on GPUs.
