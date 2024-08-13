.. meta::
    :description: This chapter provides an overview over the usage of HIP graph.
    :keywords: ROCm, HIP, graph, stream

.. understand_HIP_graph:

********************************************************************************
HIP graph
********************************************************************************

.. note::
    The HIP graph API is currently in Beta. Some features can change and might
    have outstanding issues. Not all features supported by CUDA graphs are yet
    supported. For a list of all currently supported functions see the
    :doc:`HIP graph API documentation<../doxygen/html/group___graph>`.

A HIP graph is made up of nodes and edges. The nodes of a HIP graph represent
the operations performed, while the edges mark dependencies between those
operations.

The nodes can consist of:

- empty nodes
- nested graphs
- kernel launches
- host-side function calls
- HIP memory functions (copy, memset, ...)
- HIP events
- signalling or waiting on external semaphores

The following figure visualizes the concept of graphs, compared to using streams.

.. figure:: ../data/understand/hipgraph/hip_graph.svg
    :alt: Diagram depicting the difference between using streams to execute
          kernels with dependencies, resolved by explicitly calling
          hipDeviceSynchronize, or using graphs, where the edges denote the
          dependencies.

HIP graph advantages
================================================================================

The standard way of launching work on GPUs via streams incurs a small overhead
for each iteration of the operation involved. For kernels that perform large
operations during an iteration this overhead is usually negligible. However
in many workloads, such as scientific simulations and AI, a kernel performs a
small operation for many iterations, and so the overhead of launching kernels
can be a significant cost on performance.

HIP graphs have been specifically designed to tackle this problem by only
requiring one launch from the host per iteration, and minimizing that overhead
by performing most of the initialization beforehand. Graphs can provide
additional performance benefits, by enabling optimizations that are only
possible when knowing the dependencies between the operations.

.. figure:: ../data/understand/hipgraph/hip_graph_speedup.svg
    :alt: Diagram depicting the speed up achievable with HIP graphs compared to
          HIP streams when launching many short-running kernels.
    
    Qualitative presentation of the execution time of many short-running kernels
    when launched using HIP stream versus HIP graph. This does not include the
    time needed to set up the graph.

HIP graph usage
================================================================================

Using HIP graphs to execute your work requires three different steps, where the
first two are the initial setup and only need to be executed once. First the
definition of the operations (nodes) and the dependencies (edges) between them.
The second step is the instantiation of the graph. This takes care of validating
and initializing the graph, to reduce the overhead when executing the graph.

The third step is the actual execution of the graph, which then takes care of
launching all the kernels and executing the operations while respecting their
dependencies and necessary synchronizations as specified.

As HIP graphs require some set up and initialization overhead before their first
execution, they only provide a benefit for workloads that require many iterations to complete.

Setting up HIP graphs
================================================================================

HIP graphs can be created by explicitly defining them, or using stream capture.
For the available functions see the
:doc:`HIP graph API documentation<../doxygen/html/group___graph>`.
