# Programming Model

HIP defines a model of mapping SIMT programs (Single Instruction, Multiple
Threads) onto various architectures, primarily GPUs. While the model may be
expressed in most imperative languages, (eg. Python via PyHIP) this document
will focus on the original C/C++ API of HIP.

## Threading Model

The SIMT nature of HIP is captured by the ability to execute user-provided
device programs, expressed as single-source C/C++ functions or sources compiled
online/offline to binaries in bulk.

Multiple instances of the device program (aka. kernel) may execute in parallel,
all uniquely identified by a set of integral values which are referred to as
thread IDs. The set of integers identifying a thread relate to the hierarchy in
which threads execute.

(inherent_tread_model)=

### Inherent Thread Model

The thread hiearchy inherent to how AMD GPUs operate manifest as depicted in
{numref}`inherent_thread_hierarchy`.

:::{figure-md} inherent_thread_hierarchy

<img src="../data/reference/programming_model/thread_hierarchy.svg" alt="Hierarchy of thread groups.">

Hierarchy of thread groups.
:::

- The innermost grouping is called a warp, or a wavefront in ISA terms. A warp
  is the most tightly coupled groups of threads, both physically and logically.

  When referring to threads inside a warp, they may be called lanes, and the
  integral value identifying them the lane ID. Lane IDs aren't quieried like
  other thread IDs, but are user-calculated. As a consequence they are only as
  multi-dimensional as the user interprets the calculated values to be.

  The size of a warp is architecture dependent and always fixed. Warps are
  signified by the set of communication primitives at their disposal, detailed
  under {ref}`warp_cross_lane_functions`.

- The middle grouping is called a block or thread block. The defining feature
  of a block is that all threads in a block will share an instance of memory
  which they may use to share data or synchronize with oneanother.

  The size of a block is user-configurable but is maxmized by the queryable
  capabilites of the executing hardware. The unique ID of the thread within a
  block is 3-dimensional as provided by the API. When linearizing thread IDs
  within a block, assume the "fast index" being dimension `x`, followed by the
  `y` and `z` dimensions.

- The outermost grouping is called a grid. A grid manifests as a single
  dispatch of kernels for execution. The unique ID of each block within a grid
  is 3-dimensional, as provided by the API and is queryable by every thread
  within the block.

### Cooperative Groups Thread Model

The Cooperative Groups API introduces new APIs to launch and identify threads,
as well as a matching threading model to think in terms of. It relaxes some of
the restrictions of the {ref}`inherent_tread_model` imposed by the strict 1:1
mapping of architectural details to the programming model.
