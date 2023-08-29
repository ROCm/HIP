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

The Cooperative Groups API introduces new APIs to launch, group, subdivide,
synchronize and identify threads, as well as some predefined group-collective
algorithms, but most importantly a matching threading model to think in terms
of. It relaxes some of the restrictions of the {ref}`inherent_tread_model`
imposed by the strict 1:1 mapping of architectural details to the programming
model.

The rich set of APIs introduced by Cooperative Groups allow the programmer
to define their own groups based on run-time predicates, but a set of implicit
groups manifest based on kernel launch parameters.

The thread hiearchy abstraction of Cooperative Groups manifest as depicted in
{numref}`coop_thread_hierarchy`.

:::{figure-md} coop_thread_hierarchy

<img src="../data/reference/programming_model/thread_hierarchy_coop.svg" alt="Cooperative group thread hierarchy.">

Cooperative group thread hierarchy.
:::

- Multi Grid is an abstraction of potentially multiple simultaneous launches of
  the same kernel over multiple devices. Grids inside a multi device kernel
  launch need not be of uniform size, thus allowing taking into account
  different device capabilities and preferences.

  ```{admonition} Deprecation
  :class: warning
  Types representing this level of thread groups have been deprecated in ROCm
  5.0
  ```

- Same as the {ref}`inherent_tread_model` Grid entity. The ability to
  synchronize over a grid requires the kernel to be launched using the
  Cooperative Groups API.

- The defining feature of a cluster or block cluster is that all threads in a
  cluster will share a common set of distributed shared memory which they may
  use to share data or synchronize with oneanother.

- Same as the {ref}`inherent_tread_model` Block entity.

```{note}
Explicit warp-level thread handling is absent from the Cooperative Groups API.
In order to exploit the known hardware SIMD width on which built-in
functionality translates to simpler logic, one may use the group partitioning
part of the API, typically but not necessarily `tiled_partition`.
```

## Memory Model

The hierarchy of threads introduced by {ref}`inherent_tread_model` is induced
by the memory subsystem of GPUs. _LINK_ summarizes that memory namespaces and
how they relate to the various levels of the threading model.

- Per-thread memory or local memory is read-write storage only visible to the
  threads defining the given variables. This is the default memory namespace.

- Shared memory is read-write storage visible to all the threads in a given
  block.

- Distributed shared memory is read-write storage visible to all the threads
  in a given block cluster.

- Global memory is read-write storage visible to all threads in a given grid.
  There are specialized versions of global memory with different usage
  semantics which are typically backed by the same hardware storing global.

  - Constant memory is read-only storage visible to all threads in a given
    grid. It is a limited segment of global with queryable size.

  - Texture memory is read-only storage visible to all threads in a given grid
    and accessible through additional APIs.

  - Surface is a writable version of texture memory.

## Execution Model
