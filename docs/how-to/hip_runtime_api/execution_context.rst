.. meta::
    :description: How to use HIP execution contexts to partition GPU compute resources
    :keywords: AMD, ROCm, HIP, execution context, compute unit, CU, resource partitioning, work queue

.. _execution_context:

*******************************************************************************
Execution contexts
*******************************************************************************

In the HIP runtime, an execution context is the scheduling domain used for GPU
work submitted by a process. An execution context is either the device's primary
context, which the HIP runtime manages implicitly, or a resource-partitioned
context that you create with :cpp:func:`hipGreenCtxCreate`.

By default, every kernel you launch uses the device's primary execution context.
In the primary context, kernels launched by the process compete for the GPU
resources available to the process. The runtime decides which compute units
(CUs) a kernel runs on, and kernels that run at the same time share the available
pool of CUs and work queues.

A resource-partitioned execution context lets you define a fixed slice of the
available GPU resources and bind work to it. Any kernel launched on a stream that
belongs to that context is confined to the context's CU resources, regardless of
the kernel launch configuration.

You set up resource-partitioned execution contexts entirely on the host. The
kernel source does not change, and kernels are launched with the usual HIP
launch syntax. This feature is analogous to CUDA green contexts.

.. note::

    Keep the following terminology and example-code details in mind:

    - The device resource structures use the field name ``smCount`` and the
      resource type ``hipDevResourceTypeSm``. HIP keeps these ``sm`` identifiers
      so that code written against CUDA compiles unchanged. On AMD GPUs, the
      equivalent physical resource is the compute unit. This page writes
      "compute unit" or "CU" in the text and keeps the literal identifiers in
      code.

    - The snippets on this page are written to show the calling sequence. Build
      and run them on a ROCm system before using them as the basis for production
      code. A complete, buildable program is provided as the
      `HIP-Basic execution context example <https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/execution_context>`_.

The full API listing is in :ref:`execution_context_management_reference`.

Concepts of execution contexts
===============================================================================

An execution context is a host-created scheduling domain for GPU work. The
context does not change the kernel code and it does not add a new kernel launch
parameter. Instead, it changes the resources visible to streams created on that
context.

The usual flow is:

#. Start with the CUs available to the process.
#. Split that CU resource into one or more partitions.
#. Bundle one or more resources into a descriptor.
#. Create one execution context from each descriptor.
#. Create streams from those contexts.
#. Launch kernels on the context streams.

A kernel launched on a context stream can only run on the CUs assigned to that
context. A kernel launched on a regular stream outside an execution context uses
the device's primary context and can use the CUs available to that primary
context.

The kernel's grid and block dimensions still control how much work the kernel
contains. The execution context controls where that work is allowed to execute.
If a kernel has more thread blocks than the context's CUs can run at once, the
extra thread blocks wait and are scheduled onto the same CU partition as earlier
thread blocks retire.

Execution contexts are useful when an application is intentionally designed to
assign different classes of GPU work to different resource partitions. They are
not a per-kernel launch option or an automatic scheduling behavior. To use them
effectively, the application must create the CU partitions, build execution
contexts from those partitions, create streams on those contexts, and launch each
workload on the appropriate stream. Select and tune the partition sizes based on
measured workload behavior.

Benefits of execution contexts
===============================================================================

Two characteristics of standard kernel scheduling motivate the use of execution
contexts.

First, an application cannot directly request a specific number of CUs for a
kernel launch. The number of CUs a kernel uses is determined indirectly by its
grid and block dimensions, along with its per-CU occupancy. There is no launch
parameter that specifies, for example, "run this kernel on N CUs."

Second, kernels that overlap in time draw from the same shared pool of available
CUs. If one kernel already occupies the device and a second, higher-priority
kernel is launched, the second kernel must wait until CUs become available as
the first kernel's thread blocks retire.

Consider a service that continuously runs a background kernel while occasionally
launching a short, latency-sensitive kernel. If the background kernel occupies
the available GPU resources, the latency-sensitive kernel cannot begin execution
until enough of the background kernel's thread blocks complete. Increasing the
priority of the latency-sensitive kernel's stream can reduce this delay, but it
cannot eliminate it, because the kernel must still wait for in-flight thread
blocks to drain.

Execution contexts address this contention by partitioning resources before the
work is launched. For example, the background kernel can be assigned to a context
that uses most of the CUs, while the latency-sensitive kernel can be assigned to
a separate context that uses a smaller, dedicated set of CUs. The background
kernel cannot consume the CUs assigned to the latency-sensitive context, so the
latency-sensitive kernel can start with less interference from the background
workload.

This isolation involves a trade-off: neither kernel can use all available CUs,
so each may run slightly longer when considered in isolation. In exchange, the
latency-sensitive workload is no longer blocked by the background workload's use
of shared CUs.

Use execution contexts when your application design requires predictable access
to a portion of the GPU. Resource partitioning is an explicit setup step: you
decide how many CUs each workload should receive, create contexts from those
partitions, and launch each workload on streams associated with the appropriate
context.

For example, a service can reserve a handful of CUs for a latency-sensitive
kernel so that the kernel has resources available when it launches. Another
application might deliberately cap a kernel to a smaller CU count to measure how
performance scales as the available compute resources change. In both cases, the
partition size is a design choice that should be measured and tuned for the
workload.

Partition sizing is workload dependent. Select the CU count for each execution
context when the context is created, then tune the allocation based on measured
performance.

Work queues
-------------------------------------------------------------------------------

CUs are one type of resource associated with an execution context. Work queue
configuration is another.

A work queue is an abstraction used by the driver to dispatch GPU work. Tasks
submitted to the same work queue may be serialized, even when they are logically
independent, because sharing a queue introduces an ordering dependency between
those tasks.

This distinction is important because reserving CUs alone does not always
guarantee overlap. If the latency-sensitive kernel and the background kernel are
dispatched through the same work queue, the latency-sensitive kernel may still
wait behind the background kernel even though it has dedicated CUs.

Applications do not select work queues directly. Instead, an execution context
allows the application to specify how many concurrent stream-ordered workloads it
expects. The driver treats this value as a hint and attempts to place work from
different contexts onto separate queues. The device-wide upper bound on work
queues is controlled by the ``GPU_MAX_HW_QUEUES`` environment variable.

Work queue configuration is optional. A context can be created with only a CU
resource. If you omit a work queue configuration resource, the context still
limits kernels to its assigned CUs, but the driver uses the default work queue
sharing behavior. Add a work queue configuration when you also want to give the
driver a hint about how many independent stream-ordered workloads you expect and
how work queues should be shared across contexts.

.. note::

    Partitioning CUs and configuring work queue usage reduces sources of
    interference between contexts, but it does not force independent kernels to
    execute concurrently. Actual concurrency still depends on the workloads and
    the current device state.

Relationship to hardware partitioning
-------------------------------------------------------------------------------

AMD Instinct GPUs can be divided into logical devices through hardware compute
partitioning modes such as SPX and CPX. This partitioning is configured at the
system level before an application starts, and is typically used to divide a GPU
among separate applications.

Execution contexts operate at a finer granularity and within a single process.
Hardware partitioning determines how the GPU is shared across applications,
whereas execution contexts determine how one application's streams share the CUs
available to that application.

On a GPU that is already running in a hardware-partitioned mode, an execution
context draws from the CUs in the partition assigned to the application. HIP
execution contexts are not a multi-process service equivalent; they are a
within-process resource-management mechanism.

Device resources and descriptors
===============================================================================

An execution context is built from device resources. A device resource
(``hipDevResource``) names a slice of a specific GPU, and a resource descriptor
(``hipDevResourceDesc_t``) bundles one or more resources together. The execution
context you create from a descriptor can use exactly the resources that
descriptor holds, and nothing else.

The ``hipDevResource`` structure carries a single resource, tagged by type:

.. code-block:: cpp

    typedef struct hipDevResource_st {
        hipDevResourceType type;
        // internal padding
        union {
            hipDevSmResource              sm;
            hipDevWorkqueueConfigResource wqConfig;
            hipDevWorkqueueResource       wq;
        };
        struct hipDevResource_st* nextResource;
    } hipDevResource;

Three resource types are defined:

- ``hipDevResourceTypeSm`` for a set of compute units.
- ``hipDevResourceTypeWorkqueueConfig`` for a work queue configuration.
- ``hipDevResourceTypeWorkqueue`` for an existing work queue resource.

``hipDevResourceTypeInvalid`` marks an unset resource.

Query a device with :cpp:func:`hipDeviceGetDevResource` to obtain its resources: a
CU resource covering every CU available to the process, and, on runtimes that
support it, a work queue configuration covering the device's work queues and the
matching work queue resource. Reading the work queue configuration from a device
is not supported on every ROCm runtime, so check the returned status. You can also
ask an execution context or a stream what resources it holds, using
:cpp:func:`hipExecutionCtxGetDevResource` and :cpp:func:`hipStreamGetDevResource`.
An execution context can hold several resource types at once; a stream only ever
carries a CU resource.

Compute unit resource
-------------------------------------------------------------------------------

The CU resource (``hipDevSmResource``) describes a group of compute units:

.. code-block:: cpp

    typedef struct hipDevSmResource {
        unsigned int smCount;                // number of CUs in this resource
        unsigned int minSmPartitionSize;     // smallest CU count this resource can be split into
        unsigned int smCoscheduledAlignment; // CUs guaranteed to be co-scheduled together
        unsigned int flags;                  // 0 (default) or hipDevSmResourceGroupBackfill
    } hipDevSmResource;

You never fill these fields in yourself. :cpp:func:`hipDeviceGetDevResource` sets
them when you query a device, and the split APIs set them on the resources they
produce. Treat ``minSmPartitionSize`` and ``smCoscheduledAlignment`` as
architecture-dependent values to read at runtime, not constants to hard-code.

The resource returned by :cpp:func:`hipDeviceGetDevResource` is intersected with
any global CU mask set through the ``ROC_GLOBAL_CU_MASK`` environment variable,
so it reflects the CUs your process can actually use.

Workgroup processor alignment
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

On AMD GPUs, CUs are grouped into workgroup processors (WGPs), and cooperating
CUs are scheduled together. ``smCoscheduledAlignment`` reports this granularity,
which is also the minimum partition granularity:

.. list-table::
    :header-rows: 1

    * - Mode
      - ``smCoscheduledAlignment``
    * - WGP mode, typical on RDNA and recent CDNA
      - 2 CUs
    * - CU mode
      - 1 CU

When the alignment is greater than one, request CU counts in multiples of
``smCoscheduledAlignment`` to avoid wasting units. Read
``smCoscheduledAlignment`` at runtime rather than assuming a value, since it
depends on the device and its mode.

.. note::

    Creating an execution context does not automatically prevent other contexts
    or streams created outside an execution context from using the same CUs. To
    isolate workloads, split the available CU resource into disjoint partitions
    and launch each workload only on streams associated with its assigned
    execution context. The :ref:`execution_context_example` demonstrates this
    pattern with separate contexts for a background kernel and a
    latency-sensitive kernel.

Work queue configuration resource
-------------------------------------------------------------------------------

The work queue configuration resource (``hipDevWorkqueueConfigResource``) is one
you populate directly:

.. code-block:: cpp

    typedef struct hipDevWorkqueueConfigResource {
        int                        device;             // device that owns the work queues
        unsigned int               wqConcurrencyLimit; // expected concurrent stream-ordered workloads
        hipDevWorkqueueConfigScope sharingScope;       // how work queues are shared
    } hipDevWorkqueueConfigResource;

``sharingScope`` takes one of two values. ``hipDevWorkqueueConfigScopeDeviceCtx``,
the default, shares work queues across all contexts.
``hipDevWorkqueueConfigScopeGreenCtxBalanced`` asks the driver to keep work
queues from different execution contexts apart where it can, guided by
``wqConcurrencyLimit``.

There is no split API for work queue resources: set the fields yourself, or read
a device's configuration with :cpp:func:`hipDeviceGetDevResource`. The plain work
queue resource (``hipDevResourceTypeWorkqueue``) exposes no fields you can set.

.. tip::

    Zero-initialize every device resource structure before you use it.

Creating an execution context
===============================================================================

Building an execution context takes four steps:

#. Read the resources you want to start from, usually the device's full available
   set.
#. Split the CU resource into the partitions you need.
#. Bundle the resulting resources into a descriptor.
#. Create the execution context from that descriptor.

Once the context exists, create a stream on it. Work you launch on that stream,
including a kernel launched with the triple-chevron syntax, is limited to the
context's resources.

Step 1: Read the available resources
-------------------------------------------------------------------------------

Start by populating a ``hipDevResource`` from a device, an execution context, or
a stream:

.. code-block:: cpp

    hipError_t hipDeviceGetDevResource(hipDevice_t device, hipDevResource* resource,
                                       hipDevResourceType type);
    hipError_t hipExecutionCtxGetDevResource(hipExecutionCtx_t ctx, hipDevResource* resource,
                                             hipDevResourceType type);
    hipError_t hipStreamGetDevResource(hipStream_t hStream, hipDevResource* resource,
                                       hipDevResourceType type);

Each accepts any resource type, except :cpp:func:`hipStreamGetDevResource`, which
is limited to CU resources.

Reading a device's CUs looks like this:

.. literalinclude:: ../../tools/example_codes/execution_context.hip
   :start-after: // [read-cu-resource-start]
   :end-before: // [read-cu-resource-end]
   :language: cpp
   :dedent:

Reading the work queue configuration follows the same pattern, but not every
ROCm runtime supports querying it from a device. Check the returned status rather
than aborting, and only use the configuration when the query succeeds:

.. literalinclude:: ../../tools/example_codes/execution_context.hip
   :start-after: // [read-wq-config-start]
   :end-before: // [read-wq-config-end]
   :language: cpp
   :dedent:

When the query succeeds, ``wqConcurrencyLimit`` reflects ``GPU_MAX_HW_QUEUES`` or
its default for the device.

Step 2: Split the CU resource
-------------------------------------------------------------------------------

Divide the CU resource with one of two APIs. :cpp:func:`hipDevSmResourceSplitByCount`
produces equal-sized partitions; :cpp:func:`hipDevSmResourceSplit` produces
partitions of different sizes in a single call. Either way, CUs that do not fit
the requested partitions land in an optional remainder. Both APIs only operate
on CU resources.

Equal-sized partitions
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: cpp

    hipError_t hipDevSmResourceSplitByCount(hipDevResource* result, unsigned int* nbGroups,
                                            const hipDevResource* input, hipDevResource* remainder,
                                            unsigned int flags, unsigned int minCount);

You pass in the number of groups you want (``*nbGroups``) and the minimum CUs per
group (``minCount``). The call may return fewer groups than you asked for, each
with at least ``minCount`` CUs, because the hardware imposes granularity and
alignment rules. The exact rounding depends on the device's
``minSmPartitionSize`` and ``smCoscheduledAlignment``, so read those at runtime.

.. list-table:: Why the result can differ from the request
    :header-rows: 1

    * - Situation
      - Outcome
    * - You request more groups than fit at ``minCount``
      - The count is reduced to what fits; leftover CUs go to the remainder.
    * - ``minCount`` is not a multiple of the alignment
      - Each group is rounded up to a valid size; fewer CUs remain.

A request for five groups:

.. code-block:: cpp

    hipDevResource avail = {};
    // Populate avail with hipDeviceGetDevResource.

    unsigned int min_cu_count = 8;
    unsigned int group_count  = 5; // may be lowered by the call

    hipDevResource result[5] = {};
    hipDevResource remaining = {};

    HIP_CHECK(hipDevSmResourceSplitByCount(&result[0], &group_count, &avail,
                                           &remaining, 0 /* flags */, min_cu_count));

    std::cout << "Got " << group_count << " groups of " << result[0].sm.smCount
              << " CUs, " << remaining.sm.smCount << " CUs left over\n";

Points to keep in mind:

- Pass ``result = nullptr`` to find out how many groups you would get, without
  producing them.
- Pass ``remainder = nullptr`` to discard the leftover CUs.
- The remainder is not guaranteed to have the same size or scheduling properties
  as the equal-sized groups.
- ``flags`` is ``0`` by default. ``hipDevSmResourceSplitIgnoreSmCoscheduling`` and
  ``hipDevSmResourceSplitMaxPotentialClusterSize`` are also defined.
- To repartition a resulting resource, first turn it into a descriptor and an
  execution context using steps 3 and 4.

.. note::

    ``hipDevSmResourceSplitIgnoreSmCoscheduling`` is defined but not yet
    supported by the runtime. Passing it returns ``hipErrorNotSupported``.

Different-sized partitions
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

When contexts need different CU counts, one
:cpp:func:`hipDevSmResourceSplitByCount` call is not enough, since it only makes
equal groups. :cpp:func:`hipDevSmResourceSplit` builds groups of independent
sizes at once:

.. code-block:: cpp

    hipError_t hipDevSmResourceSplit(hipDevResource* result, unsigned int nbGroups,
                                     const hipDevResource* input, hipDevResource* remainder,
                                     unsigned int flags, hipDevSmResourceGroupParams* groupParams);

Each of the ``nbGroups`` output resources is shaped by a matching
``groupParams`` entry. A remainder is optional. Each produced group contains at
least one valid scheduling granule; a group is never empty.

.. code-block:: cpp

    typedef struct hipDevSmResourceGroupParams_st {
        unsigned int smCount;                     // CU count, or 0 for discovery mode
        unsigned int coscheduledSmCount;          // co-scheduled CU count for clusters
        unsigned int preferredCoscheduledSmCount; // preferred co-scheduled CU count (hint)
        unsigned int flags;                       // 0 or hipDevSmResourceGroupBackfill
    } hipDevSmResourceGroupParams;

Give each group an ``smCount`` that is a multiple of the device's
``smCoscheduledAlignment``. If your kernels use thread block clusters, set
``coscheduledSmCount`` to the largest cluster the group must support, since a
cluster's thread blocks are always co-scheduled. ``preferredCoscheduledSmCount``
is a hint to fold groups into larger ones when possible, and setting ``flags`` to
``hipDevSmResourceGroupBackfill`` lets a group absorb extra CUs beyond its
requested size.

To let the runtime pick a size, use discovery mode: set an entry's ``smCount`` to
zero, and the call fills in a valid count. Entries are processed in order, from
index 0 to ``nbGroups - 1``, so earlier entries claim CUs first.

.. list-table:: hipDevSmResourceSplit arguments
    :header-rows: 1

    * - Argument
      - Meaning
    * - ``result``
      - ``nullptr`` for a dry run, or a valid pointer to receive the groups.
    * - ``nbGroups``
      - How many groups to create.
    * - ``input``
      - The CU resource being split.
    * - ``remainder``
      - ``nullptr`` to drop leftover CUs.
    * - ``flags``
      - ``0``.
    * - ``groupParams[i].smCount``
      - ``0`` for discovery, or a specific CU count.
    * - ``groupParams[i].coscheduledSmCount``
      - ``0`` for the default, or a co-scheduled CU count.
    * - ``groupParams[i].preferredCoscheduledSmCount``
      - ``0`` for the default, or a preferred co-scheduled CU count.
    * - ``groupParams[i].flags``
      - ``0`` or ``hipDevSmResourceGroupBackfill``.

What the return value means depends on ``result``:

- With a valid ``result``, the call succeeds only if every requested group was
  created; otherwise it returns an error.
- With ``result = nullptr``, the call can report success even for a
  configuration that would fail with a real output. Use this to probe what the
  device allows.

When the call succeeds with a valid ``result``, each ``result[i].sm.smCount`` is
aligned to the device's ``smCoscheduledAlignment`` and is within the valid range
for the input resource.

The table below sketches ``groupParams`` for a few common goals, with CU counts
left as placeholders you fill in for your device.

.. list-table:: Example splits
    :header-rows: 1

    * - Goal
      - ``nbGroups``
      - ``remainder``
      - ``smCount``
      - ``coscheduledSmCount``
      - ``flags``
    * - One group of X CUs, with remaining CUs discarded. Clusters allowed.
      - 1
      - ``nullptr``
      - X
      - 0
      - 0
    * - One group of X CUs, with remaining CUs returned as a remainder. No clusters.
      - 1
      - not ``nullptr``
      - X
      - 2
      - 0
    * - Two groups of X and Y CUs with clusters of a chosen size.
      - 2
      - ``nullptr``
      - X, then Y
      - chosen size
      - 0
    * - As many CUs as possible in one group, plus a remainder.
      - 1
      - not ``nullptr``
      - 0, discovery
      - chosen size
      - 0

A two-way uneven split:

.. code-block:: cpp

    hipDevResource cu_resources = {};
    HIP_CHECK(hipDeviceGetDevResource(0, &cu_resources, hipDevResourceTypeSm));

    hipDevResource result[2] = {};
    hipDevSmResourceGroupParams group_params[2] = {
        {/*smCount=*/16, /*coscheduledSmCount=*/0, /*preferredCoscheduledSmCount=*/0, /*flags=*/0},
        {/*smCount=*/8,  /*coscheduledSmCount=*/0, /*preferredCoscheduledSmCount=*/0, /*flags=*/0}};

    HIP_CHECK(hipDevSmResourceSplit(&result[0], 2, &cu_resources,
                                    nullptr /* remainder */, 0 /* flags */, &group_params[0]));

Leaving ``coscheduledSmCount`` or ``preferredCoscheduledSmCount`` at zero
requests the architecture default, which matches the device's
``smCoscheduledAlignment``. To see the value that was chosen, read the
``groupParams`` entry back after a successful call.

For a complete example that performs the full sequence -- query the device
resource, split the CU resource, generate descriptors, create execution
contexts, create streams, and launch kernels -- see the
`HIP-Basic execution context example <https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/execution_context>`_.
The example uses separate contexts for a background kernel and a
latency-sensitive kernel, which is the same setup pattern shown here.

Step 2.1 (optional): Add work queue configuration
-------------------------------------------------------------------------------

This step is optional. A context can be created with only a CU resource. If you
omit a work queue configuration resource, the context still limits kernels to
its assigned CUs, but the driver uses the default work queue sharing behavior.
Add a work queue configuration when you also want to give the driver a hint about
how many independent stream-ordered workloads you expect and how work queues
should be shared across contexts.

To configure work queues alongside CUs, fill in a work queue configuration
resource yourself and place it next to the CU resources you plan to bundle:

.. code-block:: cpp

    hipDevResource resources[2] = {};
    // Populate resources[0] with a split API (one group).

    resources[1].type                        = hipDevResourceTypeWorkqueueConfig;
    resources[1].wqConfig.device             = 0;
    resources[1].wqConfig.sharingScope       = hipDevWorkqueueConfigScopeGreenCtxBalanced;
    resources[1].wqConfig.wqConcurrencyLimit = 4;

A concurrency limit of four tells the driver you expect up to four concurrent
stream-ordered workloads, and it assigns work queues to respect that where it
can.

Step 3: Bundle resources into a descriptor
-------------------------------------------------------------------------------

Gather the resources for the context into a descriptor with
:cpp:func:`hipDevResourceGenerateDesc`:

.. code-block:: cpp

    hipError_t hipDevResourceGenerateDesc(hipDevResourceDesc_t* phDesc,
                                          hipDevResource* resources, unsigned int nbResources);

The resources you bundle must sit next to each other in the array:

.. code-block:: cpp

    hipDevResource result[5] = {};
    // Populate result with a split API.

    hipDevResourceDesc_t desc = {};
    HIP_CHECK(hipDevResourceGenerateDesc(&desc, &result[2], 3)); // bundles result[2], [3], [4]

The call requires that:

- Every resource belongs to the same device.
- CU resources combined together come from the same split call and share the same
  ``coscheduledSmCount``, unless they are remainders.
- Work queue configuration is optional. If included, the descriptor can contain
  either one work queue configuration resource or one work queue resource.

Step 4: Create the context
-------------------------------------------------------------------------------

Turn the descriptor into an execution context with :cpp:func:`hipGreenCtxCreate`.
The context can use only the resources the descriptor holds:

.. code-block:: cpp

    hipError_t hipGreenCtxCreate(hipExecutionCtx_t* ctx, hipDevResourceDesc_t desc,
                                 int device, unsigned int flags);

Pass ``0`` for ``flags``. Initialize the device's primary context first, with
``hipInitDevice`` or :cpp:func:`hipSetDevice`, so that primary context setup does
not add overhead to this call:

.. code-block:: cpp

    int current_device = 0;
    HIP_CHECK(hipSetDevice(current_device));

    hipDevResourceDesc_t desc = {};
    // Generate desc with hipDevResourceGenerateDesc.

    hipExecutionCtx_t exec_ctx = {};
    HIP_CHECK(hipGreenCtxCreate(&exec_ctx, desc, current_device, 0));

To confirm what the context received, call
:cpp:func:`hipExecutionCtxGetDevResource` on it for each resource type.

You can create several contexts by repeating these steps. Usually each context
uses a disjoint set of CUs, which gives the clearest isolation between workloads.
You can also deliberately include the same CU resource in more than one
descriptor, creating an overlapping region. This can improve utilization when
one workload is bursty: for example, a background context can use a small shared
CU group while a latency-sensitive context is idle, while the latency-sensitive
context still has its own private CUs when it becomes active. When both contexts
are active, the shared CUs can become a source of interference, so use
overlapping partitions only when that trade-off is acceptable.

Running work on a context
===============================================================================

To send a kernel to an execution context, create a stream on the context with
:cpp:func:`hipExecutionCtxStreamCreate`. Anything launched on that stream is
bound to the context's resources:

.. code-block:: cpp

    hipError_t hipExecutionCtxStreamCreate(hipStream_t* stream, hipExecutionCtx_t ctx,
                                           unsigned int flags, int priority);

.. code-block:: cpp

    hipStream_t stream;
    int priority = 0;
    HIP_CHECK(hipExecutionCtxStreamCreate(&stream, exec_ctx, hipStreamDefault, priority));

    my_kernel<<<grid_dim, block_dim, 0, stream>>>();
    HIP_CHECK(hipGetLastError());

On an execution context, the default stream flag behaves like
``hipStreamNonBlocking``.

You can query the CU partition backing any stream with
:cpp:func:`hipStreamGetDevResource`. It returns the execution context's CU
partition for a context stream, the explicit mask for a stream created with
:cpp:func:`hipExtStreamCreateWithCUMask`, and the full available device resource
for a stream created outside an execution context. Only
``hipDevResourceTypeSm`` is supported; other types return
``hipErrorInvalidResourceType``.

Graphs
-------------------------------------------------------------------------------

With a :doc:`graph <./hipgraph>`, the stream you launch the graph on does not
decide the resources, unlike a direct launch; that stream only tracks
dependencies. Instead, each node's execution context is fixed when the node is
created. Under stream capture, a node inherits the execution context of the
captured stream. When you build a graph through the graph APIs, set the execution
context on each node explicitly.

Thread block clusters
-------------------------------------------------------------------------------

A kernel that uses thread block clusters runs on an execution context stream like
any other kernel and is bound to the context's CUs. Use the occupancy queries,
:cpp:func:`hipOccupancyMaxPotentialClusterSize` and
:cpp:func:`hipOccupancyMaxActiveClusters`, to size clusters. When you give one
of these a launch configuration whose ``stream`` belongs to an execution
context, it accounts for that context's CUs.

Other context operations
===============================================================================

To synchronize with events across a whole context, use
:cpp:func:`hipExecutionCtxRecordEvent` and :cpp:func:`hipExecutionCtxWaitEvent`.
Recording captures all of the context's outstanding work in one event; waiting
makes later work on the context depend on that event. When a context has several
streams, this is simpler than recording or waiting on each stream separately.

:cpp:func:`hipExecutionCtxSynchronize` blocks the host until the context finishes
its work. Called on the device's primary context, obtained with
:cpp:func:`hipDeviceGetExecutionCtx`, it also waits on every execution context
created on that device.

:cpp:func:`hipExecutionCtxGetDevice` returns the device behind a context, and
:cpp:func:`hipExecutionCtxGetId` returns its unique identifier. Release a context
you created with :cpp:func:`hipExecutionCtxDestroy`.

Destroy a context's streams before the context itself. A stream whose context
has been destroyed is orphaned: operations on it other than
:cpp:func:`hipStreamDestroy` return ``hipErrorContextIsDestroyed``.

Migrating from CUDA green contexts
===============================================================================

HIP execution contexts follow the CUDA green context model, so ports are mostly
mechanical. The handle type differs: HIP uses ``hipExecutionCtx_t`` where CUDA
uses ``CUgreenCtx``. The main function correspondences are:

.. list-table::
    :header-rows: 1

    * - CUDA
      - HIP
    * - ``cuDeviceGetDevResource``
      - :cpp:func:`hipDeviceGetDevResource`
    * - ``cuDevSmResourceSplitByCount``
      - :cpp:func:`hipDevSmResourceSplitByCount`
    * - ``cuDevResourceGenerateDesc``
      - :cpp:func:`hipDevResourceGenerateDesc`
    * - ``cuGreenCtxCreate``
      - :cpp:func:`hipGreenCtxCreate`
    * - ``cuGreenCtxDestroy``
      - :cpp:func:`hipExecutionCtxDestroy`
    * - ``cuGreenCtxStreamCreate``
      - :cpp:func:`hipExecutionCtxStreamCreate`
    * - ``cuGreenCtxRecordEvent``
      - :cpp:func:`hipExecutionCtxRecordEvent`
    * - ``cuGreenCtxWaitEvent``
      - :cpp:func:`hipExecutionCtxWaitEvent`
    * - ``cuStreamGetGreenCtx``
      - :cpp:func:`hipStreamGetDevResource`

The most important behavioral difference is alignment. CUDA aligns partitions to
SM granularity, while HIP aligns partitions to the granularity reported by
``smCoscheduledAlignment``, which is typically 2 CUs in WGP mode. Query this
value and size partitions accordingly instead of porting fixed SM counts
directly.

.. _execution_context_example:

Example: reserving CUs for a critical kernel
===============================================================================

This example reserves CUs for an urgent kernel so it is not blocked by a
long-running one, and measures the difference. A large background kernel runs
concurrently with a small, latency-sensitive critical kernel, and the critical
kernel's runtime is timed in two configurations:

- **Baseline**: both kernels run on streams created outside an execution context
  and share all available CUs, so the critical kernel contends with the
  background kernel.
- **Partitioned**: the CUs are split into two execution contexts, so the
  critical kernel runs on its own CUs while the background kernel is confined to
  the rest.

Without execution contexts, the critical kernel waits for the background
kernel's thread blocks to retire before CUs open up, even with a higher stream
priority. With execution contexts, each kernel uses a distinct set of CUs, so the
critical kernel can start with less interference. Both kernels give up access to
the full available CU set, so each may run slightly longer alone, but the
critical kernel is no longer held back by the background workload.

A busy kernel stands in for a compute-bound workload. Oversubscribing the grid,
launching many more thread blocks than the device runs at once, keeps every CU
occupied for the whole measurement:

.. literalinclude:: ../../tools/example_codes/execution_context.hip
   :start-after: // [busy-kernel-start]
   :end-before: // [busy-kernel-end]
   :language: cpp

A helper launches the background kernel, then times the critical kernel with HIP
events while the background kernel is still running. It returns a ``timings``
struct holding the measured GPU time of each kernel; the critical kernel's time
is the latency of interest. Passing two streams that belong to disjoint execution
contexts confines each kernel to its own CUs, while passing two streams created
outside an execution context lets them contend. The ``workload`` argument carries
the grid sizes and iteration counts that size both kernels:

.. literalinclude:: ../../tools/example_codes/execution_context.hip
   :start-after: // [timing-helper-start]
   :end-before: // [timing-helper-end]
   :language: cpp

The baseline runs both kernels on non-blocking streams created outside an
execution context, so they share the available device resources:

.. literalinclude:: ../../tools/example_codes/execution_context.hip
   :start-after: // [baseline-start]
   :end-before: // [baseline-end]
   :language: cpp

The partitioned run splits the CUs into a group for the background kernel and a
disjoint group for the critical kernel, then follows the four-step setup: read
the device's CUs, split them, bundle each group into a descriptor, and create an
execution context per group. A stream on each context confines its kernel to that
context's CUs. The ``time_partitioned_case`` helper carries out these steps for a
requested critical-CU count and returns the same ``timings`` struct as the
baseline, writing back the aligned CU counts the split actually produced:

.. literalinclude:: ../../tools/example_codes/execution_context.hip
   :start-after: // [partitioned-start]
   :end-before: // [partitioned-end]
   :language: cpp

Comparing the baseline timing with the partitioned timing shows the critical
kernel finishing sooner once it has its own CUs. Settle on the CU split by
measuring your own workload. The
`HIP-Basic execution context example <https://github.com/ROCm/rocm-examples/tree/amd-staging/HIP-Basic/execution_context>`_
contains a complete, buildable version that sweeps several partition sizes, such
as an eighth, a quarter, and half of the device, and also provides a CUDA green
context backend.
