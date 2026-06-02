
.. meta::
   :description: HIP Device Properties.
   :keywords: AMD, ROCm, HIP, device attributes

.. _hip_device_properties:

HIP Device Properties and Topology on CDNA Architectures
==========================================================

Understanding GPU device properties is essential for writing performant and
scalable HIP applications. Modern AMD GPUs such as MI300 (CDNA3) expose a
a hierarchical, chiplet-based topology with multiple layers of compute, cache, and memory resources.

Key architectural components include:

- XCCs (Accelerated Compute Cores)
- XCDs (Compute Dies)
- Partitioned L2 cache regions
- High-bandwidth memory (HBM) channels and subsystems.
- NUMA-like HBM memory domains

This document combines conceptual explanations with HIP and HSA examples to
help developers understand and leverage these hardware characteristics.

------------------------------------------------------------
Basic HIP Device Properties
------------------------------------------------------------

Use HIP runtime APIs to query fundamental device attributes:

.. code-block:: cpp

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);

    printf("Device: %s\n", props.name);
    printf("Global Memory: %zu\n", props.totalGlobalMem);
    printf("Compute Units: %d\n", props.multiProcessorCount);
    printf("L2 Cache Size: %zu\n", props.l2CacheSize);

.. note::

   HIP exposes aggregate properties at the device level. Detailed topology (XCC/XCD/NUMA) is not directly exposed. Additional insight requires HSA APIs or profiling tools.

------------------------------------------------------------
CDNA Topology Overview
------------------------------------------------------------

CDNA GPUs employ a chiplet-based architecture:

- Multiple XCDs (compute dies) per package
- Each XCD contains one or more XCCs
- L2 cache is distributed and partitioned across XCD/XCC units
- Multiple HBM stacks form NUMA-like memory domains

These characteristics influence memory locality, scheduling behavior, and
overall performance.

------------------------------------------------------------
XCC (Accelerated Compute Core)
------------------------------------------------------------

An XCC is the fundamental execution unit in CDNA architectures.

Each XCC includes:

- Compute Units (CUs)
- Command Processor (CP)
- Run List Controller (RLC)
- Cache subsystems (TCC)

Related architectural elements:

- **XCD**: Physical die containing one or more XCCs
- **XCP**: Logical compute partition consisting of one or more XCCs
- **AID**: Interconnect die linking XCDs and enabling high-bandwidth communication

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
XCC Partitioning Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

XCC partitioning is configured at system initialization:

- **SPX (Single Partition)**: All XCCs form a single GPU
- **TPX (Tile Partition)**: XCDs grouped into multiple logical GPUs
- **CPX (Core Partition)**: Each XCC exposed as an independent GPU

Example:

- In CPX mode on MI300X: 8 XCCs ⇒ 8 HIP devices

Partitioning directly affects how applications perceive and utilize hardware
resources.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
XCC in HIP Runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Device visibility:

- Each compute partition (XCP) appears as an independent HIP device

Driver-level register access example:

.. code-block:: cpp

    // Access a specific XCC
    RegUtility* xcc0 = proc->Get(REG_ACCESS_GC_0);

    // Broadcast to all XCCs in the partition
    RegUtility* all = proc->Get(REG_ACCESS_GC_BROADCAST);

Key runtime identifiers:

- ``Virtual_XCC_ID``: identifies an XCC instance
- ``NUM_XCC_IN_XCP``: number of XCCs in a partition

------------------------------------------------------------
Compute Dies (XCDs)
------------------------------------------------------------

XCDs aggregate XCCs into higher-level compute domains.

Approximate XCC/XCD count using HSA:

.. code-block:: cpp

    #include <hsa/hsa.h>

    int gpu_agent_count = 0;

    hsa_status_t callback(hsa_agent_t agent, void* data)
    {
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);

        if (type == HSA_DEVICE_TYPE_GPU)
            (*(int*)data)++;

        return HSA_STATUS_SUCCESS;
    }

    hsa_init();
    hsa_iterate_agents(callback, &gpu_agent_count);

    printf("Approx compute partitions: %d\n", gpu_agent_count);

Take MI300X as an example, the out put will be:

.. code-block:: cpp

    Approx compute partitions: 8

Interpretation:

- CPX mode: one agent ≈ one XCC
- SPX mode: one agent ≈ full device

------------------------------------------------------------
L2 Cache Regions
------------------------------------------------------------

HIP provides only total L2 cache size:

.. code-block:: cpp

    size_t total_l2 = props.l2CacheSize;

Notes:

- L2 cache is partitioned across XCC/XCD domains
- Per-region size must be inferred
- Cross-domain accesses may increase latency

------------------------------------------------------------
NUMA Memory Domains
------------------------------------------------------------

HBM memory behaves similarly to NUMA systems.

Common configurations:

- **NPS1**: Fully interleaved memory (maximum bandwidth)
- **NPS4 / NPS8**: Localized memory (improved latency locality)

Implications:

- Memory placement affects performance
- Remote memory accesses incur higher latency

------------------------------------------------------------
Memory Channels
------------------------------------------------------------

The number of memory channels is not directly exposed.

Estimate using memory bus width:

.. code-block:: cpp

    int channel_width_bits = 128;  // architectural assumption
    int channels = props.memoryBusWidth / channel_width_bits;

    printf("Estimated memory channels: %d\n", channels);

------------------------------------------------------------
Atomic Throughput
------------------------------------------------------------

Atomic throughput limits must be measured empirically.

.. code-block:: cpp

    __global__ void atomicKernel(int* data)
    {
        int idx = threadIdx.x;
        atomicAdd(&data[idx % 1024], 1);
    }

Guidelines:

- Avoid contention on a single memory address
- Distribute atomics across multiple locations
- Use hierarchical reductions to reduce pressure on global memory

------------------------------------------------------------
Multi-XCC Profiling
------------------------------------------------------------

In multi-XCC configurations:

- Each XCC may execute workloads at different times
- Profiling tools often report only master XCC timing


------------------------------------------------------------
Optimization Strategy
------------------------------------------------------------

To optimize performance on CDNA GPUs such as MI300:

- Partition workloads across XCCs/XCDs where possible
- Apply NUMA-aware memory allocation strategies
- Align data structures to cache line boundaries
- Minimize cross-XCC communication and synchronization
- Balance memory traffic across channels
- Benchmark atomic scalability under load

------------------------------------------------------------
Summary
------------------------------------------------------------

Topology-aware programming using XCC, XCD, memory hierarchy, and HIP APIs is
critical for achieving peak performance on CDNA GPUs such as MI300.

To fully exploit the architecture, developers should combine:

- HIP runtime APIs
- HSA topology queries
- Empirical benchmarking

to understand and optimize hardware utilization.
