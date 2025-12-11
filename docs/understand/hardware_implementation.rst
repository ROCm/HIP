.. meta::
  :description: This chapter describes the hardware implementation of AMD GPUs supported by HIP.
  :keywords: AMD, ROCm, HIP, hardware, GPU, architecture, compute unit, VALU, SALU, cache, memory hierarchy, CDNA, RDNA, GCN

.. _hardware_implementation:

*******************************************************************************
Hardware implementation
*******************************************************************************

This chapter describes the hardware architecture of AMD GPUs supported by HIP,
focusing on the internal organization and operation of GPU hardware components.
Understanding these hardware details helps you optimize GPU applications and
achieve maximum performance.

Overall GPU architecture
========================

AMD GPUs consist of interconnected blocks of digital circuits that work together
to execute complex parallel computing tasks. The architecture is organized
hierarchically to enable massive parallelism while managing resources efficiently.

Command processor and control
-----------------------------

The command processor (CP) serves as the primary interface between the CPU and
GPU, receiving and distributing commands for execution. The CP consists of two
main components:

* **Command processor fetcher (CPF)**: Fetches commands from memory and passes
  them to the CPC for processing.
* **Command processor packet processor (CPC)**: A microcontroller that decodes the
  fetched commands and dispatches kernels to the workgroup processors for
  scheduling.

The command processor handles several types of operations:

* Kernel launches, which are forwarded to asynchronous compute engines (ACEs)
* Memory transfers, which are delegated to direct memory access (DMA) engines
* Synchronization operations and memory fences

**DMA engines** handle memory transfers between CPU and GPU memory without CPU
involvement after initialization. Most GPUs contain two DMA engines, enabling
concurrent bidirectional transfers to better utilize PCIe bandwidth. The DMA
engines fetch data in small chunks and can process transfers in parallel but
cannot handle multiple copy commands on the same engine simultaneously.

**Asynchronous compute engines (ACEs)** break down kernels into workgroups for
distribution to shader processor input (SPI) blocks. Multiple ACEs enable
concurrent kernel execution, with each ACE capable of dispatching one kernel
at a time. ACEs process commands from different queues asynchronously, enabling
overlap between different kernel executions and memory operations.

Hierarchical organization
-------------------------

The GPU organizes compute resources in a three-level hierarchy that enables
modular design and resource sharing:

1. **Shader engines (SE)**: Top-level organizational units containing multiple
   shader arrays and shared resources
2. **Shader arrays**: Groups of compute units sharing instruction and scalar
   caches
3. **Compute units (CU)**: Basic execution units containing the ALUs and
   registers for thread execution

.. figure:: ../data/understand/hardware_implementation/selayout.png
   :align: center
   :alt: Diagram showing the hierarchical organization of compute units grouped
         into shader engines on AMD GPUs
   :width: 800

   Hierarchical organization of compute units into shader engines

This hierarchical design allows different GPU configurations using the same
underlying architecture. For example, the R9 Fury X contains 16 shader arrays
with four CUs each, while the RX 480 contains 12 shader arrays with three CUs
each, but both use the same gfx803 chip design.

Shader engine components
========================

Shader engines group multiple compute units together with shared resources that
improve efficiency and reduce redundancy. Each shader engine contains several
key components shared across its compute units.

Workgroup manager (SPI)
-----------------------

The workgroup manager, also called the shader processor input (SPI), bridges
the command processor and compute units. After the CP processes a kernel
dispatch, the SPI:

* Receives workgroups from the ACEs
* Schedules workgroups onto available compute units
* Initializes registers with kernel parameters
* Ensures all wavefronts of a workgroup execute on the same CU for
  synchronization
* Monitors resource availability and queues workgroups when resources are
  exhausted

The SPI tracks four critical resources that limit concurrent execution:

* Wavefront slots (execution contexts)
* Vector general-purpose registers (VGPRs)
* Scalar general-purpose registers (SGPRs)
* Local data share (LDS) memory

Workgroup-to-CU mapping is non-deterministic and based on available resources.
You should not assume any specific mapping pattern, as the same kernel launched
multiple times can have different workgroup distributions.

Scalar L1 data cache (sL1D)
---------------------------

The scalar L1 data cache serves scalar memory operations from multiple CUs
within a shader array. The sL1D is shared between CUs (3 CUs in the Graphics
Core Next (GCN) and MI100, 2 CUs in the MI200 series) and caches data that is
uniform across a wavefront, including:

* Kernel arguments and pointers
* Grid and block dimensions
* Constants accessed uniformly across threads
* Data from ``__constant__`` memory when accessed uniformly

Unlike the vector L1 cache, the sL1D doesn't use a "hit-on-miss" approach,
meaning subsequent requests to the same pending cache line count as duplicated
misses rather than hits.

L1 instruction cache (L1I)
--------------------------

The L1 instruction cache is a read-only cache shared between multiple CUs in a
shader array. Like the sL1D, it is backed by the L2 cache and doesn't use the
"hit-on-miss" approach. The L1I stores kernel instructions fetched by the
compute units, reducing instruction fetch latency and L2 cache pressure.

Compute unit architecture
=========================

The compute unit (CU) is the fundamental execution block of AMD GPUs.
It's responsible for executing kernels through its various specialized components
and pipelines.

.. figure:: ../data/understand/hardware_implementation/gcn_compute_unit.png
   :align: center
   :alt: Detailed diagram of an AMD CDNA compute unit showing internal
         components and data flow
   :width: 800

   Internal architecture of an AMD CDNA compute unit

Sequencer and scheduling
------------------------

The instruction sequencer (SQ) serves as the control center of each compute
unit, managing instruction flow through the execution pipelines. The sequencer
maintains wavefront state and coordinates instruction execution across different
functional units.

**Wavefront organization**: The sequencer organizes active wavefronts into four
pools, each containing slots for up to ten wavefronts (eight on the CDNA2 MI200
series). Each slot includes:

* Wavefront-level registers (program counter, execution mask, and others)
* Instruction buffer for prefetched instructions
* State information for scheduling decisions

This organization theoretically allows up to 40 concurrent wavefronts per CU,
though actual occupancy is typically limited by register and LDS usage.

**Instruction fetching**: The fetch arbiter selects one wavefront per cycle to
fetch instructions from memory, prioritizing the oldest wavefronts. Each CU can
fetch up to 32 bytes (4-8 instructions) per cycle.

**Instruction issuing**: The issue arbiter determines which instructions execute
each cycle, selecting wavefronts from one pool per cycle in round-robin fashion.
The arbiter can issue multiple instructions per cycle to different execution
units, with a theoretical maximum of five instructions per cycle:

* One VALU instruction
* One vector memory operation
* One SALU/scalar memory operation
* One LDS operation
* One branch operation

Instructions always issue at wavefront granularity, with all threads in the
wavefront executing the same instruction in lockstep. Context switching between
wavefronts occurs every cycle with zero overhead, as all wavefront contexts
remain resident on the CU.

Execution pipelines
-------------------

Each compute unit contains multiple specialized execution pipelines that process
different types of instructions in parallel, enabling efficient utilization of
the hardware resources.

Vector arithmetic logic unit (VALU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The VALU executes vector instructions across entire wavefronts, with each thread
potentially operating on different data. The VALU consists of:

* **Four SIMD processors**: Each containing 16 single-precision ALUs (or
  equivalent), for 64 total ALUs per CU
* **Vector register files**: 256-512 KiB of VGPR storage split across the four
  SIMDs
* **Instruction buffers**: Storage for up to 8-10 wavefronts per SIMD

On architectures with 64-thread wavefronts and 16-instruction wide SIMD units,
executing one instruction takes four cycles (one cycle per 16 threads). The four
SIMD design ensures full utilization when sufficient wavefronts are available, as
a new instruction can issue to each SIMD every cycle.

The VALU serves as the primary arithmetic engine, executing the majority of 
computation in GPU kernels. Data flows into these pipelines, undergoes arithmetic 
transformation, and exits as results — with the goal of maximizing the number of 
such transformations per clock cycle.

For CDNA architectures with matrix operations, the VALU also dispatches
matrix fused multiply-add (MFMA) instructions to specialized matrix units.

Scalar arithmetic logic unit (SALU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The SALU executes instructions uniformly across all threads in a wavefront,
handling operations like:

* Control flow (branches, loops)
* Address calculations
* Loading kernel arguments and constants
* Managing wavefront-uniform values

The SALU includes:

* A scalar processor for arithmetic and logic operations
* 12.5 KiB of SGPR storage per CU
* A scalar memory (SMEM) unit for memory operations

Scalar operations reduce pressure on vector units and registers by handling
uniform computations efficiently.

Vector memory unit (VMEM)
^^^^^^^^^^^^^^^^^^^^^^^^^

The VMEM unit handles all vector memory operations, including loads, stores,
and atomic operations. Each thread supplies its own address and data, though
the hardware optimizes access through memory coalescing when threads access
nearby addresses. The VMEM unit connects to the vector L1 cache and implements
both address generation and coalescing logic.

Branch unit
^^^^^^^^^^^

The branch unit executes jumps and branches for control flow changes affecting
entire wavefronts. Note that the branch unit handles wavefront-level control
flow, not execution mask updates for thread divergence, which are handled
through predication.

Special function unit (SFU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The special function units accelerate certain arithmetic operations that are too 
complex or costly to implement purely within the standard vector ALUs.

SFUs are responsible for executing transcendental and reciprocal mathematical 
functions — operations such as ``exp``, ``log``, ``sin``, ``cos``, ``rcp`` 
(reciprocal), and ``rsqrt`` (reciprocal square root). These are heavily used in 
scientific, physics, and machine learning workloads, particularly in activation 
functions like GELU, sigmoid, or softmax.

Each compute unit includes a set of specialized pipelines or transcendental 
function units (TFUs) that handle these operations with dedicated hardware. 
While their throughput is lower than that of the primary SIMD pipelines, they 
enable these functions to execute efficiently without consuming general ALU 
bandwidth.

From the compiler's perspective, these operations map to specific AMDGPU ISA 
instructions, such as:

* ``v_exp_f32`` — compute exponential base e
* ``v_log_f32`` — compute natural logarithm  
* ``v_sin_f32``, ``v_cos_f32`` — compute sine or cosine
* ``v_rsq_f32``, ``v_rcp_f32`` — compute reciprocal or reciprocal square root

In CDNA3-based GPUs (like MI300), SFU throughput and latency have been tuned for 
deep learning primitives. For instance, exponentiation (``exp``) and logarithm 
(``log``) functions are now pipelined to complete in a few cycles per lane, 
allowing vectorized activation functions in large-scale matrix workloads to 
execute without significant stalls.

For programmers targeting ROCm or HIP, these SFU-accelerated operations are 
typically accessed through math intrinsics such as ``__expf``, ``__logf``, or 
``__sinf``, which the compiler lowers to the corresponding AMDGPU ISA instructions 
at compile time.

Load/store unit (LSU)
^^^^^^^^^^^^^^^^^^^^^

The load/store units handle the transfer of data between the compute units and 
the GPU's memory subsystems. They are responsible for issuing, tracking, and 
retiring memory operations — including loads from and stores to global memory, 
local shared memory, and caches — for thousands of concurrent threads.

Each compute unit includes a set of LSUs tightly integrated with its vector and 
scalar pipelines. These units handle memory instructions generated by active 
wavefronts — such as ``buffer_load``, ``buffer_store``, and ``flat_load_dword`` 
— and route them through the GPU's hierarchical memory system.

The LSU's responsibilities include:

* Managing vector memory accesses for SIMD instructions
* Coordinating local data share (LDS) reads and writes
* Accessing the L0/L1 caches and forwarding requests to the L2 cache and HBM
* Handling synchronization and atomic operations between threads and workgroups

LSUs manage thousands of outstanding memory requests per GPU, dynamically 
scheduling them to hide memory latency. While arithmetic pipelines continue 
executing other wavefronts, the LSUs maintain queues of pending transactions 
and reorder responses as data returns from memory.

On modern accelerators like MI300X, these LSUs achieve terabytes-per-second of 
aggregate memory bandwidth, coordinating thousands of active threads performing 
memory-intensive operations such as tensor loading, matrix tiling, and gradient 
updates.

Matrix fused multiply-add (MFMA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CDNA architectures (MI100 and newer) include specialized matrix acceleration
units for high-throughput matrix operations. These units execute independently
from other VALU operations, allowing overlap between matrix and vector
computations. MFMA units support various data types including ``INT8``, ``FP16``,
``BF16``, and ``FP32``, with different throughput characteristics for each.

Matrix cores are GPU execution units that perform large-scale matrix operations 
in a single instruction. In AMD architectures, these units are formally known as 
MFMA (matrix fused multiply-add) units — the core hardware blocks responsible for 
accelerating deep learning, HPC, and dense linear-algebra workloads on modern 
Instinct GPUs.

Operating on entire tiles of matrices per instruction allows MFMA units to deliver 
far greater arithmetic throughput and energy efficiency than scalar or vector ALUs. 
Rather than fetching and decoding thousands of per-element multiply-add instructions, 
each MFMA instruction processes an entire matrix fragment — drastically reducing 
power per operation and increasing overall throughput.

An example MFMA instruction from the AMDGPU ISA is:

.. code-block:: none

   v_mfma_f32_16x16x4f16 v[0:15], v[16:31], v[32:47], v[0:15]

This instruction performs a matrix multiplication and accumulation D=A×B+C,
where the fragments A, B, and C are stored in VGPRs. The suffix ``16x16x4f16`` 
indicates a tile size of 16×16, with an inner dimension of 4, operating on 
half-precision (FP16) inputs and accumulating into 32-bit floating-point outputs.

Since their introduction in CDNA1, and further expanded in CDNA2 and CDNA3, AMD's 
matrix cores have become the primary source of peak floating-point performance in 
datacenter GPUs. For example, an MI300X accelerator achieves its multi-petaFLOP 
throughput primarily through MFMA units.

The MFMA units use both standard VGPRs and additional accumulation VGPRs
(AGPRs) on supported architectures, providing up to 512 KiB of combined
register storage per CU.

Local data share (LDS)
----------------------

The local data share provides fast on-CU scratchpad memory for communication
between threads in a workgroup.

.. figure:: ../data/understand/hardware_implementation/lds.svg
   :align: center
   :alt: Diagram showing the organization of local data share with banks and
         connections to SIMD units
   :width: 800

   Local data share organization and SIMD connections

**Organization**: The LDS contains 32 banks, each 4-bytes wide, providing
128 bytes/cycle total bandwidth. Banks can be accessed independently each cycle
for reads, writes, or atomic operations. The SIMDs connect to the LDS in pairs,
with each pair sharing a 64-byte bidirectional port.

**Access patterns**: A single wavefront can achieve up to 64 bytes/cycle
throughput (16 lanes per cycle). The actual bandwidth depends on data size and
access patterns:

* 4-byte values: 8 cycles for 64 threads (50% peak bandwidth)
* 16-byte values: 20 cycles for 64 threads (80% peak bandwidth)

**Conflict resolution**: The LDS includes hardware to detect and resolve bank
conflicts when multiple threads access different addresses in the same bank.
Conflicts are resolved by serializing accesses across multiple cycles. Address
conflicts (multiple threads atomically updating the same address) are similarly
serialized. Broadcasting from the same address to multiple threads is handled
efficiently without conflicts.

Vector L1 cache
---------------

Each CU contains a dedicated vector L1 data cache (vL1D) serving vector memory
operations. Key characteristics include:

* Write-through design (writes go directly to L2)
* Optimization for high-bandwidth streaming access patterns
* Coherent with other CUs through software management
* Typical size of 16 KB per CU

The vector cache tags are checked for all vector memory operations, with misses
forwarded to the L2 cache. The write-through design simplifies coherence at the
cost of write bandwidth.

Memory hierarchy and system
===========================

The GPU memory system provides the bandwidth and capacity needed for massive
parallel computation while managing data coherence and access efficiency.

Memory organization
-------------------

.. figure:: ../data/understand/hardware_implementation/cdna2_gcd.png
  :alt: Block diagram showing four compute engines with L2 cache, memory
        controllers, and Infinity Fabric interconnect on CDNA2
  :width: 800

  CDNA2 Graphics Compute Die organization showing memory subsystem

AMD GPUs typically use high-bandwidth memory (HBM) for data-intensive workloads,
providing significantly higher bandwidth than traditional GDDR memory at the
cost of slightly higher latency. The memory system includes:

* **Memory channels**: Multiple independent memory controllers (typically 8-16)
* **L2 cache banks**: Distributed cache banks serving as the coherence point
* **Infinity Fabric**: High-speed interconnect for data routing

L2 cache architecture
---------------------

The L2 cache serves as the coherence point for all GPU memory accesses and is
shared by all compute units. The L2 consists of multiple independent channels
(32 on CDNA GPUs at 256-byte interleaving) that operate in parallel.

.. figure:: ../data/understand/hardware_implementation/l2perf_model.png
   :align: center
   :alt: Diagram showing L2 cache to Infinity Fabric transaction flow with
         request categorization and routing
   :width: 800

   L2 cache to Infinity Fabric transaction flow

**Key characteristics**:

* **Channel organization**: Each channel handles a portion of the address space,
  with addresses interleaved across channels for load balancing.
* **Hit-on-miss behavior**: If a request arrives for a pending cache line fill,
  it counts as a hit, improving the effective hit rate.
* **Write coalescing**: Multiple writes to the same cache line are combined.
* **Atomic operation support**: Atomics execute directly in the L2 cache for
  coherence.

**L2-Fabric interface**: Requests missing in L2 are routed through Infinity
Fabric to the appropriate memory location, which could be:

* Local HBM on the same GPU
* Remote GPU memory (in multi-GPU systems)
* System memory (CPU DRAM)

The interface categorizes requests by type (read/write), size (32B/64B), and
destination for optimal routing.

Memory coherence
----------------

GPU memory coherence differs significantly from CPU designs to optimize for
throughput over latency:

**Write-through L1 caches**: All writes update both L1 and L2, ensuring L2
always has the latest data. This eliminates the need for complex coherence
protocols between L1 caches but requires higher write bandwidth.

**Software-managed coherence**: Coherence between CUs requires explicit
synchronization through:

* Memory fences for ordering
* Cache invalidation instructions
* Atomic operations (executed at L2 level)
* Kernel boundaries (implicit synchronization)

**Write combining**: To handle partial cache line updates from different CUs,
the GPU uses write masks indicating which bytes to update. This prevents false
sharing issues while maintaining correctness.

Memory coalescing
-----------------

Memory coalescing combines memory accesses from multiple threads into fewer
transactions, significantly improving bandwidth utilization. The coalescing
hardware in the VMEM unit analyzes addresses from all threads in a wavefront
and groups them into the minimum number of cache line requests.

**Coalesced access pattern**: When consecutive threads access consecutive memory
addresses, the hardware can combine all 64 thread requests into as few as 4-8
cache line requests (depending on data size and alignment).

**Non-coalesced access pattern**: When threads access widely separated addresses,
each thread can generate a separate memory transaction, reducing effective
bandwidth by up to 16x or more.

To achieve optimal memory performance:

* Ensure consecutive threads access consecutive memory addresses
* Align data structures to cache line boundaries (64B or 128B)
* Use structure-of-arrays rather than array-of-structures layouts
* Consider padding to avoid bank conflicts

Architecture variants
=====================

AMD supports multiple GPU architecture families optimized for different use
cases while maintaining HIP compatibility.

Graphics Core Next (GCN)
------------------------

GCN represents the foundational architecture for modern AMD GPUs, establishing
key design principles still used today:

* 64-thread wavefronts
* Four SIMD units per CU with 16 lanes each
* Scalar unit for wavefront-uniform operations
* LDS with 32 banks

Multiple GCN generations (GCN1-5) introduced incremental improvements in
process technology, clock speeds, and instruction set features while maintaining
the core architectural philosophy.

.. _cdna_architecture:

CDNA architecture
-----------------

CDNA (Compute DNA) specializes in high-performance computing and machine
learning workloads. Building on GCN principles, CDNA adds significant compute
enhancements:

.. figure:: ../data/understand/hardware_implementation/cdna3_cu.png
  :alt: Block diagram showing CDNA3 compute unit with matrix core unit, shader
        cores, L1 cache, and local data share
  :width: 800

  CDNA3 compute unit with matrix acceleration

**Matrix Core Unit**: Specialized hardware for matrix multiply-accumulate
operations, providing up to 16 times more throughput than vector units for supported
operations. Matrix cores support multiple precisions (``INT8``, ``FP16``, ``BF16``, ``FP32``)
with varying performance characteristics.

**Accumulation VGPRs (AGPRs)**: Additional register file space (up to 256 KB)
dedicated to matrix accumulation, doubling the available register storage for
matrix operations. Data movement between VGPRs and AGPRs uses specialized
instructions (``v_accvgpr_*``).

**Enhanced memory bandwidth**: CDNA GPUs typically use HBM2/HBM2e/HBM3 memory,
providing up to 3.2 TB/s bandwidth on high-end models.

**Multi-die designs**: CDNA2 (MI250) and CDNA3 (MI300) use chiplet
architectures with multiple dies connected through high-speed links, scaling
to higher compute and memory capacities.

.. _rdna_architecture:

RDNA architecture
-----------------

RDNA optimizes for graphics and lower-latency compute workloads through
fundamental architectural changes:

.. figure:: ../data/understand/hardware_implementation/rdna3_cu.png
  :alt: Block diagram showing RDNA3 work group processor with dual compute
        units, shared caches, and 32-wide SIMD units
  :width: 800

  RDNA3 work group processor architecture

**Wave32 execution**: Primary execution mode uses 32-thread wavefronts,
reducing divergence penalties and register pressure. Wave64 mode is available
for backward compatibility.

**Dual compute units**: The work group processor (WGP) replaces standalone CUs,
containing two closely coupled compute units sharing resources:

* Each CU has two 32-wide SIMD units (vs. four 16-wide in GCN)
* Wavefronts execute in a single cycle on 32-wide SIMDs
* Reduced instruction latency improves responsiveness

**Three-level cache hierarchy**:

* **L0 cache**: Per-CU cache (equivalent to GCN's L1)
* **L1 cache**: Shared between CUs in a WGP (new intermediate level)
* **L2 cache**: Global cache shared across all WGPs

**128-byte cache lines**: Doubled from 64 bytes in GCN, aligning with Wave32
access patterns (32 threads × 4 bytes = 128 bytes).

These RDNA optimizations target gaming workloads where latency matters more
than pure throughput, though the architecture remains capable for general
compute tasks.

Performance considerations
==========================

Understanding hardware characteristics helps you optimize GPU applications for
maximum performance.

Occupancy and resource limits
-----------------------------

Occupancy measures the ratio of active wavefronts to maximum possible
wavefronts on a CU. Higher occupancy generally improves latency hiding but
is limited by:

* **Register usage**: Each wavefront requires VGPRs and SGPRs from finite pools
* **LDS allocation**: Shared memory used per workgroup
* **Wavefront slots**: Fixed number of execution contexts per CU
* **Workgroup size**: Smaller workgroups can waste resources

Balancing these resources is critical for achieving optimal occupancy. Tools
like ``rocprof`` can help analyze occupancy and identify limiting factors.

Latency hiding through multithreading
-------------------------------------

GPUs hide memory and instruction latency through massive hardware
multithreading rather than complex CPU techniques like out-of-order execution
or speculation. With sufficient wavefronts:

* Memory latency is hidden by executing other wavefronts during waits
* Pipeline latencies are covered by round-robin wavefront scheduling
* No context switch overhead as all contexts remain resident

The hardware can switch between wavefronts every cycle, maintaining high ALU
utilization even with long-latency operations in flight.

Memory bandwidth utilization
----------------------------

Effective memory bandwidth depends on access patterns:

* **Coalesced access**: Can achieve 70-90% of peak bandwidth
* **Random access**: Might achieve only 5-15% of peak bandwidth
* **Bank conflicts**: Can serialize LDS access, reducing throughput

Memory-bound kernels should focus on:

* Maximizing coalescing through proper data layout
* Prefetching and data reuse in LDS
* Balancing computation with memory access
* Using appropriate cache policies

Hardware-specific optimizations
-------------------------------

Different AMD GPU architectures benefit from tailored optimizations:

**For GCN/CDNA**:

* Optimize for 64-thread wavefront granularity
* Leverage matrix cores for applicable algorithms
* Consider AGPR usage for register spilling

**For RDNA**:

* Design for 32-thread wavefront execution
* Utilize improved divergence handling
* Take advantage of additional cache level

**Architecture-agnostic**:

* Minimize divergent control flow
* Ensure memory access coalescing
* Balance resource usage for occupancy
* Overlap computation with memory access

Summary
=======

AMD GPU hardware architecture provides massive parallelism through hierarchical
organization of compute resources, specialized execution units, and a
sophisticated memory system. Understanding these hardware details—from the
command processor through shader engines to individual compute units and the
memory hierarchy—enables you to write more efficient GPU applications.

Key hardware concepts for optimization include:

* Workgroup scheduling and resource management by the SPI
* Instruction scheduling and wavefront execution in compute units
* Memory coalescing and cache behavior
* Architecture-specific features (matrix cores, Wave32/64 modes)
* Resource limits affecting occupancy

For details on mapping parallel algorithms to this hardware, see the
:ref:`programming_model` chapter. For specific optimization techniques, consult
the performance optimization guides in the ROCm documentation.
