# Glossary of terms

## Host and device

* **host**, **host CPU**: Executes the HIP runtime API and is capable of initiating kernel launches to one or more devices.
* **device**: A GPU or accelerator that executes HIP kernels. In the context of HIP, this typically refers to AMD GPUs or NVIDIA GPUs when using HIP on CUDA.
* **default device**: Each host thread maintains a default device. Most HIP runtime APIs (including memory allocation, copy commands, kernel launches) do not accept an explicit device argument but instead implicitly use the default device. The default device can be set with `hipSetDevice`.
* **active host thread**: The thread which is running the HIP APIs.

## Architecture and hardware

* **AMD device architecture**: The organization and execution model of AMD GPUs, defining how computation happens inside the hardware through programmable compute units organized into shader engines.
* **compute unit (CU)**: The fundamental programmable engine of AMD GPUs. Each CU manages thousands of lightweight threads, orchestrating their execution, memory access, and synchronization. A CU consists of SIMD units, scalar units, register files, LDS, and caches.
* **GPU cores**: The primary arithmetic engines within compute units, including SIMD lanes and matrix fused multiply-add (MFMA) units that execute mathematical and logical operations.
* **SIMD (Single Instruction, Multiple Data)**: Hardware units that execute the same instruction across multiple data elements simultaneously. AMD GPUs typically have four SIMD units per CU.
* **VALU (Vector Arithmetic Logic Unit)**: Executes vector instructions across entire wavefronts, with each thread potentially operating on different data.
* **SALU (Scalar Arithmetic Logic Unit)**: Executes instructions uniformly across all threads in a wavefront, handling control flow and wavefront-uniform operations.
* **SFU (Special Function Unit)**: Accelerates transcendental and reciprocal mathematical functions like exp, log, sin, cos, rcp, and rsqrt.
* **LSU (Load/Store Unit)**: Handles data transfers between compute units and GPU memory subsystems, managing thousands of outstanding memory requests.
* **MFMA (Matrix Fused Multiply-Add)**: Specialized hardware units in CDNA architectures that perform large-scale matrix operations in a single instruction, providing the primary source of peak floating-point performance.
* **DME (Data Movement Engine)**: Specialized hardware units in CDNA3/4 that accelerate access to multi-dimensional tensor data, performing high-throughput copies between global memory and on-chip memory.
* **shader engine**: Top-level organizational unit in AMD GPUs containing multiple shader arrays and shared resources.
* **GFX IP**: Graphics IP version identifier (like gfx908, gfx90a, gfx942) that specifies the precise features, register layout, and machine instruction format a GPU supports.
* **CDNA**: Compute DNA architecture specialized for high-performance computing and machine learning workloads, featuring matrix cores and enhanced memory bandwidth.
* **RDNA**: Radeon DNA architecture optimized for graphics and lower-latency compute workloads, featuring Wave32 execution and work group processors.
* **GCN**: Graphics Core Next, the foundational architecture for modern AMD GPUs that established key design principles still used today.

## Memory hierarchy

* **VGPR (Vector General-Purpose Registers)**: Per-thread registers that hold data processed by SIMD lanes, such as individual elements of matrices or vectors.
* **SGPR (Scalar General-Purpose Registers)**: Registers holding values shared across an entire wavefront, such as loop counters, constants, or addresses.
* **AGPR (Accumulation VGPRs)**: Additional register file space in CDNA architectures dedicated to matrix accumulation, doubling available register storage for matrix operations.
* **register file**: Primary on-chip memory that holds data between arithmetic and memory operations, built from extremely fast SRAM.
* **LDS (Local Data Share)**: Fast on-chip scratchpad memory shared among threads in a workgroup, providing low-latency communication within a block.
* **L1 data cache**: Private on-chip memory associated with each compute unit, providing fast access to recently used data.
* **global memory**: General read-write accessible memory visible to all threads on a device, backed by high-bandwidth memory (HBM).
* **constant memory**: Read-only storage visible to all threads, optimized for uniform access patterns across a wavefront.
* **texture memory**: Special read-only memory optimized for spatial locality and 2D/3D access patterns with filtering capabilities.
* **surface memory**: Read-write version of texture memory.
* **HBM (High Bandwidth Memory)**: Vertically stacked DRAM technology providing terabytes per second of memory bandwidth in modern GPUs.
* **memory coalescing**: Hardware optimization that combines memory accesses from multiple threads into fewer transactions when accessing consecutive addresses.
* **bank conflict**: Performance penalty when multiple threads access different addresses in the same LDS bank, causing serialization.

## Execution model

* **work-item**: A single instance of a kernel, representing one thread of parallel execution.
* **wavefront**: A group of threads (typically 32 or 64) that execute the same instruction simultaneously in lockstep on a SIMD unit.
* **work-group**: A collection of wavefronts that can synchronize and share LDS memory, executing on the same compute unit.
* **grid**: The total collection of work-groups launched for a kernel, defining the problem size.
* **HIP kernel**: A function marked with `__global__` that executes on the GPU device in parallel across many threads.
* **HIP thread hierarchy**: The three-level organization of threads (thread, block, grid) that defines parallel execution scope.
* **warp size**: The number of threads in a wavefront - typically 64 for AMD CDNA/GCN architectures and 32 for RDNA architectures.
* **occupancy**: The ratio of active wavefronts to maximum possible wavefronts on a compute unit, affecting latency hiding ability.
* **wavefront scheduler**: Hardware component that decides which wavefront to execute each clock cycle, enabling rapid context switching.
* **wavefront divergence**: Performance penalty when threads within a wavefront take different execution paths due to conditional statements.
* **branch efficiency**: The ratio of non-divergent to total branches, indicating control flow uniformity.

## Performance concepts

* **performance bottleneck**: The limiting factor preventing higher kernel performance, typically either compute-bound or memory-bound.
* **roofline model**: Visual performance analysis framework relating achievable performance to hardware limits based on arithmetic intensity.
* **compute-bound**: Kernel performance limited by arithmetic throughput rather than memory bandwidth.
* **memory-bound**: Kernel performance limited by memory bandwidth rather than compute capacity.
* **arithmetic intensity**: Ratio of floating-point operations to memory traffic (FLOPs/byte), determining compute vs memory boundedness.
* **overhead**: Fixed costs associated with operations like kernel launches, memory allocations, or synchronization.
* **latency hiding**: GPU technique of switching between wavefronts to mask memory and instruction latency through massive multithreading.
* **memory bandwidth**: Rate of data transfer between GPU compute units and memory, measured in GB/s or TB/s.
* **arithmetic bandwidth**: Rate of arithmetic operations, measured in FLOPS (floating-point operations per second).
* **active cycle**: Percentage of cycles where at least one instruction is executing on a compute unit.
* **pipe utilization**: Percentage of execution cycles where the pipeline actively processes instructions.
* **peak rate**: Theoretical maximum performance of a GPU in FLOPS or bandwidth.
* **issue efficiency**: Ratio of issued instructions to maximum possible, indicating scheduling effectiveness.
* **CU utilization**: Percentage of compute units actively executing work.
* **register pressure**: Condition where high register usage limits occupancy and performance.
* **Little's Law**: Performance principle relating concurrency, latency, and throughput: Concurrency = Latency × Throughput.

## Software and tools

* **ROCm programming model**: The software model for programming AMD GPUs, including the HIP API and associated runtime.
* **ROCm software platform**: Complete software stack for AMD GPU computing, including drivers, runtime, compilers, libraries, and tools.
* **HIP C++**: C++ dialect for writing portable GPU code that can run on both AMD and NVIDIA platforms.
* **HIP runtime API**: Set of functions for managing GPU devices, memory, streams, and kernel execution.
* **HIP compiler driver**: Tool that compiles HIP source code into device binaries and host code.
* **HIP runtime compiler (hipRTC)**: API for compiling HIP kernels at runtime from source strings.
* **HIP-Clang**: Heterogeneous AMDGPU Compiler with capability to compile HIP programs on AMD platform.
* **clr**: Repository for AMD Compute Language Runtime, containing source codes for HIP and OpenCL runtimes.
  * `hipamd`: Implementation of HIP language on AMD platform
  * `rocclr`: Common runtime providing virtual device interfaces for different backends
  * `opencl`: Implementation of OpenCL on AMD platform
* **hipify tools**: Tools to convert CUDA code to portable C++ code.
* **hipconfig**: Tool to report various configuration properties of the target platform.
* **rocm-smi**: System management interface for monitoring and configuring AMD GPUs.
* **AMD uProf**: System-wide performance analysis tool for CPU and GPU profiling.
* **ROCm profiler**: Performance profiling tools for analyzing GPU kernel execution.
* **ROCm binary utilities**: Tools for inspecting and manipulating GPU binary code objects.
* **rocBLAS**: Optimized BLAS (Basic Linear Algebra Subprograms) library for AMD GPUs.
* **MIOpen**: Machine learning primitives library optimized for AMD GPUs.
* **AMDGPU assembly**: Low-level assembly language for AMD GPUs.
* **AMDGPU IR**: Intermediate representation used in the AMD GPU compilation pipeline.
* **nvcc**: NVIDIA CUDA compiler (do not capitalize).

## Execution states and synchronization

* **wavefront execution state**: Current status of a wavefront - active, ready, stalled, or sleeping.
* **active wavefront**: Wavefront currently executing on a SIMD unit.
* **stalled wavefront**: Wavefront waiting for a dependency like memory or synchronization.
* **stream**: FIFO queue of GPU commands that execute in order, enabling asynchronous execution.
* **event**: Synchronization primitive for coordinating execution between streams or with the host.
* **barrier**: Synchronization point where threads wait for others to reach before proceeding.
* **memory fence**: Operation ensuring memory consistency and ordering across threads.
