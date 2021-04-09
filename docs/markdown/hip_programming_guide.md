# HIP Programming Guide

## Host Memory

### Introduction
hipHostMalloc allocates pinned host memory which is mapped into the address space of all GPUs in the system.
There are two use cases for this host memory:
- Faster HostToDevice and DeviceToHost Data Transfers:
The runtime tracks the hipHostMalloc allocations and can avoid some of the setup required for regular unpinned memory.  For exact measurements on a specific system, experiment with --unpinned and --pinned switches for the hipBusBandwidth tool.
- Zero-Copy GPU Access:
GPU can directly access the host memory over the CPU/GPU interconnect, without need to copy the data.  This avoids the need for the copy, but during the kernel access each memory access must traverse the interconnect, which can be tens of times slower than accessing the GPU's local device memory.  Zero-copy memory can be a good choice when the memory accesses are infrequent (perhaps only once).  Zero-copy memory is typically "Coherent" and thus not cached by the GPU but this can be overridden if desired and is explained in more detail below.

### Memory allocation flags
hipHostMalloc always sets the hipHostMallocPortable and hipHostMallocMapped flags.  Both usage models described above use the same allocation flags, and the difference is in how the surrounding code uses the host memory.
See the hipHostMalloc API for more information.


### Coherency Controls
ROCm defines two coherency options for host memory:
- Coherent memory : Supports fine-grain synchronization while the kernel is running.  For example, a kernel can perform atomic operations that are visible to the host CPU or to other (peer) GPUs.  Synchronization instructions include threadfence_system and C++11-style atomic operations.    However, coherent memory cannot be cached by the GPU and thus may have lower performance.
- Non-coherent memory : Can be cached by GPU, but cannot support synchronization while the kernel is running.  Non-coherent memory can be optionally synchronized only at command (end-of-kernel or copy command) boundaries.  This memory is appropriate for high-performance access when fine-grain synchronization is not required.

HIP provides the developer with controls to select which type of memory is used via allocation flags passed to hipHostMalloc and the HIP_HOST_COHERENT environment variable. By default, the environment variable HIP_HOST_COHERENT is set to 0 in HIP.
- hipHostMallocCoherent=0, hipHostMallocNonCoherent=0: Use HIP_HOST_COHERENT environment variable,
  - If HIP_HOST_COHERENT is defined as 1, the host memory allocation is coherent.
  - If HIP_HOST_COHERENT is not defined, or defined as 0, the host memory allocation is non-coherent.
- hipHostMallocCoherent=1, hipHostMallocNonCoherent=0: The host memory allocation will be coherent.  HIP_HOST_COHERENT env variable is ignored.
- hipHostMallocCoherent=0, hipHostMallocNonCoherent=1: The host memory allocation will be non-coherent.  HIP_HOST_COHERENT env variable is ignored.
- hipHostMallocCoherent=1, hipHostMallocNonCoherent=1: Illegal.


### Visibility of Zero-Copy Host Memory
Coherent host memory is automatically visible at synchronization points.
Non-coherent

| HIP API              | Synchronization Effect                                                         | Fence                | Coherent Host Memory Visibiity | Non-Coherent Host Memory Visibility|
| ---                  | ---                                                                            | ---                  | ---                            | --- |
| hipStreamSynchronize | host waits for all commands in the specified stream to complete                | system-scope release | yes                        | yes   |
| hipDeviceSynchronize | host waits for all commands in all streams on the specified device to complete | system-scope release | yes                        | yes   |
| hipEventSynchronize  | host waits for the specified event to complete                                 | device-scope release | yes                        | depends - see below|
| hipStreamWaitEvent   | stream waits for the specified event to complete                               | none                 | yes                        | no   |


### hipEventSynchronize
Developers can control the release scope for hipEvents:
- By default, the GPU performs a device-scope acquire and release operation with each recorded event.  This will make host and device memory visible to other commands executing on the same device.

A stronger system-level fence can be specified when the event is created with hipEventCreateWithFlags:
- hipEventReleaseToSystem : Perform a system-scope release operation when the event is recorded.  This will make both Coherent and Non-Coherent host memory visible to other agents in the system, but may involve heavyweight operations such as cache flushing.  Coherent memory will typically use lighter-weight in-kernel synchronization mechanisms such as an atomic operation and thus does not need to use hipEventReleaseToSystem.
- hipEventDisableTiming: Events created with this flag would not record profiling data and provide best performance if used for synchronization.

Note, for HIP Events used in kernel dispatch using hipExtLaunchKernelGGL/hipExtLaunchKernel, events passed in the API are not explicitly recorded and should only be used to get elapsed time for that specific launch.
In case events are used across multiple dispatches, for example, start and stop events from different hipExtLaunchKernelGGL/hipExtLaunchKernel calls, they will be treated as invalid unrecorded events, HIP will throw error "hipErrorInvalidHandle" from hipEventElapsedTime.

### Summary and Recommendations:

- Coherent host memory is the default and is the easiest to use since the memory is visible to the CPU at typical synchronization points.  This memory allows in-kernel synchronization commands such as threadfence_system to work transparently.
- HIP/ROCm also supports the ability to cache host memory in the GPU using the "Non-Coherent" host memory allocations. This can provide performance benefit, but care must be taken to use the correct synchronization.

## Device-Side Malloc

HIP-Clang currenntly doesn't supports device-side malloc and free.

## Use of Long Double Type

In HIP-Clang, long double type is 80-bit extended precision format for x86_64, which is not supported by AMDGPU.  HIP-Clang treats long double type as IEEE double type for AMDGPU. Using long double type in HIP source code will not cause issue as long as data of long double type is not transferred between host and device. However, long double type should not be used as kernel argument type.

## Use of _Float16 Type

If a host function is to be used between clang (or hipcc) and gcc for x86_64, i.e. its definition is compiled by one compiler but the caller is compiled by a different compiler, _Float16 or aggregates containing _Float16 should not be used as function argument or return type. This is due to lack of stable ABI for _Float16 on x86_64. Passing _Float16 or aggregates containing _Float16 between clang and gcc could cause undefined behavior.

## FMA and contractions

By default HIP-Clang assumes -ffp-contract=fast-honor-pragmas.
Users can use '#pragma clang fp contract(on|off|fast)' to control fp contraction of a block of code.
For x86_64, FMA is off by default since the generic x86_64 target does not
support FMA by default. To turn on FMA on x86_64, either use -mfma or -march=native
on CPU's supporting FMA.

When contractions are enabled and the CPU has not enabled FMA instructions, the
GPU can produce different numerical results than the CPU for expressions that
can be contracted. Tolerance should be used for floating point comparsions.

## Math functions with special rounding modes

HIP does not support math functions with rounding modes ru (round up), rd (round down), and rz (round towards zero). HIP only supports math function with rounding mode rn (round to nearest). The math functions with postfixes _ru, _rd and _rz are implemented in the same way as math functions with postfix _rn. They serve as a workaround to get programs using them compiled.

## [Supported Clang Options](clang_options.md)
