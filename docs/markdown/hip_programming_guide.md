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
hipHostMalloc always sets the hipHostMallocPortable and hipHostMallocMapped flags. Both usage models described above use the same allocation flags, and the difference is in how the surrounding code uses the host memory.

hipHostMallocNumaUser is the flag to allow host memory allocation to follow numa policy set by user.

See the hipHostMalloc API for more information.

### Numa-aware host memory allocation
Numa policy determines how memory is allocated.
Target of Numa policy is to select a CPU that is closest to each GPU.
Numa distance is the measurement of how far between GPU and CPU devices.

By default, each GPU selects a Numa CPU node that has the least Numa distance between them, that is, host memory will be automatically allocated closest on the memory pool of Numa node of the current GPU device. Using hipSetDevice API to a different GPU will still be able to access the host allocation, but can have longer Numa distance.

### Managed memory allocation
Managed memory, except the `__managed__` keyword, are supported in HIP combined host/device compilation.
The allocation will be automatically managed by AMD HMM (Heterogeneous Memory Management).

In HIP application, there should be the capability check before make managed memory API call hipMallocManaged.

For example,
```
int managed_memory = 0;
HIPCHECK(hipDeviceGetAttribute(&managed_memory,
  hipDeviceAttributeManagedMemory,p_gpuDevice));

if (!managed_memory ) {
  printf ("info: managed memory access not supported on the device %d\n Skipped\n", p_gpuDevice);
}
else {
  HIPCHECK(hipSetDevice(p_gpuDevice));
  HIPCHECK(hipMallocManaged(&Hmm, N * sizeof(T)));
. . .
}
```
Please note, the managed memory capability check may not be necessary, but if HMM is not supported, then managed malloc will fall back to using system memory and other managed memory API calls will have undefined behavior.
For more details on managed memory APIs, please refer to the documentation HIP-API.pdf.

### HIP Stream Memory Operations

HIP supports Stream Memory Operations to enable direct synchronization between Network Nodes and GPU. Following new APIs are added,
  hipStreamWaitValue32
  hipStreamWaitValue64
  hipStreamWriteValue32
  hipStreamWriteValue64

Note, CPU access to the semaphore's memory requires volatile keyword to disable CPU compiler's optimizations on memory access.

For more details, please check the documentation HIP-API.pdf.

### Coherency Controls
ROCm defines two coherency options for host memory:
- Coherent memory : Supports fine-grain synchronization while the kernel is running.  For example, a kernel can perform atomic operations that are visible to the host CPU or to other (peer) GPUs.  Synchronization instructions include threadfence_system and C++11-style atomic operations. However, coherent memory cannot be cached by the GPU and thus may have lower performance.
- Non-coherent memory : Can be cached by GPU, but cannot support synchronization while the kernel is running.  Non-coherent memory can be optionally synchronized only at command (end-of-kernel or copy command) boundaries.  This memory is appropriate for high-performance access when fine-grain synchronization is not required.

HIP provides the developer with controls to select which type of memory is used via allocation flags passed to hipHostMalloc and the HIP_HOST_COHERENT environment variable. By default, the environment variable HIP_HOST_COHERENT is set to 0 in HIP.
The control logic in the current version of HIP is as follows:
- No flags are passed in: the host memory allocation is coherent, the HIP_HOST_COHERENT environment variable is ignored.
- hipHostMallocCoherent=1: The host memory allocation will be coherent, the HIP_HOST_COHERENT environment variable is ignored.
- hipHostMallocMapped=1: The host memory allocation will be coherent, the HIP_HOST_COHERENT environment variable is ignored.
- hipHostMallocNonCoherent=1, hipHostMallocCoherent=0, and hipHostMallocMapped=0: The host memory will be non-coherent, the HIP_HOST_COHERENT environment variable is ignored.
- hipHostMallocCoherent=0, hipHostMallocNonCoherent=0, hipHostMallocMapped=0, but one of the other HostMalloc flags is set:
  - If HIP_HOST_COHERENT is defined as 1, the host memory allocation is coherent.
  - If HIP_HOST_COHERENT is not defined, or defined as 0, the host memory allocation is non-coherent.
- hipHostMallocCoherent=1, hipHostMallocNonCoherent=1: Illegal.

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

## Direct Dispatch
HIP runtime has Direct Dispatch enabled by default in ROCM 4.4. With this feature we move away from our conventional producer-consumer model where the runtime creates a worker thread(consumer) for each HIP Stream, where as the host thread(producer) enqueues commands to a command queue(per stream).

For Direct Dispatch, the runtime would directly queue a packet to the AQL queue (user mode queue to GPU) in case of Dispatch and some of the synchronization. This has shown to the total latency of the HIP Dispatch API and latency to launch first wave on the GPU.

In addition, eliminating the threads in runtime has reduced the variance in the dispatch numbers as the thread scheduling delays and atomics/locks synchronization latencies are reduced.

This feature can be disabled via setting the following environment variable,
AMD_DIRECT_DISPATCH=0

## HIP Runtime Compilation
HIP now supports runtime compilation (hipRTC), the usage of which will provide the possibility of optimizations and performance improvement compared with other APIs via regular offline static compilation.

hipRTC APIs accept HIP source files in character string format as input parameters and create handles of programs by compiling the HIP source files without spawning separate processes.

For more details on hipRTC APIs, refer to HIP-API.pdf in GitHub (https://github.com/RadeonOpenCompute/ROCm).

The link here(https://github.com/ROCm-Developer-Tools/HIP/blob/main/tests/src/hiprtc/saxpy.cpp) shows an example how to program HIP application using runtime compilation mechanism, and detail hipRTC programming guide is also available in Github (https://github.com/ROCm-Developer-Tools/HIP/blob/main/docs/markdown/hip_rtc.md).


## Device-Side Malloc

HIP-Clang currently doesn't supports device-side malloc and free.

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

## Creating Static Libraries

HIP-Clang supports generating two types of static libraries. The first type of static library does not export device functions, and only exports and launches host functions within the same library. The advantage of this type is the ability to link with a non-hipcc compiler such as gcc. The second type exports device functions to be linked by other code objects. However this requires using hipcc as the linker.

In addition, the first type of library contains host objects with device code embedded as fat binaries. It is generated using the flag --emit-static-lib. The second type of library contains relocatable device objects and is generated using ar.

Here is an example to create and use static libraries:
- Type 1 using --emit-static-lib:
    ```
    hipcc hipOptLibrary.cpp --emit-static-lib -fPIC -o libHipOptLibrary.a
    gcc test.cpp -L. -lhipOptLibrary -L/path/to/hip/lib -lamdhip64 -o test.out
    ```
- Type 2 using system ar:
    ```
    hipcc hipDevice.cpp -c -fgpu-rdc -o hipDevice.o
    ar rcsD libHipDevice.a hipDevice.o
    hipcc libHipDevice.a test.cpp -fgpu-rdc -o test.out
    ```

For more information, please see samples/2_Cookbook/15_static_library/host_functions and samples/2_Cookbook/15_static_library/device_functions.

## [Supported Clang Options](clang_options.md)
