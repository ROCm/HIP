# HIP Programming Manual

## Host Memory

### Introduction
hipHostMalloc allocates pinned host memory which is mapped into the address space of all GPUs in the system, the memory can be accessed directly by the GPU device, and can be read or written with much higher bandwidth than pageable memory obtained with functions such as malloc().
There are two use cases for this host memory:
- Faster HostToDevice and DeviceToHost Data Transfers:
The runtime tracks the hipHostMalloc allocations and can avoid some of the setup required for regular unpinned memory.  For exact measurements on a specific system, experiment with --unpinned and --pinned switches for the hipBusBandwidth tool.
- Zero-Copy GPU Access:
GPU can directly access the host memory over the CPU/GPU interconnect, without need to copy the data.  This avoids the need for the copy, but during the kernel access each memory access must traverse the interconnect, which can be tens of times slower than accessing the GPU's local device memory.  Zero-copy memory can be a good choice when the memory accesses are infrequent (perhaps only once).  Zero-copy memory is typically "Coherent" and thus not cached by the GPU but this can be overridden if desired.

### Memory allocation flags
There are flags parameter which can specify options how to allocate the memory, for example,
hipHostMallocPortable, the memory is considered allocated by all contexts, not just the one on which the allocation is made.
hipHostMallocMapped, will map the allocation into the address space for the current device, and the device pointer can be obtained with the API hipHostGetDevicePointer().
hipHostMallocNumaUser is the flag to allow host memory allocation to follow Numa policy by user. Please note this flag is currently only applicable on Linux, under development on Windows.

All allocation flags are independent, and can be used in any combination without restriction, for instance, hipHostMalloc can be called with both hipHostMallocPortable and hipHostMallocMapped flags set. Both usage models described above use the same allocation flags, and the difference is in how the surrounding code uses the host memory.

### Numa-aware host memory allocation
Numa policy determines how memory is allocated.
Target of Numa policy is to select a CPU that is closest to each GPU.
Numa distance is the measurement of how far between GPU and CPU devices.

By default, each GPU selects a Numa CPU node that has the least Numa distance between them, that is, host memory will be automatically allocated closest on the memory pool of Numa node of the current GPU device. Using hipSetDevice API to a different GPU will still be able to access the host allocation, but can have longer Numa distance.
Note, Numa policy is so far implemented on Linux, and under development on Windows.


### Coherency Controls
ROCm defines two coherency options for host memory:
- Coherent memory : Supports fine-grain synchronization while the kernel is running.  For example, a kernel can perform atomic operations that are visible to the host CPU or to other (peer) GPUs.  Synchronization instructions include threadfence_system and C++11-style atomic operations.
In order to achieve this fine-grained coherence, many AMD GPUs use a limited cache policy, such as leaving these allocations uncached by the GPU, or making them read-only.

- Non-coherent memory : Can be cached by GPU, but cannot support synchronization while the kernel is running.  Non-coherent memory can be optionally synchronized only at command (end-of-kernel or copy command) boundaries.  This memory is appropriate for high-performance access when fine-grain synchronization is not required.

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
- By default, the GPU performs a device-scope acquire and release operation with each recorded event.  This will make host and device memory visible to other commands executing on the same device.

A stronger system-level fence can be specified when the event is created with hipEventCreateWithFlags:
- hipEventReleaseToSystem : Perform a system-scope release operation when the event is recorded.  This will make both Coherent and Non-Coherent host memory visible to other agents in the system, but may involve heavyweight operations such as cache flushing.  Coherent memory will typically use lighter-weight in-kernel synchronization mechanisms such as an atomic operation and thus does not need to use hipEventReleaseToSystem.
- hipEventDisableTiming: Events created with this flag would not record profiling data and provide best performance if used for synchronization.

### Summary and Recommendations:

- Coherent host memory is the default and is the easiest to use since the memory is visible to the CPU at typical synchronization points.  This memory allows in-kernel synchronization commands such as threadfence_system to work transparently.
- HIP/ROCm also supports the ability to cache host memory in the GPU using the "Non-Coherent" host memory allocations. This can provide performance benefit, but care must be taken to use the correct synchronization.

### Managed memory allocation
Managed memory, including the `__managed__` keyword, is supported in HIP combined host/device compilation, on Linux, not on Windows (under development).

Managed memory, via unified memory allocation, allows data be shared and accessible to both the CPU and GPU using a single pointer.
The allocation will be managed by AMD GPU driver using the linux HMM (Heterogeneous Memory Management) mechanism, the user can call managed memory API hipMallocManaged to allocate a large chuch of HMM memory, execute kernels on device and fetch data between the host and device as needed.

In HIP application,  It is recommend to do the capability check before calling the managed memory APIs. For example:

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

Note, managed memory management is implemented on Linux, not supported on Windows yet.

### HIP Stream Memory Operations

HIP supports Stream Memory Operations to enable direct synchronization between Network Nodes and GPU. Following new APIs are added,
  hipStreamWaitValue32
  hipStreamWaitValue64
  hipStreamWriteValue32
  hipStreamWriteValue64

Note, CPU access to the semaphore's memory requires volatile keyword to disable CPU compiler's optimizations on memory access.
For more details, please check the documentation HIP-API.pdf.

Please note, HIP stream does not gurantee concurrency on AMD hardware for the case of multiple (at least 6) long running streams executing concurrently, using hipStreamSynchronize(nullptr) for synchronization.

## Direct Dispatch
HIP runtime has Direct Dispatch enabled by default in ROCM 4.4 on Linux.
With this feature we move away from our conventional producer-consumer model where the runtime creates a worker thread(consumer) for each HIP Stream, and the host thread(producer) enqueues commands to a command queue(per stream).

For Direct Dispatch, HIP runtime would directly enqueue a packet to the AQL queue (user mode queue on GPU) on the Dispatch API call from the application. That has shown to reduce the latency to launch the first wave on the idle GPU and total time of tiny dispatches synchronized with the host.

In addition, eliminating the threads in runtime has reduced the variance in the dispatch numbers as the thread scheduling delays and atomics/locks synchronization latencies are reduced.

This feature can be disabled via setting the following environment variable,
AMD_DIRECT_DISPATCH=0

Note, Direct Dispatch is implemented on Linux. It is currently not supported on Windows.

## HIP Runtime Compilation
HIP now supports runtime compilation (HIPRTC), the usage of which will provide the possibility of optimizations and performance improvement compared with other APIs via regular offline static compilation.

HIPRTC APIs accept HIP source files in character string format as input parameters and create handles of programs by compiling the HIP source files without spawning separate processes.

For more details on HIPRTC APIs, refer to [HIP Runtime API Reference](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/index.html).

For Linux developers, the link here (https://github.com/ROCm/hip-tests/blob/develop/samples/2_Cookbook/23_cmake_hiprtc/saxpy.cpp) shows an example how to program HIP application using runtime compilation mechanism, and detail HIPRTC programming guide is also available in Github (https://github.com/ROCm/HIP/blob/develop/docs/user_guide/hip_rtc.md).

## HIP Graph
HIP graph is supported. For more details, refer to the HIP API Guide.

## Device-Side Malloc

HIP-Clang now supports device-side malloc and free.
This implementation does not require the use of `hipDeviceSetLimit(hipLimitMallocHeapSize,value)` nor respects any setting. The heap is fully dynamic and can grow until the available free memory on the device is consumed.

## Use of Per-thread default stream

The per-thread default stream is supported in HIP. It is an implicit stream local to both the thread and the current device. This means that the command issued to the per-thread default stream by the thread does not implicitly synchronize with other streams (like explicitly created streams), or default per-thread stream on other threads.
The per-thread default stream is a blocking stream and will synchronize with the default null stream if both are used in a program.
The per-thread default stream can be enabled via adding a compilation option,
"-fgpu-default-stream=per-thread".

And users can explicitly use "hipStreamPerThread" as per-thread default stream handle as input in API commands. There are test codes as examples in the link (https://github.com/ROCm/hip-tests/tree/develop/catch/unit/streamperthread).

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

Note: Currently, HIP only supports basic math functions with rounding mode rn (round to nearest). HIP does not support basic math functions with rounding modes ru (round up), rd (round down), and rz (round towards zero).

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

For more information, please see [HIP samples](https://github.com/ROCm/hip-tests/tree/develop/samples/2_Cookbook/15_static_library/host_functions) and [samples](https://github.com/ROCm/hip-tests/tree/rocm-5.5.x/samples/2_Cookbook/15_static_library/device_functions).
