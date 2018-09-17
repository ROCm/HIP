# HIP Programming Guide

## Host Memory

### Introduction
hipHostMemory allocates pinned host memory which is mapped into the address space of all GPUs in the system.
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

IP provides the developer with controls to select which type of memory is used via allocation flags passed to hipHostMalloc and the HIP_HOST_COHERENT environment variable:
- hipHostllocCoherent=0, hipHostMallocNonCoherent=0: Use HIP_HOST_COHERENT environment variable: 
    - If HIP_HOST_COHERENT is 1 or undefined, the host memory allocation is coherent.
    - If host memory is `defined and 0: the host memory allocation is non-coherent.
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

### Summary and Recommendations:

- Coherent host memory is the default and is the easiest to use since the memory is visible to the CPU at typical synchronization points.  This memory allows in-kernel synchronization commands such as threadfence_system to work transparently.
- HIP/ROCm also supports the ability to cache host memory in the GPU using the "Non-Coherent" host memory allocations. This can provide performance benefit, but care must be taken to use the correct synchronization.


## Unpinned Memory Transfer Optimization
Please note that this document lists possible ways for experimenting with HIP stack to gain performance. Performance may vary from platform to platform.
 
### On Small BAR Setup

There are two possible ways to transfer data from host-to-device (H2D) and device-to-host(D2H)
 * Using Staging Buffers
 * Using PinInPlace

### On Large BAR Setup

There are three possible ways to transfer data from host-to-device (H2D)
 * Using Staging Buffers
 * Using PinInPlace
 * Direct Memcpy
 
 And there are two possible ways to transfer data from device-to-host (D2H)
 * Using Staging Buffers
 * Using PinInPlace
 
Some GPUs may not be able to directly access host memory, and in these cases we need to
stage the copy through an optimized pinned staging buffer, to implement H2D and D2H copies.The copy is broken into buffer-sized chunks to limit the size of the buffer and also to provide better performance by overlapping the CPU copies with the DMA copies.

PinInPlace is another algorithm which pins the host memory "in-place", and copies it with the DMA engine.  

Unpinned memory transfer mode can be controlled using environment variable HCC_UNPINNED_COPY_MODE. 

By default HCC_UNPINNED_COPY_MODE is set to 0, which uses default threshold values to decide which transfer way to use based on data size.

Setting HCC_UNPINNED_COPY_MODE = 1, forces all unpinned transfer to use PinInPlace logic.

Setting HCC_UNPINNED_COPY_MODE = 2, forces all unpinned transfer to use Staging buffers.

Setting HCC_UNPINNED_COPY_MODE = 3, forces all unpinned transfer to use direct memcpy on large BAR systems.
 
Following environment variables can be used to control the transfer thresholds:

-   HCC_H2D_STAGING_THRESHOLD - Threshold in KB for H2D copy. For sizes smaller than threshold direct copy logic would be used else staging buffers logic. By default it is set to 64.
   
-   HCC_H2D_PININPLACE_THRESHOLD - Threshold in KB for H2D copy. For sizes smaller than threshold staging buffers logic would be used else PinInPlace logic. By default it is set to 4096.

-   HCC_D2H_PININPLACE_THRESHOLD  - Threshold in KB for D2H copy. For sizes smaller than threshold staging buffer logic would be used else PinInPlace logic. By default it is set to 1024.

## Device-Side Malloc

hip-hcc and hip-clang supports device-side malloc and free. Users can allocate
memory dynamically in a kernel. The allocated memory are in global address
space, however, different threads get different memory allocations for the same
call of malloc. The allocated memory can be accessed or freed by other threads
or other kernels. It persists in the life time of the HIP program until it is
freed.

The memory are allocated in pages. Users can define macro
`__HIP_SIZE_OF_PAGE` for controlling the page size in bytes and macro
`__HIP_NUM_PAGES` for controlling the total number of pages that can be
allocated.