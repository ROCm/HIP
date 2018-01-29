Table of Contents
=================

  * [Profiling HIP Code](#profiling-hip-code" aria-hidden="true"><span aria-hidden="true)
      * [Using HIP_DB](#using-hip_db" aria-hidden="true"><span aria-hidden="true)
      * [Using ltrace](#using-ltrace" aria-hidden="true"><span aria-hidden="true)
      * [Chicken bits](#chicken-bits" aria-hidden="true"><span aria-hidden="true)
      * [Debugging HIP Applications](#debugging-hip-applications" aria-hidden="true"><span aria-hidden="true)
      * [General Debugging Tips](#general-debugging-tips" aria-hidden="true"><span aria-hidden="true)
        * [Print env var state](#print-env-var-state" aria-hidden="true"><span aria-hidden="true)

### Using HIP_DB

This flag is primarily targeted to assist HIP development team in the development of the HIP runtime, but in some situations may be useful to HIP application developers as well.
The HIP debug information is designed to print important information during the execution of a HIP API.  HIP provides
different color-coded levels of debug information:
  - api  : Print the beginning and end of each HIP API, including the arguments and return codes.  This is equivalent to setting HIP_TRACE_API=1.
  - sync : Print multi-thread and other synchronization debug information.
  - copy : Print which engine is doing the copy, which copy flavor is selected, information on source and destination memory.
  - mem  : Print information about memory allocation - which pointers are allocated, where they are allocated, peer mappings, and more.

HIP_DB format is flags separated by '+' sign, or a hex code for the bitmask.  Generally the + format is preferred.  
For example:
```
$ HIP_DB=api+copy+mem  my-application
$ HIP_DB=0xF  my-application
```

### Using ltrace
ltrace is a standard linux tool which provides a message to stderr on every dynamic library call.  Since ROCr and the ROCt (the ROC thunk, which is the thin user-space interface to the ROC kernel driver) are both dynamic libraries, this provides an easy way to trace the activity in these libraries.  Tracing can be a powerful way to quickly observe the flow of the application before diving into the details with a command-line debugger.
The trace can also show performance issues related to accidental calls to expensive API calls on the critical path.

ltrace can be easily combined with the HIP_DB switches to visualize the runtime behavior of the entire ROCm software stack.  Here's a sample command-line and output:

```
$ HIP_DB=api ltrace -C -e 'hsa*'   <applicationName> <applicationArguments>

...

<<hip-api tid:1.17 hipMemcpy (0x7f7776d3e010, 0x503d1d000, 4194304, hipMemcpyDeviceToHost)
libmcwamp_hsa.so->hsa_signal_store_relaxed(0x1804000, 0, 0, 0x400000) = 0
libmcwamp_hsa.so->hsa_signal_store_relaxed(0x1816000, 0, 0x7f777f85f2a0, 0x400000) = 0
libmcwamp_hsa.so->hsa_amd_memory_lock(0x7f7776d3e010, 0x400000, 0x1213b70, 1 <unfinished ...>
libhsa-runtime64.so.1->hsaKmtRegisterMemoryToNodes(0x7f7776d3e010, 0x400000, 1, 0x1220c10) = 0
libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f7776d3e010, 0x400000, 0x7ffc32865400, 64) = 0
<... hsa_amd_memory_lock resumed> )              = 0
libmcwamp_hsa.so->hsa_signal_store_relaxed(0x1804000, 1, 0x7f777e95a770, 0x12205b0) = 0
libmcwamp_hsa.so->hsa_amd_memory_async_copy(0x50411d010, 0x11e70d0, 0x503d1d000, 0x11e70d0) = 0
libmcwamp_hsa.so->hsa_signal_wait_acquire(0x1804000, 2, 1, -1) = 0
libmcwamp_hsa.so->hsa_amd_memory_unlock(0x7f7776d3e010, 0x1213c6c, 0x12c3c600000000, 0x1804000 <unfinished ...>
libhsa-runtime64.so.1->hsaKmtUnmapMemoryToGPU(0x7f7776d3e010, 0x7f7776d3e010, 0x12c3c600000000, 0x1804000) = 0
libhsa-runtime64.so.1->hsaKmtDeregisterMemory(0x7f7776d3e010, 0x7f7776d3e010, 0x7f777f60f9e8, 0x1220580) = 0
<... hsa_amd_memory_unlock resumed> )            = 0
  hip-api tid:1.17 hipMemcpy                      ret= 0 (hipSuccess)>>
```

Some key information from the trace above.
  - Thy trace snippet shows the execution of a hipMemcpy API, bracketed by the first and last message in the trace output.  The messages show the thread id and API sequence number (`1.17`).  ltrace output intermixes messages from all threads, so the HIP debug information can be useful to determine which threads are executing.
  - The code flows through HIP APIs into ROCr (HSA) APIs (hsa*) and into the thunk (hsaKmt*) calls.
  - The HCC runtime is "libmcwamp_hsa.so" and the HSA/ROCr runtime is "libhsa-runtime64.so".
  - In this particular case, the memory copy is for unpinned memory, and the selected copy algorithm is to pin the host memory "in-place" before performing the copy.  The signaling APIs and calls to pin ("lock", "register") the memory are readily apparent in the trace output.


### Chicken bits
Chicken bits are environment variables which cause the HIP, HCC, or HSA driver to disable some feature or optimization.
These are not intended for production but can be useful diagnose synchronization problems in the application (or driver).

Some of the most useful chicken bits are described here. These bits are supported on the ROCm path:

HIP provides 3 environment variables in the HIP_*_BLOCKING family.  These introduce additional synchronization and can be useful to isolate synchronization problems. Specifically, if the code works with this flag set, then it indicates the kernels are executing correctly, and any failures likely are causes by improper or missing synchronization.  These flags will have performance impact and are not intended for production use.

- HIP_LAUNCH_BLOCKING=1 : Waits on the host after each kernel launch.  Equivalent to setting CUDA_LAUNCH_BLOCKING.
- HIP_LAUNCH_BLOCKING_KERNELS: A comma-separated list of kernel names.  The HIP runtime will wait on the host after one of the named kernels executes.  This provides a more targeted version of HIP_LAUNCH_BLOCKING and may be useful to isolate exactly which kernel needs further analysis if HIP_LAUNCH_BLOCKING=1 improves functionality.  There is no indication if kernel names are spelled incorrectly.  One mechanism to verify that the blocking is working is to run with HIP_DB=api+sync and search for debug messages with "LAUNCH_BLOCKING".
- HIP_API_BLOCKING : Forces hipMemcpyAsync and hipMemsetAsync to be host-synchronous, meaning they will wait for the requested operation to complete before returning to the caller.

These options cause HCC to serialize.  Useful if you have libraries or code which is calling HCC kernels directly rather than using HIP.  
- HCC_SERIALIZE_KERNEL : 0x1=pre-serialize before each kernel launch, 0x2=post-serialize after each kernel launch., 0x3= pre- and post- serialize.
- HCC_SERIALIZE_COPY    : 0x1=pre-serialize before each async copy, 0x2=post-serialize after each async copy., 0x3= pre- and post- serialize.

- HSA_ENABLE_SDMA=0     : Causes host-to-device and device-to-host copies to use compute shader blit kernels rather than the dedicated DMA copy engines.  Compute shader copies have low latency (typically < 5us) and can achieve approximately 80% of the bandwidth of the DMA copy engine.  This flag is useful to isolate issues with the hardware copy engines.
- HSA_ENABLE_INTERRUPT=0 : Causes completion signals to be detected with memory-based polling rather than interrupts.  Can be useful to diagnose interrupt storm issues in the driver.
- HSA_DISABLE_CACHE=1  : Disables the GPU L2 data cache.

### Debugging HIP Applications

- The variable "tls_tidInfo" contains the API sequence number (_apiSeqNum)- a monotonically increasing count of the HIP APIs called from this thread.  This can be useful for setting conditional breakpoints.  Also, each new HIP thread is mapped to monotically increasing shortTid ID.  Both of these fields are displayed in the HIP debug info. 
```
(gdb) p tls_tidInfo
$32 = {_shortTid = 1, _apiSeqNum = 803}
```

- HCC tracks all of the application memory allocations, including those from HIP and HC's "am_alloc".
If the HCC runtime is built with debug information (HCC_RUNTIME_DEBUG=ON when building HCC), then calling the function 'hc::am_memtracker_print()' will show all memory allocations. 
An optional argument specifies a void * targetPointer - the print routine will mark the allocation which contains the specified pointer with "-->" in the printed output.
This example shows a sample GDB session where we print the memory allocated by this process and mark a specified address by using the gdb "call" function..
The gdb syntax also supports using the variable name (in this case 'dst'):
```
(gdb) p dst
$33 = (void *) 0x5ec7e9000
(gdb) call hc::am_memtracker_print(dst)
TargetAddress:0x5ec7e9000
   0x504cfc000-0x504cfc00f::  allocSeqNum:1 hostPointer:0x504cfc000 devicePointer:0x504cfc000 sizeBytes:16 isInDeviceMem:0 isAmManaged:1 appId:0 appAllocFlags:0 appPtr:(nil)
...
-->0x5ec7e9000-0x5f7e28fff::  allocSeqNum:488 hostPointer:(nil) devicePointer:0x5ec7e9000 sizeBytes:191102976 isInDeviceMem:1 isAmManaged:1 appId:0 appAllocFlags:0 appPtr:(nil)

```

To debug an explicit address, cast the address to (void*) :
```
(gdb) call hc::am_memtracker_print((void*)0x508c7f000)
```
- Debugging GPUVM fault.
For example:
```
Memory access fault by GPU node-1 on address 0x5924000. Reason: Page not present or supervisor privilege.

Program received signal SIGABRT, Aborted.
[Switching to Thread 0x7fffdffb5700 (LWP 14893)]
0x00007ffff2057c37 in __GI_raise (sig=sig@entry=6) at ../nptl/sysdeps/unix/sysv/linux/raise.c:56
56      ../nptl/sysdeps/unix/sysv/linux/raise.c: No such file or directory.
(gdb) bt
#0  0x00007ffff2057c37 in __GI_raise (sig=sig@entry=6) at ../nptl/sysdeps/unix/sysv/linux/raise.c:56
#1  0x00007ffff205b028 in __GI_abort () at abort.c:89
#2  0x00007ffff6f960eb in ?? () from /opt/rocm/hsa/lib/libhsa-runtime64.so.1
#3  0x00007ffff6f99ea5 in ?? () from /opt/rocm/hsa/lib/libhsa-runtime64.so.1
#4  0x00007ffff6f78107 in ?? () from /opt/rocm/hsa/lib/libhsa-runtime64.so.1
#5  0x00007ffff744f184 in start_thread (arg=0x7fffdffb5700) at pthread_create.c:312
#6  0x00007ffff211b37d in clone () at ../sysdeps/unix/sysv/linux/x86_64/clone.S:111
(gdb) info threads
  Id   Target Id         Frame
  4    Thread 0x7fffdd521700 (LWP 14895) "caffe" pthread_cond_wait@@GLIBC_2.3.2 () at ../nptl/sysdeps/unix/sysv/linux/x86_64/pthread_cond_wait.S:185
  3    Thread 0x7fffddd22700 (LWP 14894) "caffe" pthread_cond_wait@@GLIBC_2.3.2 () at ../nptl/sysdeps/unix/sysv/linux/x86_64/pthread_cond_wait.S:185
* 2    Thread 0x7fffdffb5700 (LWP 14893) "caffe" 0x00007ffff2057c37 in __GI_raise (sig=sig@entry=6) at ../nptl/sysdeps/unix/sysv/linux/raise.c:56
  1    Thread 0x7ffff7fa6ac0 (LWP 14892) "caffe" 0x00007ffff6f934d5 in ?? () from /opt/rocm/hsa/lib/libhsa-runtime64.so.1
(gdb) thread 1
[Switching to thread 1 (Thread 0x7ffff7fa6ac0 (LWP 14892))]
#0  0x00007ffff6f934d5 in ?? () from /opt/rocm/hsa/lib/libhsa-runtime64.so.1
(gdb) bt
#0  0x00007ffff6f934d5 in ?? () from /opt/rocm/hsa/lib/libhsa-runtime64.so.1
#1  0x00007ffff6f929ba in ?? () from /opt/rocm/hsa/lib/libhsa-runtime64.so.1
#2  0x00007fffe080beca in HSADispatch::waitComplete() () from /opt/rocm/hcc/lib/libmcwamp_hsa.so
#3  0x00007fffe080415f in HSADispatch::dispatchKernelAsync(Kalmar::HSAQueue*, void const*, int, bool) () from /opt/rocm/hcc/lib/libmcwamp_hsa.so
#4  0x00007fffe080238e in Kalmar::HSAQueue::dispatch_hsa_kernel(hsa_kernel_dispatch_packet_s const*, void const*, unsigned long, hc::completion_future*) () from /opt/rocm/hcc/lib/libmcwamp_hsa.so
#5  0x00007ffff7bb7559 in hipModuleLaunchKernel () from /opt/rocm/hip/lib/libhip_hcc.so
#6  0x00007ffff2e6cd2c in mlopen::HIPOCKernel::run (this=0x7fffffffb5a8, args=0x7fffffffb2a8, size=80) at /root/MIOpen/src/hipoc/hipoc_kernel.cpp:15  
...
```

### General Debugging Tips
- The fault will be caught by the runtime but was actually generated by an asynchronous command running on the GPU.    So, the GDB backtrace will show a path in the runtime, ie inside "GI_Raise" as shown in the example above.
- To determine the true location of the fault, force the kernels to execute synchronously by seeing the environment variables HCC_SERIALIZE_KERNEL=3 HCC_SERIALIZE_COPY=3.  This will force HCC to wait for the kernel to finish executing before retuning.  If the fault occurs during the execution of a kernel, you can see the code which launched the kernel inside the backtrace.  A bit of guesswork is required to determine which thread is actually causing the issue - typically it will the thread which is waiting inside the libhsa-runtime64.so.
- VM faults inside kernels can be caused byi:
   - incorrect code (ie a for loop which extends past array boundaries), i
   - memory issues  - kernel arguments which are invalid (null pointers, unregistered host pointers, bad pointers).
   - synchronization issues
   - compiler issues (incorrect code generation from the compiler)
   - runtime issues 

-- General debug tips:
- 'gdb --args' can be used to conviently pass the executable and arguments to gdb.
- From inside GDB, you can set environment variables "set env".  Note the command does not use an '=' sign:
```
(gdb) set env HIP_DB 1
```

#### Print env var state
Setting HIP_PRINT_ENV=1 and then running a HIP application will print the HIP environment variables, their current values, and usage info.
Setting HCC_PRINT_ENV=1 and then running a HCC application will print the HCC environment variables, their current values, and usage info.
