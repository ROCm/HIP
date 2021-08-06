# HIP Debugging
There are some techniques provided in HIP for developers to trace and debug codes during execution, this section describes some details and practical suggestions on debugging.

Table of Contents
=================

  * [ Debugging Tools](#debugging-tools)
      * [Using ltrace](#using-ltrace)
      * [Using ROCgdb](#using-rocgdb)
      * [Other Debugging Tools](#Other-debugging-tools)
  * [ Debugging HIP Application](#debugging-hip-application)
  * [ Useful Environment Variables](#useful-environment-variables)
      * [Kernel Enqueue Serialization](#kernel-enqueue-serialization)
      * [Making Device visible](#making-device-visible)
      * [Dump code object](#dump-code-object)
      * [HSA related environment variables](#HSA-related-environment-variables)
  * [ General Debugging Tips](#general-debugging-tips)

## Debugging tools

### Using ltrace
ltrace is a standard linux tool which provides a message to stderr on every dynamic library call.
Since ROCr and the ROCt (the ROC thunk, which is the thin user-space interface to the ROC kernel driver) are both dynamic libraries, this provides an easy way to trace the activity in these libraries.
Tracing can be a powerful way to quickly observe the flow of the application before diving into the details with a command-line debugger.
ltrace is a helpful tool to visualize the runtime behavior of the entire ROCm software stack.
The trace can also show performance issues related to accidental calls to expensive API calls on the critical path.

Here's a simple sample with command-line to trace hip APIs and output:

```
$ ltrace -C -e "hip*" ./hipGetChanDesc
hipGetChanDesc->hipCreateChannelDesc(0x7ffdc4b66860, 32, 0, 0) = 0x7ffdc4b66860
hipGetChanDesc->hipMallocArray(0x7ffdc4b66840, 0x7ffdc4b66860, 8, 8) = 0
hipGetChanDesc->hipGetChannelDesc(0x7ffdc4b66848, 0xa63990, 5, 1) = 0
hipGetChanDesc->hipFreeArray(0xa63990, 0, 0x7f8c7fe13778, 0x7ffdc4b66848) = 0
PASSED!
+++ exited (status 0) +++
```

Another sample below with command-line only trace hsa APIs and output:

```
$ ltrace -C -e "hsa*" ./hipGetChanDesc
libamdhip64.so.4->hsa_init(0, 0x7fff325a69d0, 0x9c80e0, 0 <unfinished ...>
libhsa-runtime64.so.1->hsaKmtOpenKFD(0x7fff325a6590, 0x9c38c0, 0, 1) = 0
libhsa-runtime64.so.1->hsaKmtGetVersion(0x7fff325a6608, 0, 0, 0) = 0
libhsa-runtime64.so.1->hsaKmtReleaseSystemProperties(3, 0x80084b01, 0, 0) = 0
libhsa-runtime64.so.1->hsaKmtAcquireSystemProperties(0x7fff325a6610, 0, 0, 1) = 0
libhsa-runtime64.so.1->hsaKmtGetNodeProperties(0, 0x7fff325a66a0, 0, 0) = 0
libhsa-runtime64.so.1->hsaKmtGetNodeMemoryProperties(0, 1, 0x9c42b0, 0x936012) = 0
...
<... hsaKmtCreateEvent resumed> )                = 0
libhsa-runtime64.so.1->hsaKmtAllocMemory(0, 4096, 64, 0x7fff325a6690) = 0
libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f1202749000, 4096, 0x7fff325a6690, 0) = 0
libhsa-runtime64.so.1->hsaKmtCreateEvent(0x7fff325a6700, 0, 0, 0x7fff325a66f0) = 0
libhsa-runtime64.so.1->hsaKmtAllocMemory(1, 0x100000000, 576, 0x7fff325a67d8) = 0
libhsa-runtime64.so.1->hsaKmtAllocMemory(0, 8192, 64, 0x7fff325a6790) = 0
libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f120273c000, 8192, 0x7fff325a6790, 0) = 0
libhsa-runtime64.so.1->hsaKmtAllocMemory(0, 4096, 4160, 0x7fff325a6450) = 0
libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f120273a000, 4096, 0x7fff325a6450, 0) = 0
libhsa-runtime64.so.1->hsaKmtSetTrapHandler(1, 0x7f120273a000, 4096, 0x7f120273c000) = 0
<... hsa_init resumed> )                         = 0
libamdhip64.so.4->hsa_system_get_major_extension_table(513, 1, 24, 0x7f1202597930) = 0
libamdhip64.so.4->hsa_iterate_agents(0x7f120171f050, 0, 0x7fff325a67f8, 0 <unfinished ...>
libamdhip64.so.4->hsa_agent_get_info(0x94f110, 17, 0x7fff325a67e8, 0) = 0
libamdhip64.so.4->hsa_amd_agent_iterate_memory_pools(0x94f110, 0x7f1201722816, 0x7fff325a67f0, 0x7f1201722816 <unfinished ...>
libamdhip64.so.4->hsa_amd_memory_pool_get_info(0x9c7fb0, 0, 0x7fff325a6744, 0x7fff325a67f0) = 0
libamdhip64.so.4->hsa_amd_memory_pool_get_info(0x9c7fb0, 1, 0x7fff325a6748, 0x7f1200d82df4) = 0
...
<... hsa_amd_agent_iterate_memory_pools resumed> ) = 0
libamdhip64.so.4->hsa_agent_get_info(0x9dbf30, 17, 0x7fff325a67e8, 0) = 0
<... hsa_iterate_agents resumed> )               = 0
libamdhip64.so.4->hsa_agent_get_info(0x9dbf30, 0, 0x7fff325a6850, 3) = 0
libamdhip64.so.4->hsa_agent_get_info(0x9dbf30, 0xa000, 0x9e7cd8, 0) = 0
libamdhip64.so.4->hsa_agent_iterate_isas(0x9dbf30, 0x7f1201720411, 0x7fff325a6760, 0x7f1201720411) = 0
libamdhip64.so.4->hsa_isa_get_info_alt(0x94e7c8, 0, 0x7fff325a6728, 1) = 0
libamdhip64.so.4->hsa_isa_get_info_alt(0x94e7c8, 1, 0x9e7f90, 0) = 0
libamdhip64.so.4->hsa_agent_get_info(0x9dbf30, 4, 0x9e7ce8, 0) = 0
...
<... hsa_amd_memory_pool_allocate resumed> )     = 0
libamdhip64.so.4->hsa_ext_image_create(0x9dbf30, 0xa1c4c8, 0x7f10f2800000, 3 <unfinished ...>
libhsa-runtime64.so.1->hsaKmtAllocMemory(0, 4096, 64, 0x7fff325a6740) = 0
libhsa-runtime64.so.1->hsaKmtQueryPointerInfo(0x7f1202736000, 0x7fff325a65e0, 0, 0) = 0
libhsa-runtime64.so.1->hsaKmtMapMemoryToGPUNodes(0x7f1202736000, 4096, 0x7fff325a66e8, 0) = 0
<... hsa_ext_image_create resumed> )             = 0
libamdhip64.so.4->hsa_ext_image_destroy(0x9dbf30, 0x7f1202736000, 0x9dbf30, 0 <unfinished ...>
libhsa-runtime64.so.1->hsaKmtUnmapMemoryToGPU(0x7f1202736000, 0x7f1202736000, 4096, 0x9c8050) = 0
libhsa-runtime64.so.1->hsaKmtFreeMemory(0x7f1202736000, 4096, 0, 0) = 0
<... hsa_ext_image_destroy resumed> )            = 0
libamdhip64.so.4->hsa_amd_memory_pool_free(0x7f10f2800000, 0x7f10f2800000, 256, 0x9e76f0) = 0
PASSED!
```

### Using ROCgdb
HIP developers on ROCm can use AMD's ROCgdb for debugging and profiling.
ROCgdb is the ROCm source-level debugger for Linux, based on GDB, the GNU source-level debugger, equivalent of cuda-gdb, can be used with debugger frontends, such as eclipse, vscode, or gdb-dashboard.
For details, see (https://github.com/ROCm-Developer-Tools/ROCgdb).

Below is a sample how to use ROCgdb run and debug HIP application, rocgdb is installed with ROCM package in the folder /opt/rocm/bin.

```
$ export PATH=$PATH:/opt/rocm/bin
$ rocgdb ./hipTexObjPitch
GNU gdb (rocm-dkms-no-npi-hipclang-6549) 10.1
Copyright (C) 2020 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
...
For bug reporting instructions, please see:
<https://github.com/ROCm-Developer-Tools/ROCgdb/issues>.
Find the GDB manual and other documentation resources online at:
    <http://www.gnu.org/software/gdb/documentation/>.

...
Reading symbols from ./hipTexObjPitch...
(gdb) break main
Breakpoint 1 at 0x4013d1: file /home/test/hip/tests/src/texture/hipTexObjPitch.cpp, line 98.
(gdb) run
Starting program: /home/test/hip/build/directed_tests/texture/hipTexObjPitch 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Breakpoint 1, main ()
    at /home/test/hip/tests/src/texture/hipTexObjPitch.cpp:98
98	    texture2Dtest<float>();
(gdb)c

```

### Other Debugging Tools
There are also other debugging tools availble online developers can google and choose the one best suits the debugging requirements.

## Debugging HIP Applications

Below is an example to show how to get useful information from the debugger while running an application, which caused an issue of GPUVM fault.

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

## Useful Environment Variables
HIP provides some environment variables which allow HIP, hip-clang, or HSA driver to disable some feature or optimization.
These are not intended for production but can be useful diagnose synchronization problems in the application (or driver).

Some of the most useful environment variables are described here. They are supported on the ROCm path.

### Kernel Enqueue Serialization
Developers can control kernel command serialization from the host using the environment variable,

AMD_SERIALIZE_KERNEL, for serializing kernel enqueue.
 AMD_SERIALIZE_KERNEL = 1, Wait for completion before enqueue,
 AMD_SERIALIZE_KERNEL = 2, Wait for completion after enqueue,
 AMD_SERIALIZE_KERNEL = 3, Both.

Or
AMD_SERIALIZE_COPY, for serializing copies.

 AMD_SERIALIZE_COPY = 1, Wait for completion before enqueue,
 AMD_SERIALIZE_COPY = 2, Wait for completion after enqueue,
 AMD_SERIALIZE_COPY = 3, Both.

So HIP runtime can wait for GPU idle before/after any GPU command depending on the environment setting.

### Making Device visible
For system with multiple devices, it's possible to make only certain device(s) visible to HIP via setting environment varible,
HIP_VISIBLE_DEVICES, only devices whose index is present in the sequence are visible to HIP.

For example,
```
$ HIP_VISIBLE_DEVICES=0,1
```

or in the appliation,
```
if (totalDeviceNum > 2) {
  setenv("HIP_VISIBLE_DEVICES", "0,1,2", 1);
  assert(getDeviceNumber(false) == 3);
  ... ...
}
```

### Dump code object
Developers can dump code object to anylize compiler related issues via setting environment variable,
GPU_DUMP_CODE_OBJECT

### HSA related environment variables
HSA provides some environment varibles help to analize issues in driver or hardware, for example,

HSA_ENABLE_SDMA=0
It causes host-to-device and device-to-host copies to use compute shader blit kernels rather than the dedicated DMA copy engines.
Compute shader copies have low latency (typically < 5us) and can achieve approximately 80% of the bandwidth of the DMA copy engine.
This environment variable is useful to isolate issues with the hardware copy engines.

HSA_ENABLE_INTERRUPT=0
Causes completion signals to be detected with memory-based polling rather than interrupts.
This environment variable can be useful to diagnose interrupt storm issues in the driver.

## General Debugging Tips
- 'gdb --args' can be used to conviently pass the executable and arguments to gdb.
- From inside GDB, you can set environment variables "set env".  Note the command does not use an '=' sign:

```
(gdb) set env AMD_SERIALIZE_KERNEL 3
```
- The fault will be caught by the runtime but was actually generated by an asynchronous command running on the GPU. So, the GDB backtrace will show a path in the runtime.
- To determine the true location of the fault, force the kernels to execute synchronously by seeing the environment variables AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3.  This will force HIP runtime to wait for the kernel to finish executing before retuning.  If the fault occurs during the execution of a kernel, you can see the code which launched the kernel inside the backtrace.  A bit of guesswork is required to determine which thread is actually causing the issue - typically it will the thread which is waiting inside the libhsa-runtime64.so.
- VM faults inside kernels can be caused by:
   - incorrect code (ie a for loop which extends past array boundaries),
   - memory issues  - kernel arguments which are invalid (null pointers, unregistered host pointers, bad pointers),
   - synchronization issues,
   - compiler issues (incorrect code generation from the compiler),
   - runtime issues.

