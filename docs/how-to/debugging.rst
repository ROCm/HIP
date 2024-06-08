.. meta::
   :description: How to debug using HIP.
   :keywords: AMD, ROCm, HIP, debugging, ltrace, ROCdgb, Windgb

*************************************************************************
Debugging with HIP
*************************************************************************

AMD debugging tools include *ltrace* and *ROCdgb*. External tools are available and can be found
online. For example, if you're using Windows, you can use *Microsoft Visual Studio* and *Windgb*.

You can trace and debug your code using the following tools and techniques.

Tracing
================================================

You can use tracing to quickly observe the flow of an application before reviewing the detailed
information provided by a command-line debugger. Tracing can be used to identify issues ranging
from accidental API calls to calls made on a critical path.

ltrace is a standard Linux tool that provides a message to ``stderr`` on every dynamic library call. You
can use ltrace to visualize the runtime behavior of the entire ROCm software stack.

Here's a simple command-line example that uses ltrace to trace HIP APIs and output:

.. code:: console

    $ ltrace -C -e "hip*" ./hipGetChanDesc
    hipGetChanDesc->hipCreateChannelDesc(0x7ffdc4b66860, 32, 0, 0) = 0x7ffdc4b66860
    hipGetChanDesc->hipMallocArray(0x7ffdc4b66840, 0x7ffdc4b66860, 8, 8) = 0
    hipGetChanDesc->hipGetChannelDesc(0x7ffdc4b66848, 0xa63990, 5, 1) = 0
    hipGetChanDesc->hipFreeArray(0xa63990, 0, 0x7f8c7fe13778, 0x7ffdc4b66848) = 0
    PASSED!
    +++ exited (status 0) +++


Here's another example that uses ltrace to trace hsa APIs and output:

.. code:: console

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

Debugging
================================================

You can use ROCgdb for debugging and profiling.

ROCgdb is the ROCm source-level debugger for Linux and is based on GNU Project debugger (GDB).
the GNU source-level debugger, equivalent of cuda-gdb, can be used with debugger frontends, such as eclipse, vscode, or gdb-dashboard.
For details, see (https://github.com/ROCm/ROCgdb).

Below is a sample how to use ROCgdb run and debug HIP application, rocgdb is installed with ROCM package in the folder /opt/rocm/bin.

.. code:: console

    $ export PATH=$PATH:/opt/rocm/bin
    $ rocgdb ./hipTexObjPitch
    GNU gdb (rocm-dkms-no-npi-hipclang-6549) 10.1
    Copyright (C) 2020 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    ...
    For bug reporting instructions, please see:
    <https://github.com/ROCm/ROCgdb/issues>.
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

Debugging HIP applications
--------------------------------------------------------------------------------------------

The following Linux example shows how to get useful information from the debugger while running a
simple memory copy test, which caused a segmentation fault issue.

.. code:: console

    test: simpleTest2<?> numElements=4194304 sizeElements=4194304 bytes
    Segmentation fault (core dumped)

    (gdb) run
    Starting program: /home/test/hipamd/build/directed_tests/runtimeApi/memory/hipMemcpy_simple
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

    Breakpoint 1, main (argc=1, argv=0x7fffffffdea8)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:147
    147     int main(int argc, char* argv[]) {
    (gdb) c
    Continuing.
    [New Thread 0x7ffff64c4700 (LWP 146066)]

    Thread 1 "hipMemcpy_simpl" received signal SIGSEGV, Segmentation fault.
    0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104
    104             A_h1[i] = 3.14f + 1000 * i;
    (gdb) bt
    #0  0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104
    #1  0x000000000020e96c in main (argc=<optimized out>, argv=<optimized out>)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:163
    (gdb) info thread
    Id   Target Id                                            Frame
    * 1    Thread 0x7ffff64c5880 (LWP 146060) "hipMemcpy_simpl" 0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104
    2    Thread 0x7ffff64c4700 (LWP 146066) "hipMemcpy_simpl" 0x00007ffff6b0850b in ioctl
        () from /lib/x86_64-linux-gnu/libc.so.6
    (gdb) thread 2
    [Switching to thread 2 (Thread 0x7ffff64c4700 (LWP 146066))]
    #0  0x00007ffff6b0850b in ioctl () from /lib/x86_64-linux-gnu/libc.so.6
    (gdb) bt
    #0  0x00007ffff6b0850b in ioctl () from /lib/x86_64-linux-gnu/libc.so.6
    #1  0x00007ffff6604568 in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
    #2  0x00007ffff65fe73a in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
    #3  0x00007ffff659e4d6 in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
    #4  0x00007ffff65807de in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
    #5  0x00007ffff65932a2 in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
    #6  0x00007ffff654f547 in ?? () from /opt/rocm/lib/libhsa-runtime64.so.1
    #7  0x00007ffff7f76609 in start_thread () from /lib/x86_64-linux-gnu/libpthread.so.0
    #8  0x00007ffff6b13293 in clone () from /lib/x86_64-linux-gnu/libc.so.6
    (gdb) thread 1
    [Switching to thread 1 (Thread 0x7ffff64c5880 (LWP 146060))]
    #0  0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104
    104             A_h1[i] = 3.14f + 1000 * i;
    (gdb) bt
    #0  0x000000000020f78e in simpleTest2<float> (numElements=4194304, usePinnedHost=true)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:104
    #1  0x000000000020e96c in main (argc=<optimized out>, argv=<optimized out>)
        at /home/test/hip/tests/src/runtimeApi/memory/hipMemcpy_simple.cpp:163
    (gdb)
    ...

Debugging HIP applications using Windows tools can be more informative than on Linux. Windows
tools provides more visibility into debug codes, which makes it easier to inspect variables, watch
multiple details, and examine call stacks.

Useful environment variables
===================================================

HIP provides environment variables that allow HIP, hip-clang, or HSA drivers to prevent certain features
and optimizations. These are not intended for production, but can be useful to diagnose
synchronization problems in the application (or driver).

Some of the more widely used environment variables are described in this section. These are
supported on the Linux ROCm path and Windows.

Kernel enqueue serialization
---------------------------------------------------------------------------------

You can control kernel command serialization from the host:

``AMD_SERIALIZE_KERNEL``, for serializing kernel enqueue
 ``AMD_SERIALIZE_KERNEL = 1``, Wait for completion before enqueue
 ``AMD_SERIALIZE_KERNEL = 2``, Wait for completion after enqueue
 ``AMD_SERIALIZE_KERNEL = 3``, Both

Or

``AMD_SERIALIZE_COPY``, for serializing copies
 ``AMD_SERIALIZE_COPY = 1``, Wait for completion before enqueue
 ``AMD_SERIALIZE_COPY = 2``, Wait for completion after enqueue
 ``AMD_SERIALIZE_COPY = 3``, Both

So HIP runtime can wait for GPU idle before/after any GPU command depending on the environment
setting.

Making device visible
---------------------------------------------------------------------------------

For systems with multiple devices, you can choose to make only certain device(s) visible to HIP using
``HIP_VISIBLE_DEVICES`` (or ``CUDA_VISIBLE_DEVICES`` on an NVIDIA platform). Once enabled, HIP can
only view devices that have indices present in the sequence. For example:

.. code:: console

    $ HIP_VISIBLE_DEVICES=0,1

Or in the application:

.. code:: cpp

    if (totalDeviceNum > 2) {
    setenv("HIP_VISIBLE_DEVICES", "0,1,2", 1);
    assert(getDeviceNumber(false) == 3);
    ... ...
    }

Dump code object
---------------------------------------------------------------------------------

To analyze compiler-related issues, you can use the dump code object:
``GPU_DUMP_CODE_OBJECT``.

HSA-related environment variables (Linux)
-----------------------------------------------------------------------------------------------

HSA provides environment variables that help analyze issues in drivers or hardware.

* To isolate issues with hardware copy engines, you can use ``HSA_ENABLE_SDMA``.

    ``HSA_ENABLE_SDMA=0`` causes host-to-device and device-to-host copies to use compute shader
    blit kernels, rather than the dedicated DMA copy engines. Compute shader copies have low latency
    (typically < 5 us) and can achieve approximately 80% of the bandwidth of the DMA copy engine.

* To diagnose interrupt storm issues in the driver, you can use ``HSA_ENABLE_INTERRUPT``.

    ``HSA_ENABLE_INTERRUPT=0`` causes completion signals to be detected with memory-based
    polling, rather than interrupts.

HIP environment variable summary
--------------------------------

Here are some of the more commonly used environment variables:

.. # COMMENT: The following lines define a break for use in the table below. 
.. |br| raw:: html 

    <br />

.. list-table::

    * - **Environment variable**
      - **Default value**
      - **Usage**

    * - AMD_LOG_LEVEL 
        |br| Enable HIP log on different Level
      - 0
      - 0: Disable log. 
        |br| 1: Enable log on error level
        |br| 2: Enable log on warning and below levels
        |br| 0x3: Enable log on information and below levels
        |br| 0x4: Decode and display AQL packets

    * - AMD_LOG_MASK
        |br| Enable HIP log on different Level
      - 0x7FFFFFFF
      - 0x1: Log API calls
        |br| 0x02: Kernel and Copy Commands and Barriers
        |br| 0x4: Synchronization and waiting for commands to finish
        |br| 0x8: Enable log on information and below levels
        |br| 0x20: Queue commands and queue contents
        |br| 0x40: Signal creation, allocation, pool
        |br| 0x80: Locks and thread-safety code
        |br| 0x100: Copy debug
        |br| 0x200: Detailed copy debug
        |br| 0x400: Resource allocation, performance-impacting events
        |br| 0x800: Initialization and shutdown
        |br| 0x1000: Misc debug, not yet classified
        |br| 0x2000: Show raw bytes of AQL packet
        |br| 0x4000: Show code creation debug
        |br| 0x8000: More detailed command info, including barrier commands
        |br| 0x10000: Log message location
        |br| 0xFFFFFFFF: Log always even mask flag is zero

    * - HIP_LAUNCH_BLOCKING
        |br|  Used for serialization on kernel execution.
      - 0
      - 0: Disable. Kernel executes normally.
        |br| 1: Enable. Serializes kernel enqueue, behaves the same as AMD_SERIALIZE_KERNEL.
    
    * - HIP_VISIBLE_DEVICES (or CUDA_VISIBLE_DEVICES)
        |br|  Only devices whose index is present in the sequence are visible to HIP
      -
      - 0,1,2: Depending on the number of devices on the system

    * - GPU_DUMP_CODE_OBJECT
        |br| Dump code object
      - 0
      - 0: Disable
        |br| 1: Enable

    * - AMD_SERIALIZE_KERNEL
        |br|  Serialize kernel enqueue
      - 0
      - 1: Wait for completion before enqueue
        |br| 2: Wait for completion after enqueue
        |br| 3: Both

    * - AMD_SERIALIZE_COPY
        |br| Serialize copies
      - 0
      - 1: Wait for completion before enqueue
        |br| 2: Wait for completion after enqueue
        |br| 3: Both

    * - HIP_HOST_COHERENT
        |br| Coherent memory in hipHostMalloc
      - 0
      - 0: memory is not coherent between host and GPU
        |br| 1: memory is coherent with host

    * - AMD_DIRECT_DISPATCH
        |br| Enable direct kernel dispatch (Currently for Linux; under development for Windows)
      - 1
      - 0: Disable
        |br| 1: Enable

    * - GPU_MAX_HW_QUEUES
        |br| The maximum number of hardware queues allocated per device
      - 4
      - The variable controls how many independent hardware queues HIP runtime can create per process,
        per device. If an application allocates more HIP streams than this number, then HIP runtime reuses
        the same hardware queues for the new streams in a round-robin manner. Note that this maximum
        number does not apply to hardware queues that are created for CU-masked HIP streams, or
        cooperative queues for HIP Cooperative Groups (single queue per device).

General debugging tips
======================================================

* ``gdb --args`` can be used to pass the executable and arguments to ``gdb``.
* You can set environment variables (``set env``) from within GDB on Linux:

    .. code:: bash

        (gdb) set env AMD_SERIALIZE_KERNEL 3

    .. note::
        This ``gdb`` command does not use an equal (=) sign.

* The GDB backtrace shows a path in the runtime. This is because a fault is caught by the runtime, but it is generated by an asynchronous command running on the GPU.
* To determine the true location of a fault, you can force the kernels to run synchronously by setting the environment variables ``AMD_SERIALIZE_KERNEL=3`` and ``AMD_SERIALIZE_COPY=3``. This forces HIP runtime to wait for the kernel to finish running before retuning. If the fault occurs when a kernel is running, you can see the code that launched the kernel inside the backtrace. The thread that's causing the issue is typically the one inside ``libhsa-runtime64.so``.
* VM faults inside kernels can be caused by:

    * Incorrect code (e.g., a for loop that extends past array boundaries)
    * Memory issues, such as invalid kernel arguments (null pointers, unregistered host pointers, bad pointers)
    * Synchronization issues
    * Compiler issues (incorrect code generation from the compiler)
    * Runtime issues
