# Profiling and Debugging HIP Code

This section describes the profiling and debugging capabilities that HIP provides.  
Profiling information can viewed in the CodeXL visualization tool or printed directly to stderr as the application runs.
This document starts with some of the general capabilities of CodeXL and then describes some of the additional HIP marker and debug features.

 * [CodeXL Profiling](#codexl-profiling)
   * [Collecting and Viewing Traces](#collecting-and-viewing-traces)
     * [Using rocm-profiler timestamp profiling](#using-rocm-profiler-timestamp-profiling)
     * [Using rocm-profiler performance counter collection:](#using-rocm-profiler-performance-counter-collection)
     * [Using CodeXL to view profiling results:](#using-codexl-to-view-profiling-results)
     * [More information on CodeXL](#more-information-on-codexl)
   * [HIP Markers](#hip-markers)
     * [Profiling HIP APIs](#profiling-hip-apis)
     * [Adding markers to applications](#adding-markers-to-applications)
   * [Additional HIP Profiling Features](#additional-hip-profiling-features)
     * [Demangling C   Kernel Names](#demangling-c-kernel-names)
     * [Controlling when profiling starts and ends](#controlling-when-profiling-starts-and-ends)
     * [Reducing timeline trace output file size](#reducing-timeline-trace-output-file-size)
     * [How to enable profiling at HIP build time](#how-to-enable-profiling-at-hip-build-time)
 * [Tracing and Debug](#tracing-and-debug)
   * [Tracing HIP APIs](#tracing-hip-apis)
     * [Color](#color)
   * [Using HIP_DB](#using-hip_db)
   * [Using ltrace](#using-ltrace)
   * [Chicken bits](#chicken-bits)

## CodeXL Profiling

### Collecting and Viewing Traces

#### Using rocm-profiler timestamp profiling
rocm-profiler is a command-line tool for tracing any application that uses ROCr API, including HCC and HIP.
rocm-profiler's timeline trace will show the beginning and end for all kernel commands, data transfer commands, and HSA Runtime (ROCr) API calls.  The trace results are saved into a file, which by convention uses the "atp" extension.  Here is an example that shows how to run the command-line profiler:
```shell
$ /opt/rocm/bin/rocm-profiler -o <outputATPFileName> -A -T  <applicationName> <applicationArguments>
```

#### Using rocm-profiler performance counter collection:
rocm-profiler can record performance counter information to provide greater insight inside a kernel, such as the memory bandwidth, ALU busy percentage, and cache statistics.
Collecting the common set of useful counters requires passing the counter configuration files for two passes:
```
$ /opt/rocm/bin/rocm-profiler -C -O --counterfile /opt/rocm/profiler/counterfiles/counters_HSA_Fiji_pass1 --counterfile /opt/rocm/profiler/counterfiles/counters_HSA_Fiji_pass2  <applicationName> <applicationArguments>
```


#### Using CodeXL to view profiling results:
The trace can be loaded and viewed in the CodeXL visualization tool:

- Open the CodeXL GUI, create an new project, and switch to "Profile Mode":
  - $ CodeXL & 
  - [File->New Project, leave fields as is, just click "OK"]
  - [Profile->Switch to Profile Mode]
- Load timestamp tracing results into a timeline view:
  - Right click on the project in the CodeXL Explorer view
  - Click "Import Session..."
  - Select to $HOME/apitrace.atp (or appropriate .atp file if you used another file name)

- Load the performance counter results
  - Right click on the project in the CodeXL Explorer view
  - Click "Import Session..."
  - Select  $HOME/Session1.csv  (or appropriate .csv file if you used another file name)


#### More information on CodeXL
rocm-profiler --help will show additional options and usage guidelines.

See this [blog](http://gpuopen.com/getting-up-to-speed-with-the-codexl-gpu-profiler-and-radeon-open-compute/) for more information on profiling ROCm apps (including HIP) with CodeXL.

The 2.2 version of Windows CodeXL does not correctly handle Linux line-endings.  If you are collecting a trace on Linux and then viewing it with the 2.2 Windows CodeXL, first convert the line ending in the .atp file to Windows-style line endings.

### HIP Markers
#### Profiling HIP APIs  
HIP can generate markers at function beginning and end which are displayed on the CodeXL timeline view.
HIP 1.0 compiles marker support by default, and you can enable it by setting the HIP_PROFILE_API environment variable and then running the rocm-profiler:

```shell

# Use profile to generate timeline view:
export HIP_PROFILE_API=1
$ /opt/rocm/bin/rocm-profiler -A -T <applicationName> <applicationArguments>

Or
$ /opt/rocm/bin/rocm-profiler -e HIP_PROFILE_API=1 -A -T <applicationName> <applicationArguments>
```

HIP_PROFILE_API supports two levels of information.
- HIP_PROFILE_API=1 : Short format.  Print name of API but no arguments.  For example:  
`hipMemcpy`
- HIP_PROFILE_API=2 : Long format.  Print name of API + values of all function arguments. For example:  
`hipMemcpy (0x7f32154db010, 0x50446e000, 4000000, hipMemcpyDeviceToHost)`

#### Adding markers to applications

Markers can be used to define application-specific events that will be recorded in the ATP file and displayed in the CodeXL gui.
This can be particularly useful for visualizing how the higher-level phases of application behavior relate to the lower level HIP APIs, kernel launches, and data transfers.
For example, an instrumented machine learning framework could show the beginning and ending of each layer in the network.

Markers have a specific begin and end time, and can be nested.  Nested calls are displayed hierarchically in the CodeXL gui, with each level of the hierarchy occupying a different row.

The HIP APis are defined in "hip_profile.h":
```
#include <hip/hip_profile.h>

HIP_BEGIN_MARKER(const char *markerName, const char *groupName);
HIP_END_MARKER();

HIP_BEGIN_MARKER("Setup", "MyAppGroup");
// ...
// application code for setup
// ...
HIP_END_MARKER();
```

For C++ codes, HIP also provides a scoped marker which records the start time when constructed and the end time when the scoped marker is destructed at the end of the scope.  This provides a convenient, single-line mechanism to record an event that neatly corresponds to a region of code.  

```cxx
void FunctionFoo(...) 
{
  HIP_SCOPED_MARKER("FunctionFoo", "MyAppGroup"); // Marker starts recording here.

  // ...
  // Function implementation
  // ...

  // Marker destroyed here and records end time stamp.
};
``` 

The HIP marker API is only supported on ROCm platform.  The marker macros are defined on CUDA platforms and will compile, but are silently ignored at runtime.

This [HIP sample](samples/2_Cookbook/2_Profiler/) shows the profiler marker API used in a small application.

More information on the marker API can be found in the profiler header file and PDF in a ROCM installation:
- /opt/rocm/profiler/CXLActivityLogger/include/CXLActivityLogger.h
- /opt/rocm/profiler/CXLActivityLogger/doc/CXLActivityLogger.pdf

### Additional HIP Profiling Features
#### Demangling C++ Kernel Names
HIP includes the `hipdemangleatp` tool which can post-process an ATP file to "demangle" C++ names.
Mangled kernel names encode the C++ arguments and other information, and are guaranteed to be unique even for cases such as operator overloading.  However, the mangled names can be quite verbose.  For example:

`ZZ39gemm_NoTransA_MICRO_NBK_M_N_K_TS16XMTS4RN2hc16accelerator_viewEPKflS3_lPfliiiiiiffEN3_EC__719__cxxamp_trampolineElililiiiiiiS3_iS3_S4_ff`

`hipdemangleatp` will convert this into the more readable:
`gemm_NoTransA_MICRO_NBK_M_N_K_TS16XMTS4`

The `hipdemangleatp` tool operates on the ATP file "in-place" and thus replaces the input file with the demangled version.  

```
$ hipdemangleatp myfile.atp
```

The kernel name is also shown in some of the summary htlm files (Top10 kernels).  These can be regenerated from the demangled ATP file by re-running rocm-profiler:
```
$ rocm-profiler -T --atpfile myfile.atp
```

A future version of CodeXL may directly integrate demangle functionality.


#### Controlling when profiling starts and ends
hipProfilerStart() and hipProfilerEnd() can be inserted into an application to control which phases of the applications are profiled.
These APIs can be used to skip initialization code or to focus profiling on a desired region, and are particularly useful for large long-running applications.
See the API documentation for more information.  These APIs work on both ROCm and CUDA paths.

On ROCm, the following environment variables can be used to control when profiling occurs:

```
HIP_DB_START_API  : Comma-separated list of tid.api_seq_num for when to start debug and profiling.
HIP_DB_STOP_API   : Comma-separated list of tid.api_seq_num for when to stop debug and profiling.
```

HIP/ROCm assigns a monotonically increasing sequence number to the APIs called from each thread.  The thread and API sequence number can be used in the above API to control when tracing starts and stops.  These flags also control the HIP_DB messages (described below).

When using these options, start the profiler with profiling disabled:
```
# ROCm:
$ rocm-profiler --startdisabled ...

# CUDA:
$ nvprof --profile-from-start-off ...
```

This feature is under development.

#### Reducing timeline trace output file size
If the application is already recording the HIP APIs, the HSA APIs are somewhat redundant and the ATP file size can be substantially reduced by not recording these APIs.  HIP includes a text file that lists all of the HSA APis and can assist in this filtering:

```
$ rocm-profiler -F hip/bin/hsa-api-filter-cxl.txt 
```

This file can be copied and edited to provide more selective HSA event recording.


#### How to enable profiling at HIP build time
Recent pre-built packages of HIP are always built with profiling support enabled.
For developer builds, you must enable marker support manually when compiling HIP.

1. Build HIP with ATP markers enabled
HIP pre-built packages are enabled with ATP marker support by default.
To enable ATP marker support when building HIP from source, use the option ```-DCOMPILE_HIP_ATP_MARKER=1``` during the cmake configure step.   Build and install HIP.
```shell
$ mkdir build && cd build
$ cmake .. -DCOMPILE_HIP_ATP_MARKER
$ make install
```

2. Install ROCm-Profiler
Installing HIP from the [rocm](http://gpuopen.com/getting-started-with-boltzmann-components-platforms-installation/) pre-built packages, installs the ROCm-Profiler as well.
Alternatively, you can build ROCm-Profiler using the instructions [here](https://github.com/RadeonOpenCompute/ROCm-Profiler#building-the-rocm-profiler).

3. Recompile the target application

Then follow the steps above to collect a marker-enabled trace.


## Tracing and Debug

### Tracing HIP APIs
The HIP runtime can print the HIP function strings to stderr using HIP_TRACE_API environment variable.
The trace prints two messages for each API - one at the beginning of the API call (line starts with "<<") and one at the end of the API call (line ends with ">>").
Here's an example for one API followed by a description for the sections of the trace:

```
<<hip-api tid:1.6 hipMemcpy (0x7f32154db010, 0x50446e000, 4000000, hipMemcpyDeviceToHost)
  hip-api tid:1.6 hipMemcpy                      ret= 0 (hipSuccess)>>
```

- `<<hip-api` is the header used for all HIP API debug messages.  The message is also shown in a specific color.  This can be used to distinguish this API from other HIP or application messages.
-  `tid:1.6` indicates that this API call came from thread #1 and is the 6th API call in that thread.   When the first API in a new thread is called, HIP will associates a short sequential ID with that thread.  You can see the full thread ID (reported by C++) as 0x7f6183b097c0 in the example below.  
- `hipMemcpy` is the name of the API.
- The first line then prints a comma-separated list of the arguments to the function.  APIs which return values to the caller by writing to pointers will show the pointer addresses rather than the pointer contents.  This behavior may change in the future.
- The second line shows the completion of the API, including the numeric return value (`ret= 0`) as well as an string representation for the error code (`hipSuccess`).  If the returned error code is non-zero, then the csecond line message is shown in red (unless HIP_TRACE_API_COLOR is "none" - see below).


Heres a specific example showing the output of the [square](samples/0_Intro/square) program running on HIP:

```
$ HIP_TRACE_API=1  ./square.hip.out 
  hip-api tid:1:HIP initialized short_tid#1 (maps to full_tid: 0x7f6183b097c0)
<<hip-api tid:1.1 hipGetDeviceProperties (0x7ffddb673e08, 0)
  hip-api tid:1.1 hipGetDeviceProperties         ret= 0 (hipSuccess)>>
info: running on device gfx803
info: allocate host mem (  7.63 MB)
info: allocate device mem (  7.63 MB)
<<hip-api tid:1.2 hipMalloc (0x7ffddb673fb8, 4000000)
  hip-api tid:1.2 hipMalloc                      ret= 0 (hipSuccess)>>
<<hip-api tid:1.3 hipMalloc (0x7ffddb673fb0, 4000000)
  hip-api tid:1.3 hipMalloc                      ret= 0 (hipSuccess)>>
info: copy Host2Device
<<hip-api tid:1.4 hipMemcpy (0x50409d000, 0x7f32158ac010, 4000000, hipMemcpyHostToDevice)
  hip-api tid:1.4 hipMemcpy                      ret= 0 (hipSuccess)>>
info: launch 'vector_square' kernel
1.5 hipLaunchKernel 'HIP_KERNEL_NAME(vector_square)' gridDim:{512,1,1} groupDim:{256,1,1} sharedMem:+0 stream#0.0
info: copy Device2Host
<<hip-api tid:1.6 hipMemcpy (0x7f32154db010, 0x50446e000, 4000000, hipMemcpyDeviceToHost)
  hip-api tid:1.6 hipMemcpy                      ret= 0 (hipSuccess)>>
info: check result
PASSED!
```


#### Color
Note this trace mode uses colors. "less -r" can handle raw control characters and will display the debug output in proper colors.
You can change the color used for the trace mode with the HIP_TRACE_API_COLOR environment variable.  Possible values are None/Red/Green/Yellow/Blue/Magenta/Cyan/White.
None will disable use of color control codes for both the opening and closing and may be useful when saving the trace file or when a pure text trace is desired.



### Using HIP_DB

This flag is primarily targeted to assist HIP development team in the development of the HIP runtime, but in some situations may be useful to HIP application developers as well.
The HIP debug information is designed to print important information during the execution of a HIP API.  HIP provides
different color-coded levels of debug informaton:
  - api  : Print the beginning and end of each HIP API, including the arguments and return codes.  This is equivalent to setting HIP_TRACE_API=1.
  - sync : Print multi-thread and other synchronization debug information.
  - copy : Print which engine is doing the copy, which copy flavor is selected, information on source and destination memory.
  - mem  : Print information about memory allocation - which pointers are allocated, where they are allocated, peer mappings, and more.

DB_MEM format is flags separated by '+' sign, or a hex code for the bitmask.  Generally the + format is preferred.  
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
  - The trace snippet shows the execution of a hipMemcpy API, bracketed by the first and last message in the trace output.  The messages show the thread id and API sequence number (`1.17`).  ltrace output intermixes messages from all threads, so the HIP debug information can be useful to determine which threads are executing.
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
- HCC_SERIALZIE_KERNELS : 0x1=pre-serialize before each kernel launch, 0x2=post-serialize after each kernel launch., 0x3= pre- and post- serialize.
- HCC_SERIALIZE_COPY    : 0x1=pre-serialize before each async copy, 0x2=post-serialize after each async copy., 0x3= pre- and post- serialize.

- HSA_ENABLE_SDMA=0     : Causes host-to-device and device-to-host copies to use compute shader blit kernels rather than the dedicated DMA copy engines.  Compute shader copies have low latency (typically < 5us) and can achieve approximately 80% of the bandwidth of the DMA copy engine.  This flag is useful to isolate issues with the hardware copy engines.
- HSA_ENABLE_INTERRUPT=0 : Causes completion signals to be detected with memory-based polling rather than interrupts.  Can be useful to diagnose interrupt storm issues in the driver.
- HSA_DISABLE_CACHE=1  : Disables the GPU L2 data cache.
