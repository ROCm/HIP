# Profiling HIP Code

HIP provides several capabilities to support debugging and profiling.  Profiling information can be displayed to stderr or viewed in the CodeXl visualization tool.

### Usign CodeXL to profile a HIP Application
By defauly, CodeXL can trace all kernel commands, data transfer commands, and HSA Runtime (ROCr) API calls.  
/opt/rocm/bin/rocm-profiler -o <outputATPFileName> -A <applicationName> <applicationArguments>

### Using CodeXL markers for HIP Functions
HIP can generate markers at function being/end which are displayed on the CodeXL timeline view.
HIP 1.0 compiles marker support by default, and you can enable it by setting the HIP_PROFILE_API environment variable and then running the rocm-profiler:

```shell

# Use profile to generate timeline view:
export HIP_PROFILE_API=1
/opt/rocm/bin/rocm-profiler -o <outputATPFileName> -A <applicationName> <applicationArguments>

Or
/opt/rocm/bin/rocm-profiler -e HIP_PROFILE_API=1 -o <outputATPFileName> -A <applicationName> <applicationArguments>
```

#### Developer Builds
For developer builds, you must enable marker support manually when compiling HIP.

1. Build HIP with ATP markers enabled
HIP pre-built packages are enabled with ATP marker support by default.
To enable ATP marker support when building HIP from source, use the option ```-DCOMPILE_HIP_ATP_MARKER=1``` during the cmake configure step.

2. Install ROCm-Profiler
Installing HIP from the [rocm](http://gpuopen.com/getting-started-with-boltzmann-components-platforms-installation/) pre-built packages, installs the ROCm-Profiler as well.
Alternatively, you can build ROCm-Profiler using the instructions [here](https://github.com/RadeonOpenCompute/ROCm-Profiler#building-the-rocm-profiler).

3. Recompile the target application

Then follow the steps above to collect a marker-enabled trace.


### Using HIP_TRACE_API
You can also print the HIP function strings to stderr using HIP_TRACE_API environment variable. This can also be combined with the more detailed debug information provided
by the HIP_DB switch. For example:
```shell
# Trace to stderr showing being/end of each function (with arguments) + intermediate debug trace during the execution of each function.
HIP_TRACE_API=1 HIP_DB=0x2 ./myHipApp
```

#### Color
Note this trace mode uses colors. "less -r" can handle raw control characters and will display the debug output in proper colors.  
You can change the color used for the trace mode with the HIP_TRACE_API_COLOR environment variable.  Possible values are None/Red/Green/Yellow/Blue/Magenta/Cyan/White.
None will disable use of color control codes and may be useful when saving the trace file or when a pure text trace is desired.

#### 


### Using HIP_DB

This flag is primarily targeted to assist HIP development team in the development of the HIP runtime, but in some situations may be useful to HIP application developers as well.
The HIP debug information is designed to print important information during the execution of a HIP API.  HIP provides
different color-coded levels of debug informaton:
  - api  : Print the beginning and end of each HIP API, including the arguments and return codes.
  - sync : Print multi-thread and other synchronization debug information.
  - copy : Print which engine is doing the copy, which copy flavor is selected, information on source and destination memory.
  - mem  : Print information about memory allocation - which pointers are allocated, where they are allocated, peer mappings, and more.

DB_MEM format is flags separated by '+' sign, or a hex code for the bitmask.  Generally the + format is preferred.  
For example:
```shell
HIP_DB=api+copy+mem  my-application
HIP_DB=0xF  my-application
```
HIP_DB=1 same as HIP_TRACE_API=1




Trace provides quick look at API.
Explain output of 
Reference the cookbook example.
Command-line profile.
/// disable profiling at the start of the application you can start CodeXLGpuProfiler with the --startdisabled flag.

Can use strace interleaved with HSA Debug calls .

HIP_PROFILE_API=1
HIP_PROFILE_API=2  : Will show the full API in the trace.  This can be useful for lower-level debugging when you want to see all the parameters that are passed to a specific API.

demangle atp

Write how to collect performance counters.
- include how to compute bandwidth for copy and kernel activity.

- How to disable HSA APIs.
- Do I need to use profiler with HSA enabled?  Do I need to enable HSA profiling on the command line?

Offline compile, how to visualize.  
