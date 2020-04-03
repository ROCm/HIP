# Profiling HIP Code

This section describes the tracing and debugging capabilities that HIP provides.  
<!-- toc -->

- [Tracing and Debug](#tracing-and-debug)
  * [Tracing HIP APIs](#tracing-hip-apis)
    + [Color](#color)

<!-- tocstop -->

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


Heres a specific example showing the output of the [square](https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/0_Intro/square) program running on HIP:

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

HIP_TRACE_API supports multiple levels of debug information:
   - 0x1 = print all HIP APIs.  This is the most verbose setting; the flags below allow selecting a subset.
   - 0x2 = print HIP APIs which initiate GPU kernel commands.  Includes hipLaunchKernel, hipLaunchModuleKernel
   - 0x4 = print HIP APIs which initiate GPU memory commands.  Includes hipMemcpy*, hipMemset*.
   - 0x8 = print HIP APIs which allocate or free memory.  Includes hipMalloc, hipHostMalloc, hipFree, hipHostFree.

These can be combined.  For example, HIP_TRACE_API=6 shows a concise view of the HIP commands (both kernel and memory) that are sent to the GPU.


#### Color
Note this trace mode uses colors. "less -r" can handle raw control characters and will display the debug output in proper colors.
You can change the color used for the trace mode with the HIP_TRACE_API_COLOR environment variable.  Possible values are None/Red/Green/Yellow/Blue/Magenta/Cyan/White.
None will disable use of color control codes for both the opening and closing and may be useful when saving the trace file or when a pure text trace is desired.



