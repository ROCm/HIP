# FAQ

<!-- toc -->

- [What APIs and features does HIP support?](#what-apis-and-features-does-hip-support)
- [What is not supported?](#what-is-not-supported)
  * [Run-time features](#run-time-features)
  * [Kernel language features](#kernel-language-features)
- [Is HIP a drop-in replacement for CUDA?](#is-hip-a-drop-in-replacement-for-cuda)
- [What specific version of CUDA does HIP support?](#what-specific-version-of-cuda-does-hip-support)
- [What libraries does HIP support?](#what-libraries-does-hip-support)
- [How does HIP compare with OpenCL?](#how-does-hip-compare-with-opencl)
- [What hardware does HIP support?](#what-hardware-does-hip-support)
- [Does Hipify automatically convert all source code?](#does-hipify-automatically-convert-all-source-code)
- [What is NVCC?](#what-is-nvcc)
- [What is HCC?](#what-is-hcc)
- [Why use HIP rather than supporting CUDA directly?](#why-use-hip-rather-than-supporting-cuda-directly)
- [Can I develop HIP code on an Nvidia CUDA platform?](#can-i-develop-hip-code-on-an-nvidia-cuda-platform)
- [Can I develop HIP code on an AMD HCC platform?](#can-i-develop-hip-code-on-an-amd-hcc-platform)
- [Can a HIP binary run on both AMD and Nvidia platforms?](#can-a-hip-binary-run-on-both-amd-and-nvidia-platforms)
- [What's the difference between HIP and hc?](#whats-the-difference-between-hip-and-hc)
- [On HCC, can I link HIP code with host code compiled with another compiler such as gcc, icc, or clang ?](#on-hcc-can-i-link-hip-code-with-host-code-compiled-with-another-compiler-such-as-gcc-icc-or-clang-)
- [HIP detected my platform (hcc vs nvcc) incorrectly - what should I do?](#hip-detected-my-platform-hcc-vs-nvcc-incorrectly---what-should-i-do)
- [Can I install both CUDA SDK and HCC on same machine?](#can-i-install-both-cuda-sdk-and-hcc-on-same-machine)
- [How do I trace HIP application flow?](#how-do-i-trace-hip-application-flow)
  * [Using CodeXL markers for HIP Functions](#using-codexl-markers-for-hip-functions)
  * [Using HIP_TRACE_API](#using-hip_trace_api)

<!-- tocstop -->

### What APIs and features does HIP support?
HIP provides the following:
- Devices (hipSetDevice(), hipGetDeviceProperties(), etc.)
- Memory management (hipMalloc(), hipMemcpy(), hipFree(), etc.)
- Streams (hipStreamCreate(),hipStreamSynchronize(), hipStreamWaitEvent(),  etc.)
- Events (hipEventRecord(), hipEventElapsedTime(), etc.)
- Kernel launching (hipLaunchKernel is a standard C/C++ function that replaces <<< >>>)
- HIP Module API to control when adn how code is loaded.
- CUDA-style kernel coordinate functions (threadIdx, blockIdx, blockDim, gridDim)
- Cross-lane instructions including shfl, ballot, any, all
- Most device-side math built-ins
- Error reporting (hipGetLastError(), hipGetErrorString())

The HIP API documentation describes each API and its limitations, if any, compared with the equivalent CUDA API.

### What is not supported?
#### Runtime/Driver API features
At a high-level, the following features are not supported:
- Textures 
- Dynamic parallelism (CUDA 5.0)
- Managed memory (CUDA 6.5)
- Graphics interoperation with OpenGL or Direct3D
- CUDA Driver API (Under Development)
- CUDA IPC Functions (Under Development)

- CUDA array, mipmappedArray and pitched memory
- MemcpyToSymbol functions
- Queue priority controls

See the [API Support Table](CUDA_Runtime_API_functions_supported_by_HIP.md) for more detailed information.

#### Kernel language features
- Device-side dynamic memory allocations (malloc, free, new, delete) (CUDA 4.0)
- Virtual functions, indirect functions and try/catch (CUDA 4.0)
- `__prof_trigger` 
- PTX assembly (CUDA 4.0).  HCC supports inline GCN assembly.
- Several kernel features are under development.  See the [HIP Kernel Language](hip_kernel_language.md) for more information.  These include:
  - printf
  - assert
  - `__restrict__`
  - `__launch_bounds__`
  - `__threadfence*_`, `__syncthreads*`
  - Unbounded loop unroll



### Is HIP a drop-in replacement for CUDA?
No. HIP provides porting tools which do most of the work do convert CUDA code into portable C++ code that uses the HIP APIs.
Most developers will port their code from CUDA to HIP and then maintain the HIP version. 
HIP code provides the same performance as native CUDA code, plus the benefits of running on AMD platforms.

### What specific version of CUDA does HIP support?
HIP APIs and features do not map to a specific CUDA version.  HIP provides a strong subset of functionality provided in CUDA, and the hipify tools can 
scan code to identify any unsupported CUDA functions - this is very useful for identifying the specific features required by a given application.

However, we can provide a rough summary of the features included in each CUDA SDK and the support level in HIP:

- CUDA 4.0 and earlier :  
    - HIP supports CUDA 4.0 except for the limitations described above.
- CUDA 5.0 : 
    - Dynamic Parallelism (not supported) 
    - cuIpc functions (under development).
- CUDA 5.5 : 
    - CUPTI (not directly supported, [AMD GPUPerfAPI](http://developer.amd.com/tools-and-sdks/graphics-development/gpuperfapi/) can be used as an alternative in some cases)
- CUDA 6.0
    - Managed memory (under development)
- CUDA 6.5
    - __shfl instriniscs (supported)
- CUDA 7.0
    - Per-thread-streams (under development)
    - C++11 (HCC supports all of C++11, all of C++14 and some C++17 features)
- CUDA 7.5
    - float16 (under development)
- CUDA 8.0
    - No new language features.

### What libraries does HIP support?
HIP includes growing support for the 4 key math libraries using hcBlas, hcFft, hcrng, and hcsparse).
These offer pointer-based memory interfaces (as opposed to opaque buffers) and can be easily interfaces with other HCC code.  Developers should use conditional compliation if portability to nvcc systems is desired - using calls to cu* routines on one path and hc* routines on the other.  

- [hcblas](https://bitbucket.org/multicoreware/hcblas)
- [hcfft](https://bitbucket.org/multicoreware/hcfft)
- [hcsparse](https://bitbucket.org/multicoreware/hcsparse)
- [hcrng](https://bitbucket.org/multicoreware/hcrng)
   
Additionally, some of the cublas routines are automatically converted to hipblas equivalents by the clang-hipify tool.  These APIs use cublas or hcblas depending on the platform, and replace the need
to use conditional compilation. 

### How does HIP compare with OpenCL?
Both AMD and Nvidia support OpenCL 1.2 on their devices, so developers can write portable code.
HIP offers several benefits over OpenCL:
- Developers can code in C++ as well as mix host and device C++ code in their source files. HIP C++ code can use templates, lambdas, classes and so on.
- The HIP API is less verbose than OpenCL and is familiar to CUDA developers.
- Because both CUDA and HIP are C++ languages, porting from CUDA to HIP is significantly easier than porting from CUDA to OpenCL.
- HIP uses the best available development tools on each platform: on Nvidia GPUs, HIP code compiles using NVCC and can employ the nSight profiler and debugger (unlike OpenCL on Nvidia GPUs).
- HIP provides pointers and host-side pointer arithmetic.
- HIP provides device-level control over memory allocation and placement.
- HIP offers an offline compilation model.

### What hardware does HIP support?
- For AMD platforms, HIP runs on the same hardware that the HCC "hc" mode supports.  See the ROCM documentation for the list of supported platforms.
- For Nvidia platforms, HIP requires Unified Memory and should run on a device which runs the CUDA SDK 6.0 or newer. We have tested the Nvidia Titan and K40.

### Does Hipify automatically convert all source code?
Typically, Hipify can automatically convert almost all run-time code, and the coordinate indexing device code. 
Most device code needs no additional conversion, since HIP and CUDA have similar names for math and built-in functions. 
HIP currently requires manual addition of one more arguments to the kernel so that the host can communicate the execution configuration to the device. 
Additional porting may be required to deal with architecture feature queries or with CUDA capabilities that HIP doesn't support. 
Developers should always expect to perform some platform-specific tuning and optimization.

### What is NVCC?
NVCC is Nvidia's compiler driver for compiling "CUDA C++" code into PTX or device code for Nvidia GPUs. It's a closed-source binary product that comes with CUDA SDKs.

### What is HCC?
HCC is AMD's compiler driver which compiles "heterogenous C++" code into HSAIL or GCN device code for AMD GPUs.  It's an open-source compiler based on recent versions of CLANG/LLVM.

### Why use HIP rather than supporting CUDA directly?
While HIP is a strong subset of the CUDA, it is a subset.  The HIP layer allows that subset to be clearly defined and documented.
Developers who code to the HIP API can be assured there code will remain portable across Nvidia and AMD platforms.  
In addition, HIP defines portable mechanisms to query architectural features, and supports a larger 64-bit wavesize which expands the return type for cross-lane functions like ballot and shuffle from 32-bit ints to 64-bit ints.  

### Can I develop HIP code on an Nvidia CUDA platform?
Yes!  HIP's CUDA path only exposes the APIs and functionality that work on both NVCC and HCC back-ends.
"Extra" APIs, parameters, and features which exist in CUDA but not in HCC will typically result in compile-time or run-time errors.
Developers need to use the HIP API for most accelerator code, and bracket any CUDA-specific code with appropriate ifdefs.
Developers concerned about portability should of course run on both platforms, and should expect to tune for performance.
In some cases CUDA has a richer set of modes for some APIs, and some C++ capabilities such as virtual functions - see the HIP @API documentation for more details.

### Can I develop HIP code on an AMD HCC platform?
Yes! HIP's HCC path only exposes the APIs and functions that work on both NVCC and HCC back ends. "Extra" APIs, parameters and features that appear in HCC but not CUDA will typically cause compile- or run-time errors. Developers must use the HIP API for most accelerator code and bracket any HCC-specific code with appropriate ifdefs. Those concerned about portability should, of course, test their code on both platforms and should tune it for performance. Typically, HCC supports a more modern set of C++11/C++14/C++17 features, so HIP developers who want portability should be careful when using advanced C++ features on the hc path.

### Can a HIP binary run on both AMD and Nvidia platforms?
HIP is a source-portable language that can be compiled to run on either the HCC or NVCC platform. HIP tools don't create a "fat binary" that can run on either platform, however.


### What's the difference between HIP and hc?
HIP is a portable C++ language that supports a strong subset of the CUDA run-time APIs and device-kernel language. It's designed to simplify CUDA conversion to portable C++. HIP provides a C-compatible run-time API, C-compatible kernel-launch mechanism, C++ kernel language and pointer-based memory management.

A C++ dialect, hc is supported by the AMD HCC compiler. It provides C++ run time, C++ kernel-launch APIs (parallel_for_each), C++ kernel language, and several memory-management options, including pointers, arrays and array_view (with implicit data synchronization). It's intended to be a leading indicator of the ISO C++ standard.


### On HCC, can I link HIP code with host code compiled with another compiler such as gcc, icc, or clang ?
Yes!  HIP/HCC generates the object code which conforms to the GCC ABI, and also links with libstdc++.  This means you can compile host code with the compiler of your choice and link this
with GPU code compiler with HIP.  Larger projects often contain a mixture of accelerator code (initially written in CUDA with nvcc) plus host code (compiled with gcc, icc, or clang).   These projects
can convert the accelerator code to HIP, compile that code with hipcc, and link with object code from the preferred compiler.


### HIP detected my platform (hcc vs nvcc) incorrectly - what should I do?
HIP will set the platform to HCC if it sees that the AMD graphics driver is installed and has detected an AMD GPU.
Sometimes this isn't what you want - you can force HIP to recognize the platform by setting HIP_PLATFORM to hcc (or nvcc)
```
export HIP_PLATFORM=hcc

```
One symptom of this problem is the message "error: 'unknown error'(11) at square.hipref.cpp:56".  This can occur if you have a CUDA installation on an AMD platform, and HIP incorrectly detects the platform as nvcc.  HIP may be able to compile the application using the nvcc tool-chain, but will generate this error at runtime since the platform does not have a CUDA device. The fix is to set HIP_PLATFORM=hcc and rebuild the issue. 

If you see issues related to incorrect platform detection, please file an issue with the GitHub issue tracker so we can improve HIP's platform detection logic.

### Can I install both CUDA SDK and HCC on same machine?
Yes. You can use HIP_PLATFORM to choose which path hipcc targets.  This configuration can be useful when using HIP to develop an application which is portable to both AMD and NVIDIA.

### How do I trace HIP application flow?
#### Using CodeXL markers for HIP Functions
HIP can generate markers at function being/end which are displayed on the CodeXL timeline view.
To do this, you need to install ROCm-Profiler and enable HIP to generate the markers:

1. Install ROCm-Profiler
Installing HIP from the [rocm](http://gpuopen.com/getting-started-with-boltzmann-components-platforms-installation/) pre-built packages, installs the ROCm-Profiler as well.
Alternatively, you can build ROCm-Profiler using the instructions [here](https://github.com/RadeonOpenCompute/ROCm-Profiler#building-the-rocm-profiler).

2. Build HIP with ATP markers enabled
HIP pre-built packages are enabled with ATP marker support by default.
To enable ATP marker support when building HIP from source, use the option ```-DCOMPILE_HIP_ATP_MARKER=1``` during the cmake configure step.

3. Set HIP_ATP_MARKER
```shell
export HIP_ATP_MARKER=1
```

4. Recompile the target application

5. Run with profiler enabled to generate ATP file.
```shell
# Use profile to generate timeline view:
/opt/rocm/bin/rocm-profiler -o <outputATPFileName> -A <applicationName> <applicationArguments>
```

#### Using HIP_TRACE_API
You can also print the HIP function strings to stderr using HIP_TRACE_API environment variable. This can also be combined with the more detailed debug information provided
by the HIP_DB switch. For example:
```shell
# Trace to stderr showing being/end of each function (with arguments) + intermediate debug trace during the execution of each function.
HIP_TRACE_API=1 HIP_DB=0x2 ./myHipApp
```

Note this trace mode uses colors. "less -r" can handle raw control characters and will display the debug output in proper colors.

### What if HIP generates error of "symbol multiply defined!" only on AMD machine?
Unlike CUDA, in HCC, for functions defined in the header files, the keyword of "__forceinline__" does not imply "static".
Thus, if failed to define "static" keyword, you might see a lot of "symbol multiply defined!" errors at compilation.
The workaround is to explicitly add the keyword of "static" before any functions that were defined as "__forceinline__".

