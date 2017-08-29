# FAQ

<!-- toc -->

- [What APIs and features does HIP support?](#what-apis-and-features-does-hip-support)
- [What is not supported?](#what-is-not-supported)
  * [Runtime/Driver API features](#runtimedriver-api-features)
  * [Kernel language features](#kernel-language-features)
- [Is HIP a drop-in replacement for CUDA?](#is-hip-a-drop-in-replacement-for-cuda)
- [What specific version of CUDA does HIP support?](#what-specific-version-of-cuda-does-hip-support)
- [What libraries does HIP support?](#what-libraries-does-hip-support)
- [How does HIP compare with OpenCL?](#how-does-hip-compare-with-opencl)
- [How does porting CUDA to HIP compare to porting CUDA to OpenCL?](#how-does-porting-cuda-to-hip-compare-to-porting-cuda-to-opencl)
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
- [On CUDA, can I mix CUDA code with HIP code?](#on-cuda-can-i-mix-cuda-code-with-hip-code)
- [On HCC, can I use HC functionality with HIP?](#on-hcc-can-i-use-hc-functionality-with-hip)
- [How do I trace HIP application flow?](#how-do-i-trace-hip-application-flow)
- [What if HIP generates error of "symbol multiply defined!" only on AMD machine?](#what-if-hip-generates-error-of-symbol-multiply-defined-only-on-amd-machine)
- [How do I disable HIP Generic Grid Launch option?](#how-do-i-disable-hip-generic-grid-launch-option)

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
)t a high-level, the following features are not supported:
- Textures (partial support available)
- Dynamic parallelism (CUDA 5.0)
- Managed memory (CUDA 6.5)
- Graphics interoperability with OpenGL or Direct3D
- CUDA IPC Functions (Under Development)
- CUDA array, mipmappedArray and pitched memory
- Queue priority controls

See the [API Support Table](CUDA_Runtime_API_functions_supported_by_HIP.md) for more detailed information.

#### Kernel language features
- C++-style device-side dynamic memory allocations (free, new, delete) (CUDA 4.0)
- Virtual functions, indirect functions and try/catch (CUDA 4.0)
- `__prof_trigger` 
- PTX assembly (CUDA 4.0).  HCC supports inline GCN assembly.
- Several kernel features are under development.  See the [HIP Kernel Language](hip_kernel_language.md) for more information.  These include:
  - printf


### Is HIP a drop-in replacement for CUDA?
No. HIP provides porting tools which do most of the work to convert CUDA code into portable C++ code that uses the HIP APIs.
Most developers will port their code from CUDA to HIP and then maintain the HIP version. 
HIP code provides the same performance as native CUDA code, plus the benefits of running on AMD platforms.

### What specific version of CUDA does HIP support?
HIP APIs and features do not map to a specific CUDA version. HIP provides a strong subset of functionality provided in CUDA, and the hipify tools can 
scan code to identify any unsupported CUDA functions - this is useful for identifying the specific features required by a given application.

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
    - float16 (supported)
- CUDA 8.0
    - Page Migration including cudaMemAdvise, cudaMemPrefetch, other cudaMem* APIs(not supported)


### What libraries does HIP support?
HIP includes growing support for the 4 key math libraries using hcBlas, hcFft, hcrng and hcsparse, as well as MIOpen for machine intelligence applications.
These offer pointer-based memory interfaces (as opposed to opaque buffers) and can be easily interfaced with other HIP applications.  
The hip interfaces support both ROCm and CUDA paths, with familiar library interfaces.

- [hipBlas](https://github.com/ROCmSoftwarePlatform/hipBLAS), which utilizes [rocBlas](https://github.com/ROCmSoftwarePlatform/rocBLAS).
- [hipfft](https://github.com/ROCmSoftwarePlatform/hcFFT)
- [hipsparse](https://github.com/ROCmSoftwarePlatform/hcSPARSE)
- [hiprng](https://github.com/ROCmSoftwarePlatform/hcrng)
   
Additionally, some of the cublas routines are automatically converted to hipblas equivalents by the hipify-clang tool.  These APIs use cublas or hcblas depending on the platform, and replace the need
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

### How does porting CUDA to HIP compare to porting CUDA to OpenCL?
Both HIP and CUDA are dialects of C++, and thus porting between them is relatively straightforward.
Both dialects support templates, classes, lambdas, and other C++ constructs.
As one example, the hipify tool was originally a Perl script that used simple text conversions from CUDA to HIP.
HIP and CUDA provide similar math library calls as well.  In summary, the HIP philosophy was to make the HIP language close enough to CUDA that the porting effort is relatively simple.
This reduces the potential for error, and also makes it easy to automate the translation.  HIP's goal is to quickly get the ported program running on both platforms with little manual intervention,
so that the programmer can focus on performance optimizations.

There have been several tools that have attempted to convert CUDA into OpenCL, such as CU2CL.  OpenCL is a C99-based kernel language (rather than C++) and also does not support single-source compilation.  
As a result, the OpenCL syntax is different from CUDA, and the porting tools have to perform some heroic transformations to bridge this gap.
The tools also struggle with more complex CUDA applications, in particular those that use templates, classes, or other C++ features inside the kernel.  


### What hardware does HIP support?
- For AMD platforms, HIP runs on the same hardware that the HCC "hc" mode supports.  See the ROCm documentation for the list of supported platforms.
- For Nvidia platforms, HIP requires Unified Memory and should run on any device supporting CUDA SDK 6.0 or newer. We have tested the Nvidia Titan and Tesla K40.

### Does Hipify automatically convert all source code?
Typically, hipify can automatically convert almost all run-time code, and the coordinate indexing device code ( threadIdx.x -> hipThreadIdx_x ).  
Most device code needs no additional conversion, since HIP and CUDA have similar names for math and built-in functions. 
The hipify-clang tool will automatically modify the kernel signature as needed (automating a step that used to be done manually)
Additional porting may be required to deal with architecture feature queries or with CUDA capabilities that HIP doesn't support. 
In general, developers should always expect to perform some platform-specific tuning and optimization.

### What is NVCC?
NVCC is Nvidia's compiler driver for compiling "CUDA C++" code into PTX or device code for Nvidia GPUs. It's a closed-source binary compiler that is provided by the CUDA SDK.

### What is HCC?
HCC is AMD's compiler driver which compiles "heterogeneous C++" code into HSAIL or GCN device code for AMD GPUs.  It's an open-source compiler based on recent versions of CLANG/LLVM.

### Why use HIP rather than supporting CUDA directly?
While HIP is a strong subset of the CUDA, it is a subset.  The HIP layer allows that subset to be clearly defined and documented.
Developers who code to the HIP API can be assured their code will remain portable across Nvidia and AMD platforms.  
In addition, HIP defines portable mechanisms to query architectural features, and supports a larger 64-bit wavesize which expands the return type for cross-lane functions like ballot and shuffle from 32-bit ints to 64-bit ints.  

### Can I develop HIP code on an Nvidia CUDA platform?
Yes.  HIP's CUDA path only exposes the APIs and functionality that work on both NVCC and HCC back-ends.
"Extra" APIs, parameters, and features which exist in CUDA but not in HCC will typically result in compile- or run-time errors.
Developers need to use the HIP API for most accelerator code, and bracket any CUDA-specific code with preprocessor conditionals.
Developers concerned about portability should of course run on both platforms, and should expect to tune for performance.
In some cases CUDA has a richer set of modes for some APIs, and some C++ capabilities such as virtual functions - see the HIP @API documentation for more details.

### Can I develop HIP code on an AMD HCC platform?
Yes. HIP's HCC path only exposes the APIs and functions that work on both NVCC and HCC back ends. "Extra" APIs, parameters and features that appear in HCC but not CUDA will typically cause compile- or run-time errors. Developers must use the HIP API for most accelerator code and bracket any HCC-specific code with preprocessor conditionals. Those concerned about portability should, of course, test their code on both platforms and should tune it for performance. Typically, HCC supports a more modern set of C++11/C++14/C++17 features, so HIP developers who want portability should be careful when using advanced C++ features on the hc path.

### Can a HIP binary run on both AMD and Nvidia platforms?
HIP is a source-portable language that can be compiled to run on either the HCC or NVCC platform. HIP tools don't create a "fat binary" that can run on either platform, however.


### What's the difference between HIP and hc?
HIP is a portable C++ language that supports a strong subset of the CUDA run-time APIs and device-kernel language. It's designed to simplify CUDA conversion to portable C++. HIP provides a C-compatible run-time API, C-compatible kernel-launch mechanism, C++ kernel language and pointer-based memory management.

A C++ dialect, hc is supported by the AMD HCC compiler. It provides C++ run time, C++ kernel-launch APIs (parallel_for_each), C++ kernel language, and several memory-management options, including pointers, arrays and array_view (with implicit data synchronization). It's intended to be a leading indicator of the ISO C++ standard.


### On HCC, can I link HIP code with host code compiled with another compiler such as gcc, icc, or clang ?
Yes.  HIP/HCC generates the object code which conforms to the GCC ABI, and also links with libstdc++.  This means you can compile host code with the compiler of your choice and link the generated object code
with GPU code compiled with HIP.  Larger projects often contain a mixture of accelerator code (initially written in CUDA with nvcc) and host code (compiled with gcc, icc, or clang).   These projects
can convert the accelerator code to HIP, compile that code with hipcc, and link with object code from their preferred compiler.




### HIP detected my platform (hcc vs nvcc) incorrectly - what should I do?
HIP will set the platform to HCC if it sees that the AMD graphics driver is installed and has detected an AMD GPU.
Sometimes this isn't what you want - you can force HIP to recognize the platform by setting HIP_PLATFORM to hcc (or nvcc)
```
export HIP_PLATFORM=hcc
```

One symptom of this problem is the message "error: 'unknown error'(11) at square.hipref.cpp:56".  This can occur if you have a CUDA installation on an AMD platform, and HIP incorrectly detects the platform as nvcc.  HIP may be able to compile the application using the nvcc tool-chain, but will generate this error at runtime since the platform does not have a CUDA device. The fix is to set HIP_PLATFORM=hcc and rebuild. 

If you see issues related to incorrect platform detection, please file an issue with the GitHub issue tracker so we can improve HIP's platform detection logic.

### Can I install both CUDA SDK and HCC on same machine?
Yes. You can use HIP_PLATFORM to choose which path hipcc targets.  This configuration can be useful when using HIP to develop an application which is portable to both AMD and NVIDIA.


### On CUDA, can I mix CUDA code with HIP code?
Yes.  Most HIP data structures (hipStream_t, hipEvent_t) are typedefs to CUDA equivalents and can be intermixed.  Both CUDA and HIP use integer device ids.
One notable exception is that hipError_t is a new type, and cannot be used where a cudaError_t is expected.  In these cases, refactor the code to remove the expectation.  Alternatively, hip_runtime_api.h defines functions which convert between the error code spaces:

hipErrorToCudaError
hipCUDAErrorTohipError
hipCUResultTohipError

If platform portability is important, use #ifdef __HIP_PLATFORM_NVCC__ to guard the CUDA-specific code.

### On HCC, can I use HC functionality with HIP?
Yes.  
The code can include hc.hpp and use HC functions inside the kernel.  A typical use-case is to use AMD-specific hardware features such as the permute, swizzle, or DPP operations.
See the 'bit_extract' sample for an example. 

Also these functions can be used to extract HCC accelerator and accelerator_view structures from the HIP deviceId and hipStream_t:
hipHccGetAccelerator(int deviceId, hc::accelerator *acc);
hipError_t hipHccGetAcceleratorView(hipStream_t stream, hc::accelerator_view **av);

If platform portability is important, use #ifdef __HIP_PLATFORM_HIPCC__ to guard the HCC-specific code.


### How do I trace HIP application flow?
See the [HIP Profiling Guide](hip_porting_guide.md) for more information.

### What if HIP generates error of "symbol multiply defined!" only on AMD machine?
Unlike CUDA, in HCC, for functions defined in the header files, the keyword of "__forceinline__" does not imply "static".
Thus, if failed to define "static" keyword, you might see a lot of "symbol multiply defined!" errors at compilation.
The workaround is to explicitly add the keyword of "static" before any functions that were defined as "__forceinline__".

### How do I disable HIP Generic Grid Launch option? 
Generic Grid Launch(GGL) is currently the default method for hip kernel launch.
To disable it and use the legancy grid launch method, please either change the default value of GENERIC_GRID_LAUNCH to 0 in the following to header files and rebuild HIP:
$HIP/include/hip/hcc_detail/hip_runtime_api.h 
$HIP/include/hip/hcc_detail/host_defines.h
Or pass "-DGENERIC_GRID_LAUNCH=0" to hipcc at application compilation time.
