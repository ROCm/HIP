# FAQ

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [What APIs does HIP support ?](#what-apis-does-hip-support-)
- [What is not supported?](#what-is-not-supported)
  - [Run-time features](#run-time-features)
  - [Kernel language features](#kernel-language-features)
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
- [HIP detected my platform (hcc vs nvcc) incorrectly - what should I do?](#hip-detected-my-platform-hcc-vs-nvcc-incorrectly---what-should-i-do)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
=================


### What APIs does HIP support ?
HIP provides the following:
- Devices (hipSetDevice(), hipGetDeviceProperties(), etc)
- Memory management (hipMalloc(), hipMemcpy(), hipFree())
- Streams (hipStreamCreate(), etc.)---under development
- Events (hipEventRecord(), hipEventElapsedTime(), etc.)
- Kernel launching (hipLaunchKernel is a standard C/C++ function that replaces <<< >>>)
- CUDA-style kernel indexing
- Device-side math built-ins
- Error reporting (hipGetLastError(), hipGetErrorString())

The HIP documentation describes each API and its limitations, if any, compared with the equivalent CUDA API.

### What is not supported?
#### Run-time features
- Textures 
- Dynamic parallelism
- Managed memory
- Graphics interoperation with OpenGL or Direct3D
- CUDA array, mipmappedArray and pitched memory
- CUDA Driver API
   
#### Kernel language features
- Device-side dynamic memory allocations (malloc, free, new, delete)
- Virtual functions, indirect functions and try/catch
- `__prof_trigger` 
- PTX assembly
- See the [HIP Kernel Language](hip_kernel_language.md) for more information.

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
- For AMD platforms, HIP runs on the same hardware that the HCC "hc" mode supports - specifically AMD Kaveri, Carrizo and Fiji.
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
HCC is AMD's compiler driver which compiles "heterogenous C++" code into HSAIL or GCN device code for AMD GPUs.  HCC is an open-source compiler based on recent versions of CLANG/LLVM.

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


### HIP detected my platform (hcc vs nvcc) incorrectly - what should I do?

HIP will set the platform to HCC if it sees that the AMD graphics driver is installed and has detected an AMD GPU.
Sometimes this isn't what you want - you can force HIP to recognize the platform by setting HIP_PLATFORM to hcc (or nvcc)
```
export HIP_PLATFORM=hcc

```
One symptom of this problem is the message "error: 'unknown error'(11) at square.hipref.cpp:56".  This can occur if you have a CUDA installation on an AMD platform, and HIP incorrectly detects the platform as nvcc.  HIP may be able to compile the application using the nvcc tool-chain, but will generate this error at runtime since the platform does not have a CUDA device. The fix is to set HIP_PLATFORM=hcc and rebuild the issue. 

If you see issues related to incorrect platform detection, please file an issue with the GitHub issue tracker so we can improve HIP's platform detection logic.
