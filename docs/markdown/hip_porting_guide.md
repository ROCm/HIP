# HIP Porting Guide 
In addition to providing a portable C++ programmming environement for GPUs, HIP is designed to ease
the porting of existing CUDA code into the HIP environment.  This section describes the available tools 
and provides practical suggestions on how to port CUDA code and work through common issues. 

###Table of Contents
=================

 * [HIP Porting Guide](#hip-porting-guide)
      * [Table of Contents](#table-of-contents)
    * [Porting a New Cuda Project TO](#porting-a-new-cuda-project)
      * [General Tips](#general-tips" aria-hidden="true"><span aria-hidden="true)
      * [Scanning existing CUDA code to scope the porting effort](#scanning-existing-cuda-code-to-scope-the-porting-effort" aria-hidden="true"><span aria-hidden="true)
    * [Distinguishing Compiler Modes](#distinguishing-compiler-modes" aria-hidden="true"><span aria-hidden="true)
      * [Identifying HIP Target Platform](#identifying-hip-target-platform" aria-hidden="true"><span aria-hidden="true)
      * [Identifying the Compiler: hcc or nvcc](#identifying-the-compiler-hcc-or-nvcc" aria-hidden="true"><span aria-hidden="true)
      * [Identifying Current Compilation Pass: Host or Device](#identifying-current-compilation-pass-host-or-device" aria-hidden="true"><span aria-hidden="true)
      * [Compiler Defines: Summary](#compiler-defines-summary" aria-hidden="true"><span aria-hidden="true)
    * [Identifying Architecture Features](#identifying-architecture-features" aria-hidden="true"><span aria-hidden="true)
      * [HIP_ARCH Defines](#hip_arch-defines" aria-hidden="true"><span aria-hidden="true)
      * [Device-Architecture Properties](#device-architecture-properties" aria-hidden="true"><span aria-hidden="true)
      * [Table of Architecture Properties](#table-of-architecture-properties" aria-hidden="true"><span aria-hidden="true)
    * [Finding HIP](#finding-hip" aria-hidden="true"><span aria-hidden="true)
    * [Compiler Options](#compiler-options" aria-hidden="true"><span aria-hidden="true)
    * [Linking Issues](#linking-issues" aria-hidden="true"><span aria-hidden="true)
      * [Linking With hipcc](#linking-with-hipcc" aria-hidden="true"><span aria-hidden="true)
      * [-lm Option](#-lm-option" aria-hidden="true"><span aria-hidden="true)
    * [Linking Code With Other Compilers](#linking-code-with-other-compilers" aria-hidden="true"><span aria-hidden="true)
      * [libc   and libstdc  ](#libc-and-libstdc" aria-hidden="true"><span aria-hidden="true)
      * [HIP Headers (hip_runtime.h, hip_runtime_api.h)](#hip-headers-hip_runtimeh-hip_runtime_apih" aria-hidden="true"><span aria-hidden="true)
      * [Using a Standard C   Compiler](#using-a-standard-c-compiler" aria-hidden="true"><span aria-hidden="true)
        * [cuda.h](#cudah" aria-hidden="true"><span aria-hidden="true)
      * [Choosing HIP File Extensions](#choosing-hip-file-extensions" aria-hidden="true"><span aria-hidden="true)
      * [Workarounds](#workarounds" aria-hidden="true"><span aria-hidden="true)
        * [warpSize](#warpsize" aria-hidden="true"><span aria-hidden="true)
        * [Textures and Cache Control](#textures-and-cache-control" aria-hidden="true"><span aria-hidden="true)
    * [More Tips](#more-tips" aria-hidden="true"><span aria-hidden="true)
      * [hcc CPU Mode](#hcc-cpu-mode" aria-hidden="true"><span aria-hidden="true)
      * [HIPTRACE Mode](#hiptrace-mode" aria-hidden="true"><span aria-hidden="true)
      * [Environment Variables](#environment-variables" aria-hidden="true"><span aria-hidden="true)
      * [Debugging hipcc](#debugging-hipcc" aria-hidden="true"><span aria-hidden="true)
      * [What Does This Error Mean?](#what-does-this-error-mean" aria-hidden="true"><span aria-hidden="true)
        * [/usr/include/c  /v1/memory:5172:15: error: call to implicitly deleted default constructor of 'std::__1::bad_weak_ptr' throw bad_weak_ptr();](#usrincludecv1memory517215-error-call-to-implicitly-deleted-default-constructor-of-std__1bad_weak_ptr-throw-bad_weak_ptr" aria-hidden="true"><span aria-hidden="true)
        * [grid_launch kernel dispatch - fallback](#grid_launch-kernel-dispatch---fallback" aria-hidden="true"><span aria-hidden="true)
        * [Editor Highlighting](#editor-highlighting)


## Porting a New Cuda Project

### General Tips
- Starting the port on a Cuda machine is often the easiest approach, since you can incrementally port pieces of the code to HIP while leaving the rest in Cuda. (Recall that on Cuda machines HIP is just a thin layer over Cuda, so the two code types can interoperate on nvcc platforms.) Also, the HIP port can be compared with the original Cuda code for function and performance.
- Once the Cuda code is ported to HIP and is running on the Cuda machine, compile the HIP code using hcc on an AMD machine.
- HIP ports can replace Cuda versions---HIP can deliver the same performance as a native Cuda implementation, with the benefit of portability to both Nvidia and AMD architectures as well as a path to future C++ standard support. You can handle platform-specific features through conditional compilation or by adding them to the open-source HIP infrastructure.
- Use **bin/hipconvertinplace.sh** to hipify all code files in the Cuda source directory.

### Scanning existing CUDA code to scope the porting effort
The hipexamine.sh tool will scan a source directory to determine which files contain CUDA code and how much of that code can be automatically hipified, 
```
> cd examples/rodinia_3.0/cuda/kmeans
> $HIP_DIR/bin/hipexamine.sh .
info: hipify ./kmeans.h =====>
info: hipify ./unistd.h =====>
info: hipify ./kmeans.c =====>
info: hipify ./kmeans_cuda_kernel.cu =====>
  info: converted 40 CUDA->HIP refs( dev:0 mem:0 kern:0 builtin:37 math:0 stream:0 event:0 err:0 def:0 tex:3 other:0 ) warn:0 LOC:185
info: hipify ./getopt.h =====>
info: hipify ./kmeans_cuda.cu =====>
  info: converted 49 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:0 math:0 stream:0 event:0 err:0 def:0 tex:12 other:0 ) warn:0 LOC:311
info: hipify ./rmse.c =====>
info: hipify ./cluster.c =====>
info: hipify ./getopt.c =====>
info: hipify ./kmeans_clustering.c =====>
info: TOTAL-converted 89 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:37 math:0 stream:0 event:0 err:0 def:0 tex:15 other:0 ) warn:0 LOC:3607
  kernels (1 total) :   kmeansPoint(1)
```

hipexamine scans each code file (cpp, c, h, hpp, etc) found in the specified directory:

   * Files with no CUDA code (ie kmeans.h) print one line summary just listing the source file name.
   * Files with CUDA code print a summary of what was found - for example the kmeans_cuda_kernel.cu file:
```
info: hipify ./kmeans_cuda_kernel.cu =====>
  info: converted 40 CUDA->HIP refs( dev:0 mem:0 kern:0 builtin:37 math:0 stream:0 event:0 
```
* Some of the most interesting information in kmeans_cuda_kernel.cu :
       * How many CUDA calls were converted to HIP (40)
       * Breakdown of the different CUDA functionality used (dev:0 mem:0 etc).  This file uses many CUDA builtins (37) and texture functions (3).
       * Warning for code that looks like CUDA API but was not converted (0 in this file).
       * Count Lines-of-Code (LOC) - 185 for this file. 

* hipexamine also presents a summary at the end of the process for the statistics collected across all files. This has similar format to the per-file reporting, and also includes a list of all kernels which have been called.  An example from above:

```shell
info: TOTAL-converted 89 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:37 math:0 stream:0 event:0 err:0 def:0 tex:15 other:0 ) warn:0 LOC:3607
  kernels (1 total) :   kmeansPoint(1)
```
 
### Converting a project "in-place"

```shell
> hipify --inplace
```

For each input file FILE, this script will:
  - If "FILE.prehip file does not exist, copy the original code to a new file with extension ".prehip".  Then Hipify the code file.
  - If "FILE.prehip" file exists, hipify FILE.prehip and save to FILE.  

This is useful for testing improvements to the hipify toolset.


The "hipconvertinplace.sh" script will perform inplace conversion for all code files in the specified directory.
This can be quite handy when dealing with an existing CUDA code base since the script preserves the existing directory structure
and filenames - so includes work.  After converting in-place, you can review the code to add additional parameters to
directory names.


```shell
> hipconvertinplace.sh MY_SRC_DIR
```




 
## Distinguishing Compiler Modes
 
 
### Identifying HIP Target Platform
All HIP projects target either the hcc or nvcc platform. The platform affects which headers are included and which libraries are used for linking. 
 
- `HIPCC_PLATFORM_HCC` is defined if the HIP platform targets hcc
- `HIPCC_PLATFORM_NVCC` is defined if the HIP platform targets nvcc
 
Many projects use a mixture of an accelerator compiler (hcc or nvcc) and a standard compiler (e.g., g++). These defines are set for both accelerator and standard compilers and thus are often the best option when writing code that uses conditional compilation.
 
 
### Identifying the Compiler: hcc or nvcc
Often, its useful to know whether the underlying compiler is hcc or nvcc. This knowledge can guard platform-specific code (features that only work on the nvcc or hcc path but not both) or aid in platform-specific performance tuning.   
 
```
#ifdef __HCC__
// Compiled with hcc 
 
```
 
```
#ifdef __NVCC__
// Compiled with nvcc  
//  Could be compiling with Cuda language extensions enabled (for example, a ".cu file)
//  Could be in pass-through mode to an underlying host compile OR (for example, a .cpp file)
 
```
 
```
#ifdef __CUDACC__
// Compiled with nvcc (Cuda language extensions enabled) 
```
 
hcc directly generates the host code (using the Clang x86 target) and passes the code to another host compiler. Thus, it lacks the equivalent of the \__CUDA_ACC define.
 
The macro `__HIPCC__` is set if either `__HCC__` or `__CUDACC__` is defined. This configuration is useful in determining when code is being compiled using an accelerator-enabled compiler (hcc or nvcc) as opposed to a standard host compiler (GCC, ICC, Clang, etc.).
 
### Identifying Current Compilation Pass: Host or Device
 
Both nvcc and hcc make two passes over the code: one for host code and one for device code. `__HIP_DEVICE_COMPILE__` is set to a nonzero value when the compiler (hcc or nvcc) is compiling code for a device inside a `__global__` kernel or for a device function. `__HIP_DEVICE_COMPILE__` can replace #ifdef checks on the `__CUDA_ARCH__` define.
 
```
// #ifdef __CUDA_ARCH__  
#ifdef __HIP_DEVICE_COMPILE__ 
```
 
Unlike `__CUDA_ARCH__`, the `__HIP_DEVICE_COMPILE__` value is 0 or 1, and it doesnt represent the feature capability of the target device.  


### Compiler Defines: Summary
|Define  		|  hcc      | nvcc 		|  Other (GCC, ICC, Clang, etc.) 
|--- | --- | --- |---|
|HIP-related defines:|
|`__HIP_PLATFORM_HCC___`| Defined | Undefined |  Defined if targeting hcc platform; undefined otherwise |
|`__HIP_PLATFORM_NVCC___`| Undefined | Defined |  Defined if targeting nvcc platform; undefined otherwise |
|`__HIP_DEVICE_COMPILE__`     | 1 if compiling for device; 0 if compiling for host  |1 if compiling for device; 0 if compiling for host  | Undefined 
|`__HIPCC__`		| Defined   | Defined 		|  Undefined
|`__HIP_ARCH_*` | 0 or 1 depending on feature support (see below) | 0 or 1 depending on feature support (see below) | 0 
|nvcc-related defines:|
|`__CUDACC__` 		| Undefined | Defined if compiling for Cuda device; undefined otherwise 		|  Undefined
|`__NVCC__` 		| Undefined | Defined 		|  Undefined
|`__CUDA_ARCH__`		| Undefined | Unsigned representing compute capability (e.g., "130") if in device code; 0 if in host code  | Undefined 
|hcc-related defines:|
|`__HCC__`  		| Defined   | Undefined   	|  Undefined
|`__HCC_ACCELERATOR__`  	| Nonzero if in device code; otherwise undefined | Undefined | Undefined 
|`__clang__`		| Defined   | Undefined 	|  Defined if using Clang; otherwise undefined


## Identifying Architecture Features

### HIP_ARCH Defines

Some Cuda code tests `__CUDA_ARCH__` for a specific value to determine whether the machine supports a certain architectural feature. For instance,

```
#if (__CUDA_ARCH__ >= 130) 
// doubles are supported
```
This type of code requires special attention, since hcc/AMD and nvcc/Cuda devices have different architectural capabilities. Moreover, you cant determine the presence of a feature using a simple comparison against an architectures version number. HIP provides a set of defines and device properties to query whether a specific architectural feature is supported.

The `__HIP_ARCH_*` defines can replace comparisons of `__CUDA_ARCH__` values: 
```
//#if (__CUDA_ARCH__ >= 130)   // non-portable
if __HIP_ARCH_HAS_DOUBLES__ {  // portable HIP feature query
   // doubles are supported
}
```

For host code, the `__HIP_ARCH__*` defines are set to 0. You should only use the __HIP_ARCH__ fields in device code.

### Device-Architecture Properties

Host code should query the architecture feature flags in the device properties that hipGetDeviceProperties returns, rather than testing the "major" and "minor" fields directly:

```
hipGetDeviceProperties(&deviceProp, device);
//if ((deviceProp.major == 1 && deviceProp.minor < 2))  // non-portable
if (deviceProp.arch.hasSharedInt32Atomics) {            // portable HIP feature query
    // has shared int32 atomic operations ...
}
```

### Table of Architecture Properties
The table below shows the full set of architectural properties that HIP supports.

|Define (use only in device code) | Device Property (run-time query) | Comment |
|------- | ---------   | -----   |
|32-bit atomics:||
|`__HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__`        |    hasGlobalInt32Atomics      |32-bit integer atomics for global memory
|`__HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__`    |    hasGlobalFloatAtomicExch   |32-bit float atomic exchange for global memory
|`__HIP_ARCH_HAS_SHARED_INT32_ATOMICS__`        |    hasSharedInt32Atomics      |32-bit integer atomics for shared memory
|`__HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__`    |    hasSharedFloatAtomicExch   |32-bit float atomic exchange for shared memory
|`__HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__`            |     hasFloatAtomicAdd         |32-bit float atomic add in global and shared memory
|64-bit atomics:                           |                                     |                                                     
|`__HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__`        |    hasGlobalInt64Atomics      |64-bit integer atomics for global memory                                                 
|`__HIP_ARCH_HAS_SHARED_INT64_ATOMICS__`        |    hasSharedInt64Atomics      |64-bit integer atomics for shared memory
|Doubles:                                  |                                     |
|`__HIP_ARCH_HAS_DOUBLES__`                    |     hasDoubles                 |Double-precision floating point                                                     
|Warp cross-lane operations:              |                                      |                                                     
|`__HIP_ARCH_HAS_WARP_VOTE__`                  |     hasWarpVote                |Warp vote instructions (any, all)                                                      
|`__HIP_ARCH_HAS_WARP_BALLOT__`                |     hasWarpBallot              |Warp ballot instructions 
|`__HIP_ARCH_HAS_WARP_SHUFFLE__`               |     hasWarpShuffle             |Warp shuffle operations (shfl\_\*)                                                     
|`__HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__`          |     hasFunnelShift             |Funnel shift two input words into one
|Sync:                                    |                                      |
|`__HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__`        |     hasThreadFenceSystem       |threadfence\_system
|`__HIP_ARCH_HAS_SYNC_THREAD_EXT__`            |     hasSyncThreadsExt         |syncthreads\_count, syncthreads\_and, syncthreads\_or 
|                                                       
|Miscellaneous:                                     |                                     |                                                     
|`__HIP_ARCH_HAS_SURFACE_FUNCS__`              |   hasSurfaceFuncs              |                                                     
|`__HIP_ARCH_HAS_3DGRID__`                     |   has3dGrid                    | Grids and groups are 3D                                                     
|`__HIP_ARCH_HAS_DYNAMIC_PARALLEL__`           |   hasDynamicParallelism        | 
                                                                                 

## Finding HIP

Makefiles can use the following syntax to conditionally provide a default HIP_PATH if one does not exist:

```
HIP_PATH ?= $(shell hipconfig --path)
```

## Compiler Options

hipcc is a portable compiler driver that will call nvcc or hcc (depending on the target system) and attach all required include and library options. It passes options through to the target compiler. Tools that call hipcc must ensure the compiler options are appropriate for the target compiler. The `hipconfig` script may helpful in making
infrastructure that identifies the target platform and sets options appropriately. It returns either "nvcc" or "hcc." The following sample shows the script in a makefile:

```
HIP_PLATFORM=$(shell hipconfig --compiler)

ifeq (${HIP_PLATFORM}, nvcc)
	HIPCC_FLAGS = -gencode=arch=compute_20,code=sm_20 
endif
ifeq (${HIP_PLATFORM}, hcc)
	HIPCC_FLAGS = -Wno-deprecated-register
endif

```


## Linking Issues

### Linking With hipcc

hipcc adds the necessary libraries for HIP as well as for the accelerator compiler (nvcc or hcc). We recommend linking with hipcc.

### -lm Option
 
hipcc adds -lm by default to the link command.


## Linking Code With Other Compilers

Cuda code often uses nvcc for accelerator code (defining and launching kernels, typically defined in .cu or .cuh files). It also uses a standard compiler (G++) for the rest of the application. nvcc is a preprocessor that employs a standard host compiler (e.g., GCC) to generate the host code. Code compiled using this tool can employ the intersection of language features supported by both nvcc and the host compiler. In some cases, you must take care to ensure the data types and alignment of the host compiler are identical to those of the device compiler. Only some host compilers are supported---for example, recent nvcc versions lack Clang host-compiler capability.  

hcc generates both device and host code using the same Clang-based compiler. The code uses the same API as GCC, which allows code generated by different GCC-compatible compilers to be linked together. For example, code compiled using hcc can link with code compiled using "standard" compilers (such as GCC, ICC and Clang). You must take care to ensure all compilers use the same standard C++ header and library formats.


### libc++ and libstdc++

hcc uses the LLVM "libc++" standard library, whereas GCC uses the "libstdc++" standard library. Generally, libc++ provides a broader set of C++ features; libstdc++ is the standard for more compilers.

When cross-linking C++ code, any C++ functions that use types from the C++ standard library (including std::string, std::vector and other containers) must use the same standard-library implementation. They include the following:  

- Functions or kernels defined in hcc that are called from a standard compiler
- Functions defined in a standard compiler that are called from hcc.

Note that C++ code that doesnt use the standard library, or pure-C code, can be linked across compiler boundaries. The following suggestions may help you to meet these requirements:

- Use the libc++ header and library for standard compilers. For GCC you can do so by passing the following:
    - `g++ -std=c++11 -I/usr/include/c++/v1 `; note that the host code will compile as C++ 11 code.
- Use hipcc to compile the entire application 
- Partition the code into separate files so that standard C++ types dont apply across compiler boundaries. For example, placing the kernels in a separate file compiled using hcc will resolve the issue (provided they avoid using types from the C++ library).

A future hcc version may support libstdc++ in addition to libc++.


### HIP Headers (hip_runtime.h, hip_runtime_api.h)

The hip_runtime.h and hip_runtime_api.h files define the types, functions and enumerations needed to compile a HIP program:

- hip_runtime_api.h: defines all the HIP runtime APIs (e.g., hipMalloc) and the types required to call them. A source file that is only calling HIP APIs but neither defines nor launches any kernels can include hip_runtime_api.h. hip_runtime_api.h uses no custom C++ features and can be compiled using a standard C++ compiler.
- hip_runtime.h: included in hip_runtime_api.h. It additionally provides the types and defines required to create and launch kernels. hip_runtime.h does use custom C++ features, but they are guarded by ifdef checks. It can be compiled using a standard C++ compiler but will expose a subset of the available functions.

Cuda has slightly different contents for these two files. In some cases you may need to convert hipified code to include the richer hip_runtime.h instead of hip_runtime_api.h.

### Using a Standard C++ Compiler
You can compile hip\_runtime\_api.h using a standard C or C++ compiler (e.g., GCC or ICC). The HIP include paths and defines (`__HIP_PLATFORM_HCC__` or `__HIP_PLATFORM_NVCC__`) must pass to the standard compiler; hipconfig then returns the necessary options:
```
> hipconfig --cxx_config 
 -D__HIP_PLATFORM_HCC__ -I/home/user1/hip/include
```

You can capture the hipconfig output and passed it to the standard compiler; below is a sample makefile syntax:

```
CPPFLAGS += $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)
```

nvcc includes some headers by default. Files that call HIP run-time APIs or define HIP kernels must explicitly include HIP headers. If the compilation process reports that it cannot find necessary APIs (for example, "error: identifier hipSetDevice is undefined"),
ensure that the file includes hip_runtime.h (or hip_runtime_api.h, if appropriate). The hipify script automatically converts "cuda_runtime.h" to "hip_runtime.h," and it converts "cuda_runtime_api.h" to "hip_runtime_api.h", but it may miss nested headers or macros.  

#### cuda.h

The hcc path provides an empty cuda.h file. Some existing Cuda programs include this file but don't require any of the functions.

### Choosing HIP File Extensions

Many existing Cuda projects use the ".cu" and ".cuh" file extensions to indicate code that should be run through the nvcc compiler. 
For quick HIP ports, leaving these file extensions unchanged is often easier, as it minimizes the work required to change file names in the directory and #include statements in the files.

For new projects or ports which can be re-factored, we recommend the use of the extension ".hip.cpp" for header files, and
".hip.h" or ".hip.hpp" for header files.
This indicates that the code is standard C++ code, but also provides a unique indication for make tools to
run hipcc when appropriate.

### Workarounds

#### warpSize
Code should not assume a warp size of 32 or 64.  See [Warp Cross-Lane Functions](hip_kernel_language.md#warp-cross-lane-functions) for information on how to write portable wave-aware code.


#### Textures and Cache Control

Compute programs sometimes use textures either to access dedicated texture caches or to use the texture-sampling hardware for interpolation and clamping. The former approach uses simple point samplers with linear interpolation, essentially only reading a single point. The latter approach uses the sampler hardware to interpolate and combine multiple
point samples. AMD hardware, as well as recent competing hardware,
has a unified texture/L1 cache, so it no longer has a dedicated texture cache. But the nvcc path often caches global loads in the L2 cache, and some programs may benefit from explicit control of the L1 cache contents.  We recommend the __ldg instruction for this purpose.

HIP currently lacks texture support; a future revision will add this capability. Also, AMD compilers currently load all data into both the L1 and L2 caches, so __ldg is treated as a no-op. 

We recommend the following for functional portability:

- For programs that use textures only to benefit from improved caching, use the __ldg instruction
- Alternatively, use conditional compilation (see [Identify HIP Target Platform](#identify-hip-target-platform)) 
   - For the `__HIP_PLATFORM_NVCC__` path, use the full texture path
   - For the `__HIP_PLATFORM_HCC__` path, pass an additional pointer to the kernel and reference it using regular device memory-load instructions rather than texture loads. Some applications may already take this step, since it allows experimentation with caching behavior.

``` 
texture<float, 1, cudaReadModeElementType> t_features;

void __global__ MyKernel(float *d_features /* pass pointer parameter, if not already available */...) 
{
    // ... 

#ifdef __HIP_PLATFORM_NVCC__
    float tval = tex1Dfetch(t_features,addr);
#else
    float tval = d_features[addr];
#endif
        
}

// Host code:
void myFunc () 
{
    // ...

#ifdef __HIP_PLATFORM_NVCC__
    cudaChannelFormatDesc chDesc0 = cudaCreateChannelDesc<float>();
    t_features.filterMode = cudaFilterModePoint;   
    t_features.normalized = false;
    t_features.channelDesc = chDesc0;

	cudaBindTexture(NULL, &t_features, d_features, &chDesc0, npoints*nfeatures*sizeof(float));
#endif

``` 


Cuda programs that employ sampler hardware must either wait for hcc texture support or use more-sophisticated workarounds.

## More Tips
### hcc CPU Mode
Recent hcc versions support CPU accelerator targets. This feature enables some interesting possibilities for HIP porting:

- hcc can run on any machine, including perhaps a cross-compiling environment on a machine also running nvcc
- Standard CPU debuggers can debug CPU code
- A single code path can run on an AMD or Nvidia GPU or CPU, but the CPU accelerator is a low-performance target---its just a single core and lacks SIMD acceleration  

### HIPTRACE Mode

On an hcc/AMD platform, set the HIP_TRACE_API environment variable to see a textural API trace. Use the following bit mask:
    
- 0x1 = trace APIs
- 0x2 = trace synchronization operations
- 0x4 = trace memory allocation / deallocation

### Environment Variables

On hcc/AMD platforms, set the HIP_PRINT_ENV environment variable to 1 and run an application that calls a HIP API to see all HIP-supported environment variables and their current values:

- HIP_PRINT_ENV = 1: print HIP environment variables
- HIP_TRACE_API = 0: trace each HIP API call. Print the function name and return code to stderr as the program executes.
- HIP_LAUNCH_BLOCKING = 0: make HIP APIs host-synchronous so they are blocked until any kernel launches or data-copy commands are complete (an alias is CUDA_LAUNCH_BLOCKING)



### Debugging hipcc
To see the detailed commands that hipcc issues, set the environment variable HIPCC_VERBOSE to 1. Doing so will print to stderr the hcc (or nvcc) commands that hipcc generates. 

```
export HIPCC_VERBOSE=1
make
...
hipcc-cmd: /opt/hcc/bin/hcc  -hc -I/opt/hcc/include -stdlib=libc++ -I../../../../hc/include -I../../../../include/hcc_detail/cuda -I../../../../include -x c++ -I../../common -O3 -c backprop_cuda.cu
```

### What Does This Error Mean?

#### /usr/include/c++/v1/memory:5172:15: error: call to implicitly deleted default constructor of 'std::__1::bad_weak_ptr' throw bad_weak_ptr();

If you pass a ".cu" file, hcc will attempt to compile it as a Cuda language file. You must tell hcc that its in fact a C++ file: use the "-x c++" option.


#### grid_launch kernel dispatch - fallback
HIP uses an hcc language feature called "grid_launch". The [[hc_grid_launch]] attribute that can be attached to a function definition, and the first parameter is of type grid_launch_parm.
When a [[hc_grid_launch]] function is called, hcc runtime uses the grid_launch_parm to control the execution configuration of the kernel 
(including the grid and group dimensions, the queue, and dynamic group memory allocations).   By default, the hipLaunchKernel macro creates a grid_launch_parm structure and launches a
[[hc_grid_launch]] kernel.  grid_launch is a relatively new addition to hcc so this section describes how to fall back to a traditional calling sequence which invokes a standard host function
which calls a hc::parallel_for_each to launch the kernel.  

First, set DISABLE_GRID_LAUNCH:
include/hip_common.h
```
// Set this define to disable GRID_LAUNCH
#define DISABLE_GRID_LAUNCH
```

Inside any kernel use the KERNELBEGIN as the first line in the kernel function, and KERNELEND as the last line.  For example:
```
__global__ void
MyKernel(hipLaunchParm lp, float *C, const float *A, size_t N)
{
    KERNELBEGIN; // Required if hc_grid_launch is disabled

	int tid = hipBlockIdx_x*MAX_THREADS_PER_BLOCK + hipThreadIdx_x;

    if (tid < N) {
        C[tid] = A[tid];
    }

    KERNELEND; // Required if hc_grid_launch is disabled
}
```

#### HIP Environment Variables

On the HCC path, HIP provides a number of environment variables that control the behavior of HIP.  Some of these are useful for appliction development (for example HIP_VISIBLE_DEVICES, HIP_LAUNCH_BLOCKING),
some are useful for performance tuning or experimentation (for example HIP_STAGING*), and some are useful for debugging (HIP_DB).  You can see the environment variables supported by HIP as well as
their current values and usage with the environment var "HIP_PRINT_ENV" - set this and then run any HIP application.  For example:

```
$ HIP_PRINT_ENV=1 ./myhipapp
HIP_PRINT_ENV                  =  1 : Print HIP environment variables.
HIP_LAUNCH_BLOCKING            =  0 : Make HIP APIs 'host-synchronous', so they block until any kernel launches or data copy commands complete. Alias: CUDA_LAUNCH_BLOCKING.
HIP_DB                         =  0 : Print various debug info.  Bitmask, see hip_hcc.cpp for more information.
HIP_TRACE_API                  =  0 : Trace each HIP API call.  Print function name and return code to stderr as program executes.
HIP_STAGING_SIZE               = 64 : Size of each staging buffer (in KB)
HIP_STAGING_BUFFERS            =  2 : Number of staging buffers to use in each direction. 0=use hsa_memory_copy.
HIP_PININPLACE                 =  0 : For unpinned transfers, pin the memory in-place in chunks before doing the copy.  Under development.
HIP_STREAM_SIGNALS             =  2 : Number of signals to allocate when new stream is created (signal pool will grow on demand)
HIP_VISIBLE_DEVICES            =  0 : Only devices whose index is present in the secquence are visible to HIP applications and they are enumerated in the order of secquence
HIP_DISABLE_HW_KERNEL_DEP      =  1 : Disable HW dependencies before kernel commands  - instead wait for dependency on host. -1 means ignore these dependencies. (debug mode)
HIP_DISABLE_HW_COPY_DEP        =  1 : Disable HW dependencies before copy commands  - instead wait for dependency on host. -1 means ifnore these dependencies (debug mode)

```


#### Editor Highlighting
See the utils/vim or utils/gedit directories to add handy highlighting to hip files.
