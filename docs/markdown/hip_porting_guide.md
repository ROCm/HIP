# HIP Porting Guide 
In addition to providing a portable C++ programming environment for GPUs, HIP is designed to ease
the porting of existing CUDA code into the HIP environment.  This section describes the available tools 
and provides practical suggestions on how to port CUDA code and work through common issues. 

## Table of Contents

<!-- toc -->

- [Porting a New Cuda Project](#porting-a-new-cuda-project)
  * [General Tips](#general-tips)
  * [Scanning existing CUDA code to scope the porting effort](#scanning-existing-cuda-code-to-scope-the-porting-effort)
  * [Converting a project "in-place"](#converting-a-project-in-place)
- [Distinguishing Compiler Modes](#distinguishing-compiler-modes)
  * [Identifying HIP Target Platform](#identifying-hip-target-platform)
  * [Identifying the Compiler: hcc, hip-clang, or nvcc](#identifying-the-compiler-hcc-hip-clang-or-nvcc)
  * [Identifying Current Compilation Pass: Host or Device](#identifying-current-compilation-pass-host-or-device)
  * [Compiler Defines: Summary](#compiler-defines-summary)
- [Identifying Architecture Features](#identifying-architecture-features)
  * [HIP_ARCH Defines](#hip_arch-defines)
  * [Device-Architecture Properties](#device-architecture-properties)
  * [Table of Architecture Properties](#table-of-architecture-properties)
- [Finding HIP](#finding-hip)
- [hipLaunchKernel](#hiplaunchkernel)
- [Compiler Options](#compiler-options)
- [Linking Issues](#linking-issues)
  * [Linking With hipcc](#linking-with-hipcc)
  * [-lm Option](#-lm-option)
- [Linking Code With Other Compilers](#linking-code-with-other-compilers)
  * [libc++ and libstdc++](#libc-and-libstdc)
  * [HIP Headers (hip_runtime.h, hip_runtime_api.h)](#hip-headers-hip_runtimeh-hip_runtime_apih)
  * [Using a Standard C++ Compiler](#using-a-standard-c-compiler)
    + [cuda.h](#cudah)
  * [Choosing HIP File Extensions](#choosing-hip-file-extensions)
- [Workarounds](#workarounds)
  * [warpSize](#warpsize)
- [memcpyToSymbol](#memcpytosymbol)
- [threadfence_system](#threadfence_system)
  * [Textures and Cache Control](#textures-and-cache-control)
- [More Tips](#more-tips)
  * [HIPTRACE Mode](#hiptrace-mode)
  * [Environment Variables](#environment-variables)
  * [Debugging hipcc](#debugging-hipcc)
  * [What Does This Error Mean?](#what-does-this-error-mean)
    + [/usr/include/c++/v1/memory:5172:15: error: call to implicitly deleted default constructor of 'std::__1::bad_weak_ptr' throw bad_weak_ptr();](#usrincludecv1memory517215-error-call-to-implicitly-deleted-default-constructor-of-std__1bad_weak_ptr-throw-bad_weak_ptr)
  * [HIP Environment Variables](#hip-environment-variables)
  * [Editor Highlighting](#editor-highlighting)
  * [CUDA to HIP Math Library Equivalents](#library-equivalents)
  

<!-- tocstop -->

## Porting a New Cuda Project

### General Tips
- Starting the port on a Cuda machine is often the easiest approach, since you can incrementally port pieces of the code to HIP while leaving the rest in Cuda. (Recall that on Cuda machines HIP is just a thin layer over Cuda, so the two code types can interoperate on nvcc platforms.) Also, the HIP port can be compared with the original Cuda code for function and performance.
- Once the Cuda code is ported to HIP and is running on the Cuda machine, compile the HIP code using hcc on an AMD machine.
- HIP ports can replace Cuda versions: HIP can deliver the same performance as a native Cuda implementation, with the benefit of portability to both Nvidia and AMD architectures as well as a path to future C++ standard support. You can handle platform-specific features through conditional compilation or by adding them to the open-source HIP infrastructure.
- Use **[bin/hipconvertinplace.sh](https://github.com/ROCm-Developer-Tools/HIP/blob/master/bin/hipconvertinplace.sh)** to hipify all code files in the Cuda source directory.

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

hipexamine scans each code file (cpp, c, h, hpp, etc.) found in the specified directory:

   * Files with no CUDA code (ie kmeans.h) print one line summary just listing the source file name.
   * Files with CUDA code print a summary of what was found - for example the kmeans_cuda_kernel.cu file:
```
info: hipify ./kmeans_cuda_kernel.cu =====>
  info: converted 40 CUDA->HIP refs( dev:0 mem:0 kern:0 builtin:37 math:0 stream:0 event:0 
```
* Interesting information in kmeans_cuda_kernel.cu :
  * How many CUDA calls were converted to HIP (40)
  * Breakdown of the CUDA functionality used (dev:0 mem:0 etc). This file uses many CUDA builtins (37) and texture functions (3).
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


The [hipconvertinplace.sh](https://github.com/ROCm-Developer-Tools/HIP/blob/master/bin/hipconvertinplace.sh) script will perform inplace conversion for all code files in the specified directory.
This can be quite handy when dealing with an existing CUDA code base since the script preserves the existing directory structure
and filenames - and includes work.  After converting in-place, you can review the code to add additional parameters to
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
 
 
### Identifying the Compiler: hcc, hip-clang or nvcc
Often, it's useful to know whether the underlying compiler is hcc, hip-clang or nvcc. This knowledge can guard platform-specific code (features that only work on the nvcc, hip-clang or hcc path but not all) or aid in platform-specific performance tuning.   
 
```
#ifdef __HCC__
// Compiled with hcc 
 
```
```
#ifdef __HIP__
// Compiled with hip-clang 
 
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
 
hcc and hip-clang directly generates the host code (using the Clang x86 target) and passes the code to another host compiler. Thus, they have no equivalent of the \__CUDA_ACC define.
 
The macro `__HIPCC__` is set if either `__HCC__`, `__HIP__` or `__CUDACC__` is defined. This configuration is useful in determining when code is being compiled using an accelerator-enabled compiler (hcc or nvcc) as opposed to a standard host compiler (GCC, ICC, Clang, etc.).
 
### Identifying Current Compilation Pass: Host or Device
 
Both nvcc and hcc make two passes over the code: one for host code and one for device code. `__HIP_DEVICE_COMPILE__` is set to a nonzero value when the compiler (hcc or nvcc) is compiling code for a device inside a `__global__` kernel or for a device function. `__HIP_DEVICE_COMPILE__` can replace #ifdef checks on the `__CUDA_ARCH__` define.
 
```
// #ifdef __CUDA_ARCH__  
#if __HIP_DEVICE_COMPILE__
```
 
Unlike `__CUDA_ARCH__`, the `__HIP_DEVICE_COMPILE__` value is 1 or undefined, and it doesn't represent the feature capability of the target device.  


### Compiler Defines: Summary
|Define  		|  hcc      |  hip-clang  | nvcc 		|  Other (GCC, ICC, Clang, etc.) 
|--- | --- | --- | --- |---|
|HIP-related defines:|
|`__HIP_PLATFORM_HCC___`| Defined | Defined | Undefined |  Defined if targeting hcc platform; undefined otherwise |
|`__HIP_PLATFORM_NVCC___`| Undefined | Undefined | Defined |  Defined if targeting nvcc platform; undefined otherwise |
|`__HIP_DEVICE_COMPILE__`     | 1 if compiling for device; undefined if compiling for host  | 1 if compiling for device; undefined if compiling for host  |1 if compiling for device; undefined if compiling for host  | Undefined 
|`__HIPCC__`		| Defined   | Defined | Defined 		|  Undefined
|`__HIP_ARCH_*` | 0 or 1 depending on feature support (see below) |0 or 1 depending on feature support (see below) | 0 or 1 depending on feature support (see below) | 0 
|nvcc-related defines:|
|`__CUDACC__` 		| Undefined | Undefined | Defined if source code is compiled by nvcc; undefined otherwise 		|  Undefined
|`__NVCC__` 		| Undefined | Undefined | Defined 		|  Undefined
|`__CUDA_ARCH__`		| Undefined | Undefined | Unsigned representing compute capability (e.g., "130") if in device code; 0 if in host code  | Undefined 
|hcc-related defines:|
|`__HCC__`  		| Defined   | Undefined | Undefined   	|  Undefined
|`__HCC_ACCELERATOR__`  	| Nonzero if in device code; otherwise undefined | Undefined | Undefined | Undefined 
|hip-clang-related defines:|
|`__HIP__`  		| Undefined | Defined   | Undefined   	|  Undefined
|hcc/hip-clang common defines:|
|`__clang__`		| Defined   | Defined | Undefined 	|  Defined if using Clang; otherwise undefined


## Identifying Architecture Features

### HIP_ARCH Defines

Some Cuda code tests `__CUDA_ARCH__` for a specific value to determine whether the machine supports a certain architectural feature. For instance,

```
#if (__CUDA_ARCH__ >= 130) 
// doubles are supported
```
This type of code requires special attention, since hcc/AMD and nvcc/Cuda devices have different architectural capabilities. Moreover, you can't determine the presence of a feature using a simple comparison against an architecture's version number. HIP provides a set of defines and device properties to query whether a specific architectural feature is supported.

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
|Miscellaneous:                                |                                |                                                     
|`__HIP_ARCH_HAS_SURFACE_FUNCS__`              |   hasSurfaceFuncs              |                                                     
|`__HIP_ARCH_HAS_3DGRID__`                     |   has3dGrid                    | Grids and groups are 3D                                                     
|`__HIP_ARCH_HAS_DYNAMIC_PARALLEL__`           |   hasDynamicParallelism        | 
                                                                                 

## Finding HIP

Makefiles can use the following syntax to conditionally provide a default HIP_PATH if one does not exist:

```
HIP_PATH ?= $(shell hipconfig --path)
```

## hipLaunchKernel 

hipLaunchKernel is a variadic macro which accepts as parameters the launch configurations (grid dims, group dims, stream, dynamic shared size) followed by a variable number of kernel arguments.
This sequence is then expanded into the appropriate kernel launch syntax depending on the platform.  
While this can be a convenient single-line kernel launch syntax, the macro implementation can cause issues when nested inside other macros.  For example, consider the following:

```
// Will cause compile error:
#define MY_LAUNCH(command, doTrace) \
{\
    if (doTrace) printf ("TRACE: %s\n", #command); \
    (command);   /* The nested ( ) will cause compile error */\
}

MY_LAUNCH (hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad), true, "firstCall");
```

Avoid nesting macro parameters inside parenthesis - here's an alternative that will work:

```
#define MY_LAUNCH(command, doTrace) \
{\
    if (doTrace) printf ("TRACE: %s\n", #command); \
    command;\ 
}

MY_LAUNCH (hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad), true, "firstCall");
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

Cuda code often uses nvcc for accelerator code (defining and launching kernels, typically defined in .cu or .cuh files). 
It also uses a standard compiler (g++) for the rest of the application. nvcc is a preprocessor that employs a standard host compiler (e.g., gcc) to generate the host code. 
Code compiled using this tool can employ only the intersection of language features supported by both nvcc and the host compiler. 
In some cases, you must take care to ensure the data types and alignment of the host compiler are identical to those of the device compiler. Only some host compilers are supported---for example, recent nvcc versions lack Clang host-compiler capability.  

hcc generates both device and host code using the same Clang-based compiler. The code uses the same API as gcc, which allows code generated by different gcc-compatible compilers to be linked together. For example, code compiled using hcc can link with code compiled using "standard" compilers (such as gcc, ICC and Clang). Take care to ensure all compilers use the same standard C++ header and library formats.


### libc++ and libstdc++

Version 0.86 of hipcc now uses libstdc++ by default for the HCC platform.  This improves cross-linking support between G++ and hcc, in particular for interfaces that use
  standard C++ libraries (ie std::vector, std::string). 

If you pass "--stdlib=libc++" to hipcc, hipcc will use the libc++ library.  Generally, libc++ provides a broader set of C++ features while libstdc++ is the standard 
for more compilers (notably including g++).    

When cross-linking C++ code, any C++ functions that use types from the C++ standard library (including std::string, std::vector and other containers) must use the same standard-library implementation. They include the following:  

- Functions or kernels defined in hcc that are called from a standard compiler
- Functions defined in a standard compiler that are called from hcc.

Applications with these interfaces should use the default libstdc++ linking.    

Applications which are compiled entirely with hipcc, and which benefit from advanced C++ features not supported in libstdc++, and which do not require portability to nvcc, may choose to use libc++.


### HIP Headers (hip_runtime.h, hip_runtime_api.h)

The hip_runtime.h and hip_runtime_api.h files define the types, functions and enumerations needed to compile a HIP program:

- hip_runtime_api.h: defines all the HIP runtime APIs (e.g., hipMalloc) and the types required to call them. A source file that is only calling HIP APIs but neither defines nor launches any kernels can include hip_runtime_api.h. hip_runtime_api.h uses no custom hc language features and can be compiled using a standard C++ compiler.
- hip_runtime.h: included in hip_runtime_api.h. It additionally provides the types and defines required to create and launch kernels. hip_runtime.h does use custom hc language features, but they are guarded by ifdef checks. It can be compiled using a standard C++ compiler but will expose a subset of the available functions.

Cuda has slightly different contents for these two files. In some cases you may need to convert hipified code to include the richer hip_runtime.h instead of hip_runtime_api.h.

### Using a Standard C++ Compiler
You can compile hip\_runtime\_api.h using a standard C or C++ compiler (e.g., gcc or ICC). The HIP include paths and defines (`__HIP_PLATFORM_HCC__` or `__HIP_PLATFORM_NVCC__`) must pass to the standard compiler; hipconfig then returns the necessary options:
```
> hipconfig --cxx_config 
 -D__HIP_PLATFORM_HCC__ -I/home/user1/hip/include
```

You can capture the hipconfig output and passed it to the standard compiler; below is a sample makefile syntax:

```
CPPFLAGS += $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)
```

nvcc includes some headers by default.  However, HIP does not include default headers, and instead all required files must be explicitly included.  
Specifically, files that call HIP run-time APIs or define HIP kernels must explicitly include the appropriate HIP headers. 
If the compilation process reports that it cannot find necessary APIs (for example, "error: identifier hipSetDevice is undefined"),
ensure that the file includes hip_runtime.h (or hip_runtime_api.h, if appropriate). 
The hipify script automatically converts "cuda_runtime.h" to "hip_runtime.h," and it converts "cuda_runtime_api.h" to "hip_runtime_api.h", but it may miss nested headers or macros.  

#### cuda.h

The hcc path provides an empty cuda.h file. Some existing Cuda programs include this file but don't require any of the functions.

### Choosing HIP File Extensions

Many existing Cuda projects use the ".cu" and ".cuh" file extensions to indicate code that should be run through the nvcc compiler. 
For quick HIP ports, leaving these file extensions unchanged is often easier, as it minimizes the work required to change file names in the directory and #include statements in the files.

For new projects or ports which can be re-factored, we recommend the use of the extension ".hip.cpp" for source files, and
".hip.h" or ".hip.hpp" for header files.
This indicates that the code is standard C++ code, but also provides a unique indication for make tools to
run hipcc when appropriate.

## Workarounds

### warpSize
Code should not assume a warp size of 32 or 64.  See [Warp Cross-Lane Functions](hip_kernel_language.md#warp-cross-lane-functions) for information on how to write portable wave-aware code.

## memcpyToSymbol

HIP support for hipMemcpyToSymbol is complete.  This feature allows a kernel
to define a device-side data symbol which can be accessed on the host side.  The symbol
can be in __constant or device space.

For example:

Device Code:
```
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include<iostream>

#define HIP_ASSERT(status) \
    assert(status == hipSuccess)

#define LEN 512
#define SIZE 2048

__constant__ int Value[LEN];

__global__ void Get(hipLaunchParm lp, int *Ad)
{
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    Ad[tid] = Value[tid];
}

int main()
{
    int *A, *B, *Ad;
    A = new int[LEN];
    B = new int[LEN];
    for(unsigned i=0;i<LEN;i++)
    {
        A[i] = -1*i;
        B[i] = 0;
    }

    HIP_ASSERT(hipMalloc((void**)&Ad, SIZE));

    HIP_ASSERT(hipMemcpyToSymbol(HIP_SYMBOL(Value), A, SIZE, 0, hipMemcpyHostToDevice));
    hipLaunchKernel(Get, dim3(1,1,1), dim3(LEN,1,1), 0, 0, Ad);
    HIP_ASSERT(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));

    for(unsigned i=0;i<LEN;i++)
    {
        assert(A[i] == B[i]);
    }
    std::cout<<"Passed"<<std::endl;
}
```
 
## threadfence_system
Threadfence_system makes all device memory writes, all writes to mapped host memory, and all writes to peer memory visible to CPU and other GPU devices.
Some implementations can provide this behavior by flushing the GPU L2 cache.
HIP/HCC does not provide this functionality.  As a workaround, users can set the environment variable `HSA_DISABLE_CACHE=1` to 
disable the GPU L2 cache. This will affect all accesses and for all kernels and so may have 
a performance impact.

### Textures and Cache Control

Compute programs sometimes use textures either to access dedicated texture caches or to use the texture-sampling hardware for interpolation and clamping. The former approach uses simple point samplers with linear interpolation, essentially only reading a single point. The latter approach uses the sampler hardware to interpolate and combine multiple
point samples. AMD hardware, as well as recent competing hardware,
has a unified texture/L1 cache, so it no longer has a dedicated texture cache. But the nvcc path often caches global loads in the L2 cache, and some programs may benefit from explicit control of the L1 cache contents.  We recommend the __ldg instruction for this purpose.

AMD compilers currently load all data into both the L1 and L2 caches, so __ldg is treated as a no-op. 

We recommend the following for functional portability:

- For programs that use textures only to benefit from improved caching, use the __ldg instruction
- Programs that use texture object APIs, work well on HIP
- For program that use texture reference APIs, use conditional compilation (see [Identify HIP Target Platform](#identify-hip-target-platform)) 
   - For the `__HIP_PLATFORM_HCC__` path, pass an additional argument to the kernel and in texture fetch API inside kernel as shown below:-

``` 
texture<float, 2, hipReadModeElementType> tex;

__global__ void tex2DKernel(float* outputData,
#ifdef __HIP_PLATFORM_HCC__
                             hipTextureObject_t textureObject,
#endif
                             int width,
                             int height)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
#ifdef __HIP_PLATFORM_HCC__
    outputData[y*width + x] = tex2D(tex, textureObject, x, y);
#else
    outputData[y*width + x] = tex2D(tex, x, y);
#endif
}

// Host code:
void myFunc () 
{
    // ...

#ifdef __HIP_PLATFORM_HCC__
    hipLaunchKernelGGL(tex2DKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, dData, tex.textureObject, width, height);
#else
    hipLaunchKernelGGL(tex2DKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, dData, width, height);
#endif


``` 

## More Tips
### HIPTRACE Mode

On an hcc/AMD platform, set the HIP_TRACE_API environment variable to see a textural API trace. Use the following bit mask:
    
- 0x1 = trace APIs
- 0x2 = trace synchronization operations
- 0x4 = trace memory allocation / deallocation

### Environment Variables

On hcc/AMD platforms, set the HIP_PRINT_ENV environment variable to 1 and run an application that calls a HIP API to see all HIP-supported environment variables and their current values:

- HIP_PRINT_ENV = 1: print HIP environment variables
- HIP_TRACE_API = 1: trace each HIP API call. Print the function name and return code to stderr as the program executes.
- HIP_LAUNCH_BLOCKING = 0: make HIP APIs host-synchronous so they are blocked until any kernel launches or data-copy commands are complete (an alias is CUDA_LAUNCH_BLOCKING)

- KMDUMPISA = 1 : Will dump the GCN ISA for all kernels into the local directory. (This flag is provided by HCC).


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

If you pass a ".cu" file, hcc will attempt to compile it as a Cuda language file. You must tell hcc that it's in fact a C++ file: use the "-x c++" option.


### HIP Environment Variables

On the HCC path, HIP provides a number of environment variables that control the behavior of HIP.  Some of these are useful for application development (for example HIP_VISIBLE_DEVICES, HIP_LAUNCH_BLOCKING),
some are useful for performance tuning or experimentation (for example HIP_STAGING*), and some are useful for debugging (HIP_DB).  You can see the environment variables supported by HIP as well as
their current values and usage with the environment var "HIP_PRINT_ENV" - set this and then run any HIP application.  For example:

```
$ HIP_PRINT_ENV=1 ./myhipapp
HIP_PRINT_ENV                  =  1 : Print HIP environment variables.
HIP_LAUNCH_BLOCKING            =  0 : Make HIP APIs 'host-synchronous', so they block until any kernel launches or data copy commands complete. Alias: CUDA_LAUNCH_BLOCKING.
HIP_DB                         =  0 : Print various debug info.  Bitmask, see hip_hcc.cpp for more information.
HIP_TRACE_API                  =  0 : Trace each HIP API call.  Print function name and return code to stderr as program executes.
HIP_TRACE_API_COLOR            = green : Color to use for HIP_API.  None/Red/Green/Yellow/Blue/Magenta/Cyan/White
HIP_PROFILE_API                 =  0 : Add HIP function begin/end to ATP file generated with CodeXL
HIP_VISIBLE_DEVICES            =  0 : Only devices whose index is present in the secquence are visible to HIP applications and they are enumerated in the order of secquence

```


### Editor Highlighting
See the utils/vim or utils/gedit directories to add handy highlighting to hip files.


### Library Equivalents

| CUDA Library | ROCm Library | Comment |
|------- | ---------   | -----   |
| cuBLAS        |    rocBLAS     | Basic Linear Algebra Subroutines 
| cuFFT        |    rocFFT     | Fast Fourier Transfer Library   
| cuSPARSE     |    rocSPARSE   | Sparse BLAS  + SPMV 
| cuSolver     |    rocSolver   | Lapack library
| AMG-X    |    rocALUTION   | Sparse iterative solvers and preconditioners with Geometric and Algebraic MultiGrid
| Thrust    |    hipThrust | C++ parallel algorithms library
| CUB     |    rocPRIM | Low Level Optimized Parallel Primitives
| cuDNN    |    MIOpen | Deep learning Solver Library 
| cuRAND    |    rocRAND | Random Number Generator Library
| EIGEN    |    EIGEN – HIP port | C++ template library for linear algebra: matrices, vectors, numerical solvers, 
| NCCL    |    RCCL  | Communications Primitives Library based on the MPI equivalents
