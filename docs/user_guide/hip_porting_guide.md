# HIP Porting Guide
In addition to providing a portable C++ programming environment for GPUs, HIP is designed to ease
the porting of existing CUDA code into the HIP environment.  This section describes the available tools
and provides practical suggestions on how to port CUDA code and work through common issues.

## Porting a New CUDA Project

### General Tips
- Starting the port on a CUDA machine is often the easiest approach, since you can incrementally port pieces of the code to HIP while leaving the rest in CUDA. (Recall that on CUDA machines HIP is just a thin layer over CUDA, so the two code types can interoperate on nvcc platforms.) Also, the HIP port can be compared with the original CUDA code for function and performance.
- Once the CUDA code is ported to HIP and is running on the CUDA machine, compile the HIP code using the HIP compiler on an AMD machine.
- HIP ports can replace CUDA versions: HIP can deliver the same performance as a native CUDA implementation, with the benefit of portability to both Nvidia and AMD architectures as well as a path to future C++ standard support. You can handle platform-specific features through conditional compilation or by adding them to the open-source HIP infrastructure.
- Use **[hipconvertinplace-perl.sh](https://github.com/ROCm/HIPIFY/blob/amd-staging/bin/hipconvertinplace-perl.sh)** to hipify all code files in the CUDA source directory.

### Scanning existing CUDA code to scope the porting effort
The **[hipexamine-perl.sh](https://github.com/ROCm/HIPIFY/blob/amd-staging/bin/hipexamine-perl.sh)** tool will scan a source directory to determine which files contain CUDA code and how much of that code can be automatically hipified.
```
> cd examples/rodinia_3.0/cuda/kmeans
> $HIP_DIR/bin/hipexamine-perl.sh.
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

hipexamine-perl scans each code file (cpp, c, h, hpp, etc.) found in the specified directory:

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

* hipexamine-perl also presents a summary at the end of the process for the statistics collected across all files. This has similar format to the per-file reporting, and also includes a list of all kernels which have been called.  An example from above:

```shell
info: TOTAL-converted 89 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:37 math:0 stream:0 event:0 err:0 def:0 tex:15 other:0 ) warn:0 LOC:3607
  kernels (1 total) :   kmeansPoint(1)
```

### Converting a project "in-place"

```shell
> hipify-perl --inplace
```

For each input file FILE, this script will:
  - If "FILE.prehip file does not exist, copy the original code to a new file with extension ".prehip". Then hipify the code file.
  - If "FILE.prehip" file exists, hipify FILE.prehip and save to FILE.

This is useful for testing improvements to the hipify toolset.


The [hipconvertinplace-perl.sh](https://github.com/ROCm/HIPIFY/blob/amd-staging/bin/hipconvertinplace-perl.sh) script will perform inplace conversion for all code files in the specified directory.
This can be quite handy when dealing with an existing CUDA code base since the script preserves the existing directory structure
and filenames - and includes work.  After converting in-place, you can review the code to add additional parameters to
directory names.


```shell
> hipconvertinplace-perl.sh MY_SRC_DIR
```

### Library Equivalents

| CUDA Library | ROCm Library | Comment |
|------- | ---------   | -----   |
| cuBLAS        |    rocBLAS     | Basic Linear Algebra Subroutines
| cuFFT        |    rocFFT     | Fast Fourier Transfer Library
| cuSPARSE     |    rocSPARSE   | Sparse BLAS  + SPMV
| cuSolver     |    rocSOLVER   | Lapack library
| AMG-X    |    rocALUTION   | Sparse iterative solvers and preconditioners with Geometric and Algebraic MultiGrid
| Thrust    |    rocThrust | C++ parallel algorithms library
| CUB     |    rocPRIM | Low Level Optimized Parallel Primitives
| cuDNN    |    MIOpen | Deep learning Solver Library
| cuRAND    |    rocRAND | Random Number Generator Library
| EIGEN    |    EIGEN - HIP port | C++ template library for linear algebra: matrices, vectors, numerical solvers,
| NCCL    |    RCCL  | Communications Primitives Library based on the MPI equivalents



## Distinguishing Compiler Modes


### Identifying HIP Target Platform
All HIP projects target either AMD or NVIDIA platform. The platform affects which headers are included and which libraries are used for linking.

- `HIP_PLATFORM_AMD` is defined if the HIP platform targets AMD.
Note, `HIP_PLATFORM_HCC` was previously defined if the HIP platform targeted AMD, it is deprecated.

- `HIP_PLATFORM_NVDIA` is defined if the HIP platform targets NVIDIA.
Note, `HIP_PLATFORM_NVCC` was previously defined if the HIP platform targeted NVIDIA, it is deprecated.

### Identifying the Compiler: hip-clang or nvcc
Often, it's useful to know whether the underlying compiler is HIP-Clang or nvcc. This knowledge can guard platform-specific code or aid in platform-specific performance tuning.

```
#ifdef __HIP_PLATFORM_AMD__
// Compiled with HIP-Clang
#endif
```

```
#ifdef __HIP_PLATFORM_NVIDIA__
// Compiled with nvcc
//  Could be compiling with CUDA language extensions enabled (for example, a ".cu file)
//  Could be in pass-through mode to an underlying host compile OR (for example, a .cpp file)

```

```
#ifdef __CUDACC__
// Compiled with nvcc (CUDA language extensions enabled)
```

Compiler directly generates the host code (using the Clang x86 target) and passes the code to another host compiler. Thus, they have no equivalent of the \__CUDA_ACC define.


### Identifying Current Compilation Pass: Host or Device

nvcc makes two passes over the code: one for host code and one for device code.
HIP-Clang will have multiple passes over the code: one for the host code, and one for each architecture on the device code.
`__HIP_DEVICE_COMPILE__` is set to a nonzero value when the compiler (HIP-Clang or nvcc) is compiling code for a device inside a `__global__` kernel or for a device function. `__HIP_DEVICE_COMPILE__` can replace #ifdef checks on the `__CUDA_ARCH__` define.

```
// #ifdef __CUDA_ARCH__
#if __HIP_DEVICE_COMPILE__
```

Unlike `__CUDA_ARCH__`, the `__HIP_DEVICE_COMPILE__` value is 1 or undefined, and it doesn't represent the feature capability of the target device.

### Compiler Defines: Summary
|Define  		|   HIP-Clang  | nvcc 		|  Other (GCC, ICC, Clang, etc.)
|--- | --- | --- |---|
|HIP-related defines:|
|`__HIP_PLATFORM_AMD__`| Defined | Undefined |  Defined if targeting AMD platform; undefined otherwise |
|`__HIP_PLATFORM_NVIDIA__`| Undefined  | Defined |  Defined if targeting NVIDIA platform; undefined otherwise |
|`__HIP_DEVICE_COMPILE__`     | 1 if compiling for device; undefined if compiling for host  |1 if compiling for device; undefined if compiling for host  | Undefined
|`__HIPCC__`		|  Defined | Defined 		|  Undefined
|`__HIP_ARCH_*` |0 or 1 depending on feature support (see below) | 0 or 1 depending on feature support (see below) | 0
|nvcc-related defines:|
|`__CUDACC__` 		 | Defined if source code is compiled by nvcc; undefined otherwise 		|  Undefined
|`__NVCC__` 		 | Undefined | Defined 		|  Undefined
|`__CUDA_ARCH__`		 | Undefined | Unsigned representing compute capability (e.g., "130") if in device code; 0 if in host code  | Undefined
|hip-clang-related defines:|
|`__HIP__`  		 | Defined   | Undefined   	|  Undefined
|HIP-Clang common defines:|
|`__clang__`		| Defined   | Defined | Undefined 	|  Defined if using Clang; otherwise undefined

## Identifying Architecture Features

### HIP_ARCH Defines

Some CUDA code tests `__CUDA_ARCH__` for a specific value to determine whether the machine supports a certain architectural feature. For instance,

```
#if (__CUDA_ARCH__ >= 130)
// doubles are supported
```
This type of code requires special attention, since AMD and CUDA devices have different architectural capabilities. Moreover, you can't determine the presence of a feature using a simple comparison against an architecture's version number. HIP provides a set of defines and device properties to query whether a specific architectural feature is supported.

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

## Identifying HIP Runtime

HIP can depend on rocclr, or cuda as runtime

- AMD platform
On AMD platform, HIP uses Radeon Open Compute Common Language Runtime, called ROCclr.
ROCclr is a virtual device interface that HIP runtimes interact with different backends which allows runtimes to work on Linux , as well as Windows without much efforts.

- NVIDIA platform
On Nvidia platform, HIP is just a thin layer on top of CUDA.
On non-AMD platform, HIP runtime determines if cuda is available and can be used. If available, HIP_PLATFORM is set to nvidia and underneath CUDA path is used.


## hipLaunchKernelGGL

hipLaunchKernelGGL is a macro that can serve as an alternative way to launch kernel, which accepts parameters of launch configurations (grid dims, group dims, stream, dynamic shared size) followed by a variable number of kernel arguments.
It can replace <<< >>>, if the user so desires.

## Compiler Options

hipcc is a portable compiler driver that will call nvcc or HIP-Clang (depending on the target system) and attach all required include and library options. It passes options through to the target compiler. Tools that call hipcc must ensure the compiler options are appropriate for the target compiler.
The `hipconfig` script may helpful in identifying the target platform, compiler and runtime. It can also help set options appropriately.

### Compiler options supported on AMD platforms

Here are the main compiler options supported on AMD platforms by HIP-Clang.

| Option                            | Description |
| ------                            | ----------- |
| --amdgpu-target=<gpu_arch>        | [DEPRECATED] This option is being replaced by `--offload-arch=<target>`. Generate code for the given GPU target.  Supported targets are gfx701, gfx801, gfx802, gfx803, gfx900, gfx906, gfx908, gfx1010, gfx1011, gfx1012, gfx1030, gfx1031.  This option could appear multiple times on the same command line to generate a fat binary for multiple targets. |
| --fgpu-rdc                        | Generate relocatable device code, which allows kernels or device functions calling device functions in different translation units. |
| -ggdb                             | Equivalent to `-g` plus tuning for GDB.  This is recommended when using ROCm's GDB to debug GPU code. |
| --gpu-max-threads-per-block=<num> | Generate code to support up to the specified number of threads per block.  |
| -O<n>                             | Specify the optimization level. |
| -offload-arch=<target>            | Specify the AMD GPU [target ID](https://clang.llvm.org/docs/ClangOffloadBundler.html#target-id). |
| -save-temps                       | Save the compiler generated intermediate files. |
| -v                                | Show the compilation steps. |

## Linking Issues

### Linking With hipcc

hipcc adds the necessary libraries for HIP as well as for the accelerator compiler (nvcc or AMD compiler). We recommend linking with hipcc since it automatically links the binary to the necessary HIP runtime libraries.  It also has knowledge on how to link and to manage the GPU objects.

### -lm Option

hipcc adds -lm by default to the link command.


## Linking Code With Other Compilers

CUDA code often uses nvcc for accelerator code (defining and launching kernels, typically defined in .cu or .cuh files).
It also uses a standard compiler (g++) for the rest of the application. nvcc is a preprocessor that employs a standard host compiler (gcc) to generate the host code.
Code compiled using this tool can employ only the intersection of language features supported by both nvcc and the host compiler.
In some cases, you must take care to ensure the data types and alignment of the host compiler are identical to those of the device compiler. Only some host compilers are supported---for example, recent nvcc versions lack Clang host-compiler capability.

HIP-Clang generates both device and host code using the same Clang-based compiler. The code uses the same API as gcc, which allows code generated by different gcc-compatible compilers to be linked together. For example, code compiled using HIP-Clang can link with code compiled using "standard" compilers (such as gcc, ICC and Clang). Take care to ensure all compilers use the same standard C++ header and library formats.


### libc++ and libstdc++

hipcc links to libstdc++ by default. This provides better compatibility between g++ and HIP.

If you pass "--stdlib=libc++" to hipcc, hipcc will use the libc++ library.  Generally, libc++ provides a broader set of C++ features while libstdc++ is the standard for more compilers (notably including g++).

When cross-linking C++ code, any C++ functions that use types from the C++ standard library (including std::string, std::vector and other containers) must use the same standard-library implementation. They include the following:

- Functions or kernels defined in HIP-Clang that are called from a standard compiler
- Functions defined in a standard compiler that are called from HIP-Clanng.

Applications with these interfaces should use the default libstdc++ linking.

Applications which are compiled entirely with hipcc, and which benefit from advanced C++ features not supported in libstdc++, and which do not require portability to nvcc, may choose to use libc++.


### HIP Headers (hip_runtime.h, hip_runtime_api.h)

The hip_runtime.h and hip_runtime_api.h files define the types, functions and enumerations needed to compile a HIP program:

- hip_runtime_api.h: defines all the HIP runtime APIs (e.g., hipMalloc) and the types required to call them. A source file that is only calling HIP APIs but neither defines nor launches any kernels can include hip_runtime_api.h. hip_runtime_api.h uses no custom hc language features and can be compiled using a standard C++ compiler.
- hip_runtime.h: included in hip_runtime_api.h. It additionally provides the types and defines required to create and launch kernels. hip_runtime.h can be compiled using a standard C++ compiler but will expose a subset of the available functions.

CUDA has slightly different contents for these two files. In some cases you may need to convert hipified code to include the richer hip_runtime.h instead of hip_runtime_api.h.

### Using a Standard C++ Compiler
You can compile hip\_runtime\_api.h using a standard C or C++ compiler (e.g., gcc or ICC). The HIP include paths and defines (`__HIP_PLATFORM_AMD__` or `__HIP_PLATFORM_NVIDIA__`) must pass to the standard compiler; hipconfig then returns the necessary options:
```
> hipconfig --cxx_config
 -D__HIP_PLATFORM_AMD__ -I/home/user1/hip/include
```

You can capture the hipconfig output and passed it to the standard compiler; below is a sample makefile syntax:

```
CPPFLAGS += $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)
```

nvcc includes some headers by default.  However, HIP does not include default headers, and instead all required files must be explicitly included.
Specifically, files that call HIP run-time APIs or define HIP kernels must explicitly include the appropriate HIP headers.
If the compilation process reports that it cannot find necessary APIs (for example, "error: identifier hipSetDevice is undefined"),
ensure that the file includes hip_runtime.h (or hip_runtime_api.h, if appropriate).
The hipify-perl script automatically converts "cuda_runtime.h" to "hip_runtime.h," and it converts "cuda_runtime_api.h" to "hip_runtime_api.h", but it may miss nested headers or macros.

#### cuda.h

The HIP-Clang path provides an empty cuda.h file. Some existing CUDA programs include this file but don't require any of the functions.

### Choosing HIP File Extensions

Many existing CUDA projects use the ".cu" and ".cuh" file extensions to indicate code that should be run through the nvcc compiler.
For quick HIP ports, leaving these file extensions unchanged is often easier, as it minimizes the work required to change file names in the directory and #include statements in the files.

For new projects or ports which can be re-factored, we recommend the use of the extension ".hip.cpp" for source files, and
".hip.h" or ".hip.hpp" for header files.
This indicates that the code is standard C++ code, but also provides a unique indication for make tools to
run hipcc when appropriate.

## Workarounds

### warpSize
Code should not assume a warp size of 32 or 64.  See [Warp Cross-Lane Functions](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#warp-cross-lane-functions) for information on how to write portable wave-aware code.

### Kernel launch with group size > 256
Kernel code should use ``` __attribute__((amdgpu_flat_work_group_size(<min>,<max>)))```.

For example:
```
__global__ void dot(double *a,double *b,const int n) __attribute__((amdgpu_flat_work_group_size(1, 512)))
```

## memcpyToSymbol

HIP support for hipMemcpyToSymbol is complete.  This feature allows a kernel
to define a device-side data symbol which can be accessed on the host side.  The symbol
can be in __constant or device space.

Note that the symbol name needs to be encased in the HIP_SYMBOL macro, as shown in the code example below. This also applies to hipMemcpyFromSymbol, hipGetSymbolAddress, and hipGetSymbolSize.

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
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
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
    hipLaunchKernelGGL(Get, dim3(1,1,1), dim3(LEN,1,1), 0, 0, Ad);
    HIP_ASSERT(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));

    for(unsigned i=0;i<LEN;i++)
    {
        assert(A[i] == B[i]);
    }
    std::cout<<"Passed"<<std::endl;
}
```

## CU_POINTER_ATTRIBUTE_MEMORY_TYPE

To get pointer's memory type in HIP/HIP-Clang, developers should use hipPointerGetAttributes API. First parameter of the API is hipPointerAttribute_t which has 'type' as member variable. 'type' indicates input pointer is allocated on device or host.

For example:
```
double * ptr;
hipMalloc(reinterpret_cast<void**>(&ptr), sizeof(double));
hipPointerAttribute_t attr;
hipPointerGetAttributes(&attr, ptr); /*attr.type will have value as hipMemoryTypeDevice*/

double* ptrHost;
hipHostMalloc(&ptrHost, sizeof(double));
hipPointerAttribute_t attr;
hipPointerGetAttributes(&attr, ptrHost); /*attr.type will have value as hipMemoryTypeHost*/
```
Please note, hipMemoryType enum values are different from cudaMemoryType enum values.

For example, on AMD platform, hipMemoryType is defined in hip_runtime_api.h,
```
typedef enum hipMemoryType {
    hipMemoryTypeHost = 0,    ///< Memory is physically located on host
    hipMemoryTypeDevice = 1,  ///< Memory is physically located on device. (see deviceId for specific device)
    hipMemoryTypeArray = 2,   ///< Array memory, physically located on device. (see deviceId for specific device)
    hipMemoryTypeUnified = 3, ///< Not used currently
    hipMemoryTypeManaged = 4  ///< Managed memory, automaticallly managed by the unified memory system
} hipMemoryType;
```
Looking into CUDA toolkit, it defines cudaMemoryType as following,
```
enum cudaMemoryType
{
  cudaMemoryTypeUnregistered = 0, // Unregistered memory.
  cudaMemoryTypeHost = 1, // Host memory.
  cudaMemoryTypeDevice = 2, // Device memory.
  cudaMemoryTypeManaged = 3, // Managed memory
}
```
In this case, memory type translation for hipPointerGetAttributes needs to be handled properly on nvidia platform to get the correct memory type in CUDA, which is done in the file nvidia_hip_runtime_api.h.

So in any HIP applications which use HIP APIs involving memory types, developers should use #ifdef in order to assign the correct enum values depending on Nvidia or AMD platform.

As an example, please see the code from the [link](https://github.com/ROCm/hip-tests/tree/develop/catch/unit/memory/hipMemcpyParam2D.cc).

With the #ifdef condition, HIP APIs work as expected on both AMD and NVIDIA platforms.

Note, cudaMemoryTypeUnregstered is currently not supported in hipMemoryType enum, due to HIP functionality backward compatibility.

## threadfence_system
Threadfence_system makes all device memory writes, all writes to mapped host memory, and all writes to peer memory visible to CPU and other GPU devices.
Some implementations can provide this behavior by flushing the GPU L2 cache.
HIP/HIP-Clang does not provide this functionality.  As a workaround, users can set the environment variable `HSA_DISABLE_CACHE=1` to disable the GPU L2 cache. This will affect all accesses and for all kernels and so may have a performance impact.

### Textures and Cache Control

Compute programs sometimes use textures either to access dedicated texture caches or to use the texture-sampling hardware for interpolation and clamping. The former approach uses simple point samplers with linear interpolation, essentially only reading a single point. The latter approach uses the sampler hardware to interpolate and combine multiple samples. AMD hardware, as well as recent competing hardware, has a unified texture/L1 cache, so it no longer has a dedicated texture cache. But the nvcc path often caches global loads in the L2 cache, and some programs may benefit from explicit control of the L1 cache contents.  We recommend the __ldg instruction for this purpose.

AMD compilers currently load all data into both the L1 and L2 caches, so __ldg is treated as a no-op.

We recommend the following for functional portability:

- For programs that use textures only to benefit from improved caching, use the __ldg instruction
- Programs that use texture object and reference APIs, work well on HIP


## More Tips

### HIP Logging

On an AMD platform, set the AMD_LOG_LEVEL environment variable to log HIP application execution information.

The value of the setting controls different logging level,

```
enum LogLevel {
LOG_NONE = 0,
LOG_ERROR = 1,
LOG_WARNING = 2,
LOG_INFO = 3,
LOG_DEBUG = 4
};
```

Logging mask is used to print types of functionalities during the execution of HIP application.
It can be set as one of the following values,

```
enum LogMask {
  LOG_API       = 1,      //!< (0x1)     API call
  LOG_CMD       = 2,      //!< (0x2)     Kernel and Copy Commands and Barriers
  LOG_WAIT      = 4,      //!< (0x4)     Synchronization and waiting for commands to finish
  LOG_AQL       = 8,      //!< (0x8)     Decode and display AQL packets
  LOG_QUEUE     = 16,     //!< (0x10)    Queue commands and queue contents
  LOG_SIG       = 32,     //!< (0x20)    Signal creation, allocation, pool
  LOG_LOCK      = 64,     //!< (0x40)    Locks and thread-safety code.
  LOG_KERN      = 128,    //!< (0x80)    Kernel creations and arguments, etc.
  LOG_COPY      = 256,    //!< (0x100)   Copy debug
  LOG_COPY2     = 512,    //!< (0x200)   Detailed copy debug
  LOG_RESOURCE  = 1024,   //!< (0x400)   Resource allocation, performance-impacting events.
  LOG_INIT      = 2048,   //!< (0x800)   Initialization and shutdown
  LOG_MISC      = 4096,   //!< (0x1000)  Misc debug, not yet classified
  LOG_AQL2      = 8192,   //!< (0x2000)  Show raw bytes of AQL packet
  LOG_CODE      = 16384,  //!< (0x4000)  Show code creation debug
  LOG_CMD2      = 32768,  //!< (0x8000)  More detailed command info, including barrier commands
  LOG_LOCATION  = 65536,  //!< (0x10000) Log message location
  LOG_MEM       = 131072, //!< (0x20000) Memory allocation
  LOG_MEM_POOL  = 262144, //!< (0x40000) Memory pool allocation, including memory in graphs
  LOG_ALWAYS    = -1      //!< (0xFFFFFFFF) Log always even mask flag is zero
};
```

### Debugging hipcc
To see the detailed commands that hipcc issues, set the environment variable HIPCC_VERBOSE to 1. Doing so will print to stderr the HIP-clang (or nvcc) commands that hipcc generates.

```
export HIPCC_VERBOSE=1
make
...
hipcc-cmd: /opt/rocm/bin/hipcc --offload-arch=native -x hip backprop_cuda.cu
```

### Editor Highlighting
See the utils/vim or utils/gedit directories to add handy highlighting to hip files.


