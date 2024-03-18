# Frequently asked questions

## What APIs and features does HIP support?
HIP provides the following:
- Devices (hipSetDevice(), hipGetDeviceProperties(), etc.)
- Memory management (hipMalloc(), hipMemcpy(), hipFree(), etc.)
- Streams (hipStreamCreate(),hipStreamSynchronize(), hipStreamWaitEvent(),  etc.)
- Events (hipEventRecord(), hipEventElapsedTime(), etc.)
- Kernel launching (hipLaunchKernel/hipLaunchKernelGGL is the preferred way of launching kernels. hipLaunchKernelGGL is a standard C/C++ macro that can serve as an alternative way to launch kernels, replacing the CUDA triple-chevron (<<< >>>) syntax).
- HIP Module API to control when adn how code is loaded.
- CUDA-style kernel coordinate functions (threadIdx, blockIdx, blockDim, gridDim)
- Cross-lane instructions including shfl, ballot, any, all
- Most device-side math built-ins
- Error reporting (hipGetLastError(), hipGetErrorString())

The HIP API documentation describes each API and its limitations, if any, compared with the equivalent CUDA API.

## What is not supported?

### Runtime/Driver API features
At a high-level, the following features are not supported:
- Textures (partial support available)
- Dynamic parallelism (CUDA 5.0)
- Graphics interoperability with OpenGL or Direct3D
- CUDA IPC Functions (Under Development)
- CUDA array, mipmappedArray and pitched memory
- Queue priority controls

See the [API Support Table](https://github.com/ROCm/HIPIFY/blob/amd-staging/docs/tables/CUDA_Runtime_API_functions_supported_by_HIP.md) for more detailed information.

### Kernel language features
- C++-style device-side dynamic memory allocations (free, new, delete) (CUDA 4.0)
- Virtual functions, indirect functions and try/catch (CUDA 4.0)
- `__prof_trigger`
- PTX assembly (CUDA 4.0).  HIP-Clang supports inline GCN assembly.
- Several kernel features are under development.  See the {doc}`/reference/kernel_language` for more information.


## Is HIP a drop-in replacement for CUDA?
No. HIP provides porting tools which do most of the work to convert CUDA code into portable C++ code that uses the HIP APIs.
Most developers will port their code from CUDA to HIP and then maintain the HIP version.
HIP code provides the same performance as native CUDA code, plus the benefits of running on AMD platforms.

## What specific version of CUDA does HIP support?
HIP APIs and features do not map to a specific CUDA version. HIP provides a strong subset of the functionality provided in CUDA, and the hipify tools can scan code to identify any unsupported CUDA functions - this is useful for identifying the specific features required by a given application.

However, we can provide a rough summary of the features included in each CUDA SDK and the support level in HIP. Each bullet below lists the major new language features in each CUDA release and then indicate which are supported/not supported in HIP:

- CUDA 4.0 and earlier :
    - HIP supports CUDA 4.0 except for the limitations described above.
- CUDA 5.0 :
    - Dynamic Parallelism (not supported)
    - cuIpc functions (under development).
- CUDA 6.0 :
    - Managed memory (under development)
- CUDA 6.5 :
    - __shfl intriniscs (supported)
- CUDA 7.0 :
    - Per-thread default streams (supported)
    - C++11 (Hip-Clang supports all of C++11, all of C++14 and some C++17 features)
- CUDA 7.5 :
    - float16 (supported)
- CUDA 8.0 :
    - Page Migration including cudaMemAdvise, cudaMemPrefetch, other cudaMem* APIs(not supported)
- CUDA 9.0 :
    - Cooperative Launch, Surface Object Management, Version Management

## What libraries does HIP support?
HIP includes growing support for the four key math libraries using hipBlas, hipFFt, hipRAND and hipSPARSE, as well as MIOpen for machine intelligence applications.
These offer pointer-based memory interfaces (as opposed to opaque buffers) and can be easily interfaced with other HIP applications.
The hip interfaces support both ROCm and CUDA paths, with familiar library interfaces.

- [hipBlas](https://github.com/ROCmSoftwarePlatform/hipBLAS), which utilizes [rocBlas](https://github.com/ROCmSoftwarePlatform/rocBLAS).
- [hipFFt](https://github.com/ROCmSoftwarePlatform/hipfft)
- [hipsSPARSE](https://github.com/ROCmSoftwarePlatform/hipsparse)
- [hipRAND](https://github.com/ROCmSoftwarePlatform/hipRAND)
- [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen)

Additionally, some of the cublas routines are automatically converted to hipblas equivalents by the HIPIFY tools. These APIs use cublas or hcblas depending on the platform and replace the need to use conditional compilation.

## How does HIP compare with OpenCL?
Both AMD and Nvidia support OpenCL 1.2 on their devices so that developers can write portable code.
HIP offers several benefits over OpenCL:
- Developers can code in C++ as well as mix host and device C++ code in their source files. HIP C++ code can use templates, lambdas, classes and so on.
- The HIP API is less verbose than OpenCL and is familiar to CUDA developers.
- Because both CUDA and HIP are C++ languages, porting from CUDA to HIP is significantly easier than porting from CUDA to OpenCL.
- HIP uses the best available development tools on each platform: on Nvidia GPUs, HIP code compiles using NVCC and can employ the nSight profiler and debugger (unlike OpenCL on Nvidia GPUs).
- HIP provides pointers and host-side pointer arithmetic.
- HIP provides device-level control over memory allocation and placement.
- HIP offers an offline compilation model.

## How does porting CUDA to HIP compare to porting CUDA to OpenCL?
Both HIP and CUDA are dialects of C++, and thus porting between them is relatively straightforward.
Both dialects support templates, classes, lambdas, and other C++ constructs.
As one example, the hipify-perl tool was originally a Perl script that used simple text conversions from CUDA to HIP.
HIP and CUDA provide similar math library calls as well.  In summary, the HIP philosophy was to make the HIP language close enough to CUDA that the porting effort is relatively simple.
This reduces the potential for error, and also makes it easy to automate the translation.  HIP's goal is to quickly get the ported program running on both platforms with little manual intervention, so that the programmer can focus on performance optimizations.

There have been several tools that have attempted to convert CUDA into OpenCL, such as CU2CL.  OpenCL is a C99-based kernel language (rather than C++) and also does not support single-source compilation.
As a result, the OpenCL syntax is different from CUDA, and the porting tools have to perform some heroic transformations to bridge this gap.
The tools also struggle with more complex CUDA applications, in particular, those that use templates, classes, or other C++ features inside the kernel.

## What hardware does HIP support?
- For AMD platforms, see the [ROCm documentation](https://github.com/RadeonOpenCompute/ROCm#supported-gpus) for the list of supported platforms.
- For Nvidia platforms, HIP requires Unified Memory and should run on any device supporting CUDA SDK 6.0 or newer. We have tested the Nvidia Titan and Tesla K40.

## Do HIPIFY tools automatically convert all source code?
Typically, HIPIFY tools can automatically convert almost all run-time code.
Most device code needs no additional conversion since HIP and CUDA have similar names for math and built-in functions.
The hipify-clang tool will automatically modify the kernel signature as needed (automating a step that used to be done manually).
Additional porting may be required to deal with architecture feature queries or with CUDA capabilities that HIP doesn't support.
In general, developers should always expect to perform some platform-specific tuning and optimization.

## What is NVCC?
NVCC is Nvidia's compiler driver for compiling "CUDA C++" code into PTX or device code for Nvidia GPUs. It's a closed-source binary compiler that is provided by the CUDA SDK.

## What is HIP-Clang?
HIP-Clang is a Clang/LLVM based compiler to compile HIP programs which can run on AMD platform.

## Why use HIP rather than supporting CUDA directly?
While HIP is a strong subset of the CUDA, it is a subset.  The HIP layer allows that subset to be clearly defined and documented.
Developers who code to the HIP API can be assured their code will remain portable across Nvidia and AMD platforms.
In addition, HIP defines portable mechanisms to query architectural features and supports a larger 64-bit wavesize which expands the return type for cross-lane functions like ballot and shuffle from 32-bit ints to 64-bit ints.

## Can I develop HIP code on an Nvidia CUDA platform?
Yes.  HIP's CUDA path only exposes the APIs and functionality that work on both NVCC and AMDGPU back-ends.
"Extra" APIs, parameters, and features which exist in CUDA but not in HIP-Clang will typically result in compile-time or run-time errors.
Developers need to use the HIP API for most accelerator code and bracket any CUDA-specific code with preprocessor conditionals.
Developers concerned about portability should, of course, run on both platforms, and should expect to tune for performance.
In some cases, CUDA has a richer set of modes for some APIs, and some C++ capabilities such as virtual functions - see the HIP @API documentation for more details.

## Can I develop HIP code on an AMD HIP-Clang platform?
Yes. HIP's HIP-Clang path only exposes the APIs and functions that work on AMD runtime back ends. "Extra" APIs, parameters and features that appear in HIP-Clang but not CUDA will typically cause compile- or run-time errors. Developers must use the HIP API for most accelerator code and bracket any HIP-Clang specific code with preprocessor conditionals. Those concerned about portability should, of course, test their code on both platforms and should tune it for performance. Typically, HIP-Clang supports a more modern set of C++11/C++14/C++17 features, so HIP developers who want portability should be careful when using advanced C++ features on the HIP-Clang path.

## How to use HIP-Clang to build HIP programs?
The environment variable can be used to set compiler path:
- HIP_CLANG_PATH: path to hip-clang. When set, this variable let hipcc to use hip-clang for compilation/linking.

There is an alternative environment variable to set compiler path:
- HIP_ROCCLR_HOME: path to root directory of the HIP-ROCclr runtime. When set, this variable let hipcc use hip-clang from the ROCclr distribution.
NOTE: If HIP_ROCCLR_HOME is set, there is no need to set HIP_CLANG_PATH since hipcc will deduce them from HIP_ROCCLR_HOME.

## What is AMD clr?
AMD clr (Common Language Runtime) is a repository for the AMD platform, which contains source codes for AMD's compute languages runtimes as follows,

- hipamd - contains implementation of HIP language for AMD GPU.
- rocclr - contains virtual device interfaces that compute runtimes interact with backends, such as ROCr on Linux and PAL on Windows.
- opencl - contains implementation of OpenCL™ on the AMD platform.

## What is hipother?
A new repository 'hipother' is added in the ROCm 6.1 release, which is branched out from HIP.
hipother supports the HIP back-end implementation on some non-AMD platforms, like NVIDIA.

## Can I get HIP open source repository for Windows?
No, there is no HIP repository open publicly on Windows.

## Can a HIP binary run on both AMD and Nvidia platforms?
HIP is a source-portable language that can be compiled to run on either AMD or NVIDIA platform. HIP tools don't create a "fat binary" that can run on either platform, however.

## On HIP-Clang, can I link HIP code with host code compiled with another compiler such as gcc, icc, or clang ?
Yes.  HIP generates the object code which conforms to the GCC ABI, and also links with libstdc++.  This means you can compile host code with the compiler of your choice and link the generated object code
with GPU code compiled with HIP.  Larger projects often contain a mixture of accelerator code (initially written in CUDA with nvcc) and host code (compiled with gcc, icc, or clang).   These projects
can convert the accelerator code to HIP, compile that code with hipcc, and link with object code from their preferred compiler.

## Can HIP API support C style application? What is the differentce between C and C++ ?
HIP is C++ runtime API that supports C style applications as well.

Some C style applications (and interfaces to other languages (Fortran, Python)) would call certain HIP APIs but not use kernel programming.
They can be compiled with a C compiler and run correctly, however, small details must be considered in the code. For example, initializtion, as shown in the simple application below, uses HIP structs dim3 with the file name "test.hip.cpp"
```
#include "hip/hip_runtime_api.h"
#include "stdio.h"

int main(int argc, char** argv) {
  dim3 grid1;
  printf("dim3 grid1; x=%d, y=%d, z=%d\n",grid1.x,grid1.y,grid1.z);
  dim3 grid2 = {1,1,1};
  printf("dim3 grid2 = {1,1,1}; x=%d, y=%d, z=%d\n",grid2.x,grid2.y,grid2.z);
  return 0;
}
```

When using a C++ compiler,
```
$ gcc -x c++  $(hipconfig --cpp_config) test3.hip.cpp -o test
$ ./test
dim3 grid1; x=1, y=1, z=1
dim3 grid2 = {1,1,1}; x=1, y=1, z=1
```
In which "dim3 grid1;" will yield a dim3 grid with all dimentional members x,y,z initalized to 1, as the default constructor behaves that way.
Further, if write,
```
dim3 grid(2); // yields {2,1,1}
dim3 grid(2,3); yields {2,3,1}
```

In comparison, when using the C compiler,
```
$ gcc -x c $(hipconfig --cpp_config) test.hip.cpp -o test
$ ./test
dim3 grid1; x=646881376, y=21975, z=1517277280
dim3 grid2 = {1,1,1}; x=1, y=1, z=1
```
In which "dim3 grid;" does not imply any initialization, no constructor is called, and dimentional values x,y,z of grid are undefined.
NOTE: To get the C++ default behavior, C programmers must additionally specify the right-hand side as shown below,
```
dim3 grid = {1,1,1}; // initialized as in C++
```


## Can I install both CUDA SDK and HIP-Clang on the same machine?
Yes. You can use HIP_PLATFORM to choose which path hipcc targets.  This configuration can be useful when using HIP to develop an application which is portable to both AMD and NVIDIA.


## HIP detected my platform (HIP-Clang vs nvcc) incorrectly - what should I do?
HIP will set the platform to AMD and use HIP-Clang as compiler if it sees that the AMD graphics driver is installed and has detected an AMD GPU.
Sometimes this isn't what you want - you can force HIP to recognize the platform by setting the following,
```
export HIP_PLATFORM=amd
```
HIP then set and use correct AMD compiler and runtime,
HIP_COMPILER=clang
HIP_RUNTIME=rocclr

To choose NVIDIA platform, you can set,
```
export HIP_PLATFORM=nvidia
```
In this case, HIP will set and use the following,
HIP_COMPILER=cuda
HIP_RUNTIME=nvcc

One symptom of this problem is the message "error: 'unknown error'(11) at square.hipref.cpp:56".  This can occur if you have a CUDA installation on an AMD platform, and HIP incorrectly detects the platform as nvcc.  HIP may be able to compile the application using the nvcc tool-chain but will generate this error at runtime since the platform does not have a CUDA device.

## On CUDA, can I mix CUDA code with HIP code?
Yes.  Most HIP data structures (hipStream_t, hipEvent_t) are typedefs to CUDA equivalents and can be intermixed.  Both CUDA and HIP use integer device ids.
One notable exception is that hipError_t is a new type, and cannot be used where a cudaError_t is expected.  In these cases, refactor the code to remove the expectation.  Alternatively, hip_runtime_api.h defines functions which convert between the error code spaces:

hipErrorToCudaError
hipCUDAErrorTohipError
hipCUResultTohipError

If platform portability is important, use #ifdef __HIP_PLATFORM_NVIDIA__ to guard the CUDA-specific code.

## How do I trace HIP application flow?
See {doc}`/developer_guide/logging` for more information.

## What is maximum limit of kernel launching parameter?
Product of block.x, block.y, and block.z should be less than 1024.
Please note, HIP does not support kernel launch with total work items defined in dimension with size gridDim x blockDim >= 2^32, so gridDim.x * blockDim.x, gridDim.y * blockDim.y and gridDim.z * blockDim.z are always less than 2^32.

## Are __shfl_*_sync functions supported on HIP platform?
__shfl_*_sync is not supported on HIP but for nvcc path CUDA 9.0 and above all shuffle calls get redirected to it's sync version.

## How to create a guard for code that is specific to the host or the GPU?
The compiler defines the `__HIP_DEVICE_COMPILE__` macro only when compiling the code for the GPU.  It could be used to guard code that is specific to the host or the GPU.

## Why _OpenMP is undefined when compiling with -fopenmp?
When compiling an OpenMP source file with `hipcc -fopenmp`, the compiler may generate error if there is a reference to the `_OPENMP` macro.  This is due to a limitation in hipcc that treats any source file type (e.g., `.cpp`) as an HIP translation unit leading to some conflicts with the OpenMP language switch.  If the OpenMP source file doesn't contain any HIP language construct, you could workaround this issue by adding the `-x c++` switch to force the compiler to treat the file as regular C++.  Another approach would be to guard the OpenMP code with `#ifdef _OPENMP` so that the code block is disabled when compiling for the GPU.  The `__HIP_DEVICE_COMPILE__` macro defined by the HIP compiler when compiling GPU code could also be used for guarding code paths specific to the host or the GPU.

## Does the HIP-Clang compiler support extern shared declarations?

Previously, it was essential to declare dynamic shared memory using the HIP_DYNAMIC_SHARED macro for accuracy, as using static shared memory in the same kernel could result in overlapping memory ranges and data-races.

Now, the HIP-Clang compiler provides support for extern shared declarations, and the HIP_DYNAMIC_SHARED option is no longer required. You may use the standard extern definition:
extern __shared__ type var[];

## I have multiple HIP enabled devices and I am getting an error code hipErrorSharedObjectInitFailed with the message "Error: shared object initialization failed"?

This error message is seen due to the fact that you do not have valid code object for all of your devices.

If you have compiled the application yourself, make sure you have given the correct device name(s) and its features via: `--offload-arch`. If you are not mentioning the `--offload-arch`, make sure that `hipcc` is using the correct offload arch by verifying the hipcc output generated by setting the environment variable `HIPCC_VERBOSE=1`.

If you have a precompiled application/library (like rocblas, tensorflow etc) which gives you such error, there are one of two possibilities.

 - The application/library does not ship code object bundles for *all* of your device(s): in this case you need to recompile the application/library yourself with correct `--offload-arch`.
 - The application/library does not ship code object bundles for *some* of your device(s), for example you have a system with an APU + GPU and the library does not ship code objects for your APU. For this you can set the environment variable `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` on NVdia platform, to only enable GPUs for which code object is available. This will limit the GPUs visible to your application and allow it to run.

Note: In previous releases, the error code is hipErrorNoBinaryForGpu with message "Unable to find code object for all current devices".
The error code handling behavior is changed. HIP runtime shows the error code hipErrorSharedObjectInitFailed with message "Error: shared object initialization failed" on unsupported GPU.

## How to use per-thread default stream in HIP?

The per-thread default stream is an implicit stream local to both the thread and the current device. It does not do any implicit synchronization with other streams (like explicitly created streams), or default per-thread stream on other threads.

The per-thread default stream is a blocking stream and will synchronize with the default null stream if both are used in a program.

In ROCm, a compilation option should be added in order to compile the translation unit with per-thread default stream enabled.
"-fgpu-default-stream=per-thread".
Once source is compiled with per-thread default stream enabled, all APIs will be executed on per thread default stream, hence there will not be any implicit synchronization with other streams.

Besides, per-thread default stream be enabled per translation unit, users can compile some files with feature enabled and some with feature disabled. Feature enabled translation unit will have default stream as per thread and there will not be any implicit synchronization done but other modules will have legacy default stream which will do implicit synchronization.

## How to use complex muliplication and division operations?

In HIP, hipFloatComplex and hipDoubleComplex are defined as complex data types,
typedef float2 hipFloatComplex;
typedef double2 hipDoubleComplex;

Any application uses complex multiplication and division operations, need to replace '*' and '/' operators with the following,
- hipCmulf() and hipCdivf() for hipFloatComplex
- hipCmul() and hipCdiv() for hipDoubleComplex

Note: These complex operations are equivalent to corresponding types/functions on the NVIDIA platform.

## Can I develop applications with HIP APIs on Windows the same on Linux?

Yes, HIP APIs are available to use on both Linux and Windows.
Due to different working mechanisms on operating systems like Windows vs Linux, HIP APIs call corresponding lower level backend runtime libraries and kernel drivers for the OS, in order to control the executions on GPU hardware accordingly. There might be a few differences on the related backend software and driver support, which might affect usage of HIP APIs. See OS support details in HIP API document.

## Does HIP support LUID?

Starting ROCm 6.0, HIP runtime supports Locally Unique Identifier (LUID).
This feature enables the local physical device(s) to interoperate with other devices. For example, DX12.

HIP runtime sets device LUID properties so the driver can query LUID to identify each device for interoperability.

Note: HIP supports LUID only on Windows OS.

## How can I know the version of HIP?

HIP version definition has been updated since ROCm 4.2 release as the following:

HIP_VERSION=HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)

HIP version can be queried from HIP API call,
hipRuntimeGetVersion(&runtimeVersion);

The version returned will always be greater than the versions in previous ROCm releases.

Note: The version definition of HIP runtime is different from CUDA. On AMD platform, the function returns HIP runtime version, while on NVIDIA platform, it returns CUDA runtime version. And there is no mapping/correlation between HIP version and CUDA version.

