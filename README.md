## What is this repository for? ###

HIP allows developers to convert CUDA code to portable C++.  The same source code can be compiled to run on NVIDIA or AMD GPUs. 
Key features include:

* HIP is very thin and has little or no performance impact over coding directly in CUDA or hcc "HC" mode.
* HIP allows coding in a single-source C++ programming language including features such as templates, C++11 lambdas, classes, namespaces, and more.
* HIP allows developers to use the "best" development environment and tools on each target platform.
* The "hipify" tool automatically converts source from CUDA to HIP.
* Developers can specialize for the platform (CUDA or hcc) to tune for performance or handle tricky cases 

New projects can be developed directly in the portable HIP C++ language and can run on either NVIDIA or AMD platforms.  Additionally, HIP provides porting tools which make it easy to port existing CUDA codes to the HIP layer, with no loss of performance as compared to the original CUDA application.  HIP is not intended to be a drop-in replacement for CUDA, and developers should expect to do some manual coding and performance tuning work to complete the port.

## Repository branches:

The HIP repository maintains several branches. The branches that are of importance are:

* master branch: This is the stable branch. All stable releases are based on this branch.
* developer-preview branch: This is the branch were the new features still under development are visible. While this maybe of interest to many, it should be noted that this branch and the features under development might not be stable.

## Release tagging:

HIP releases are typically of two types. The tag naming convention is different for both types of releases to help differentiate them.

* release_x.yy.zzzz: These are the stable releases based on the master branch. This type of release is typically made once a month.
* preview_x.yy.zzzz: These denote pre-release code and are based on the developer-preview branch. This type of release is typically made once a week.

## More Info:
- [Installation](INSTALL.md) 
- [HIP FAQ](docs/markdown/hip_faq.md)
- [HIP Kernel Language](docs/markdown/hip_kernel_language.md)
- [HIP Runtime API (Doxygen)](http://rocm-developer-tools.github.io/HIP)
- [HIP Porting Guide](docs/markdown/hip_porting_guide.md)
- [HIP Porting Driver Guide](docs/markdown/hip_porting_driver_api.md)
- [HIP Programming Guide](docs/markdown/hip_programming_guide.md)
- [HIP Profiling ](docs/markdown/hip_profiling.md)
- [HIP Debugging](docs/markdown/hip_debugging.md)
- [HIP Terminology](docs/markdown/hip_terms.md) (including Rosetta Stone of GPU computing terms across CUDA/HIP/HC/AMP/OpenCL)
- [hipify-clang](hipify-clang/README.md)
- [Developer/CONTRIBUTING Info](CONTRIBUTING.md)
- [Release Notes](RELEASE.md)

## How do I get set up?

See the [Installation](INSTALL.md) notes.

## Simple Example
The HIP API includes functions such as hipMalloc, hipMemcpy, and hipFree.  
Programmers familiar with CUDA will also be able to quickly learn and start coding with the HIP API.
Compute kernels are launched with the "hipLaunchKernel" macro call.    Here is simple example showing a
snippet of HIP API code:

```cpp
hipMalloc(&A_d, Nbytes));
hipMalloc(&C_d, Nbytes));

hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice);

const unsigned blocks = 512;
const unsigned threadsPerBlock = 256;
hipLaunchKernel(vector_square,   /* compute kernel*/
                dim3(blocks), dim3(threadsPerBlock), 0/*dynamic shared*/, 0/*stream*/,     /* launch config*/
                C_d, A_d, N);  /* arguments to the compute kernel */

hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost); 
```


The HIP kernel language defines builtins for determining grid and block coordinates, math functions, short vectors, 
atomics, and timer functions. It also specifies additional defines and keywords for function types, address spaces, and 
optimization controls.  (See the [HIP Kernel Language](docs/markdown/hip_kernel_language.md) for a full description).
Here's an example of defining a simple 'vector_square' kernel.  



```cpp
template <typename T>
__global__ void
vector_square(T *C_d, const T *A_d, size_t N)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}
```

The HIP Runtime API code and compute kernel definition can exist in the same source file - HIP takes care of generating host and device code appropriately.

## HIP Portability and Compiler Technology
HIP C++ code can be compiled with either :
- On the NVIDIA CUDA platform, HIP provides header file which translate from the HIP runtime APIs to CUDA runtime APIs.  The header file contains mostly inlined 
  functions and thus has very low overhead - developers coding in HIP should expect the same performance as coding in native CUDA.  The code is then
  compiled with nvcc, the standard C++ compiler provided with the CUDA SDK.  Developers can use any tools supported by the CUDA SDK including the CUDA
  profiler and debugger.
- On the AMD ROCm platform, HIP provides a header and runtime library built on top of hcc compiler.  The HIP runtime implements HIP streams, events, and memory APIs, 
  and is a object library that is linked with the application.  The source code for all headers and the library implementation is available on GitHub.  
  HIP developers on ROCm can use AMD's CodeXL for debugging and profiling.

Thus HIP source code can be compiled to run on either platform.  Platform-specific features can be isolated to a specific platform using conditional compilation.  Thus HIP
provides source portability to either platform.   HIP provides the _hipcc_ compiler driver which will call the appropriate toolchain depending on the desired platform.


## Examples and Getting Started:

* A sample and [blog](http://gpuopen.com/hip-to-be-squared-an-introductory-hip-tutorial) that uses hipify to convert a simple app from CUDA to HIP:

 
```shell
cd samples/01_Intro/square
# follow README / blog steps to hipify the application.
```

* A sample and [blog](http://gpuopen.com/platform-aware-coding-inside-hip/) demonstrating platform specialization:
```shell
cd samples/01_Intro/bit_extract
make
```

* Guide to [Porting a New Cuda Project](docs/markdown/hip_porting_guide.md#porting-a-new-cuda-project")

 
## More Examples
The GitHub repository [HIP-Examples](https://github.com/ROCm-Developer-Tools/HIP-Examples.git) contains a hipified version of the popular Rodinia benchmark suite.
The README with the procedures and tips the team used during this porting effort is here: [Rodinia Porting Guide](https://github.com/ROCm-Developer-Tools/HIP-Examples/blob/master/rodinia_3.0/hip/README.hip_porting)

## Tour of the HIP Directories
* **include**: 
    * **hip_runtime_api.h** : Defines HIP runtime APIs and can be compiled with many standard Linux compilers (hcc, GCC, ICC, CLANG, etc), in either C or C++ mode.
    * **hip_runtime.h** : Includes everything in hip_runtime_api.h PLUS hipLaunchKernel and syntax for writing device kernels and device functions.  hip_runtime.h can only be compiled with hcc.
    * **hcc_detail/**** , **nvcc_detail/**** : Implementation details for specific platforms.  HIP applications should not include these files directly.
    * **hcc.h** : Includes interop APIs for HIP and HCC
    
* **bin**: Tools and scripts to help with hip porting
    * **hipify** : Tool to convert CUDA code to portable CPP.  Converts CUDA APIs and kernel builtins.  
    * **hipcc** : Compiler driver that can be used to replace nvcc in existing CUDA code. hipcc will call nvcc or hcc depending on platform, and include appropriate platform-specific headers and libraries.
    * **hipconfig** : Print HIP configuration (HIP_PATH, HIP_PLATFORM, CXX config flags, etc)
    * **hipexamine.sh** : Script to scan directory, find all code, and report statistics on how much can be ported with HIP (and identify likely features not yet supported)

* **doc**: Documentation - markdown and doxygen info

## Reporting an issue
Use the [GitHub issue tracker](https://github.com/ROCm-Developer-Tools/HIP/issues).
If reporting a bug, include the output of "hipconfig --full" and samples/1_hipInfo/hipInfo (if possible).

