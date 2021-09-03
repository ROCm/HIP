## What is this repository for? ###

**HIP is a C++ Runtime API and Kernel Language that allows developers to create portable applications for AMD and NVIDIA GPUs from single source code.**

Key features include:

* HIP is very thin and has little or no performance impact over coding directly in CUDA mode.
* HIP allows coding in a single-source C++ programming language including features such as templates, C++11 lambdas, classes, namespaces, and more.
* HIP allows developers to use the "best" development environment and tools on each target platform.
* The [HIPIFY](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/README.md) tools automatically convert source from CUDA to HIP.
* Developers can specialize for the platform (CUDA or AMD) to tune for performance or handle tricky cases.

New projects can be developed directly in the portable HIP C++ language and can run on either NVIDIA or AMD platforms.  Additionally, HIP provides porting tools which make it easy to port existing CUDA codes to the HIP layer, with no loss of performance as compared to the original CUDA application.  HIP is not intended to be a drop-in replacement for CUDA, and developers should expect to do some manual coding and performance tuning work to complete the port.

## DISCLAIMER

The information contained herein is for informational purposes only, and is subject to change without notice. In addition, any stated support is planned and is also subject to change. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the parties or in AMD's Standard Terms and Conditions of Sale.

© 2020 Advanced Micro Devices, Inc. All Rights Reserved.

## Repository branches:

The HIP repository maintains several branches. The branches that are of importance are:

* develop branch: This is the default branch, on which the new features are still under development and visible. While this maybe of interest to many, it should be noted that this branch and the features under development might not be stable.
* Main branch: This is the stable branch. It is up to date with the latest release branch, for example, if the latest HIP release is rocm-4.3, main branch will be the repository based on this release.
* Release branches. These are branches corresponding to each ROCM release, listed with release tags, such as rocm-4.2, rocm-4.3, etc.

## Release tagging:

HIP releases are typically naming convention for each ROCM release to help differentiate them.

* rocm x.yy: These are the stable releases based on the ROCM release.
  This type of release is typically made once a month.*

## More Info:
- [Installation](INSTALL.md)
- [HIP FAQ](docs/markdown/hip_faq.md)
- [HIP Kernel Language](docs/markdown/hip_kernel_language.md)
- [HIP Runtime API (Doxygen)](https://github.com/RadeonOpenCompute/ROCm)
- [HIP Porting Guide](docs/markdown/hip_porting_guide.md)
- [HIP Porting Driver Guide](docs/markdown/hip_porting_driver_api.md)
- [HIP Programming Guide](docs/markdown/hip_programming_guide.md)
- [HIP Logging ](docs/markdown/hip_logging.md)
- [HIP Debugging ](docs/markdown/hip_debugging.md)
- [Code Object tooling ](docs/markdown/obj_tooling.md)
- [HIP Terminology](docs/markdown/hip_terms2.md) (including Rosetta Stone of GPU computing terms across CUDA/HIP/OpenCL)
- [HIPIFY](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/README.md)
- Supported CUDA APIs:
  * [Runtime API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUDA_Runtime_API_functions_supported_by_HIP.md)
  * [Driver API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUDA_Driver_API_functions_supported_by_HIP.md)
  * [cuComplex API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/cuComplex_API_supported_by_HIP.md)
  * [Device API](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUDA_Device_API_supported_by_HIP.md)
  * [cuBLAS](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUBLAS_API_supported_by_HIP.md)
  * [cuRAND](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CURAND_API_supported_by_HIP.md)
  * [cuDNN](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUDNN_API_supported_by_HIP.md)
  * [cuFFT](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUFFT_API_supported_by_HIP.md)
  * [cuSPARSE](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUSPARSE_API_supported_by_HIP.md)
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
atomics, and timer functions.
It also specifies additional defines and keywords for function types, address spaces, and optimization controls (See the [HIP Kernel Language](docs/markdown/hip_kernel_language.md) for a full description).
Here's an example of defining a simple 'vector_square' kernel.


```cpp
template <typename T>
__global__ void
vector_square(T *C_d, const T *A_d, size_t N)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}
```

The HIP Runtime API code and compute kernel definition can exist in the same source file - HIP takes care of generating host and device code appropriately.

## HIP Portability and Compiler Technology
HIP C++ code can be compiled with either,
- On the NVIDIA CUDA platform, HIP provides header file which translate from the HIP runtime APIs to CUDA runtime APIs.  The header file contains mostly inlined
  functions and thus has very low overhead - developers coding in HIP should expect the same performance as coding in native CUDA.  The code is then
  compiled with nvcc, the standard C++ compiler provided with the CUDA SDK.  Developers can use any tools supported by the CUDA SDK including the CUDA
  profiler and debugger.
- On the AMD ROCm platform, HIP provides a header and runtime library built on top of HIP-Clang compiler.  The HIP runtime implements HIP streams, events, and memory APIs,
  and is a object library that is linked with the application.  The source code for all headers and the library implementation is available on GitHub.
  HIP developers on ROCm can use AMD's ROCgdb (https://github.com/ROCm-Developer-Tools/ROCgdb) for debugging and profiling.

Thus HIP source code can be compiled to run on either platform.  Platform-specific features can be isolated to a specific platform using conditional compilation.  Thus HIP
provides source portability to either platform.   HIP provides the _hipcc_ compiler driver which will call the appropriate toolchain depending on the desired platform.


## Examples and Getting Started:

* A sample and [blog](https://github.com/ROCm-Developer-Tools/HIP/tree/main/samples/0_Intro/square) that uses any of [HIPIFY](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/README.md) tools to convert a simple app from CUDA to HIP:


```shell
cd samples/01_Intro/square
# follow README / blog steps to hipify the application.
```

* Guide to [Porting a New Cuda Project](docs/markdown/hip_porting_guide.md#porting-a-new-cuda-project")


## More Examples
The GitHub repository [HIP-Examples](https://github.com/ROCm-Developer-Tools/HIP-Examples.git) contains a hipified version of the popular Rodinia benchmark suite.
The README with the procedures and tips the team used during this porting effort is here: [Rodinia Porting Guide](https://github.com/ROCm-Developer-Tools/HIP-Examples/blob/master/rodinia_3.0/hip/README.hip_porting)

## Tour of the HIP Directories
* **include**:
    * **hip_runtime_api.h** : Defines HIP runtime APIs and can be compiled with many standard Linux compilers (GCC, ICC, CLANG, etc), in either C or C++ mode.
    * **hip_runtime.h** : Includes everything in hip_runtime_api.h PLUS hipLaunchKernel and syntax for writing device kernels and device functions.  hip_runtime.h can be compiled using a standard C++ compiler but will expose a subset of the available functions.
    * **amd_detail/**** , **nvidia_detail/**** : Implementation details for specific platforms. HIP applications should not include these files directly.

* **bin**: Tools and scripts to help with hip porting
    * **hipify-perl** : Script based tool to convert CUDA code to portable CPP. Converts CUDA APIs and kernel builtins.
    * **hipcc** : Compiler driver that can be used to replace nvcc in existing CUDA code. hipcc will call nvcc or HIP-Clang depending on platform and include appropriate platform-specific headers and libraries.
    * **hipconfig** : Print HIP configuration (HIP_PATH, HIP_PLATFORM, HIP_COMPILER, HIP_RUNTIME, CXX config flags, etc.)
    * **hipexamine-perl.sh** : Script to scan the directory, find all code, and report statistics on how much can be ported with HIP (and identify likely features not yet supported).
    * **hipconvertinplace-perl.sh** : Script to scan the directory, find all code, and convert the found CUDA code to HIP reporting all unconverted things.

* **doc**: Documentation - markdown and doxygen info.

## Reporting an issue
Use the [GitHub issue tracker](https://github.com/ROCm-Developer-Tools/HIP/issues).
If reporting a bug, include the output of "hipconfig --full" and samples/1_hipInfo/hipInfo (if possible).

