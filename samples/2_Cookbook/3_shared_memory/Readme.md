## Using shared memory ###

Earlier we learned how to write our first hip program, in which we compute Matrix Transpose. In this tutorial, we'll explain how to use the shared memory to improve the performance.

## Introduction:

As we mentioned earlier  that Memory bottlenecks is the main problem why we are not able to get the highest performance, therefore minimizing the latency for memory access plays prominent role in application optimization. In this tutorial, we'll learn how to use static shared memory and will explain the dynamic one latter.

## Requirement:
For hardware requirement and software installation [Installation](https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md)

## prerequiste knowledge:

Programmers familiar with CUDA, OpenCL will be able to quickly learn and start coding with the HIP API. In case you are not, don't worry. You choose to start with the best one. We'll be explaining everything assuming you are completely new to gpgpu programming.

## Simple Matrix Transpose

We will be using the Simple Matrix Transpose application from the previous tutorial and modify it to learn how to use shared memory.

## Shared Memory

Shared memory is way more faster than that of global and constant memory and accessible to all the threads in the block. If the size of shared memory is known at compile time, we can specify the size and will use the static shared memory. In the same sourcecode, we will use the `__shared__` variable type qualifier as follows:

`  __shared__ float sharedMem[1024*1024];`

Be careful while using shared memory, since all threads within the block can access the shared memory, we need to sync the operation of individual threads by using:

`  __syncthreads();`

## How to build and run:
Use the make command and execute it using ./exe
Use hipcc to build the application, which is using hcc on AMD and nvcc on nvidia.

## More Info:
- [HIP FAQ](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_faq.md)
- [HIP Kernel Language](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_kernel_language.md)
- [HIP Runtime API (Doxygen)](http://rocm-developer-tools.github.io/HIP)
- [HIP Porting Guide](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_guide.md)
- [HIP Terminology](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_terms.md) (including Rosetta Stone of GPU computing terms across CUDA/HIP/HC/AMP/OpenL)
- [HIPIFY](https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/README.md)
- [Developer/CONTRIBUTING Info](https://github.com/ROCm-Developer-Tools/HIP/blob/master/CONTRIBUTING.md)
- [Release Notes](https://github.com/ROCm-Developer-Tools/HIP/blob/master/RELEASE.md)
