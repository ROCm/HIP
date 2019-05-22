## SimpleMultiGPU ###

In Earlier tutorial array of elements bit extract(0_Intro/bit_extract) demonstrated on single GPU, this tutorial extends it to multiple GPUs

## Introduction:

This tutorial explains how to scale application across multiple GPUs

## Requirement:
For hardware requirement and software installation [Installation](https://github.com/ROCm-Developer-Tools/HIP/INSTALL.md) 

## prerequiste knowledge:

Programmers familiar with CUDA, OpenCL will be able to quickly learn and start coding with the HIP API. In case you are not, don't worry. You choose to start with the best one. We'll be explaining everything assuming you are completely new to gpgpu programming.

## bit_extract

We will be using the bit_extract application from the previous tutorial and modify it to learn how to distribute on multiple GPUs

## Multi GPU 

In HIP application that works with multiple GPUs, current set of HIP operations targeted on to which GPU device has to be set explicitly. You can set the current device with the following function:

hipError_t hipSetDevice(int deviceId);

Below loop iterates over multiple GPUs, asynchronously copying the input arrays for that device. It then launches a kernel in the same stream operating on N per GPU data elements. Finally, an asynchronous copy from the device is issued to transfer the results from the kernel back to the host. Because all functions are asynchronous, control is returned to the host thread immediately. It is then safe to switch to the next device while tasks are still running on the current device.

for (int i = 0; i < ngpus; i++)
{
	CHECK(hipSetDevice(i));
	CHECK(hipMemcpyAsync(d_input[i], h_input[i], NbytesPerGPU, hipMemcpyHostToDevice, stream[i]));

	dim3 block(256,1,1);
	dim3 grid((NPerGPU+block.x-1)/256,1,1);

	hipLaunchKernelGGL((bit_extract_kernel), dim3(grid), dim3(block), 0, stream[i], d_input[i], d_output[i], NPerGPU);

	CHECK(hipMemcpyAsync(h_output[i], d_output[i], NbytesPerGPU, hipMemcpyDeviceToHost, stream[i]));
}

## How to build and run:
Use the make command and execute it using ./exe
Use hipcc to build the application, which is using hcc on AMD and nvcc on nvidia.

## More Info:
- [HIP FAQ](https://github.com/ROCm-Developer-Tools/HIP/docs/markdown/hip_faq.md)
- [HIP Kernel Language](https://github.com/ROCm-Developer-Tools/HIP/docs/markdown/hip_kernel_language.md)
- [HIP Runtime API (Doxygen)](http://rocm-developer-tools.github.io/HIP)
- [HIP Porting Guide](https://github.com/ROCm-Developer-Tools/HIP/docs/markdown/hip_porting_guide.md)
- [HIP Terminology](https://github.com/ROCm-Developer-Tools/HIP/docs/markdown/hip_terms.md) (including Rosetta Stone of GPU computing terms across CUDA/HIP/HC/AMP/OpenL)
- [clang-hipify](https://github.com/ROCm-Developer-Tools/HIP/clang-hipify/README.md)
- [Developer/CONTRIBUTING Info](https://github.com/ROCm-Developer-Tools/HIP/CONTRIBUTING.md)
- [Release Notes](https://github.com/ROCm-Developer-Tools/HIP/RELEASE.md)

