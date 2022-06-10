## Writing first HIP program ###

This tutorial shows how to get write simple HIP application. We will write the simplest Matrix Transpose program.

## HIP Introduction:

HIP is a C++ runtime API and kernel language that allows developers to create portable applications that can run on AMD and other GPU’s. Our goal was to rise above the lowest-common-denominator paths and deliver a solution that allows you, the developer, to use essential hardware features and maximize your application’s performance on GPU hardware.

## Requirement:
For hardware requirement and software installation [Installation](https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md)

## prerequiste knowledge:

Programmers familiar with CUDA, OpenCL will be able to quickly learn and start coding with the HIP API. In case you are not, don't worry. You choose to start with the best one. We'll be explaining everything assuming you are completely new to gpgpu programming.

## Simple Matrix Transpose

Here is simple example showing how to write your first program in HIP.
In order to use the HIP framework, we need to add the "hip_runtime.h" header file. SInce its c++ api you can add any header file you have been using earlier while writing your c/c++ program. For gpgpu programming, we have host(microprocessor) and the device(gpu).

## Device-side code
We will work on device side code first, Here is simple example showing a snippet of HIP device side code:

```
__global__ void matrixTranspose(float *out,
                                float *in,
                                const int width,
                                const int height)
{
    int x = blockDim.x * blockIdx.x + bhreadIdx.x;
    int y = blockDim.y * blockIdx.y + bhreadIdx.y;

    out[y * width + x] = in[x * height + y];
}
```

`__global__` keyword is the Function-Type Qualifiers, it is used with functions that are executed on device and are called/launched from the hosts.
other function-type qualifiers are:
`__device__` functions are Executed on the device and Called from the device only
`__host__` functions are Executed on the host and Called from the host

`__host__` can combine with `__device__`, in which case the function compiles for both the host and device. These functions cannot use the HIP grid coordinate functions (for example, "threadIdx.x", will talk about it latter). A possible workaround is to pass the necessary coordinate info as an argument to the function.
`__host__` cannot combine with `__global__`.

`__global__` functions are often referred to as *kernels*, and calling one is termed *launching the kernel*.

Next keyword is `void`. HIP `__global__` functions must have a `void` return type. Global functions require the caller to specify an "execution configuration" that includes the grid and block dimensions. The execution configuration can also include other information for the launch, such as the amount of additional shared memory to allocate and the stream where the kernel should execute.

The kernel function begins with
`    int x = blockDim.x * blockIdx.x + threadIdx.x;`
`    int y = blockDim.y * blockIdx.y + threadIdx.y;`
here the keyword blockIdx.x, blockIdx.y and blockIdx.z(not used here) are the built-in functions to identify the threads in a block. The keyword blockDim.x, blockDim.y and blockDim.z(not used here) are to identify the dimensions of the block.

We are familiar with rest of the code on device-side.

## Host-side code

Now, we'll see how to call the kernel from the host. Inside the main() function, we first defined the pointers(for both, the host-side as well as device). The declaration of device pointer is similar to that of the host. Next, we have `hipDeviceProp_t`, it is the pre-defined struct for hip device properties. This is followed by `hipGetDeviceProperties(&devProp, 0)` It is used to extract the device information. The first parameter is the struct, second parameter is the device number to get properties for. Next line print the name of the device.

We allocated memory to the Matrix on host side by using malloc and initiallized it. While in order to allocate memory on device side we will be using `hipMalloc`, it's quiet similar to that of malloc instruction. After this, we will copy the data to the allocated memory on device-side using `hipMemcpy`.
`  hipMemcpy(gpuMatrix, Matrix, NUM*sizeof(float), hipMemcpyHostToDevice);`
here the first parameter is the destination pointer, second is the source pointer, third is the size of memory copy and the last specify the direction on memory copy(which is in this case froom host to device). While in order to transfer memory from device to host, use `hipMemcpyDeviceToHost` and for device to device memory copy use `hipMemcpyDeviceToDevice`.

Now, we'll see how to launch the kernel.
```
  hipLaunchKernelGGL(matrixTranspose,
                  dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  gpuTransposeMatrix , gpuMatrix, WIDTH ,HEIGHT);
```

HIP introduces a standard C++ calling convention to pass the execution configuration to the kernel (this convention replaces the `Cuda <<< >>>` syntax). In HIP,
- Kernels launch with the `"hipLaunchKernelGGL"` function
- The first five parameters to hipLaunchKernelGGL are the following:
   - **symbol kernelName**: the name of the kernel to launch.  To support template kernels which contains "," use the HIP_KERNEL_NAME macro. In current application it's "matrixTranspose".
   - **dim3 gridDim**: 3D-grid dimensions specifying the number of blocks to launch. In MatrixTranspose sample, it's "dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y)".
   - **dim3 blockDim**: 3D-block dimensions specifying the number of threads in each block.In MatrixTranspose sample, it's "dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)".
   - **size_t dynamicShared**: amount of additional shared memory to allocate when launching the kernel. In MatrixTranspose sample, it's '0'.
   - **hipStream_t**: stream where the kernel should execute. A value of 0 corresponds to the NULL stream.In MatrixTranspose sample, it's '0'.
- Kernel arguments follow these first five parameters. Here, these are "gpuTransposeMatrix , gpuMatrix, WIDTH ,HEIGHT".

Next, we'll copy the computed values/data back to the device using the `hipMemcpy`. Here the last parameter will be `hipMemcpyDeviceToHost`

After, copying the data from device to memory, we will verify it with the one we computed with the cpu reference funtion.

Finally, we will free the memory allocated earlier by using free() for host while for devices we will use `hipFree`.

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
