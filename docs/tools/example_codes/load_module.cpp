// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// [sphinx-start]
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(expression)                                \
{                                                            \
    const hipError_t err = expression;                       \
    if (err != hipSuccess)                                   \
    {                                                        \
        std::cout << "HIP Error: " << hipGetErrorString(err) \
              << " at line " << __LINE__ << std::endl;       \
        std::exit(EXIT_FAILURE);                             \
    }                                                        \
}

int main()
{
    std::size_t elements = 64*1024;
    std::size_t size_bytes = elements * sizeof(float);

    std::vector<float> A(elements), B(elements);

    // On NVIDIA platforms the driver runtime needs to be initiated
    #ifdef __HIP_PLATFORM_NVIDIA__
    hipInit(0);
    hipDevice_t device;
    hipCtx_t context;
    HIP_CHECK(hipDeviceGet(&device, 0));
    HIP_CHECK(hipCtxCreate(&context, 0, device));
    #endif

    // Allocate device memory
    hipDeviceptr_t d_A, d_B;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_A), size_bytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_B), size_bytes));

    // Copy data to device
    HIP_CHECK(hipMemcpyHtoD(d_A, A.data(), size_bytes));
    HIP_CHECK(hipMemcpyHtoD(d_B, B.data(), size_bytes));

    // Load module
    hipModule_t Module;
    // For AMD the module file has to contain architecture specific object code
    // For NVIDIA the module file has to contain PTX, found in e.g. "vcpy_isa.ptx"
    #ifdef __HIP_PLATFORM_AMD__
    HIP_CHECK(hipModuleLoad(&Module, "vcpy_isa.hsaco"));
    #elif defined(__HIP_PLATFORM_NVIDIA__)
    HIP_CHECK(hipModuleLoad(&Module, "vcpy_isa.ptx"));
    #endif
    // Get kernel function from the module via its name
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "hello_world"));

    // Create buffer for kernel arguments
    std::vector<void*> argBuffer{reinterpret_cast<void*>(d_A), reinterpret_cast<void*>(d_B)};
    std::size_t arg_size_bytes = argBuffer.size() * sizeof(void*);

    // Create configuration passed to the kernel as arguments
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, argBuffer.data(),
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size_bytes,
                      HIP_LAUNCH_PARAM_END};

    int threads_per_block = 128;
    int blocks = (elements + threads_per_block - 1) / threads_per_block;

    // Actually launch kernel
    HIP_CHECK(hipModuleLaunchKernel(Function, blocks, 1, 1, threads_per_block, 1, 1, 0, 0, NULL, config));

    HIP_CHECK(hipMemcpyDtoH(A.data(), d_A, elements));
    HIP_CHECK(hipMemcpyDtoH(B.data(), d_B, elements));

    HIP_CHECK(hipFree(reinterpret_cast<void*>(d_A)));
    HIP_CHECK(hipFree(reinterpret_cast<void*>(d_B)));

    #ifdef __HIP_PLATFORM_NVIDIA__
    HIP_CHECK(hipCtxDestroy(context));
    #endif

    return EXIT_SUCCESS;
}
// [sphinx-end]
