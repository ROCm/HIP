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

#include <cuda.h>

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define CUDA_CHECK(expression)                                          \
{                                                                       \
    const CUresult err = expression;                                    \
    if (err != CUDA_SUCCESS)                                            \
    {                                                                   \
        const char* err_str{nullptr};                                   \
        cuGetErrorString(err, &err_str);                                \
        std::cerr << "CUDA Error: " << err_str                          \
                  << " at line " << __LINE__ << std::endl;              \
        std::exit(EXIT_FAILURE);                                        \
    }                                                                   \
}

void* populate_data_pointer()
{
    auto filename = std::string{"myKernel.ptx"};
    std::fstream file{filename, std::ios::in | std::ios::binary | std::ios::ate};
    if(!file.is_open())
    {
        std::cerr << "Error opening file " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    auto filesize = file.tellg();
    auto storage = new char[filesize];

    file.seekg(0, std::ios::beg);
    file.read(storage, filesize);

    return storage;
}

int main()
{
    std::size_t elements = 64*1024;
    std::size_t size_bytes = elements * sizeof(float);

    std::vector<float> A(elements), B(elements);

    // On NVIDIA platforms the driver runtime needs to be initiated
    cuInit(0);
    CUdevice device;
    CUcontext context;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuCtxCreate(&context, 0, device));

    // Allocate device memory
    CUdeviceptr d_A, d_B;
    CUDA_CHECK(cuMemAlloc(&d_A, size_bytes));
    CUDA_CHECK(cuMemAlloc(&d_B, size_bytes));

    // Copy data to device
    CUDA_CHECK(cuMemcpyHtoD(d_A, A.data(), size_bytes));
    CUDA_CHECK(cuMemcpyHtoD(d_B, B.data(), size_bytes));

    // Load module
    
    // For NVIDIA the module file has to contain PTX, found in e.g. "myKernel.ptx"
    // [sphinx-start]
    CUmodule module;
    void* imagePtr = populate_data_pointer();

    const int numOptions = 1;
    CUjit_option options[numOptions];
    void *optionValues[numOptions];

    options[0] = CU_JIT_MAX_REGISTERS;
    unsigned maxRegs = 15;
    optionValues[0] = (void *)(&maxRegs);

    cuModuleLoadDataEx(&module, imagePtr, numOptions, options, optionValues);

    CUfunction k;
    cuModuleGetFunction(&k, module, "myKernel");
    // [sphinx-end]

    // Create buffer for kernel arguments
    std::vector<void*> argBuffer{&d_A, &d_B};
    std::size_t arg_size_bytes = argBuffer.size() * sizeof(void*);

    // Create configuration passed to the kernel as arguments
    void* config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer.data(),
                      CU_LAUNCH_PARAM_BUFFER_SIZE, &arg_size_bytes, CU_LAUNCH_PARAM_END};

    int threads_per_block = 128;
    int blocks = (elements + threads_per_block - 1) / threads_per_block;

    // Actually launch kernel
    CUDA_CHECK(cuLaunchKernel(k, blocks, 1, 1, threads_per_block, 1, 1, 0, 0, NULL, config));

    CUDA_CHECK(cuMemcpyDtoH(A.data(), d_A, elements));
    CUDA_CHECK(cuMemcpyDtoH(B.data(), d_B, elements));

    CUDA_CHECK(cuMemFree(d_A));
    CUDA_CHECK(cuMemFree(d_B));

    CUDA_CHECK(cuCtxDestroy(context));

    delete[] static_cast<char*>(imagePtr);

    return EXIT_SUCCESS;
}
