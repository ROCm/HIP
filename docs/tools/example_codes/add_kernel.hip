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

#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

///\brief Calculates \p a[i] = \p a[i] + \p b[i] where \p i stands for the thread's index in the grid.
// [sphinx-kernel-start]
__global__ void AddKernel(float* a, const float* b)
{
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    a[global_idx] += b[global_idx];
}
// [sphinx-kernel-end]

int main()
{
    // The number of float elements in each vector.
    constexpr unsigned int size = 1 << 20; // == 1'048'576 elements

    // Bytes to allocate for each device vector.
    constexpr size_t size_bytes = size * sizeof(float);

    // Number of threads per kernel block.
    constexpr unsigned int threads_per_block = 256;

    // Number of blocks per kernel grid. The expression below calculates ceil(size/block_size).
    constexpr unsigned int number_of_blocks = ceiling_div(size, threads_per_block);

    // Allocate a vector and fill it with an increasing sequence (i.e. 1, 2, 3, 4...)
    std::vector<float> h_a(size);
    std::iota(h_a.begin(), h_a.end(), 1.f);

    // Allocate b vector and fill it with a decreasing sequence (i.e. 1'048'576, 1'048'575, ..., 3, 2, 1)
    std::vector<float> h_b(size);
    std::iota(h_b.rbegin(), h_b.rend(), 1.f);

    // Allocate and copy vectors to device memory.
    float* d_a{};
    float* d_b{};
    HIP_CHECK(hipMalloc(&d_a, size_bytes));
    HIP_CHECK(hipMalloc(&d_b, size_bytes));
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), size_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), size_bytes, hipMemcpyHostToDevice));

    std::cout << "Calculating a[i] = a[i] + b[i] over " << size << " elements." << std::endl;

    // Launch the kernel on the default stream.
    // [sphinx-kernel-launch-start]
    AddKernel<<<number_of_blocks, threads_per_block>>>(d_a, d_b);
    // [sphinx-kernel-launch-end]

    // Check if the kernel launch was successful.
    HIP_CHECK(hipGetLastError());

    // Copy the results back to the host. This call blocks the host's execution until the copy is finished.
    HIP_CHECK(hipMemcpy(h_a.data(), d_a, size_bytes, hipMemcpyDeviceToHost));

    // Free device memory.
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_a));

    // Print the first few elements of the results:
    constexpr size_t elements_to_print = 10;
    std::cout << "First " << elements_to_print << " elements of the results: "
              << format_range(h_a.begin(), h_a.begin() + elements_to_print) << std::endl;

    return EXIT_SUCCESS;
}
