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

#include <cstdlib>
#include <iostream>

extern __shared__ int shared_array[];

__global__ void kernel()
{
    // initialize shared memory
    shared_array[threadIdx.x] = threadIdx.x;
    // use shared memory - synchronize to make sure, that all threads of the
    // block see all changes to shared memory
    __syncthreads();
}

int main()
{
    //shared memory in this case depends on the configurable block size
    constexpr int blockSize = 256;
    constexpr int sharedMemSize = blockSize * sizeof(int);
    constexpr int gridSize = 2;

    kernel<<<dim3(gridSize), dim3(blockSize), sharedMemSize, 0>>>();
    if(auto err = hipDeviceSynchronize(); err != hipSuccess)
        std::cerr << "HIP error " << err << ": " << hipGetErrorString(err) << std::endl;

    return EXIT_SUCCESS;
}
// [sphinx-end]