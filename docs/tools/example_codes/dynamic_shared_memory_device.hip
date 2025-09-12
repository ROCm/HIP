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

#include <cstddef>
#include <cstdlib>
#include <iostream>

// [sphinx-start]
extern __shared__ int dynamic_shared[];

__global__ void kernel(int array1SizeX, int array1SizeY, int array2Size)
{
    // at least (array1SizeX * array1SizeY + array2Size) * sizeof(int) bytes
    // dynamic shared memory need to be allocated when the kernel is launched
    int* array1 = dynamic_shared;
    // array1 is interpreted as 2D of size:
    int array1Size = array1SizeX * array1SizeY;

    int* array2 = &(array1[array1Size]);

    if(threadIdx.x < array1SizeX && threadIdx.y < array1SizeY)
    {
        // access array1 with threadIdx.x + threadIdx.y * array1SizeX
    }
    if(threadIdx.x < array2Size)
    {
        // access array2 threadIdx.x
    }
}
// [sphinx-end]

int main()
{
    std::size_t shared_memory_bytes = 512 * sizeof(int);
    kernel<<<64, 512, shared_memory_bytes>>>(512, 1, 512);
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Success!" << std::endl;
    return EXIT_SUCCESS;
}
