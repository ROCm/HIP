// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

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
