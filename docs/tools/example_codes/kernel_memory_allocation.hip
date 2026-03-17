// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>

// [sphinx-kernel-start]
__global__ void kernel_memory_allocation()
{
  // The pointer is stored in shared memory, so that all
  // threads of the block can access the pointer
  __shared__ int *memory;

  std::size_t blockSize = blockDim.x;
  constexpr std::size_t elementsPerThread = 1024;
  if(threadIdx.x == 0)
  {
    // allocate memory in one contiguous block
    memory = new int[blockDim.x * elementsPerThread];
  }
  __syncthreads();

  // load pointer into thread-local variable to avoid
  // unnecessary accesses to shared memory
  int *localPtr = memory;

  // work with allocated memory, e.g. initialization
  for(int i = 0; i < elementsPerThread; ++i)
  {
    // access in a contiguous way
    localPtr[i * blockSize + threadIdx.x] = i;
  }

  // synchronize to make sure no thread is accessing the memory before freeing
  __syncthreads();
  if(threadIdx.x == 0)
  {
    delete[] memory;
  }
}
// [sphinx-kernel-end]

int main()
{
    kernel_memory_allocation<<<64, 1024>>>();
    HIP_CHECK(hipGetLastError());
    
    std::cout << "Success!" << std::endl;
    
    return EXIT_SUCCESS;
}
