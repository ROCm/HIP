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
