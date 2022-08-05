/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip/hip_runtime.h"
#include "./defs.h"
/**
 * This kernel allocates and deallocates memory in every thread.
 */
extern "C" __global__ void ker_TestDynamicAllocInAllThreads_CodeObj(
                        int *outputBuf, int test_type, int value,
                        size_t perThreadSize) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Allocate
  size_t size = 0;
  int* ptr = nullptr;
  if (test_type == TEST_MALLOC_FREE) {
    size = perThreadSize * sizeof(int);
    ptr = reinterpret_cast<int*> (malloc(size));
  } else {
    size = perThreadSize;
    ptr = new int[perThreadSize];
  }
  if (ptr == nullptr) {
    printf("Device Allocation in thread %d Failed! \n", myId);
    return;
  }
  // Set memory
  for (size_t idx = 0; idx < perThreadSize; idx++) {
    ptr[idx] = value;
  }
  // Copy to output buffer
  for (size_t idx = 0; idx < perThreadSize; idx++) {
    outputBuf[myId*perThreadSize + idx] = ptr[idx];
  }
  // Free memory
  if (test_type == TEST_MALLOC_FREE) {
    free(ptr);
  } else {
    delete[] ptr;
  }
}
