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
 * This kernel copies the contents of memory allocated in
 * ker_Alloc_MultCodeObj<<<>>> to host and deletes the memory
 * from thread 0.
 */
extern "C" __global__ void ker_Free_MultCodeObj(int *outputBuf,
                               int **dev_mem, int test_type) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  // Check allocated memory in all threads in block before access
  if (*dev_mem == nullptr) {
    printf("Device Allocation Failed in thread = %d \n", myId);
    return;
  }

  if (0 == myId) {
    for (size_t idx = 0; idx < (blockDim.x*gridDim.x); idx++) {
      outputBuf[idx] = (*dev_mem)[idx];
    }
    if (test_type == TEST_MALLOC_FREE) {
      free(*dev_mem);
    } else {
      delete[] (*dev_mem);
    }
  }
}
