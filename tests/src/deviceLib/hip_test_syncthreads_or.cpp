/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include <hip/hip_runtime.h>
#include "test_common.h"

#define ASSERT_EQUAL(lhs, rhs) assert(lhs == rhs)

static __global__
void kernel_syncthreads_or(int *syncTestD,
                           int *allThreadsZeroD,
                           int *allThreadsOneD,
                           int *oneThreadOneD,
                           int *allThreadsMinusOneD)
{
  int blockSize = blockDim.x;
  int predicate = 10;
  // First block index starts with 0, and second block index starts
  // with blockSize
  int i = (blockIdx.x == 0) ? threadIdx.x : blockSize + threadIdx.x;

  // At very first, we need to ensure work-group level syncronization
  // properly happened, don't bother about predicate testing for now.
  // Thread 0 and thread 1 writes to shared memory. After call to api,
  // every thread reads shared memory, and store subtraction for verification
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  __syncthreads_or(predicate);
  syncTestD[i] = sm[1] - sm[0];

  // All threads pass 0 as predicate value, result should be 0
  predicate = 0;
  allThreadsZeroD[i] = __syncthreads_or(predicate);

  // All threads pass 1 as predicate value, result should be 1
  predicate = 1;
  allThreadsOneD[i] = __syncthreads_or(predicate);

  // Thread 0 pass 1, and all other threads 0 as predicate value,
  // result should be 1
  predicate = (threadIdx.x == 0) ? 1 : 0;
  oneThreadOneD[i] = __syncthreads_or(predicate);

  // All threads pass -1 as predicate value, result should be 1
  predicate = -1;
  allThreadsMinusOneD[i] = __syncthreads_or(predicate);
}

static void test_syncthreads_or(int blockSize)
{
  int nBytes = sizeof(int) * 2 * blockSize;
  int * syncTestD, *syncTestH;
  int *allThreadsZeroD, *allThreadsZeroH;
  int *allThreadsOneD, *allThreadsOneH;
  int *oneThreadOneD, *oneThreadOneH;
  int *allThreadsMinusOneD, *allThreadsMinusOneH;

  // Allocate device memory
  ASSERT_EQUAL(hipMalloc((void**)&syncTestD, nBytes), hipSuccess);
  ASSERT_EQUAL(hipMalloc((void**)&allThreadsZeroD, nBytes), hipSuccess);
  ASSERT_EQUAL(hipMalloc((void**)&allThreadsOneD, nBytes), hipSuccess);
  ASSERT_EQUAL(hipMalloc((void**)&oneThreadOneD, nBytes), hipSuccess);
  ASSERT_EQUAL(hipMalloc((void**)&allThreadsMinusOneD, nBytes), hipSuccess);

  // Allocate host memory
  ASSERT_EQUAL(hipHostMalloc((void**)&syncTestH, nBytes), hipSuccess);
  ASSERT_EQUAL(hipHostMalloc((void**)&allThreadsZeroH, nBytes), hipSuccess);
  ASSERT_EQUAL(hipHostMalloc((void**)&allThreadsOneH, nBytes), hipSuccess);
  ASSERT_EQUAL(hipHostMalloc((void**)&oneThreadOneH, nBytes), hipSuccess);
  ASSERT_EQUAL(hipHostMalloc((void**)&allThreadsMinusOneH, nBytes), hipSuccess);

  // Launch Kernel
  hipLaunchKernelGGL(kernel_syncthreads_or,
                     2,
                     blockSize,
                     0,
                     0,
                     syncTestD,
                     allThreadsZeroD,
                     allThreadsOneD,
                     oneThreadOneD,
                     allThreadsMinusOneD);

  // Copy result from device to host
  ASSERT_EQUAL(hipMemcpy(syncTestH, syncTestD, nBytes, hipMemcpyDeviceToHost),
               hipSuccess);
  ASSERT_EQUAL(hipMemcpy(allThreadsZeroH, allThreadsZeroD, nBytes, hipMemcpyDeviceToHost),
               hipSuccess);
  ASSERT_EQUAL(hipMemcpy(allThreadsOneH, allThreadsOneD, nBytes, hipMemcpyDeviceToHost),
               hipSuccess);
  ASSERT_EQUAL(hipMemcpy(oneThreadOneH, oneThreadOneD, nBytes, hipMemcpyDeviceToHost),
               hipSuccess);
  ASSERT_EQUAL(hipMemcpy(allThreadsMinusOneH, allThreadsMinusOneD, nBytes, hipMemcpyDeviceToHost),
               hipSuccess);

  // Validate results for both blocks together
  for (int i = 0; i < 2 * blockSize; ++i) {
    ASSERT_EQUAL(syncTestH[i], 10);
    ASSERT_EQUAL(allThreadsZeroH[i], 0);
    ASSERT_EQUAL(allThreadsOneH[i], 1);
    ASSERT_EQUAL(oneThreadOneH[i], 1);
    ASSERT_EQUAL(allThreadsMinusOneH[i], 1);
  }

  // Free device memory
  ASSERT_EQUAL(hipFree(syncTestD), hipSuccess);
  ASSERT_EQUAL(hipFree(allThreadsZeroD), hipSuccess);
  ASSERT_EQUAL(hipFree(allThreadsOneD), hipSuccess);
  ASSERT_EQUAL(hipFree(oneThreadOneD), hipSuccess);
  ASSERT_EQUAL(hipFree(allThreadsMinusOneD), hipSuccess);

  //Free host memory
  ASSERT_EQUAL(hipHostFree(syncTestH), hipSuccess);
  ASSERT_EQUAL(hipHostFree(allThreadsZeroH), hipSuccess);
  ASSERT_EQUAL(hipHostFree(allThreadsOneH), hipSuccess);
  ASSERT_EQUAL(hipHostFree(oneThreadOneH), hipSuccess);
  ASSERT_EQUAL(hipHostFree(allThreadsMinusOneH), hipSuccess);
}

int main()
{
  int blockSizes[] = {10, 40, 70, 130, 240, 723, 32, 64, 128, 256, 512, 1024};
  for (int i = 0; i < (sizeof(blockSizes) / sizeof(blockSizes[0])); ++i)
    test_syncthreads_or(blockSizes[i]);
  passed();
}
