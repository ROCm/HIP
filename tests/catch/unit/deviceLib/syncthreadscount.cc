/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>
#include <hip/hip_runtime.h>


static __global__ void kernel_syncthreads_count(int* syncTestD, int* allThreadsZeroD,
                                                int* allThreadsOneD, int* oddThreadsOneD,
                                                int* allThreadsMinusOneD, int* allThreadsIdD) {
  int blockSize = blockDim.x;
  int predicate = 10;
  // First block index starts with 0, and second block index starts
  // with blockSize
  int i = (blockIdx.x == 0) ? threadIdx.x : blockSize + threadIdx.x;

  // At very first, we need to ensure work-group level syncronization
  // properly happened, don't bother about predicate testing for now.
  // Thread 0 and thread 1 writes to shared memory. After call to api,
  // every thread reads shared memory, and store sum for verification
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  __syncthreads_count(predicate);
  syncTestD[i] = sm[0] + sm[1];

  // All threads pass 0 as predicate value, result should be 0
  predicate = 0;
  allThreadsZeroD[i] = __syncthreads_count(predicate);

  // All threads pass 1 as predicate value, result should be blockSize
  predicate = 1;
  allThreadsOneD[i] = __syncthreads_count(predicate);

  // Odd numbered threads pass 1, and even numbered threads pass 0, as
  // predicate value, result should be blockSize / 2
  predicate = threadIdx.x % 2;
  oddThreadsOneD[i] = __syncthreads_count(predicate);

  // All threads pass -1 as predicate value, result should blockSize
  predicate = -1;
  allThreadsMinusOneD[i] = __syncthreads_count(predicate);

  // Each thread pass its ID as predicate value, result should be blockSize - 1
  predicate = threadIdx.x;
  allThreadsIdD[i] = __syncthreads_count(predicate);
}

void test_syncthreads_count(int blockSize) {
  int nBytes = sizeof(int) * 2 * blockSize;
  int *syncTestD, *syncTestH;
  int *allThreadsZeroD, *allThreadsZeroH;
  int *allThreadsOneD, *allThreadsOneH;
  int *oddThreadsOneD, *oddThreadsOneH;
  int *allThreadsMinusOneD, *allThreadsMinusOneH;
  int *allThreadsIdD, *allThreadsIdH;

  // Allocate device memory
  HIP_CHECK(hipMalloc((void**)&syncTestD, nBytes));
  HIP_CHECK(hipMalloc((void**)&allThreadsZeroD, nBytes));
  HIP_CHECK(hipMalloc((void**)&allThreadsOneD, nBytes));
  HIP_CHECK(hipMalloc((void**)&oddThreadsOneD, nBytes));
  HIP_CHECK(hipMalloc((void**)&allThreadsMinusOneD, nBytes));
  HIP_CHECK(hipMalloc((void**)&allThreadsIdD, nBytes));

  // Allocate host memory
  HIP_CHECK(hipHostMalloc((void**)&syncTestH, nBytes));
  HIP_CHECK(hipHostMalloc((void**)&allThreadsZeroH, nBytes));
  HIP_CHECK(hipHostMalloc((void**)&allThreadsOneH, nBytes));
  HIP_CHECK(hipHostMalloc((void**)&oddThreadsOneH, nBytes));
  HIP_CHECK(hipHostMalloc((void**)&allThreadsMinusOneH, nBytes));
  HIP_CHECK(hipHostMalloc((void**)&allThreadsIdH, nBytes));

  // Launch Kernel
  hipLaunchKernelGGL(kernel_syncthreads_count, 2, blockSize, 0, 0, syncTestD, allThreadsZeroD,
                     allThreadsOneD, oddThreadsOneD, allThreadsMinusOneD, allThreadsIdD);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(syncTestH, syncTestD, nBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(allThreadsZeroH, allThreadsZeroD, nBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(allThreadsOneH, allThreadsOneD, nBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(oddThreadsOneH, oddThreadsOneD, nBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(allThreadsMinusOneH, allThreadsMinusOneD, nBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(allThreadsIdH, allThreadsIdD, nBytes, hipMemcpyDeviceToHost));

  // Validate results for both the blocks together
  for (int i = 0; i < 2 * blockSize; ++i) {
    REQUIRE(syncTestH[i] == 30);
    REQUIRE(allThreadsZeroH[i] == 0);
    REQUIRE(allThreadsOneH[i] == blockSize);
    REQUIRE(oddThreadsOneH[i] == blockSize / 2);
    REQUIRE(allThreadsMinusOneH[i] == blockSize);
    REQUIRE(allThreadsIdH[i] == (blockSize - 1));
  }

  // Free device memory
  HIP_CHECK(hipFree(syncTestD));
  HIP_CHECK(hipFree(allThreadsZeroD));
  HIP_CHECK(hipFree(allThreadsOneD));
  HIP_CHECK(hipFree(oddThreadsOneD));
  HIP_CHECK(hipFree(allThreadsMinusOneD));
  HIP_CHECK(hipFree(allThreadsIdD));

  // Free host memory
  HIP_CHECK(hipHostFree(syncTestH));
  HIP_CHECK(hipHostFree(allThreadsZeroH));
  HIP_CHECK(hipHostFree(allThreadsOneH));
  HIP_CHECK(hipHostFree(oddThreadsOneH));
  HIP_CHECK(hipHostFree(allThreadsMinusOneH));
  HIP_CHECK(hipHostFree(allThreadsIdH));
}

TEST_CASE("Unit_syncthreads_count") {
  int blockSizes[] = {10, 40, 70, 130, 240, 723, 32, 64, 128, 256, 512, 1024};
  for (unsigned long i = 0; i < (sizeof(blockSizes) / sizeof(blockSizes[0])); ++i)
    test_syncthreads_count(blockSizes[i]);
}
