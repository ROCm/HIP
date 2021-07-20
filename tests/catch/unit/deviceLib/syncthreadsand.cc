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


static __global__ void kernel_syncthreads_and(int* syncTestD, int* allThreadsZeroD,
                                              int* allThreadsOneD, int* oneThreadZeroD,
                                              int* allThreadsMinusOneD) {
  int blockSize = blockDim.x;
  int predicate = 10;
  // First block index starts with 0, and second block index starts
  // with blockSize
  int i = (blockIdx.x == 0) ? threadIdx.x : blockSize + threadIdx.x;

  // At very first, we need to ensure work-group level syncronization
  // properly happened, don't bother about predicate testing for now.
  // Thread 0 and thread 1 writes to shared memory. After call to api,
  // every thread reads shared memory, and store product for verification
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  __syncthreads_and(predicate);
  syncTestD[i] = sm[0] * sm[1];

  // All threads pass 0 as predicate value, result should be 0
  predicate = 0;
  allThreadsZeroD[i] = __syncthreads_and(predicate);

  // All threads pass 1 as predicate value, result should be 1
  predicate = 1;
  allThreadsOneD[i] = __syncthreads_and(predicate);

  // Thread 0 pass 0, and all other threads 1 as predicate value,
  // result should be 0
  predicate = (threadIdx.x == 0) ? 0 : 1;
  oneThreadZeroD[i] = __syncthreads_and(predicate);

  // All threads pass -1 as predicate value, result should be 1
  predicate = -1;
  allThreadsMinusOneD[i] = __syncthreads_and(predicate);
}

static void test_syncthreads_and(int blockSize) {
  int nBytes = sizeof(int) * 2 * blockSize;
  int *syncTestD, *syncTestH;
  int *allThreadsZeroD, *allThreadsZeroH;
  int *allThreadsOneD, *allThreadsOneH;
  int *oneThreadZeroD, *oneThreadZeroH;
  int *allThreadsMinusOneD, *allThreadsMinusOneH;

  // Allocate device memory
  HIP_CHECK(hipMalloc((void**)&syncTestD, nBytes));
  HIP_CHECK(hipMalloc((void**)&allThreadsZeroD, nBytes));
  HIP_CHECK(hipMalloc((void**)&allThreadsOneD, nBytes));
  HIP_CHECK(hipMalloc((void**)&oneThreadZeroD, nBytes));
  HIP_CHECK(hipMalloc((void**)&allThreadsMinusOneD, nBytes));

  // Allocate host memory
  HIP_CHECK(hipHostMalloc((void**)&syncTestH, nBytes));
  HIP_CHECK(hipHostMalloc((void**)&allThreadsZeroH, nBytes));
  HIP_CHECK(hipHostMalloc((void**)&allThreadsOneH, nBytes));
  HIP_CHECK(hipHostMalloc((void**)&oneThreadZeroH, nBytes));
  HIP_CHECK(hipHostMalloc((void**)&allThreadsMinusOneH, nBytes));

  // Launch Kernel
  hipLaunchKernelGGL(kernel_syncthreads_and, 2, blockSize, 0, 0, syncTestD, allThreadsZeroD,
                     allThreadsOneD, oneThreadZeroD, allThreadsMinusOneD);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(syncTestH, syncTestD, nBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(allThreadsZeroH, allThreadsZeroD, nBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(allThreadsOneH, allThreadsOneD, nBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(oneThreadZeroH, oneThreadZeroD, nBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(allThreadsMinusOneH, allThreadsMinusOneD, nBytes, hipMemcpyDeviceToHost));

  // Validate results for both blocks together
  for (int i = 0; i < 2 * blockSize; ++i) {
    REQUIRE(syncTestH[i] == 200);
    REQUIRE(allThreadsZeroH[i] == 0);
    REQUIRE(allThreadsOneH[i] == 1);
    REQUIRE(oneThreadZeroH[i] == 0);
    REQUIRE(allThreadsMinusOneH[i] == 1);
  }

  // Free device memory
  HIP_CHECK(hipFree(syncTestD));
  HIP_CHECK(hipFree(allThreadsZeroD));
  HIP_CHECK(hipFree(allThreadsOneD));
  HIP_CHECK(hipFree(oneThreadZeroD));
  HIP_CHECK(hipFree(allThreadsMinusOneD));

  // Free host memory
  HIP_CHECK(hipHostFree(syncTestH));
  HIP_CHECK(hipHostFree(allThreadsZeroH));
  HIP_CHECK(hipHostFree(allThreadsOneH));
  HIP_CHECK(hipHostFree(oneThreadZeroH));
  HIP_CHECK(hipHostFree(allThreadsMinusOneH));
}

TEST_CASE("Unit_syncthreads_and") {
  int blockSizes[] = {10, 40, 70, 130, 240, 723, 32, 64, 128, 256, 512, 1024};
  for (unsigned long i = 0; i < (sizeof(blockSizes) / sizeof(blockSizes[0])); ++i)
    test_syncthreads_and(blockSizes[i]);
}
