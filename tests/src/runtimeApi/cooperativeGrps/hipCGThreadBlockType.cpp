/*
Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include "hip/hip_cooperative_groups.h"
#include <cmath>
#include <cstdlib>

#define ASSERT_EQUAL(lhs, rhs) assert(lhs == rhs)

using namespace cooperative_groups;

static __global__
void kernel_cg_thread_block_type(int *sizeTestD,
                                 int *thdRankTestD,
                                 int *syncTestD,
                                 dim3 *groupIndexTestD,
                                 dim3 *thdIndexTestD)
{
  thread_block tb = this_thread_block();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test size
  sizeTestD[gIdx] = tb.size();

  // Test thread_rank
  thdRankTestD[gIdx] = tb.thread_rank();

  // Test sync
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  tb.sync();
  syncTestD[gIdx] = sm[1] * sm[0];

  // Test group_index
  groupIndexTestD[gIdx] = tb.group_index();

  // Test thread_index
  thdIndexTestD[gIdx] = tb.thread_index();
}

static void test_cg_thread_block_type(int blockSize)
{
  int nBytes = sizeof(int) * 2 * blockSize;
  int nDim3Bytes = sizeof(dim3) * 2 * blockSize;
  int *sizeTestD, *sizeTestH;
  int *thdRankTestD, *thdRankTestH;
  int *syncTestD, *syncTestH;
  dim3 *groupIndexTestD, *groupIndexTestH;
  dim3 *thdIndexTestD, *thdIndexTestH;

  // Allocate device memory
  ASSERT_EQUAL(hipMalloc(&sizeTestD, nBytes), hipSuccess);
  ASSERT_EQUAL(hipMalloc(&thdRankTestD, nBytes), hipSuccess);
  ASSERT_EQUAL(hipMalloc(&syncTestD, nBytes), hipSuccess);
  ASSERT_EQUAL(hipMalloc(&groupIndexTestD, nDim3Bytes), hipSuccess);
  ASSERT_EQUAL(hipMalloc(&thdIndexTestD, nDim3Bytes), hipSuccess);

  // Allocate host memory
  ASSERT_EQUAL(hipHostMalloc(&sizeTestH, nBytes), hipSuccess);
  ASSERT_EQUAL(hipHostMalloc(&thdRankTestH, nBytes), hipSuccess);
  ASSERT_EQUAL(hipHostMalloc(&syncTestH, nBytes), hipSuccess);
  ASSERT_EQUAL(hipHostMalloc(&groupIndexTestH, nDim3Bytes), hipSuccess);
  ASSERT_EQUAL(hipHostMalloc(&thdIndexTestH, nDim3Bytes), hipSuccess);

  // Launch Kernel
  hipLaunchKernelGGL(kernel_cg_thread_block_type,
                     2,
                     blockSize,
                     0,
                     0,
                     sizeTestD,
                     thdRankTestD,
                     syncTestD,
                     groupIndexTestD,
                     thdIndexTestD);

  // Copy result from device to host
  ASSERT_EQUAL(hipMemcpy(sizeTestH, sizeTestD, nBytes, hipMemcpyDeviceToHost),
               hipSuccess);
  ASSERT_EQUAL(hipMemcpy(thdRankTestH, thdRankTestD, nBytes, hipMemcpyDeviceToHost),
               hipSuccess);
  ASSERT_EQUAL(hipMemcpy(syncTestH, syncTestD, nBytes, hipMemcpyDeviceToHost),
               hipSuccess);
  ASSERT_EQUAL(hipMemcpy(groupIndexTestH, groupIndexTestD, nDim3Bytes, hipMemcpyDeviceToHost),
               hipSuccess);
  ASSERT_EQUAL(hipMemcpy(thdIndexTestH, thdIndexTestD, nDim3Bytes, hipMemcpyDeviceToHost),
               hipSuccess);

  // Validate results for both blocks together
  for (int i = 0; i < 2 * blockSize; ++i) {
    ASSERT_EQUAL(sizeTestH[i], blockSize);
    ASSERT_EQUAL(thdRankTestH[i], i % blockSize);
    ASSERT_EQUAL(syncTestH[i], 200);
    ASSERT_EQUAL(groupIndexTestH[i].x, i / blockSize);
    ASSERT_EQUAL(groupIndexTestH[i].y, 0);
    ASSERT_EQUAL(groupIndexTestH[i].z, 0);
    ASSERT_EQUAL(thdIndexTestH[i].x, i % blockSize);
    ASSERT_EQUAL(thdIndexTestH[i].y, 0);
    ASSERT_EQUAL(thdIndexTestH[i].z, 0);
  }

  // Free device memory
  ASSERT_EQUAL(hipFree(sizeTestD), hipSuccess);
  ASSERT_EQUAL(hipFree(thdRankTestD), hipSuccess);
  ASSERT_EQUAL(hipFree(syncTestD), hipSuccess);
  ASSERT_EQUAL(hipFree(groupIndexTestD), hipSuccess);
  ASSERT_EQUAL(hipFree(thdIndexTestD), hipSuccess);

  //Free host memory
  ASSERT_EQUAL(hipHostFree(sizeTestH), hipSuccess);
  ASSERT_EQUAL(hipHostFree(thdRankTestH), hipSuccess);
  ASSERT_EQUAL(hipHostFree(syncTestH), hipSuccess);
  ASSERT_EQUAL(hipHostFree(groupIndexTestH), hipSuccess);
  ASSERT_EQUAL(hipHostFree(thdIndexTestH), hipSuccess);
}

int main()
{
  // Use default device for validating the test
  int deviceId;
  ASSERT_EQUAL(hipGetDevice(&deviceId), hipSuccess);
  hipDeviceProp_t deviceProperties;
  ASSERT_EQUAL(hipGetDeviceProperties(&deviceProperties, deviceId), hipSuccess);
  int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

  if (!deviceProperties.cooperativeLaunch) {
    std::cout << "info: Device doesn't support cooperative launch! skipping the test!\n";
    if (hip_skip_tests_enabled()) {
      return hip_skip_retcode();
    } else {
      passed();
    }
    return 0;
  }

  // Test block sizes which are powers of 2
  int i = 1;
  while (true) {
    int blockSize = pow(2, i);
    if (blockSize > maxThreadsPerBlock)
      break;
    test_cg_thread_block_type(blockSize);
    ++i;
  }

  // Test some random block sizes
  for(int j = 0; j < 10 ; ++j) {
    int blockSize = rand() % maxThreadsPerBlock;
    test_cg_thread_block_type(blockSize);
  }

  passed();
}
