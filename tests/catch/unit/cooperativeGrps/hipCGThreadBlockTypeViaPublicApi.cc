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

#include <hip_test_common.hh>
#include "hip/hip_cooperative_groups.h"
#include <cstdlib>

#define ASSERT_EQUAL(lhs, rhs) assert(lhs == rhs)

using namespace cooperative_groups;

static __global__
void kernel_cg_thread_block_type_via_public_api(int *sizeTestD,
                                                int *thdRankTestD,
                                                int *syncTestD)
{
  thread_block tb = this_thread_block();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test group_size api
  sizeTestD[gIdx] = group_size(tb);

  // Test thread_rank api
  thdRankTestD[gIdx] = thread_rank(tb);

  // Test sync api
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  sync(tb);
  syncTestD[gIdx] = sm[1] * sm[0];
}

static void test_cg_thread_block_type_via_public_api(int blockSize)
{
  int nBytes = sizeof(int) * 2 * blockSize;
  int *sizeTestD, *sizeTestH;
  int *thdRankTestD, *thdRankTestH;
  int *syncTestD, *syncTestH;

  // Allocate device memory
  HIPCHECK(hipMalloc(&sizeTestD, nBytes));
  HIPCHECK(hipMalloc(&thdRankTestD, nBytes));
  HIPCHECK(hipMalloc(&syncTestD, nBytes));

  // Allocate host memory
  HIPCHECK(hipHostMalloc(&sizeTestH, nBytes));
  HIPCHECK(hipHostMalloc(&thdRankTestH, nBytes));
  HIPCHECK(hipHostMalloc(&syncTestH, nBytes));

  // Launch Kernel
  hipLaunchKernelGGL(kernel_cg_thread_block_type_via_public_api,
                     2,
                     blockSize,
                     0,
                     0,
                     sizeTestD,
                     thdRankTestD,
                     syncTestD);

  // Copy result from device to host
  HIPCHECK(hipMemcpy(sizeTestH, sizeTestD, nBytes, hipMemcpyDeviceToHost));
  HIPCHECK(hipMemcpy(thdRankTestH, thdRankTestD, nBytes, hipMemcpyDeviceToHost));
  HIPCHECK(hipMemcpy(syncTestH, syncTestD, nBytes, hipMemcpyDeviceToHost));

  // Validate results for both blocks together
  for (int i = 0; i < 2 * blockSize; ++i) {
    ASSERT_EQUAL(sizeTestH[i], blockSize);
    ASSERT_EQUAL(thdRankTestH[i], i % blockSize);
    ASSERT_EQUAL(syncTestH[i], 200);
  }

  // Free device memory
  HIPCHECK(hipFree(sizeTestD));
  HIPCHECK(hipFree(thdRankTestD));
  HIPCHECK(hipFree(syncTestD));

  //Free host memory
  HIPCHECK(hipHostFree(sizeTestH));
  HIPCHECK(hipHostFree(thdRankTestH));
  HIPCHECK(hipHostFree(syncTestH));
}

TEST_CASE("Unit_hipCGThreadBlockType_PublicApi") {
  // Use default device for validating the test
  int deviceId;
  hipDeviceProp_t deviceProperties;
  HIPCHECK(hipGetDevice(&deviceId));
  HIPCHECK(hipGetDeviceProperties(&deviceProperties, deviceId));

  if (!deviceProperties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  // Test for blockSizes in powers of 2
  int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
  for (int blockSize = 2; blockSize <= maxThreadsPerBlock; blockSize = blockSize*2) {
    test_cg_thread_block_type_via_public_api(blockSize);
  }

  // Test for random blockSizes, but the sequence is the same every execution
  srand(0);
  for (int i = 0; i < 10; i++) {
    // Test fails for only 1 thread per block
    test_cg_thread_block_type_via_public_api(max(2, rand() % maxThreadsPerBlock));
  }
}
