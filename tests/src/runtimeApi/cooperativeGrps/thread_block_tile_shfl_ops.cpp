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
// Test Description:
/* This test implements sum reduction kernel, first with each threads own rank
   as input and comparing the sum with expected sum output derieved from n(n-1)/2
   formula.
   This sample tests functionality of intrinsics provided by thread_block_tile type,
   shfl_down and shfl_xor.
*/

#include "test_common.h"
#include <hip/hip_cooperative_groups.h>
#include <stdio.h>
#include <vector>

using namespace cooperative_groups;

#define ASSERT_EQUAL(lhs, rhs) assert(lhs == rhs)

template <unsigned int tileSz>
__device__ int reduction_kernel_shfl_down(thread_block_tile<tileSz> const& g, volatile int val) {
  int sz = g.size();

  for (int i = sz / 2; i > 0; i >>= 1) {
    val += g.shfl_down(val, i);
  }

  // Choose the 0'th indexed thread that holds the reduction value to return
  if (g.thread_rank() == 0) {
    return val;
  }
  // Rest of the threads return no useful values
  else {
    return -1;
  }
}

template <unsigned int tileSz>
__device__ int reduction_kernel_shfl_xor(thread_block_tile<tileSz> const& g, int val) {
  int sz = g.size();

  for (int i = sz / 2; i > 0; i >>= 1) {
    val += g.shfl_xor(val, i);
  }

  // Choose the 0'th indexed thread that holds the reduction value to return
  if (g.thread_rank() == 0) {
    return val;
  }
  // Rest of the threads return no useful values
  else {
    return -1;
  }
}

template <unsigned int tileSz>
__global__ void kernel_cg_group_partition_static(int* result, bool runShflDown) {
  thread_block threadBlockCGTy = this_thread_block();
  int threadBlockGroupSize = threadBlockCGTy.size();
  int input, outputSum, expectedSum;

  // Choose a leader thread to print the results
  if (threadBlockCGTy.thread_rank() == 0) {
    printf(" Creating %d groups, of tile size %d threads:\n\n",
           (int)threadBlockCGTy.size() / tileSz, tileSz);
  }

  threadBlockCGTy.sync();

  thread_block_tile<tileSz> tiledPartition = tiled_partition<tileSz>(threadBlockCGTy);
  int threadRank = tiledPartition.thread_rank();

  input = tiledPartition.thread_rank();

  // (n-1)(n)/2
  expectedSum = ((tileSz - 1) * tileSz / 2);

  if (runShflDown) {
    outputSum = reduction_kernel_shfl_down(tiledPartition, input);

    if (tiledPartition.thread_rank() == 0) {
      printf(
          "   Sum of all ranks 0..%d in this tiledPartition group using shfl_down is %d (expected "
          "%d)\n",
          tiledPartition.size() - 1, outputSum, expectedSum);
      result[threadBlockCGTy.thread_rank() / (tileSz)] = outputSum;
    }
  } else {
    outputSum = reduction_kernel_shfl_xor(tiledPartition, input);

    if (tiledPartition.thread_rank() == 0) {
      printf(
          "   Sum of all ranks 0..%d in this tiledPartition group using shfl_xor is %d (expected "
          "%d)\n",
          tiledPartition.size() - 1, outputSum, expectedSum);
      result[threadBlockCGTy.thread_rank() / (tileSz)] = outputSum;
    }
  }

  return;
}

void verifyResults(int* ptr, int expectedResult, int numTiles) {
  for (int i = 0; i < numTiles; i++) {
    if (ptr[i] != expectedResult) {
      failed(" Results do not match! ");
    }
  }
}

template <unsigned int tileSz> static void test_group_partition(bool runShflDown) {
  hipError_t err;
  int blockSize = 1;
  int threadsPerBlock = 64;

  int numTiles = (blockSize * threadsPerBlock) / tileSz;
  int expectedSum = ((tileSz - 1) * tileSz / 2);
  int* expectedResult = new int[numTiles];

  for (int i = 0; i < numTiles; i++) {
    expectedResult[i] = expectedSum;
  }

  int* dResult = NULL;
  int* hResult = NULL;

  hipHostMalloc(&hResult, numTiles * sizeof(int), hipHostMallocDefault);
  memset(hResult, 0, numTiles * sizeof(int));

  hipMalloc(&dResult, numTiles * sizeof(int));

  if (runShflDown) {
    // Launch Kernel
    hipLaunchKernelGGL(kernel_cg_group_partition_static<tileSz>, blockSize, threadsPerBlock,
                       threadsPerBlock * sizeof(int), 0, dResult, runShflDown);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }
  } else {
    // Launch Kernel
    hipLaunchKernelGGL(kernel_cg_group_partition_static<tileSz>, blockSize, threadsPerBlock,
                       threadsPerBlock * sizeof(int), 0, dResult, runShflDown);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }
  }

  hipMemcpy(hResult, dResult, sizeof(int) * numTiles, hipMemcpyDeviceToHost);

  verifyResults(hResult, expectedSum, numTiles);

  // Free all allocated memory on host and device
  hipFree(dResult);
  hipFree(hResult);
  delete[] expectedResult;

  printf("\n...PASSED.\n\n");
}


int main() {
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

  bool runShflDown = true;
  std::cout << "Testing static tiled_partition for different tile sizes using shfl_down"
            << std::endl;
  /* Test static tile_partition */
  std::cout << "TEST 1:" << '\n' << std::endl;
  test_group_partition<2>(runShflDown);
  std::cout << "TEST 2:" << '\n' << std::endl;
  test_group_partition<4>(runShflDown);
  std::cout << "TEST 3:" << '\n' << std::endl;
  test_group_partition<8>(runShflDown);
  std::cout << "TEST 4:" << '\n' << std::endl;
  test_group_partition<16>(runShflDown);
  std::cout << "TEST 5:" << '\n' << std::endl;
  test_group_partition<32>(runShflDown);

  runShflDown = false;
  std::cout << "Testing static tiled_partition for different tile sizes using shfl_xor"
            << std::endl;
  /* Test static tile_partition */
  std::cout << "TEST 1:" << '\n' << std::endl;
  test_group_partition<2>(runShflDown);
  std::cout << "TEST 2:" << '\n' << std::endl;
  test_group_partition<4>(runShflDown);
  std::cout << "TEST 3:" << '\n' << std::endl;
  test_group_partition<8>(runShflDown);
  std::cout << "TEST 4:" << '\n' << std::endl;
  test_group_partition<16>(runShflDown);
  std::cout << "TEST 5:" << '\n' << std::endl;
  test_group_partition<32>(runShflDown);

  passed();
}
