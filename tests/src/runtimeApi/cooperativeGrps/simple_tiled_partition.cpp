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

// Test Description:
/* This test implements sum reduction kernel, first with each threads own rank
   as input and comparing the sum with expected sum output derieved from n(n-1)/2
   formula. The second part, partitions this parent group into child subgroups
   a.k.a tiles using using tiled_partition() collective operation. This can be called
   with a static tile size, passed in templated non-type variable-tiled_partition<tileSz>,
   or in runtime as tiled_partition(thread_group parent, tileSz). This test covers both these
   cases.
   This test tests functionality of cg group partitioning, (static and dynamic) and its respective
   API's size(), thread_rank(), and sync().
*/

#include "test_common.h"
#include <hip/hip_cooperative_groups.h>
#include <stdio.h>
#include <vector>

using namespace cooperative_groups;

#define ASSERT_EQUAL(lhs, rhs) assert(lhs == rhs)

/* Parallel reduce kernel.
 *
 * Step complexity: O(log n)
 * Work complexity: O(n)
 *
 * Note: This kernel works only with power of 2 input arrays.
 */
__device__ int reduction_kernel(thread_group g, int* x, int val) {
  int lane = g.thread_rank();
  int sz = g.size();

  for (int i = g.size() / 2; i > 0; i /= 2) {
    // use lds to store the temporary result
    x[lane] = val;
    // Ensure all the stores are completed.
    g.sync();

    if (lane < i) {
      val += x[lane + i];
    }
    // It must work on one tiled thread group at a time,
    // and it must make sure all memory operations are
    // completed before moving to the next stride.
    // sync() here just does that.
    g.sync();
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
__global__ void kernel_cg_group_partition_static(int* result, bool isGlobalMem, int* globalMem) {
  thread_block threadBlockCGTy = this_thread_block();
  int threadBlockGroupSize = threadBlockCGTy.size();

  int* workspace = NULL;

  if (isGlobalMem) {
    workspace = globalMem;
  } else {
    // Declare a shared memory
    extern __shared__ int sharedMem[];
    workspace = sharedMem;
  }

  int input, outputSum, expectedOutput;

  // we pass its own thread rank as inputs
  input = threadBlockCGTy.thread_rank();

  expectedOutput = (threadBlockGroupSize - 1) * threadBlockGroupSize / 2;

  outputSum = reduction_kernel(threadBlockCGTy, workspace, input);

  // Choose a leader thread to print the results
  if (threadBlockCGTy.thread_rank() == 0) {
    printf(" Sum of all ranks 0..%d in threadBlockCooperativeGroup is %d (expected %d)\n\n",
           (int)threadBlockCGTy.size() - 1, outputSum, expectedOutput);
    printf(" Creating %d groups, of tile size %d threads:\n\n",
           (int)threadBlockCGTy.size() / tileSz, tileSz);
  }

  threadBlockCGTy.sync();

  thread_block_tile<tileSz> tiledPartition = tiled_partition<tileSz>(threadBlockCGTy);

  // This offset allows each group to have its own unique area in the workspace array
  int workspaceOffset = threadBlockCGTy.thread_rank() - tiledPartition.thread_rank();

  outputSum = reduction_kernel(tiledPartition, workspace + workspaceOffset, input);

  if (tiledPartition.thread_rank() == 0) {
    printf(
        "   Sum of all ranks 0..%d in this tiledPartition group is %d. Corresponding parent thread "
        "rank: %d\n",
        tiledPartition.size() - 1, outputSum, input);
    result[input / (tileSz)] = outputSum;
  }
  return;
}


__global__ void kernel_cg_group_partition_dynamic(unsigned int tileSz, int* result,
                                                  bool isGlobalMem, int* globalMem) {
  thread_block threadBlockCGTy = this_thread_block();
  int threadBlockGroupSize = threadBlockCGTy.size();

  int* workspace = NULL;

  if (isGlobalMem) {
    workspace = globalMem;
  } else {
    // Declare a shared memory
    extern __shared__ int sharedMem[];
    workspace = sharedMem;
  }

  int input, outputSum, expectedOutput;

  // input to reduction, for each thread, is its' rank in the group
  input = threadBlockCGTy.thread_rank();

  expectedOutput = (threadBlockGroupSize - 1) * threadBlockGroupSize / 2;

  outputSum = reduction_kernel(threadBlockCGTy, workspace, input);

  if (threadBlockCGTy.thread_rank() == 0) {
    printf(" Sum of all ranks 0..%d in threadBlockCooperativeGroup is %d\n\n",
           (int)threadBlockCGTy.size() - 1, outputSum);
    printf(" Creating %d groups, of tile size %d threads:\n\n",
           (int)threadBlockCGTy.size() / tileSz, tileSz);
  }

  threadBlockCGTy.sync();

  thread_group tiledPartition = tiled_partition(threadBlockCGTy, tileSz);

  // This offset allows each group to have its own unique area in the workspace array
  int workspaceOffset = threadBlockCGTy.thread_rank() - tiledPartition.thread_rank();

  outputSum = reduction_kernel(tiledPartition, workspace + workspaceOffset, input);

  if (tiledPartition.thread_rank() == 0) {
    printf(
        "   Sum of all ranks 0..%d in this tiledPartition group is %d. Corresponding parent thread "
        "rank: %d\n",
        tiledPartition.size() - 1, outputSum, input);

    result[input / (tileSz)] = outputSum;
  }
  return;
}

// Search if the sum exists in the expected results array
void verifyResults(int* hPtr, int* dPtr, int size) {
  int i = 0, j = 0;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      if (hPtr[i] == dPtr[j]) {
        break;
      }
    }
    if (j == size) {
      failed(" Result verification failed!");
    }
  }
}


template <unsigned int tileSz> static void test_group_partition(bool useGlobalMem) {
  hipError_t err;
  int blockSize = 1;
  int threadsPerBlock = 64;

  int numTiles = (blockSize * threadsPerBlock) / tileSz;

  // Build an array of expected reduction sum output on the host
  // based on the sum of their respective thread ranks for verification.
  // eg: parent group has 64threads.
  // child thread ranks: 0-15, 16-31, 32-47, 48-63
  // expected sum:       120,   376,  632,  888
  int* expectedSum = new int[numTiles];
  int temp = 0, sum = 0;

  for (int i = 1; i <= numTiles; i++) {
    sum = temp;
    temp = (((tileSz * i) - 1) * (tileSz * i)) / 2;
    expectedSum[i-1] = temp - sum;
  }

  int* dResult = NULL;
  hipMalloc((void**)&dResult, numTiles * sizeof(int));

  int* globalMem = NULL;
  if (useGlobalMem) {
    hipMalloc((void**)&globalMem, threadsPerBlock * sizeof(int));
  }

  int* hResult = NULL;
  hipHostMalloc(&hResult, numTiles * sizeof(int), hipHostMallocDefault);
  memset(hResult, 0, numTiles * sizeof(int));

  if (useGlobalMem) {
    // Launch Kernel
    hipLaunchKernelGGL(kernel_cg_group_partition_static<tileSz>, blockSize, threadsPerBlock, 0, 0,
                       dResult, useGlobalMem, globalMem);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }
  } else {
    // Launch Kernel
    hipLaunchKernelGGL(kernel_cg_group_partition_static<tileSz>, blockSize, threadsPerBlock,
                       threadsPerBlock * sizeof(int), 0, dResult, useGlobalMem, globalMem);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }
  }

  hipMemcpy(hResult, dResult, numTiles * sizeof(int), hipMemcpyDeviceToHost);

  verifyResults(expectedSum, hResult, numTiles);

  // Free all allocated memory on host and device
  hipFree(dResult);
  hipFree(hResult);
  if (useGlobalMem) {
    hipFree(globalMem);
  }
  delete[] expectedSum;

  printf("\n...PASSED.\n\n");
}

static void test_group_partition(unsigned int tileSz, bool useGlobalMem) {
  hipError_t err;
  int blockSize = 1;
  int threadsPerBlock = 64;

  int numTiles = (blockSize * threadsPerBlock) / tileSz;
  // Build an array of expected reduction sum output on the host
  // based on the sum of their respective thread ranks to use for verification
  int* expectedSum = new int[numTiles];
  int temp = 0, sum = 0;
  for (int i = 1; i <= numTiles; i++) {
    sum = temp;
    temp = (((tileSz * i) - 1) * (tileSz * i)) / 2;
    expectedSum[i-1] = temp - sum;
  }

  int* dResult = NULL;
  hipMalloc(&dResult, sizeof(int) * numTiles);

  int* globalMem = NULL;
  if (useGlobalMem) {
    hipMalloc((void**)&globalMem, threadsPerBlock * sizeof(int));
  }

  int* hResult = NULL;
  hipHostMalloc(&hResult, numTiles * sizeof(int), hipHostMallocDefault);
  memset(hResult, 0, numTiles * sizeof(int));

  // Launch Kernel
  if (useGlobalMem) {
    hipLaunchKernelGGL(kernel_cg_group_partition_dynamic, blockSize, threadsPerBlock, 0, 0, tileSz,
                       dResult, useGlobalMem, globalMem);

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }
  } else {
    hipLaunchKernelGGL(kernel_cg_group_partition_dynamic, blockSize, threadsPerBlock,
                       threadsPerBlock * sizeof(int), 0, tileSz, dResult, useGlobalMem, globalMem);

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }
  }

  hipMemcpy(hResult, dResult, numTiles * sizeof(int), hipMemcpyDeviceToHost);

  verifyResults(expectedSum, hResult, numTiles);

  // Free all allocated memory on host and device
  hipFree(dResult);
  hipFree(hResult);
  if (useGlobalMem) {
    hipFree(globalMem);
  }
  delete[] expectedSum;

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
  }

  bool useGlobalMem = true;
  std::cout << "Testing static tiled_partition for different tile sizes" << std::endl;
  std::cout << "\nUsing global memory for computation\n";
  /* Test static tile_partition */
  std::cout << "TEST 1:" << '\n' << std::endl;
  test_group_partition<2>(useGlobalMem);
  std::cout << "TEST 2:" << '\n' << std::endl;
  test_group_partition<4>(useGlobalMem);
  std::cout << "TEST 3:" << '\n' << std::endl;
  test_group_partition<8>(useGlobalMem);
  std::cout << "TEST 4:" << '\n' << std::endl;
  test_group_partition<16>(useGlobalMem);
  std::cout << "TEST 5:" << '\n' << std::endl;
  test_group_partition<32>(useGlobalMem);

  useGlobalMem = false;
  std::cout << "Testing static tiled_partition for different tile sizes" << std::endl;
  std::cout << "\nUsing shared memory for computation\n";
  /* Test static tile_partition */
  std::cout << "TEST 1:" << '\n' << std::endl;
  test_group_partition<2>(useGlobalMem);
  std::cout << "TEST 2:" << '\n' << std::endl;
  test_group_partition<4>(useGlobalMem);
  std::cout << "TEST 3:" << '\n' << std::endl;
  test_group_partition<8>(useGlobalMem);
  std::cout << "TEST 4:" << '\n' << std::endl;
  test_group_partition<16>(useGlobalMem);
  std::cout << "TEST 5:" << '\n' << std::endl;
  test_group_partition<32>(useGlobalMem);


  std::cout << "Now testing dynamic tiled_partition for different tile sizes" << '\n' << std::endl;

  /* Test dynamic group partition*/
  useGlobalMem = true;
  int testNo = 1;
  std::vector<unsigned int> tileSizes = {2, 4, 8, 16, 32};
  std::cout << "\nUsing global memory for computation\n";
  for (auto i : tileSizes) {
    std::cout << "TEST " << testNo << ":" << '\n' << std::endl;
    test_group_partition(i, useGlobalMem);
    testNo++;
  }

  useGlobalMem = false;
  testNo = 1;
  std::cout << "\nUsing shared memory for computation\n";
  for (auto i : tileSizes) {
    std::cout << "TEST " << testNo << ":" << '\n' << std::endl;
    test_group_partition(i, useGlobalMem);
    testNo++;
  }

  passed();
  return 0;
}
