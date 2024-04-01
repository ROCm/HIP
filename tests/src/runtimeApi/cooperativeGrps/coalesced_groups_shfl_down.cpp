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
#define WAVE_SIZE 32

__device__ int reduction_kernel_shfl_down(coalesced_group const& g, int val) {
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

__global__ void kernel_shfl_down (int * dPtr, int *dResults, int lane_delta, int cg_sizes) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id % cg_sizes == 0) {
    coalesced_group const& g = coalesced_threads();
    int rank = g.thread_rank();
    int val = dPtr[rank];
    dResults[rank] = g.shfl_down(val, lane_delta);
    return;
  }
}

__global__ void kernel_cg_group_partition(int* result, unsigned int tileSz, int cg_sizes) {

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id % cg_sizes == 0) {
    coalesced_group  threadBlockCGTy = coalesced_threads();
    int input, outputSum, expectedSum;

    // Choose a leader thread to print the results
    if (threadBlockCGTy.thread_rank() == 0) {
      printf(" Creating %d groups, of tile size %d threads:\n\n",
             (int)threadBlockCGTy.size() / tileSz, tileSz);
    }

    threadBlockCGTy.sync();

    coalesced_group tiledPartition = tiled_partition(threadBlockCGTy, tileSz);
    int threadRank = tiledPartition.thread_rank();

    input = tiledPartition.thread_rank();

    // (n-1)(n)/2
    expectedSum = ((tileSz - 1) * tileSz / 2);

    outputSum = reduction_kernel_shfl_down(tiledPartition, input);

    if (tiledPartition.thread_rank() == 0) {
      printf(
          "   Sum of all ranks 0..%d in this tiledPartition group using shfl_down is %d (expected "
          "%d)\n",
          tiledPartition.size() - 1, outputSum, expectedSum);
      result[threadBlockCGTy.thread_rank() / (tileSz)] = outputSum;
    }
    return;
  }
}

void verifyResults(int* ptr, int expectedResult, int numTiles) {
  for (int i = 0; i < numTiles; i++) {
    if (ptr[i] != expectedResult) {
      printf(" Results do not match! ");
    }
  }
}

void compareResults(int* cpu, int* gpu, int size) {
  for (unsigned int i = 0; i < size / sizeof(int); i++) {
    if (cpu[i] != gpu[i]) {
      printf(" results do not match.");
    }
  }
}

void printResults(int* ptr, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << '\n';
}

static void test_group_partition(unsigned int tileSz) {
  hipError_t err;
  int blockSize = 1;
  int threadsPerBlock = 32;

  std::vector<unsigned int> cg_sizes = {1, 2, 3};
  for (auto i : cg_sizes) {

    int numTiles = ((blockSize * threadsPerBlock) / i) / tileSz;
    int expectedSum = ((tileSz - 1) * tileSz / 2);
    int* expectedResult = new int[numTiles];

    // numTiles = 0 when partitioning is possible. The below statement is to avoid
    // out-of-bounds error and still evaluate failure case.
    numTiles = (numTiles == 0) ? 1 : numTiles;

    for (int i = 0; i < numTiles; i++) {
      expectedResult[i] = expectedSum;
    }

    int* dResult = NULL;
    int* hResult = NULL;

    hipHostMalloc(&hResult, numTiles * sizeof(int), hipHostMallocDefault);
    memset(hResult, 0, numTiles * sizeof(int));

    hipMalloc(&dResult, numTiles * sizeof(int));


    // Launch Kernel
    hipLaunchKernelGGL(kernel_cg_group_partition, blockSize, threadsPerBlock,
                       threadsPerBlock * sizeof(int), 0, dResult, tileSz, i);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }


    hipMemcpy(hResult, dResult, sizeof(int) * numTiles, hipMemcpyDeviceToHost);

    verifyResults(hResult, expectedSum, numTiles);

    // Free all allocated memory on host and device
    hipFree(dResult);
    hipFree(hResult);
    delete[] expectedResult;

    printf("\n...PASSED.\n\n");
  }
}

static void test_shfl_down() {

  std::vector<unsigned int> cg_sizes = {1, 2, 3};
  for (auto i : cg_sizes) {

    hipError_t err;
    int blockSize = 1;
    int threadsPerBlock = WAVE_SIZE;

    int totalThreads = blockSize * threadsPerBlock;
    int group_size = totalThreads / i;
    int group_size_in_bytes = group_size * sizeof(int);

    int* hPtr = NULL;
    int* dPtr = NULL;
    int* dResults = NULL;
    int lane_delta = rand() % group_size;
    std::cout << "Testing coalesced_groups shfl_down with lane_delta " << lane_delta << "and group size "
              << WAVE_SIZE << '\n' << std::endl;

    int arrSize = blockSize * threadsPerBlock * sizeof(int);

    hipHostMalloc(&hPtr, arrSize);
    // Fill up the array
    for (int i = 0; i < WAVE_SIZE; i++) {
      hPtr[i] = rand() % 1000;
    }

    int* cpuResultsArr = (int*)malloc(group_size_in_bytes);
    for (int i = 0; i < group_size; i++) {
      cpuResultsArr[i] = (i + lane_delta >= group_size) ? hPtr[i] : hPtr[i + lane_delta];
    }
    //printf("Array passed to GPU for computation\n");
    //printResults(hPtr, WAVE_SIZE);
    hipMalloc(&dPtr, group_size_in_bytes);
    hipMalloc(&dResults, group_size_in_bytes);

    hipMemcpy(dPtr, hPtr, group_size_in_bytes, hipMemcpyHostToDevice);
    // Launch Kernel
    hipLaunchKernelGGL(kernel_shfl_down, blockSize, threadsPerBlock,
                       threadsPerBlock * sizeof(int), 0, dPtr, dResults, lane_delta, i);
    hipMemcpy(hPtr, dResults, group_size_in_bytes, hipMemcpyDeviceToHost);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }
    //printf("GPU results: \n");
    //printResults(hPtr, WAVE_SIZE);
    //printf("Printing cpu to be verified array\n");
    //printResults(cpuResultsArr, WAVE_SIZE);

    compareResults(hPtr, cpuResultsArr, group_size_in_bytes);
    std::cout << "Results verified!\n";

    hipFree(hPtr);
    hipFree(dPtr);
    free(cpuResultsArr);
  }
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

  // Test shfl_down with random group sizes
    for (int i = 0; i < 100; i++) {
      test_shfl_down();
    }

  std::cout << "Testing static tiled_partition for different tile sizes using shfl_down"
            << std::endl;

  int testNo = 1;
  std::vector<unsigned int> tileSizes = {2, 4, 8, 16, 32};
  for (auto i : tileSizes) {
    std::cout << "TEST " << testNo << ":" << '\n' << std::endl;
    test_group_partition(i);
    testNo++;
  }

  passed();
}
