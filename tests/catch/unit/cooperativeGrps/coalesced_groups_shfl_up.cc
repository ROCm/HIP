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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// Test Description:
/* This test implements prefix sum(scan) kernel, first with each threads own rank
   as input and comparing the sum with expected serial summation output on CPU.

   This sample tests functionality of intrinsics provided by coalesced_group,
   shfl_up.
*/
#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <stdio.h>
#include <vector>

using namespace cooperative_groups;

#define ASSERT_EQUAL(lhs, rhs) assert(lhs == rhs)
#define WAVE_SIZE 32
__device__ int prefix_sum_kernel(coalesced_group const& g, int val) {
  int sz = g.size();
  for (int i = 1; i < sz; i <<= 1) {
    int temp = g.shfl_up(val, i);

    if (g.thread_rank() >= i) {
      val += temp;
    }
  }
  return val;
}

__global__ void kernel_shfl_up (int * dPtr, int *dResults, int lane_delta, int cg_sizes) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id % cg_sizes == 0) {
    coalesced_group g = coalesced_threads();
    int rank = g.thread_rank();
    int val = dPtr[rank];
    dResults[rank] = g.shfl_up(val, lane_delta);
  return;
  }

}

__global__ void kernel_cg_group_partition(int* dPtr, unsigned int tileSz, int cg_sizes) {

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id % cg_sizes == 0) {
    coalesced_group threadBlockCGTy = coalesced_threads();
    int input, outputSum;

    // we pass its own thread rank as inputs
    input = threadBlockCGTy.thread_rank();

    // Choose a leader thread to print the results
    if (threadBlockCGTy.thread_rank() == 0) {
      printf(" Creating %d groups, of tile size %d threads:\n\n",
             (int)threadBlockCGTy.size() / tileSz, tileSz);
    }

    threadBlockCGTy.sync();

    coalesced_group tiledPartition = tiled_partition(threadBlockCGTy, tileSz);

    input = tiledPartition.thread_rank();

    outputSum = prefix_sum_kernel(tiledPartition, input);

    // Update the result array with the corresponsing prefix sum
    dPtr[threadBlockCGTy.thread_rank()] = outputSum;
    return;
  }
}

void serialScan(int* ptr, int size) {
  // Fill up the array
  for (int i = 0; i < size; i++) {
    ptr[i] = i;
  }

  int acc = 0;
  for (int i = 0; i < size; i++) {
    acc = acc + ptr[i];
    ptr[i] = acc;
  }
}

void printResults(int* ptr, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << '\n';
}

void verifyResults(int* cpu, int* gpu, int size) {
  for (unsigned int i = 0; i < size / sizeof(int); i++) {
    if (cpu[i] != gpu[i]) {
      INFO(" Results do not match.");
    }
  }
}

static void test_group_partition(unsigned tileSz) {
  hipError_t err;
  int blockSize = 1;
  int threadsPerBlock = WAVE_SIZE;

  int* hPtr = NULL;
  int* dPtr = NULL;
  int* cpuPrefixSum = NULL;

  std::vector<unsigned int> cg_sizes = {1, 2, 3};
  for (auto i : cg_sizes) {

    int arrSize = blockSize * threadsPerBlock * sizeof(int);

    hipHostMalloc(&hPtr, arrSize);
    hipMalloc(&dPtr, arrSize);

    // Launch Kernel
    hipLaunchKernelGGL(kernel_cg_group_partition, blockSize, threadsPerBlock,
                     threadsPerBlock * sizeof(int), 0, dPtr, tileSz, i);
    hipMemcpy(hPtr, dPtr, arrSize, hipMemcpyDeviceToHost);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }

    cpuPrefixSum = new int[tileSz];
    serialScan(cpuPrefixSum, tileSz);
    //std::cout << "\nPrefix sum results on CPU\n";
    //printResults(cpuPrefixSum, tileSz);

    //std::cout << "\nPrefix sum results on GPU\n";
    //printResults(hPtr, tileSz);
    std::cout << "\n";
    verifyResults(hPtr, cpuPrefixSum, tileSz);
    std::cout << "Results verified!\n";

    delete[] cpuPrefixSum;
    hipFree(hPtr);
    hipFree(dPtr);
  }
}

static void test_shfl_up() {

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
    int lane_delta =  (rand() % group_size);

    std::cout << "Testing coalesced_groups shfl_up with lane_delta " << lane_delta
              << " and group size " << WAVE_SIZE << '\n' << std::endl;

    int arrSize = blockSize * threadsPerBlock * sizeof(int);

    hipHostMalloc(&hPtr, arrSize);
    // Fill up the array
    for (int i = 0; i < WAVE_SIZE; i++) {
      hPtr[i] = rand() % 1000;
    }
    //printResults(hPtr, WAVE_SIZE);

    int* cpuResultsArr = (int*)malloc(group_size_in_bytes);
    for (int i = 0; i < group_size; i++) {
      cpuResultsArr[i] = (i <= (lane_delta - 1)) ?  hPtr[i] : hPtr[i - lane_delta];
    }

    //printf("Printing cpu results arr\n");
    //printResults(cpuResultsArr, WAVE_SIZE);

    hipMalloc(&dPtr, group_size_in_bytes);
    hipMalloc(&dResults, group_size_in_bytes);

    hipMemcpy(dPtr, hPtr, group_size_in_bytes, hipMemcpyHostToDevice);
    // Launch Kernel
    hipLaunchKernelGGL(kernel_shfl_up, blockSize, threadsPerBlock,
                       threadsPerBlock * sizeof(int), 0, dPtr, dResults, lane_delta, i);
    hipMemcpy(hPtr, dResults, group_size_in_bytes, hipMemcpyDeviceToHost);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
    }
    //printf("GPU computation array :\n");
    //printResults(hPtr, WAVE_SIZE);

    verifyResults(hPtr, cpuResultsArr, group_size_in_bytes);
    std::cout << "Results verified!\n";

    hipFree(hPtr);
    hipFree(dPtr);
    free(cpuResultsArr);
  }
}

TEST_CASE("Unit_coalesced_groups_shfl_down") {
  // Use default device for validating the test
  int deviceId;
  ASSERT_EQUAL(hipGetDevice(&deviceId), hipSuccess);
  hipDeviceProp_t deviceProperties;
  ASSERT_EQUAL(hipGetDeviceProperties(&deviceProperties, deviceId), hipSuccess);
  int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

  for (int i = 0; i < 100; i++) {
      test_shfl_up();
  }

  std::cout << "Testing coalesced_groups partitioning and shfl_up" << '\n' << std::endl;

  int testNo = 1;
  std::vector<unsigned int> tileSizes = {2, 4, 8, 16, 32};
  for (auto i : tileSizes) {
    std::cout << "TEST " << testNo << ":" << '\n' << std::endl;
    test_group_partition(i);
    testNo++;
  }
}

/* Kogge-Stone algorithm */