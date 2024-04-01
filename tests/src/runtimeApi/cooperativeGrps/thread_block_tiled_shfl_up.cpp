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
/* This test implements prefix sum(scan) kernel, first with each threads own rank
   as input and comparing the sum with expected serial summation output on CPU.

   This sample tests functionality of intrinsics provided by thread_block_tile type,
   shfl_up.
*/
#include "test_common.h"
#include <hip/hip_cooperative_groups.h>
#include <stdio.h>
#include <vector>

using namespace cooperative_groups;

#define ASSERT_EQUAL(lhs, rhs) assert(lhs == rhs)

template <unsigned int tileSz>
__device__ int prefix_sum_kernel(thread_block_tile<tileSz> const& g, volatile int val) {
  int sz = g.size();
#pragma unroll
  for (int i = 1; i < sz; i <<= 1) {
    int temp = g.shfl_up(val, i);

    if (g.thread_rank() >= i) {
      val += temp;
    }
  }
  return val;
}

template <unsigned int tileSz> __global__ void kernel_cg_group_partition_static(int* dPtr) {
  thread_block threadBlockCGTy = this_thread_block();
  int threadBlockGroupSize = threadBlockCGTy.size();

  int input, outputSum;

  // we pass its own thread rank as inputs
  input = threadBlockCGTy.thread_rank();

  // Choose a leader thread to print the results
  if (threadBlockCGTy.thread_rank() == 0) {
    printf(" Creating %d groups, of tile size %d threads:\n\n",
           (int)threadBlockCGTy.size() / tileSz, tileSz);
  }

  threadBlockCGTy.sync();

  thread_block_tile<tileSz> tiledPartition = tiled_partition<tileSz>(threadBlockCGTy);

  input = tiledPartition.thread_rank();

  outputSum = prefix_sum_kernel(tiledPartition, input);

  // Update the result array with the corresponsing prefix sum
  dPtr[threadBlockCGTy.thread_rank()] = outputSum;
  return;
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
      failed(" Prefix sum results do not match.");
    }
  }
}

template <unsigned int tileSz> static void test_group_partition() {
  hipError_t err;
  int blockSize = 1;
  int threadsPerBlock = 64;

  int* hPtr = NULL;
  int* dPtr = NULL;
  int* cpuPrefixSum = NULL;

  int arrSize = blockSize * threadsPerBlock * sizeof(int);

  hipHostMalloc(&hPtr, arrSize);
  hipMalloc(&dPtr, arrSize);

  // Launch Kernel
  hipLaunchKernelGGL(kernel_cg_group_partition_static<tileSz>, blockSize, threadsPerBlock,
                     threadsPerBlock * sizeof(int), 0, dPtr);
  hipMemcpy(hPtr, dPtr, arrSize, hipMemcpyDeviceToHost);
  err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to launch kernel (error code %s)!\n", hipGetErrorString(err));
  }

  cpuPrefixSum = new int[tileSz];
  serialScan(cpuPrefixSum, tileSz);
  std::cout << "\nPrefix sum results on CPU\n";
  printResults(cpuPrefixSum, tileSz);

  std::cout << "\nPrefix sum results on GPU\n";
  printResults(hPtr, tileSz);
  std::cout << "\n";
  verifyResults(hPtr, cpuPrefixSum, tileSz);
  std::cout << "Results verified!\n";

  delete[] cpuPrefixSum;
  hipFree(hPtr);
  hipFree(dPtr);
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
  std::cout << "Testing static tiled_partition for different tile sizes" << std::endl;
  /* Test static tile_partition */
  std::cout << "TEST 1:" << '\n' << std::endl;
  test_group_partition<2>();
  std::cout << "TEST 2:" << '\n' << std::endl;
  test_group_partition<4>();
  std::cout << "TEST 3:" << '\n' << std::endl;
  test_group_partition<8>();
  std::cout << "TEST 4:" << '\n' << std::endl;
  test_group_partition<16>();
  std::cout << "TEST 5:" << '\n' << std::endl;
  test_group_partition<32>();
  passed();
}

/* Kogge-Stone algorithm */