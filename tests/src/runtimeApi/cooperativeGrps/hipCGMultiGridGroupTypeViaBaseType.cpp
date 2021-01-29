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
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11 -rdc=true -gencode arch=compute_60,code=sm_60
 * TEST: %t EXCLUDE_HIP_PLATFORM nvidia
 * HIT_END
 */

#include "test_common.h"
#include "hip/hip_cooperative_groups.h"
#include <cmath>
#include <cstdlib>
#include <climits>

#define ASSERT_EQUAL(lhs, rhs) assert(lhs == rhs)
#define ASSERT_LE(lhs, rhs) assert(lhs <= rhs)
#define ASSERT_GE(lhs, rhs) assert(lhs >= rhs)

using namespace cooperative_groups;

static __global__
void kernel_cg_multi_grid_group_type_via_base_type(int *sizeTestD,
                                                   int* gridRankTestD,
                                                   int *thdRankTestD,
                                                   int *isValidTestD,
                                                   int *syncTestD,
                                                   int *syncResultD)
{
  multi_grid_group tg = this_multi_grid();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test size
  sizeTestD[gIdx] = tg.size();

  // Test thread_rank
  gridRankTestD[gIdx] = this_multi_grid().grid_rank();
  thdRankTestD[gIdx] = tg.thread_rank();

  // Test is_valid
  isValidTestD[gIdx] = tg.is_valid();

  // Test sync
  //
  // Eech thread assign 1 to their respective location
  syncTestD[gIdx] = 1;
  // Grid level sync
  this_grid().sync();
  // Thread 0 from work-group 0 of current grid (gpu) does grid level reduction
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = 1; i < gridDim.x * blockDim.x; ++i) {
      syncTestD[0] += syncTestD[i];
    }
    syncResultD[this_multi_grid().grid_rank() + 1] = syncTestD[0];
  }
  // multi-grid level sync
  tg.sync();
  // grid (gpu) 0 does final reduction across all grids (gpus)
  if (this_multi_grid().grid_rank() == 0) {
    syncResultD[0] = 0;
    for (int i = 1; i <= this_multi_grid().num_grids(); ++i) {
      syncResultD[0] += syncResultD[i];
    }
  }
}

static void test_cg_multi_grid_group_type_via_base_type(int blockSize)
{
  // Get device count
  constexpr int MaxGPUs = 8;
  int nGpu = 0;
  ASSERT_EQUAL(hipGetDeviceCount(&nGpu), hipSuccess);

  // Check if device suppurts multi gpu cooperative group support
  hipDeviceProp_t deviceProp[MaxGPUs];
  for (int i = 0; i < nGpu; i++) {
    ASSERT_EQUAL(hipSetDevice(i), hipSuccess);
    hipGetDeviceProperties(&deviceProp[i], 0);
    if (!deviceProp[i].cooperativeMultiDeviceLaunch) {
      printf("Device doesn't support multi gpu cooperative launch");
      return;
    }
    hipDeviceSynchronize();
  }

  // Create a stream each device
  hipStream_t stream[MaxGPUs];
  for (int i = 0; i < nGpu; i++) {
    ASSERT_EQUAL(hipSetDevice(i), hipSuccess);
    ASSERT_EQUAL(hipStreamCreate(&stream[i]), hipSuccess);
    hipDeviceSynchronize();
  }

  // Allocate host and device memory
  int nBytes = sizeof(int) * 2 * blockSize;
  int *sizeTestD[MaxGPUs], *sizeTestH[MaxGPUs];
  int *gridRankTestD[MaxGPUs], *gridRankTestH[MaxGPUs];
  int *thdRankTestD[MaxGPUs], *thdRankTestH[MaxGPUs];
  int *isValidTestD[MaxGPUs], *isValidTestH[MaxGPUs];
  int *syncTestD[MaxGPUs], *syncResultD;
  for (int i = 0; i < nGpu; i++) {
    ASSERT_EQUAL(hipSetDevice(i), hipSuccess);

    ASSERT_EQUAL(hipMalloc(&sizeTestD[i], nBytes), hipSuccess);
    ASSERT_EQUAL(hipMalloc(&gridRankTestD[i], nBytes), hipSuccess);
    ASSERT_EQUAL(hipMalloc(&thdRankTestD[i], nBytes), hipSuccess);
    ASSERT_EQUAL(hipMalloc(&isValidTestD[i], nBytes), hipSuccess);
    ASSERT_EQUAL(hipMalloc(&syncTestD[i], nBytes), hipSuccess);

    ASSERT_EQUAL(hipHostMalloc(&sizeTestH[i], nBytes), hipSuccess);
    ASSERT_EQUAL(hipHostMalloc(&gridRankTestH[i], nBytes), hipSuccess);
    ASSERT_EQUAL(hipHostMalloc(&thdRankTestH[i], nBytes), hipSuccess);
    ASSERT_EQUAL(hipHostMalloc(&isValidTestH[i], nBytes), hipSuccess);

    if (i == 0) {
      ASSERT_EQUAL(
        hipHostMalloc(&syncResultD, sizeof(int) * (nGpu + 1), hipHostMallocCoherent),
        hipSuccess);
    }

    hipDeviceSynchronize();
  }

  // Launch Kernel
  constexpr int NumKernelArgs = 6;
  hipLaunchParams* launchParamsList = new hipLaunchParams[nGpu];
  void* args[MaxGPUs * NumKernelArgs];
  for (int i = 0; i < nGpu; i++) {
    ASSERT_EQUAL(hipSetDevice(i), hipSuccess);

    args[i * NumKernelArgs    ] = &sizeTestD[i];
    args[i * NumKernelArgs + 1] = &gridRankTestD[i];
    args[i * NumKernelArgs + 2] = &thdRankTestD[i];
    args[i * NumKernelArgs + 3] = &isValidTestD[i];
    args[i * NumKernelArgs + 4] = &syncTestD[i];
    args[i * NumKernelArgs + 5] = &syncResultD;

    launchParamsList[i].func = reinterpret_cast<void*>(kernel_cg_multi_grid_group_type_via_base_type);
    launchParamsList[i].gridDim = 2;
    launchParamsList[i].blockDim = blockSize;
    launchParamsList[i].sharedMem = 0;
    launchParamsList[i].stream = stream[i];
    launchParamsList[i].args = &args[i * NumKernelArgs];

    hipDeviceSynchronize();
  }
  hipLaunchCooperativeKernelMultiDevice(launchParamsList, nGpu, 0);

  // Copy result from device to host
  for (int i = 0; i < nGpu; i++) {
    ASSERT_EQUAL(hipSetDevice(i), hipSuccess);

    ASSERT_EQUAL(hipMemcpy(sizeTestH[i], sizeTestD[i], nBytes, hipMemcpyDeviceToHost),
                 hipSuccess);
    ASSERT_EQUAL(hipMemcpy(gridRankTestH[i], gridRankTestD[i], nBytes, hipMemcpyDeviceToHost),
                 hipSuccess);
    ASSERT_EQUAL(hipMemcpy(thdRankTestH[i], thdRankTestD[i], nBytes, hipMemcpyDeviceToHost),
                 hipSuccess);
    ASSERT_EQUAL(hipMemcpy(isValidTestH[i], isValidTestD[i], nBytes, hipMemcpyDeviceToHost),
                 hipSuccess);

    hipDeviceSynchronize();
  }

  // Validate results
  int gridsSeen[MaxGPUs];
  for (int i = 0; i < nGpu; ++i) {
    for (int j = 0; j < 2 * blockSize; ++j) {
      ASSERT_EQUAL(sizeTestH[i][j], nGpu * 2 * blockSize);
      ASSERT_GE(gridRankTestH[i][j], 0);
      ASSERT_LE(gridRankTestH[i][j], nGpu-1);
      ASSERT_EQUAL(gridRankTestH[i][j], gridRankTestH[i][0]);
      int gridRank = gridRankTestH[i][j];
      ASSERT_EQUAL(thdRankTestH[i][j], (gridRank * 2 * blockSize) + j);
      ASSERT_EQUAL(isValidTestH[i][j], 1);
    }
    ASSERT_EQUAL(syncResultD[i+1],  2 * blockSize);

    // Validate uniqueness property of grid rank
    gridsSeen[i] = gridRankTestH[i][0];
    for (int k = 0; k < i; ++k) {
      if (gridsSeen[k] == gridsSeen[i]) {
        assert (false && "Grid rank in multi-gpu setup should be unique");
      }
    }
  }
  ASSERT_EQUAL(syncResultD[0], nGpu * 2 * blockSize);

  // Free host and device memory
  delete [] launchParamsList;
  for (int i = 0; i < nGpu; i++) {
    ASSERT_EQUAL(hipSetDevice(i), hipSuccess);

    ASSERT_EQUAL(hipFree(sizeTestD[i]), hipSuccess);
    ASSERT_EQUAL(hipFree(gridRankTestD[i]), hipSuccess);
    ASSERT_EQUAL(hipFree(thdRankTestD[i]), hipSuccess);
    ASSERT_EQUAL(hipFree(isValidTestD[i]), hipSuccess);
    ASSERT_EQUAL(hipFree(syncTestD[i]), hipSuccess);

    if (i == 0)
      ASSERT_EQUAL(hipFree(syncResultD), hipSuccess);

    ASSERT_EQUAL(hipHostFree(sizeTestH[i]), hipSuccess);
    ASSERT_EQUAL(hipHostFree(gridRankTestH[i]), hipSuccess);
    ASSERT_EQUAL(hipHostFree(thdRankTestH[i]), hipSuccess);
    ASSERT_EQUAL(hipHostFree(isValidTestH[i]), hipSuccess);

    hipDeviceSynchronize();
  }
}

int main()
{
  // Set `maxThreadsPerBlock` by taking minimum among all available devices
  int nGpu = 0;
  ASSERT_EQUAL(hipGetDeviceCount(&nGpu), hipSuccess);
  int maxThreadsPerBlock = INT_MAX;
  for (int i = 0; i < nGpu; i++) {
    hipDeviceProp_t deviceProperties;
    ASSERT_EQUAL(hipGetDeviceProperties(&deviceProperties, i), hipSuccess);
    int curDeviceMaxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
    maxThreadsPerBlock = min(maxThreadsPerBlock, curDeviceMaxThreadsPerBlock);
    hipDeviceSynchronize();
  }

  // Test block sizes which are powers of 2
  int i = 0;
  while (true) {
    int blockSize = pow(2, i);
    if (blockSize > maxThreadsPerBlock)
      break;
    test_cg_multi_grid_group_type_via_base_type(blockSize);
    ++i;
  }

  // Test some random block sizes
  for(int j = 0; j < 10 ; ++j) {
    int blockSize = rand() % maxThreadsPerBlock;
    test_cg_multi_grid_group_type_via_base_type(blockSize);
  }

  passed();
}
