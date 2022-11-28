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
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11 -rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80
 * TEST: %t
 * HIT_END
 */

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

#define ASSERT_EQUAL(lhs, rhs) HIPASSERT(lhs == rhs)
#define ASSERT_LE(lhs, rhs) HIPASSERT(lhs <= rhs)
#define ASSERT_GE(lhs, rhs) HIPASSERT(lhs >= rhs)

using namespace cooperative_groups;
constexpr int MaxGPUs = 8;

static __global__
void kernel_cg_multi_grid_group_type(int* numGridsTestD,
                                     int* gridRankTestD,
                                     int *sizeTestD,
                                     int *thdRankTestD,
                                     int *isValidTestD,
                                     int *syncTestD,
                                     int *syncResultD)
{
  multi_grid_group mg = this_multi_grid();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test num_grids
  numGridsTestD[gIdx] = mg.num_grids();

  // Test grid_rank
  gridRankTestD[gIdx] = mg.grid_rank();

  // Test size
  sizeTestD[gIdx] = mg.size();

  // Test thread_rank
  thdRankTestD[gIdx] = mg.thread_rank();

  // Test is_valid
  isValidTestD[gIdx] = mg.is_valid();

  // Test sync
  //
  // Eech thread assign 1 to their respective location
  syncTestD[gIdx] = 1;
  // Grid level sync
  this_grid().sync();
  // Thread 0 from work-group 0 of current grid (gpu) does grid level reduction
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (uint i = 1; i < gridDim.x * blockDim.x; ++i) {
      syncTestD[0] += syncTestD[i];
    }
    syncResultD[mg.grid_rank() + 1] = syncTestD[0];
  }
  // multi-grid level sync
  mg.sync();
  // grid (gpu) 0 does final reduction across all grids (gpus)
  if (mg.grid_rank() == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    syncResultD[0] = 0;
    for (uint i = 1; i <= mg.num_grids(); ++i) {
      syncResultD[0] += syncResultD[i];
    }
  }
}

static void test_cg_multi_grid_group_type(int blockSize, int nGpu)
{
  // Create a stream each device
  hipStream_t stream[MaxGPUs];
  for (int i = 0; i < nGpu; i++) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipDeviceSynchronize());  // Make sure work is done on this device
    HIPCHECK(hipStreamCreate(&stream[i]));
  }

  // Allocate host and device memory
  int nBytes = sizeof(int) * 2 * blockSize;
  int *numGridsTestD[MaxGPUs], *numGridsTestH[MaxGPUs];
  int *gridRankTestD[MaxGPUs], *gridRankTestH[MaxGPUs];
  int *sizeTestD[MaxGPUs], *sizeTestH[MaxGPUs];
  int *thdRankTestD[MaxGPUs], *thdRankTestH[MaxGPUs];
  int *isValidTestD[MaxGPUs], *isValidTestH[MaxGPUs];
  int *syncTestD[MaxGPUs], *syncResultD;
  for (int i = 0; i < nGpu; i++) {
    HIPCHECK(hipSetDevice(i));

    HIPCHECK(hipMalloc(&numGridsTestD[i], nBytes));
    HIPCHECK(hipMalloc(&gridRankTestD[i], nBytes));
    HIPCHECK(hipMalloc(&sizeTestD[i], nBytes));
    HIPCHECK(hipMalloc(&thdRankTestD[i], nBytes));
    HIPCHECK(hipMalloc(&isValidTestD[i], nBytes));
    HIPCHECK(hipMalloc(&syncTestD[i], nBytes));

    HIPCHECK(hipHostMalloc(&numGridsTestH[i], nBytes));
    HIPCHECK(hipHostMalloc(&gridRankTestH[i], nBytes));
    HIPCHECK(hipHostMalloc(&sizeTestH[i], nBytes));
    HIPCHECK(hipHostMalloc(&thdRankTestH[i], nBytes));
    HIPCHECK(hipHostMalloc(&isValidTestH[i], nBytes));

    if (i == 0) {
      HIPCHECK(hipHostMalloc(&syncResultD, sizeof(int) * (nGpu + 1), hipHostMallocCoherent));
    }
  }

  // Launch Kernel
  constexpr int NumKernelArgs = 7;
  hipLaunchParams* launchParamsList = new hipLaunchParams[nGpu];
  void* args[MaxGPUs * NumKernelArgs];
  for (int i = 0; i < nGpu; i++) {
    HIPCHECK(hipSetDevice(i));

    args[i * NumKernelArgs]     = &numGridsTestD[i];
    args[i * NumKernelArgs + 1] = &gridRankTestD[i];
    args[i * NumKernelArgs + 2] = &sizeTestD[i];
    args[i * NumKernelArgs + 3] = &thdRankTestD[i];
    args[i * NumKernelArgs + 4] = &isValidTestD[i];
    args[i * NumKernelArgs + 5] = &syncTestD[i];
    args[i * NumKernelArgs + 6] = &syncResultD;

    launchParamsList[i].func = reinterpret_cast<void*>(kernel_cg_multi_grid_group_type);
    launchParamsList[i].gridDim = 2;
    launchParamsList[i].blockDim = blockSize;
    launchParamsList[i].sharedMem = 0;
    launchParamsList[i].stream = stream[i];
    launchParamsList[i].args = &args[i * NumKernelArgs];
  }
  HIPCHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList, nGpu, 0));

  // Copy result from device to host
  for (int i = 0; i < nGpu; i++) {
    HIPCHECK(hipSetDevice(i));
    HIPCHECK(hipMemcpy(numGridsTestH[i], numGridsTestD[i], nBytes, hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(gridRankTestH[i], gridRankTestD[i], nBytes, hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(sizeTestH[i], sizeTestD[i], nBytes, hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(thdRankTestH[i], thdRankTestD[i], nBytes, hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(isValidTestH[i], isValidTestD[i], nBytes, hipMemcpyDeviceToHost));
  }

  // Validate results
  int gridsSeen[MaxGPUs];
  for (int i = 0; i < nGpu; ++i) {
    for (int j = 0; j < 2 * blockSize; ++j) {
      ASSERT_EQUAL(numGridsTestH[i][j], nGpu);
      ASSERT_GE(gridRankTestH[i][j], 0);
      ASSERT_LE(gridRankTestH[i][j], nGpu-1);
      ASSERT_EQUAL(gridRankTestH[i][j], gridRankTestH[i][0]);
      ASSERT_EQUAL(sizeTestH[i][j], nGpu * 2 * blockSize);
      int gridRank = gridRankTestH[i][j];
      ASSERT_EQUAL(thdRankTestH[i][j], (gridRank * 2 * blockSize) + j);
      ASSERT_EQUAL(isValidTestH[i][j], 1);
    }
    ASSERT_EQUAL(syncResultD[i+1],  2 * blockSize);

    // Validate uniqueness property of grid rank
    gridsSeen[i] = gridRankTestH[i][0];
    for (int k = 0; k < i; ++k) {
      if (gridsSeen[k] == gridsSeen[i]) {
        assert(false && "Grid rank in multi-gpu setup should be unique");
      }
    }
  }
  ASSERT_EQUAL(syncResultD[0], nGpu * 2 * blockSize);

  // Free host and device memory
  delete [] launchParamsList;
  for (int i = 0; i < nGpu; i++) {
    HIPCHECK(hipSetDevice(i));

    HIPCHECK(hipFree(numGridsTestD[i]));
    HIPCHECK(hipFree(gridRankTestD[i]));
    HIPCHECK(hipFree(sizeTestD[i]));
    HIPCHECK(hipFree(thdRankTestD[i]));
    HIPCHECK(hipFree(isValidTestD[i]));
    HIPCHECK(hipFree(syncTestD[i]));

    if (i == 0) {
      HIPCHECK(hipHostFree(syncResultD));
    }
    HIPCHECK(hipHostFree(numGridsTestH[i]));
    HIPCHECK(hipHostFree(gridRankTestH[i]));
    HIPCHECK(hipHostFree(sizeTestH[i]));
    HIPCHECK(hipHostFree(thdRankTestH[i]));
    HIPCHECK(hipHostFree(isValidTestH[i]));
  }
}

TEST_CASE("Unit_hipCGMultiGridGroupType") {
  int nGpu = 0;
  HIPCHECK(hipGetDeviceCount(&nGpu));
  nGpu = min(nGpu, MaxGPUs);

  // Set `maxThreadsPerBlock` by taking minimum among all available devices
  int maxThreadsPerBlock = INT_MAX;
  hipDeviceProp_t deviceProperties;
  for (int i = 0; i < nGpu; i++) {
    HIPCHECK(hipGetDeviceProperties(&deviceProperties, i));
    if (!deviceProperties.cooperativeMultiDeviceLaunch) {
      HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
      return;
    }
    maxThreadsPerBlock = min(maxThreadsPerBlock, deviceProperties.maxThreadsPerBlock);
  }

  // Test for blockSizes in powers of 2
  for (int blockSize = 2; blockSize <= maxThreadsPerBlock; blockSize = blockSize*2) {
    test_cg_multi_grid_group_type(blockSize, nGpu);
  }

  // Test for random blockSizes, but the sequence is the same every execution
  srand(0);
  for (int i = 0; i < 10; i++) {
    // Test fails for 0 thread per block
    test_cg_multi_grid_group_type(max(2, rand() % maxThreadsPerBlock), nGpu);
  }
}
