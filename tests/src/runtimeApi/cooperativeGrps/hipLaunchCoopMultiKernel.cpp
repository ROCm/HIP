/*
Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Simple test for hipLaunchCooperativeKernelMultiDevice API.

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11 -rdc=true -gencode arch=compute_70,code=sm_70
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include "hip/hip_cooperative_groups.h"
#include "test_common.h"

using namespace std::chrono;

const static uint BufferSizeInDwords = 256 * 1024 * 1024;
const static uint numQueues = 4;
const static uint numIter = 100;
constexpr uint NumKernelArgs = 4;
constexpr uint MaxGPUs = 8;

#include <stdio.h>

__global__ void test_gws(uint* buf, uint bufSize, long* tmpBuf, long* result)
{
    extern __shared__ long tmp[];
    uint groups = gridDim.x;
    uint group_id = blockIdx.x;
    uint local_id = threadIdx.x;
    uint chunk = gridDim.x * blockDim.x;

    uint i = group_id * blockDim.x + local_id;
    long sum = 0;
    while (i < bufSize) {
      sum += buf[i];
      i += chunk;
    }
    tmp[local_id] = sum;
    __syncthreads();
    i = 0;
    if (local_id == 0) {
        sum = 0;
        while (i < blockDim.x) {
          sum += tmp[i];
          i++;
        }
        tmpBuf[group_id] = sum;
    }

    // wait
    cooperative_groups::this_grid().sync();

    if (((blockIdx.x * blockDim.x) + threadIdx.x) == 0) {
        for (uint i = 1; i < groups; ++i) {
          sum += tmpBuf[i];
       }
       //*result = sum;
       result[1 + cooperative_groups::this_multi_grid().grid_rank()] = sum;
    }
    cooperative_groups::this_multi_grid().sync();
    if (cooperative_groups::this_multi_grid().grid_rank() == 0) {
      sum = 0;
      for (uint i = 1; i <= cooperative_groups::this_multi_grid().num_grids(); ++i) {
        sum += result[i];
      }
      *result = sum;
    }
}

int main() {
  float *A, *B;
  uint* dA[MaxGPUs];
  long* dB[MaxGPUs];
  long* dC;
  hipStream_t stream[MaxGPUs];

  uint32_t* init = new uint32_t[BufferSizeInDwords];
  for (uint32_t i = 0; i < BufferSizeInDwords; ++i) {
    init[i] = i;
  }

  int nGpu = 0;
  HIPCHECK(hipGetDeviceCount(&nGpu));
  size_t copySizeInDwords = BufferSizeInDwords / nGpu;
  hipDeviceProp_t deviceProp[MaxGPUs];

  for (int i = 0; i < nGpu; i++) {
    HIPCHECK(hipSetDevice(i));

    // Calculate the device occupancy to know how many blocks can be run concurrently
    hipGetDeviceProperties(&deviceProp[i], 0);
    if (!deviceProp[i].cooperativeMultiDeviceLaunch) {
      printf("Device doesn't support cooperative launch!");
      passed();
      return 0;
    }
    size_t SIZE = copySizeInDwords * sizeof(uint);

    HIPCHECK(hipMalloc((void**)&dA[i], SIZE));
    HIPCHECK(hipMalloc((void**)&dB[i], 64 * deviceProp[i].multiProcessorCount * sizeof(long)));
    if (i == 0) {
      HIPCHECK(hipHostMalloc((void**)&dC, (nGpu + 1) * sizeof(long), hipHostMallocCoherent));
    }
    HIPCHECK(hipMemcpy(dA[i], &init[i * copySizeInDwords] , SIZE, hipMemcpyHostToDevice));
    HIPCHECK(hipStreamCreate(&stream[i]));
    hipDeviceSynchronize();
  }

  dim3 dimBlock;
  dim3 dimGrid;
  dimGrid.x = 1;
  dimGrid.y = 1;
  dimGrid.z = 1;
  dimBlock.x = 64;
  dimBlock.y = 1;
  dimBlock.z = 1;

  int numBlocks = 0;
  uint workgroups[3] = {64, 128, 256};

  hipLaunchParams* launchParamsList = new hipLaunchParams[nGpu];
  std::time_t end_time;
  double time = 0;
  for (uint set = 0; set < 3; ++set) {
    void* args[MaxGPUs * NumKernelArgs];
    std::cout << "---------- Test#" << set << ", size: "<< BufferSizeInDwords <<
      " dwords ---------------\n";
    for (int i = 0; i < nGpu; i++) {
      HIPCHECK(hipSetDevice(i));
      dimBlock.x = workgroups[set];
      HIPCHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
      test_gws, dimBlock.x * dimBlock.y * dimBlock.z, dimBlock.x * sizeof(long)));

      std::cout << "GPU(" << i << ") Block size: " << dimBlock.x <<
        " Num blocks per CU: " << numBlocks << "\n";

      dimGrid.x = deviceProp[i].multiProcessorCount * std::min(numBlocks, 32);

      args[i * NumKernelArgs]     = (void*)&dA[i];
      args[i * NumKernelArgs + 1] = (void*)&copySizeInDwords;
      args[i * NumKernelArgs + 2] = (void*)&dB[i];
      args[i * NumKernelArgs + 3] = (void*)&dC;

      launchParamsList[i].func = reinterpret_cast<void*>(test_gws);
      launchParamsList[i].gridDim = dimGrid;
      launchParamsList[i].blockDim = dimBlock;
      launchParamsList[i].sharedMem = dimBlock.x * sizeof(long);
      launchParamsList[i].stream = stream[i];
      launchParamsList[i].args = &args[i * NumKernelArgs];
    }

    system_clock::time_point start = system_clock::now();
    hipLaunchCooperativeKernelMultiDevice(launchParamsList, nGpu, 0);
    system_clock::time_point end = system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    end_time = std::chrono::system_clock::to_time_t(end);

    time += elapsed_seconds.count();

    size_t processedDwords = copySizeInDwords * nGpu;
    if (*dC != (((long)(processedDwords) * (processedDwords - 1)) / 2)) {
      std::cout << "Data validation failed ("<< *dC << " != " <<
        (((long)(BufferSizeInDwords) * (BufferSizeInDwords - 1)) / 2) <<
        ") for grid size = " << dimGrid.x << " and block size = " << dimBlock.x << "\n";
      std::cout << "Test failed! \n";
    }
  }

  delete [] launchParamsList;

  std::cout << "finished computation at " << std::ctime(&end_time) <<
    "elapsed time: " << time << "s\n";

  hipSetDevice(0);
  hipFree(dC);
  for (int i = 0; i < nGpu; i++) {
    hipFree(dA[i]);
    hipFree(dB[i]);
    HIPCHECK(hipStreamDestroy(stream[i]));
  }
  delete [] init;
  passed();
  return 0;
}
