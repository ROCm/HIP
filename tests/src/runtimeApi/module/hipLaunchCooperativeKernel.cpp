/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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
// Simple test for hipLaunchCooperativeKernel API.

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip/hcc_detail/device_library_decls.h"
#include "hip/hcc_detail/hip_cooperative_groups.h"
#include <iostream>
#include <chrono>
#include "test_common.h"

using namespace std::chrono;

namespace cg = cooperative_groups;

const static uint BufferSizeInDwords = 448 * 1024 * 1024;

__global__ void test_gws(uint* buf, uint bufSize, long* tmpBuf, long* result)
{
    extern __shared__ long tmp[];
    uint offset = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x  * blockDim.x;
    cg::grid_group gg = cg::this_grid();

    long sum = 0;
    for (uint i = offset; i < bufSize; i += stride) {
        sum += buf[i];
    }
    tmp[threadIdx.x] = sum;

    __syncthreads();

    if (threadIdx.x == 0) {
        sum = 0;
        for (uint i = 0; i < blockDim.x; i++) {
            sum += tmp[i];
        }
        tmpBuf[blockIdx.x] = sum;
    }

    gg.sync();

    if (offset == 0) {
        for (uint i = 1; i < gridDim.x; ++i) {
          sum += tmpBuf[i];
       }
       *result = sum;
    }
}

int main() {
  float *A, *B, *Ad, *Bd;
  uint* dA;
  long* dB;
  long* dC;

  uint32_t* init = new uint32_t[BufferSizeInDwords];
  for (uint32_t i = 0; i < BufferSizeInDwords; ++i) {
    init[i] = i;
  }

  hipDeviceProp_t deviceProp;

  hipGetDeviceProperties(&deviceProp, 0);
  if (!deviceProp.cooperativeLaunch) {
    std::cout << "info: Device doesn't support cooperative launch! skipping the test!\n";
    passed();
    return 0;
  }

  std::cout << "info: running on bus 0x" << deviceProp.pciBusID << " " << deviceProp.name << "\n";

  size_t SIZE = BufferSizeInDwords * sizeof(uint);

  HIPCHECK(hipMalloc((void**)&dA, SIZE));
  HIPCHECK(hipHostMalloc((void**)&dC, sizeof(long)));
  HIPCHECK(hipMemcpy(dA, init, SIZE, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  dim3 dimBlock = dim3(1);
  dim3 dimGrid  = dim3(1);
  int numBlocks = 0;
  uint workgroups[4] = {32, 64, 128, 256};

  system_clock::time_point start = system_clock::now();

  for (uint i = 0; i < 4; ++i) {

    dimBlock.x = workgroups[i];
    // Calculate the device occupancy to know how many blocks can be run concurrently
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
        test_gws, dimBlock.x * dimBlock.y * dimBlock.z, dimBlock.x * sizeof(long));

    dimGrid.x = deviceProp.multiProcessorCount * std::min(numBlocks, 32);
    HIPCHECK(hipMalloc((void**)&dB, dimGrid.x * sizeof(long)));
 
    void *params[4];
    params[0] = (void*)&dA;
    params[1] = (void*)&BufferSizeInDwords;
    params[2] = (void*)&dB;
    params[3] = (void*)&dC;

    std::cout << "Testing with grid size = " << dimGrid.x << " and block size = " << dimBlock.x << "\n";
    HIPCHECK(hipLaunchCooperativeKernel(test_gws, dimGrid, dimBlock, params, dimBlock.x * sizeof(long), stream));

    HIPCHECK(hipMemcpy(init, dC, sizeof(long), hipMemcpyDeviceToHost));

    if (*dC != (((long)(BufferSizeInDwords) * (BufferSizeInDwords - 1)) / 2)) {
        std::cout << "Data validation failed for grid size = " << dimGrid.x << " and block size = " << dimBlock.x << "\n";
        HIPCHECK(hipStreamDestroy(stream));
        hipFree(dC);
        hipFree(dB);
        hipFree(dA);
        delete [] init;
        std::cout << "Test failed! \n";
        return 0;
    } else {
        std::cout << "info: data validated!\n";
    }

  }

  system_clock::time_point end = system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time) <<
    "elapsed time: " << elapsed_seconds.count() << "s\n";

  HIPCHECK(hipStreamDestroy(stream));
  hipFree(dC);
  hipFree(dB);
  hipFree(dA);
  delete [] init;
  passed();
  return 0;
}
