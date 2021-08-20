/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

/*
This testcase verifies the following scenario
1. Allocating the memory and modifying it coherently
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>

constexpr auto wait_sec = 5000;

__global__ void Kernel(float* hostRes, int clkRate) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  hostRes[tid] = tid + 1;
  __threadfence_system();
  // expecting that the data is getting flushed to host here!
  uint64_t start = clock64()/clkRate, cur;
  if (clkRate > 1) {
    do { cur = clock64()/clkRate-start;}while (cur < wait_sec);
  } else {
    do { cur = clock64()/start;}while (cur < wait_sec);
  }
}

TEST_CASE("Unit_hipHostMalloc_CoherentAccess") {
  int blocks = 2;
  float* hostRes;
  hipHostMalloc(&hostRes, blocks * sizeof(float),
                hipHostMallocMapped);
  hostRes[0] = 0;
  hostRes[1] = 0;
  int clkRate;
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));
  std::cout << clkRate << std::endl;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Kernel), dim3(1), dim3(blocks),
                     0, 0, hostRes, clkRate);
  int eleCounter = 0;
  while (eleCounter < blocks) {
    // blocks until the value changes
    while (hostRes[eleCounter] == 0) {printf("waiting for counter inc\n");}
    eleCounter++;
  }
  hipHostFree(reinterpret_cast<void *>(hostRes));
}

