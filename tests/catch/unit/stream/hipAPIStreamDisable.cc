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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include "hip/math_functions.h"

#define NUM_STREAMS 8

namespace hipAPIStreamDisableTest {
const int NN = 1 << 21;

__global__ void kernel(float* x, float* y, int n) {
  int tid = threadIdx.x;
  if (tid < 1) {
    for (int i = 0; i < n; i++) {
      x[i] = sqrt(powf(3.14159, i));
    }
    y[tid] = y[tid] + 1.0f;
  }
}

__global__ void nKernel(float* y) {
  int tid = threadIdx.x;
  y[tid] = y[tid] + 1.0f;
}
}  // namespace hipAPIStreamDisableTest

/**
 * Validate basic multistream functionalities
 */
TEST_CASE("Unit_hipStreamCreate_MultistreamBasicFunctionalities") {
  hipStream_t streams[NUM_STREAMS];
  float *data[NUM_STREAMS], *yd, *xd;
  float y = 1.0f, x = 1.0f;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&yd), sizeof(float)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&xd), sizeof(float)));
  HIP_CHECK(hipMemcpy(yd, &y, sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(xd, &x, sizeof(float), hipMemcpyHostToDevice));

  for (int i = 0; i < NUM_STREAMS; i++) {
    HIP_CHECK(hipStreamCreate(&streams[i]));
    HIP_CHECK(hipMalloc(&data[i],
            (hipAPIStreamDisableTest::NN)*sizeof(float)));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipAPIStreamDisableTest::kernel),
                       dim3(1), dim3(1), 0, streams[i], data[i], xd,
                       hipAPIStreamDisableTest::NN);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipAPIStreamDisableTest::nKernel),
                       dim3(1), dim3(1), 0, 0, yd);
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
  HIP_CHECK(hipMemcpy(&x, xd, sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&y, yd, sizeof(float), hipMemcpyDeviceToHost));
  REQUIRE(x == y);
}
