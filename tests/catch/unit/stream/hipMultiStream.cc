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
#include <hip_test_common.hh>
#include <iostream>
#include <vector>
constexpr int NN = 1 << 21;
__global__ void kernel_do_nothing(__attribute__((unused))int a) {
  // empty kernel
}
__global__ void kernel(float* x, float* y, int n) {
  size_t tid{threadIdx.x};
  if (tid < 1) {
    for (int i = 0; i < n; i++) {
      x[i] = sqrt(powf(3.14159, i));
    }
    y[tid] = y[tid] + 1.0f;
  }
}
__global__ void nKernel(float* y) {
  size_t tid{threadIdx.x};
  y[tid] = y[tid] + 1.0f;
}
TEST_CASE("Unit_hipMultiStream_sameDevice") {
  constexpr int num_streams{8};
  hipStream_t streams[num_streams];
  float *data[num_streams], *yd, *xd;
  float y{1.0f}, x{1.0f};
  HIP_CHECK(hipMalloc((void**)&yd, sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&xd, sizeof(float)));
  HIP_CHECK(hipMemcpy(yd, &y, sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(xd, &x, sizeof(float), hipMemcpyHostToDevice));
  for (int i = 0; i < num_streams; i++) {
    HIP_CHECK(hipStreamCreate(&streams[i]));
    HIP_CHECK(hipMalloc(&data[i], NN * sizeof(float)));
    hipLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, streams[i], data[i], xd, NN);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(nKernel), dim3(1), dim3(1), 0, 0, yd);
    HIP_CHECK(hipFree(data[i]));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
  HIP_CHECK(hipMemcpy(&x, xd, sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&y, yd, sizeof(float), hipMemcpyDeviceToHost));
  REQUIRE(x == Approx(y));
}

TEST_CASE("Unit_hipMultiStream_multimeDevice") {
  constexpr int nLoops = 50000;
  constexpr int nStreams = 2;
  std::vector<hipStream_t> streams(nStreams);
  int nGpu = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    INFO("No GPU for Testing");
    SUCCEED(true);
  }
  static int device = 0;
  HIP_CHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  INFO("Running on Bus: " << props.pciBusID << " " << props.name);
  for (int i = 0; i < nStreams; i++) {
    HIP_CHECK(hipStreamCreate(&streams[i]));
  }
  for (int k = 0; k <= nLoops; ++k) {
    HIP_CHECK(hipDeviceSynchronize());
    // Launch kernel with default stream
    hipLaunchKernelGGL(kernel_do_nothing, dim3(1), dim3(1), 0, 0, 1);
    // Launch kernel on all streams
    for (int i = 0; i < nStreams; i++) {
      hipLaunchKernelGGL(kernel_do_nothing, dim3(1), dim3(1), 0, streams[i], 1);
    }
    // Sync stream 1
    HIP_CHECK(hipStreamSynchronize(streams[0]));
    if (k % 10000 == 0 || k == nLoops) {
      INFO("Iter: " << k);
    }
  }
  HIP_CHECK(hipDeviceSynchronize());
  // Clean up
  for (int i = 0; i < nStreams; i++) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}
