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

/*
 * Test for checking the functionality of
 * hipError_t hipDeviceSynchronize();
 */


#include <hip_test_common.hh>

#define _SIZE sizeof(int) * 1024 * 1024
#define NUM_STREAMS 2

static __global__ void Iter(int* Ad, int num) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    // Kernel loop designed to execute very slowly.
    // so we can test timing-related
    // behavior below
    if (tx == 0) {
        for (int i = 0; i < num; i++) {
            Ad[tx] += 1;
        }
    }
}

TEST_CASE("Unit_hipDeviceSynchronize_Positive_Empty_Streams") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device: " << device);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipDeviceSynchronize_Positive_Nullstream") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device: " << device);

  int *A_h = nullptr, *A_d = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), _SIZE, hipHostMallocDefault));
  A_h[0] = 1;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), _SIZE));

  HIP_CHECK(hipMemcpyAsync(A_d, A_h, _SIZE, hipMemcpyHostToDevice, NULL));
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Iter), dim3(1), dim3(1), 0, NULL, A_d, 1 << 30);
  HIP_CHECK(hipMemcpyAsync(A_h, A_d, _SIZE, hipMemcpyDeviceToHost, NULL));

  CHECK(1 << 30 != A_h[0] - 1);
  HIP_CHECK(hipDeviceSynchronize());
  CHECK(1 << 30 == A_h[0] - 1);
}

TEST_CASE("Unit_hipDeviceSynchronize_Functional") {
  int* A[NUM_STREAMS];
  int* Ad[NUM_STREAMS];
  hipStream_t stream[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; i++) {
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A[i]), _SIZE,
                                                   hipHostMallocDefault));
      A[i][0] = 1;
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad[i]), _SIZE));
      HIP_CHECK(hipStreamCreate(&stream[i]));
  }
  for (int i = 0; i < NUM_STREAMS; i++) {
      HIP_CHECK(hipMemcpyAsync(Ad[i], A[i], _SIZE, hipMemcpyHostToDevice,
                                                               stream[i]));
  }
  for (int i = 0; i < NUM_STREAMS; i++) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(Iter), dim3(1), dim3(1), 0,
                                                stream[i], Ad[i], 1 << 30);
  }
  for (int i = 0; i < NUM_STREAMS; i++) {
      HIP_CHECK(hipMemcpyAsync(A[i], Ad[i], _SIZE, hipMemcpyDeviceToHost,
                                                               stream[i]));
  }


  // This first check but relies on the kernel running for so long that the
  // D2H async memcopy has not started yet. This will be true in an optimal
  // asynchronous implementation.
  // Conservative implementations which synchronize the hipMemcpyAsync will
  // fail, ie if HIP_LAUNCH_BLOCKING=true.

  CHECK(1 << 30 != A[NUM_STREAMS - 1][0] - 1);
  HIP_CHECK(hipDeviceSynchronize());
  CHECK(1 << 30 == A[NUM_STREAMS - 1][0] - 1);
}