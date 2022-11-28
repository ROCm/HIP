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

/* Test Case Description:
   1) This testcase verifies the hipMallocManaged basic scenario - supported on
     all devices
   2) This testcase verifies the hipMallocManaged advanced scenario - supported
     only on HMM enabled devices
   3) This testcase verifies that hipMallocManaged returns an OutOfMemory error
     for allocations much larger than the available memory - supported on all devices
*/

#include "hipMallocManagedCommon.hh"
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>


// Kernel functions

__global__ void KernelMul_MngdMem(int* Hmm, int* Dptr, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    Hmm[i] = Dptr[i] * 10;
  }
}

__global__ void KernelMulAdd_MngdMem(int* Hmm, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    Hmm[i] = Hmm[i] * 2 + 10;
  }
}


static size_t numElements{64 * 1024 * 1024};
static unsigned blocksPerCU{6};
static unsigned threadsPerBlock{256};

/*
   This testcase verifies the hipMallocManaged basic scenario - supported on all devices
 */
TEST_CASE("Unit_hipMallocManaged_Basic") {
  auto managed = HmmAttrPrint();
  if (managed != 1) {
    WARN(
        "GPU doesn't support hipDeviceAttributeManagedMemory attribute so defaulting to system "
        "memory.");
  }

  float *A, *B, *C;

  HIP_CHECK(hipMallocManaged(&A, numElements * sizeof(float)));
  HIP_CHECK(hipMallocManaged(&B, numElements * sizeof(float)));
  HIP_CHECK(hipMallocManaged(&C, numElements * sizeof(float)));
}

/*
   This testcase verifies the hipMallocManaged advanced scenario - supported only on HMM enabled
   devices
 */
TEST_CASE("Unit_hipMallocManaged_Advanced") {
  auto managed = HmmAttrPrint();
  if (managed != 1) {
    HipTest::HIP_SKIP_TEST("GPU doesn't support managed memory so skipping test.");
    return;
  }

  float *A, *B, *C;

  HIP_CHECK(hipMallocManaged(&A, numElements * sizeof(float)));
  HIP_CHECK(hipMallocManaged(&B, numElements * sizeof(float)));
  HIP_CHECK(hipMallocManaged(&C, numElements * sizeof(float)));
  HipTest::setDefaultData(numElements, A, B, C);

  hipDevice_t device = hipCpuDeviceId;

  HIP_CHECK(hipMemAdvise(A, numElements * sizeof(float), hipMemAdviseSetReadMostly, device));
  HIP_CHECK(hipMemPrefetchAsync(A, numElements * sizeof(float), 0));
  HIP_CHECK(hipMemPrefetchAsync(B, numElements * sizeof(float), 0));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemRangeGetAttribute(&device, sizeof(device),
                                    hipMemRangeAttributeLastPrefetchLocation, A,
                                    numElements * sizeof(float)));
  if (device != 0) {
    INFO("hipMemRangeGetAttribute error, device = " << device);
  }
  uint32_t read_only = 0xf;
  HIP_CHECK(hipMemRangeGetAttribute(&read_only, sizeof(read_only), hipMemRangeAttributeReadMostly,
                                    A, numElements * sizeof(float)));
  if (read_only != 1) {
    SUCCEED("hipMemRangeGetAttribute error, read_only = " << read_only);
  }

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);
  hipEvent_t event0, event1;
  HIP_CHECK(hipEventCreate(&event0));
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventRecord(event0, 0));
  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                     static_cast<const float*>(A), static_cast<const float*>(B), C, numElements);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(event1, 0));
  HIP_CHECK(hipDeviceSynchronize());
  float time = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&time, event0, event1));
  printf("Time %.3f ms\n", time);
  float maxError = 0.0f;
  HIP_CHECK(hipMemPrefetchAsync(B, numElements * sizeof(float), hipCpuDeviceId));
  HIP_CHECK(hipDeviceSynchronize());
  device = 0;
  HIP_CHECK(hipMemRangeGetAttribute(&device, sizeof(device),
                                    hipMemRangeAttributeLastPrefetchLocation, A,
                                    numElements * sizeof(float)));
  if (device != hipCpuDeviceId) {
    SUCCEED("hipMemRangeGetAttribute error device = " << device);
  }

  for (size_t i = 0; i < numElements; i++) {
    maxError = fmax(maxError, fabs(B[i] - 3.0f));
  }
  HIP_CHECK(hipFree(A));
  HIP_CHECK(hipFree(B));
  REQUIRE(maxError != 0.0f);
}

/*
   This testcase verifies that hipMallocManaged returns an OutOfMemory error for allocations much
   larger than the available memory - supported on all devices
 */
TEST_CASE("Unit_hipMallocManaged_Large") {
  auto managed = HmmAttrPrint();
  if (managed != 1) {
    WARN(
        "GPU doesn't support hipDeviceAttributeManagedMemory attribute so defaulting to system "
        "memory.");
  }

  float* A;
  HIP_CHECK_ERROR(hipMallocManaged(&A, std::numeric_limits<size_t>::max()), hipErrorOutOfMemory);
}
