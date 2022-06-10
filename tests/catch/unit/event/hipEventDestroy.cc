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

#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include "hip/hip_runtime_api.h"

static constexpr size_t vectorSize{16384};


static inline void launchVectorAdd(float*& A_h, float*& B_h, float*& C_h, hipStream_t stream) {
  float* A_d{nullptr};
  float* B_d{nullptr};
  float* C_d{nullptr};
  HipTest::initArraysForHost(&A_h, &B_h, &C_h, vectorSize, true);
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), A_h, 0));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&B_d), B_h, 0));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&C_d), C_h, 0));
  HipTest::vectorADD<<<1, 1, 0, stream>>>(A_d, B_d, C_d, vectorSize);
}


/**
 * @brief Check that destroying an event before the kernel has finished running causes no errors.
 *
 */
TEST_CASE("Unit_hipEventDestroy_Unfinished") {
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));

  float *A_h, *B_h, *C_h;
  launchVectorAdd(A_h, B_h, C_h, nullptr);

  HIP_CHECK(hipEventRecord(event));
  HIP_CHECK_ERROR(hipEventQuery(event), hipErrorNotReady);
  HIP_CHECK(hipEventDestroy(event));

  HIP_CHECK(hipDeviceSynchronize());
  HipTest::checkVectorADD(A_h, B_h, C_h, vectorSize);
  HipTest::freeArraysForHost(A_h, B_h, C_h, true);
}

/**
 * @brief Check that destroying an event enqueued to a stream causes no errors.
 *
 */
TEST_CASE("Unit_hipEventDestroy_WithWaitingStream") {
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float *A_h, *B_h, *C_h;
  launchVectorAdd(A_h, B_h, C_h, stream);

  HIP_CHECK(hipEventRecord(event, stream));
  HIP_CHECK_ERROR(hipEventQuery(event), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));

  HipTest::checkVectorADD(A_h, B_h, C_h, vectorSize);
  HipTest::freeArraysForHost(A_h, B_h, C_h, true);
}

TEST_CASE("Unit_hipEventDestroy_Negative") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-105");
  return;
#endif

  SECTION("Invalid Event") {
    hipEvent_t event{nullptr};
    HIP_CHECK_ERROR(hipEventDestroy(event), hipErrorInvalidResourceHandle);
  }

  SECTION("Destroy twice") {
    hipEvent_t event;
    HIP_CHECK(hipEventCreate(&event));
    HIP_CHECK(hipEventDestroy(event));
    HIP_CHECK_ERROR(hipEventDestroy(event), hipErrorContextIsDestroyed);
  }
}
