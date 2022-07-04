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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Test hipEventRecord serialization behavior.

#include <hip_test_common.hh>

#include <kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_context.hh>

TEST_CASE("Unit_hipEventRecord") {
  constexpr size_t N = 1024;
  constexpr int iterations = 1;

  constexpr int blocks = 1024;

  constexpr size_t Nbytes = N * sizeof(float);

  float *A_h, *B_h, *C_h;
  float *A_d, *B_d, *C_d;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);

  enum TestType {
    WithFlags_Default = hipEventDefault,
    WithFlags_Blocking = hipEventBlockingSync,
    WithFlags_DisableTiming = hipEventDisableTiming,
#if HT_AMD
    WithFlags_ReleaseToDevice = hipEventReleaseToDevice,
    WithFlags_ReleaseToSystem = hipEventReleaseToSystem,
#endif
    WithoutFlags
  };

#if HT_AMD
  auto flags = GENERATE(WithFlags_Default, WithFlags_Blocking, WithFlags_DisableTiming,
                        WithFlags_ReleaseToDevice, WithFlags_ReleaseToSystem, WithoutFlags);
#endif

#if HT_NVIDIA
  auto flags =
      GENERATE(WithFlags_Default, WithFlags_Blocking, WithFlags_DisableTiming, WithoutFlags);
#endif


  hipEvent_t start{}, stop{};

  if (flags == WithoutFlags) {
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
  } else {
    HIP_CHECK(hipEventCreateWithFlags(&start, flags));
    HIP_CHECK(hipEventCreateWithFlags(&stop, flags));
  }

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  for (int i = 0; i < iterations; i++) {
    //--- START TIMED REGION
    long long hostStart = HipTest::get_time();
    // Record the start event
    HIP_CHECK(hipEventRecord(start, NULL));

    HipTest::launchKernel<float>(HipTest::vectorADD<float>, blocks, 1, 0, 0,
                                 static_cast<const float*>(A_d), static_cast<const float*>(B_d),
                                 C_d, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    long long hostStop = HipTest::get_time();
    //--- STOP TIMED REGION

    float hostMs = HipTest::elapsed_time(hostStart, hostStop);

    INFO("host_time (chrono)                = " << hostMs);

    // Make sure timer is timing something...
    if (flags != WithFlags_DisableTiming) {
      float eventMs = 1.0f;
      HIP_CHECK(hipEventElapsedTime(&eventMs, start, stop));
      INFO("kernel_time (hipEventElapsedTime) = " << eventMs);
      REQUIRE(eventMs > 0.0f);
    }
  }

  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  HipTest::checkVectorADD(A_h, B_h, C_h, N, true);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  TestContext::get().cleanContext();
}

TEST_CASE("Unit_hipEventRecord_Negative") {
  SECTION("Nullptr event") {
    HIP_CHECK_ERROR(hipEventRecord(nullptr, nullptr), hipErrorInvalidResourceHandle);
  }
}