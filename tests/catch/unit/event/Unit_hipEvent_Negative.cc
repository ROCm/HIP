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

TEST_CASE("Unit_hipEventCreate_NullCheck") {
  hipEvent_t start_event;
  auto res = hipEventCreate(nullptr);
  REQUIRE(res != hipSuccess);
  res = hipEventCreateWithFlags(nullptr, 0);
  REQUIRE(res != hipSuccess);
  res = hipEventCreateWithFlags(&start_event, 10);
  REQUIRE(res != hipSuccess);
}

TEST_CASE("Unit_hipEventSynchronize_NullCheck") {
  auto res = hipEventSynchronize(nullptr);
  REQUIRE(res != hipSuccess);
}

TEST_CASE("Unit_hipEventQuery_NullCheck") {
  auto res = hipEventQuery(nullptr);
  REQUIRE(res != hipSuccess);
}

TEST_CASE("Unit_hipEventDestroy_NullCheck") {
  auto res = hipEventDestroy(nullptr);
  REQUIRE(res != hipSuccess);
}

TEST_CASE("Unit_hipEventCreate_IncompatibleFlags") {
  hipEvent_t event;

#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-106");
#endif

  HIP_CHECK_ERROR(hipEventCreateWithFlags(&event, hipEventInterprocess), hipErrorInvalidValue);

#if HT_AMD
  HIP_CHECK_ERROR(
      hipEventCreateWithFlags(&event, hipEventReleaseToDevice | hipEventReleaseToSystem),
      hipErrorInvalidValue);
#endif

  unsigned allFlags{hipEventReleaseToDevice | hipEventReleaseToSystem | hipEventBlockingSync |
                    hipEventDisableTiming | hipEventDefault | hipEventInterprocess};

#if HT_AMD
  HIP_CHECK_ERROR(hipEventCreateWithFlags(&event, allFlags), hipErrorInvalidValue);
#else
  /* Works on Non-AMD because hipEventReleaseToDevice / hipEventReleaseToSystem have no meaning in
   * that case */
  HIP_CHECK(hipEventCreateWithFlags(&event, allFlags));
#endif

  unsigned invalidFlag{0x08000000};
  HIP_CHECK_ERROR(hipEventCreateWithFlags(&event, invalidFlag), hipErrorInvalidValue);
}