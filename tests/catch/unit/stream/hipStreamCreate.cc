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
#include <hip_test_common.hh>
TEST_CASE("Unit_hipStreamCreate_default") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
}
TEST_CASE("Unit_hipStreamCreateWithFlags_Negative") {
  hipStream_t stream;
  auto status = hipStreamCreateWithFlags(&stream, 0xFF);
  REQUIRE(status == hipErrorInvalidValue);
  status = hipStreamCreateWithFlags(nullptr, hipStreamDefault);
  REQUIRE(status == hipErrorInvalidValue);
}
TEST_CASE("Unit_hipStreamCreateWithFlags") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamDefault));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  HIP_CHECK(hipStreamDestroy(stream));
}
TEST_CASE("Unit_hipStreamCreateWithPriority") {
  int priority_low = 0;
  int priority_high = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream;

  SECTION("Setting high prirority") {
    HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_high));
  }
  SECTION("Setting low priority") {
    HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_low));
  }
  HIP_CHECK(hipStreamDestroy(stream));
}
