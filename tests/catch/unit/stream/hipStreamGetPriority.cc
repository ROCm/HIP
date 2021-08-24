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
TEST_CASE("Unit_hipStreamGetPriority_Negative") {
  hipStream_t stream = 0;
  REQUIRE(hipStreamGetPriority(stream, nullptr) == hipErrorInvalidValue);
}
TEST_CASE("Unit_hipStreamGetPriority_default") {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  int priority = 0;
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  // valid priority
  // Lower the value higher the priority, higher the value lower the priority
  REQUIRE(priority_low >= priority);
  REQUIRE(priority >= priority_high);
  HIP_CHECK(hipStreamDestroy(stream));
}
TEST_CASE("Unit_hipStreamGetPriority_high") {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_high));
  int priority = 0;
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  REQUIRE(priority == priority_high);
  HIP_CHECK(hipStreamDestroy(stream));
}
TEST_CASE("Unit_hipStreamGetPriority_higher") {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority_high-1));
  int priority = 0;
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  REQUIRE(priority == priority_high);
  HIP_CHECK(hipStreamDestroy(stream));
}
TEST_CASE("Unit_hipStreamGetPriority_low") {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_low));
  int priority = 0;
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  REQUIRE(priority_low == priority);
  HIP_CHECK(hipStreamDestroy(stream));
}
TEST_CASE("Unit_hipStreamGetPriority_lower") {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority_low+1));
  int priority = 0;
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  REQUIRE(priority_low == priority);
  HIP_CHECK(hipStreamDestroy(stream));
}
