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
/**
Testcase Scenarios :
1) Negative tests for hipStreamGetPriority api.
2) Create stream and check default priority of stream is within range.
3) Create stream with high or low priority and check priority is set as expected.
4) Create stream with higher priority or lower priority for the priority range returned.
5) Create stream with CUMask and check priority is returned as expected.
*/

#include <hip_test_common.hh>

/**
 * Check the error returned when a null stream is used.
 */
TEST_CASE("Unit_hipStreamGetPriority_NullStream") {
  int priority = 0;
  HIP_CHECK_ERROR(hipStreamGetPriority(nullptr, &priority), hipErrorInvalidResourceHandle);
}

/**
 * Check the error returned when an invalid pointer to a priority is used.
 */
TEST_CASE("Unit_hipStreamGetPriority_InvalidPriorityPointer") {
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK_ERROR(hipStreamGetPriority(stream, nullptr), hipErrorInvalidValue);
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Create stream and check default priority of stream is within range.
 */
TEST_CASE("Unit_hipStreamGetPriority_default") {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  int priority = 0;
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  // valid priority
  // Lower the value higher the priority, higher the value lower the priority
  REQUIRE(priority_low >= priority);
  REQUIRE(priority >= priority_high);
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Create stream with high priority and check priority is set as expected.
 */
TEST_CASE("Unit_hipStreamGetPriority_high") {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_high));
  int priority = 0;
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  REQUIRE(priority == priority_high);
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Create stream with higher priority for the priority range returned.
 */
TEST_CASE("Unit_hipStreamGetPriority_higher") {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority_high - 1));
  int priority = 0;
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  REQUIRE(priority == priority_high);
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Create stream with low priority and check priority is set as expected.
 */
TEST_CASE("Unit_hipStreamGetPriority_low") {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority_low));
  int priority = 0;
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  REQUIRE(priority_low == priority);
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Create stream with lower priority for the priority range returned.
 */
TEST_CASE("Unit_hipStreamGetPriority_lower") {
  int priority_low = 0;
  int priority_high = 0;
  int devID = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(devID));
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority_low + 1));
  int priority = 0;
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  REQUIRE(priority_low == priority);
  HIP_CHECK(hipStreamDestroy(stream));
}

#if HT_AMD
/**
 * Create stream with CUMask and check priority is returned as expected.
 */
TEST_CASE("Unit_hipStreamGetPriority_StreamsWithCUMask") {
  hipStream_t stream{};
  int priority = 0;
  int priority_normal = 0;
  int priority_low = 0;
  int priority_high = 0;
  // Test is to get the Stream Priority Range
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  priority_normal = priority_low + priority_high;
  // Check if priorities are indeed supported
  REQUIRE_FALSE(priority_low == priority_high);
  // Creating a stream with hipExtStreamCreateWithCUMask and checking
  // priority.
  const uint32_t cuMask = 0xffffffff;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, 1, &cuMask));
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  REQUIRE_FALSE(priority_normal != priority);
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif
