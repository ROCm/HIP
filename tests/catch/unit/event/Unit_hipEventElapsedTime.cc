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
#include <iostream>

TEST_CASE("Unit_hipEventElapsedTime_NullCheck") {
  hipEvent_t start = nullptr, end = nullptr;
  float tms = 1.0f;
  HIP_ASSERT(hipEventElapsedTime(nullptr, start, end) == hipErrorInvalidValue);
#ifndef __HIP_PLATFORM_NVIDIA__
  // On NVCC platform API throws seg fault hence skipping
  HIP_ASSERT(hipEventElapsedTime(&tms, nullptr, end) == hipErrorInvalidHandle);
  HIP_ASSERT(hipEventElapsedTime(&tms, start, nullptr) == hipErrorInvalidHandle);
#endif
}

TEST_CASE("Unit_hipEventElapsedTime_DisableTiming") {
  float timeElapsed = 1.0f;
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreateWithFlags(&start, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&stop, hipEventDisableTiming));
  HIP_ASSERT(hipEventElapsedTime(&timeElapsed, start, stop) == hipErrorInvalidHandle);
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
}

TEST_CASE("Unit_hipEventElapsedTime_DifferentDevices") {
  int devCount = 0;
  HIP_CHECK(hipGetDeviceCount(&devCount));
  if (devCount > 1) {
    // create event on dev=0
    HIP_CHECK(hipSetDevice(0));
    hipEvent_t start;
    hipEvent_t start1;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&start1));

    HIP_CHECK(hipEventRecord(start, nullptr));
    HIP_CHECK(hipEventSynchronize(start));

    // create event on dev=1
    HIP_CHECK(hipSetDevice(1));
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&stop));

    // start1 on device 0 but null stream on device 1
    HIP_ASSERT(hipEventRecord(start1, nullptr) == hipErrorInvalidHandle);

    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipEventSynchronize(stop));

    float tElapsed = 1.0f;
    // start on device 0 but stop on device 1
    HIP_ASSERT(hipEventElapsedTime(&tElapsed,start,stop) == hipErrorInvalidHandle);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(start1));
    HIP_CHECK(hipEventDestroy(stop));
  }
}

TEST_CASE("Unit_hipEventElapsedTime") {
  hipEvent_t start;
  HIP_CHECK(hipEventCreate(&start));

  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start, nullptr));
  HIP_CHECK(hipEventSynchronize(start));

  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));

  float tElapsed = 1.0f;
  HIP_CHECK(hipEventElapsedTime(&tElapsed, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
}
