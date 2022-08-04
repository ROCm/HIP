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

// Since we can not use atomic*_system on every gpu, we will use wait based on clock rate.
// This wont be accurate since current clock rate of a GPU varies depending on many variables
// including thermals, load, utilization etc
__global__ void waitKernel(int clockRate, int seconds) {
  auto start = clock();
  auto ms = seconds * 1000;
  long long waitTill = clockRate * (long long)ms;
  while (1) {
    auto end = clock();
    if ((end - start) > waitTill) {
      return;
    }
  }
}

TEST_CASE("Unit_hipEventQuery_DifferentDevice") {
  hipEvent_t event1{}, event2{};
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  REQUIRE(event1 != nullptr);
  REQUIRE(event2 != nullptr);

  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);

  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  auto clockRate = prop.clockRate;

  // Start kernel on 1st device
  {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipEventRecord(event1, stream));

    // Start kernel and wait for 3 seconds
    // Make sure you increase this time if you add more tests here
    waitKernel<<<1, 1, 0, stream>>>(clockRate, 3);

    HIP_CHECK(hipEventRecord(event2, stream));

    HIP_CHECK(hipEventSynchronize(event1));
    HIP_CHECK(hipEventQuery(event1));  // Should be done

    HIP_CHECK_ERROR(hipEventQuery(event2),
                    hipErrorNotReady);  // Wont be done since kernel is waiting
  }

  // If other devices are available, set it
  if (HipTest::getDeviceCount() > 1) {
    HIP_CHECK(hipSetDevice(1));
  }

  // Query from same or other device depending on availability
  {
    HIP_CHECK(hipEventQuery(event1));
    HIP_CHECK_ERROR(hipEventQuery(event2), hipErrorNotReady);

    // Sync
    // if it hangs here it means GPU kernel cant see the value update by atomic store
    HIP_CHECK(hipEventSynchronize(event2));

    // Query, should be done now
    HIP_CHECK(hipEventQuery(event2));
  }

  // Query on same device if multiple devices are present
  if (HipTest::getDeviceCount() > 1) {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipEventQuery(event2));
  }

  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
  HIP_CHECK(hipStreamDestroy(stream));
}
