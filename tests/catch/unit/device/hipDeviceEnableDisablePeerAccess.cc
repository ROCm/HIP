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
#include <hip_test_helper.hh>

/*
  Positive tests:
    - for each peer change and check other peer access

  Negative tests:
    - peerDeviceId is invalid
*/
TEST_CASE("Unit_hipDeviceEnableDisablePeerAccess_positive") {
  int canAccessPeer = 0;
  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  int dev = GENERATE(range(0, HipTest::getGeviceCount()));
  int peerDev = GENERATE(range(0, HipTest::getGeviceCount()));

  if (dev != peerDev) {
    hipSetDevice(dev);
    REQUIRE(hipDeviceCanAccessPeer(&canAccessPeer, dev, peerDev) == hipSuccess);
    if (canAccessPeer == 0) {
      HipTest::HIP_SKIP_TEST("Skipping because no P2P support");
      return;
    }
    REQUIRE(hipDeviceEnablePeerAccess(peerDev, 0) == hipSuccess);
    REQUIRE(hipDeviceDisablePeerAccess(peerDev) == hipSuccess);
  }
}


TEST_CASE("Unit_hipDeviceEnableDisablePeerAccess_negative") {
  int canAccessPeer = 0;
  int deviceCount = HipTest::getGeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  SECTION("peerDeviceId is invalid") {
    REQUIRE(hipDeviceEnablePeerAccess(-1, 0) != hipSuccess);
    REQUIRE(hipDeviceEnablePeerAccess(deviceCount + 1, 0) != hipSuccess);
    REQUIRE(hipDeviceDisablePeerAccess(-1) != hipSuccess);
    REQUIRE(hipDeviceDisablePeerAccess(deviceCount + 1) != hipSuccess);
  }
}
