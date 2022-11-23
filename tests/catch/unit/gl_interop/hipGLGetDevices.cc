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
#include <hip/hip_runtime_api.h>

#include "gl_interop_common.hh"

namespace {
constexpr std::array<hipGLDeviceList, 3> kDeviceLists{
    hipGLDeviceListAll, hipGLDeviceListCurrentFrame, hipGLDeviceListNextFrame};
}  // anonymous namespace

TEST_CASE("Unit_hipGLGetDevices_Positive_Basic") {
  GLContextScopeGuard gl_context;

  const auto device_list = GENERATE(from_range(begin(kDeviceLists), end(kDeviceLists)));

  const int device_count = HipTest::getDeviceCount();

  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, device_list));

  REQUIRE(gl_device_count == 1);
  REQUIRE(gl_devices.at(0) == 0);
}

TEST_CASE("Unit_hipGLGetDevices_Positive_Parameters") {
  GLContextScopeGuard gl_context;

  const int device_count = HipTest::getDeviceCount();

  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  SECTION("pHipDeviceCount == nullptr") {
    HIP_CHECK(hipGLGetDevices(nullptr, gl_devices.data(), device_count, hipGLDeviceListAll));
    REQUIRE(gl_devices.at(0) == 0);
  }

  SECTION("pHipDevices == nullptr") {
    HIP_CHECK(hipGLGetDevices(&gl_device_count, nullptr, device_count, hipGLDeviceListAll));
    REQUIRE(gl_device_count == 1);
  }

  SECTION("hipDeviceCount == 0") {
    HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), 0, hipGLDeviceListAll));
    REQUIRE(gl_device_count == 1);
    REQUIRE(gl_devices.at(0) == -1);
  }
}

TEST_CASE("Unit_hipGLGetDevices_Negative_Parameters") {
  GLContextScopeGuard gl_context;

  const int device_count = HipTest::getDeviceCount();

  unsigned int gl_device_count = 0;
  std::vector<int> gl_devices(device_count, -1);

  SECTION("invalid deviceList") {
    HIP_CHECK_ERROR(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count,
                                    static_cast<hipGLDeviceList>(-1)),
                    hipErrorInvalidValue);
    REQUIRE(gl_device_count == 0);
    REQUIRE(gl_devices.at(0) == -1);
  }
}