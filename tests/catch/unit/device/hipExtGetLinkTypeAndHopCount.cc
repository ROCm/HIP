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

#if HT_AMD
TEST_CASE("Unit_hipExtGetLinkTypeAndHopCount_Positive_Basic") {
  const auto device1 = GENERATE(range(0, HipTest::getDeviceCount()));
  const auto device2 = GENERATE(range(0, HipTest::getDeviceCount()));
  uint32_t link_type1 = -1, hop_count1 = -1;
  uint32_t link_type2 = -1, hop_count2 = -1;

  HIP_CHECK(hipExtGetLinkTypeAndHopCount(device1, device2, &link_type1, &hop_count1));
  HIP_CHECK(hipExtGetLinkTypeAndHopCount(device2, device1, &link_type2, &hop_count2));

  if (device1 == device2)
    REQUIRE(hop_count1 == 0);
  else
    REQUIRE(hop_count1 >= 0);

  REQUIRE(hop_count1 == hop_count2);
  REQUIRE(link_type1 == link_type2);
}

TEST_CASE("Unit_hipExtGetLinkTypeAndHopCount_Negative_Parameters") {
  uint32_t link_type, hop_count;
  SECTION("device ordinance 1 too large") {
    HIP_CHECK_ERROR(
        hipExtGetLinkTypeAndHopCount(HipTest::getDeviceCount(), 0, &link_type, &hop_count),
        hipErrorInvalidDevice);
  }

  SECTION("device ordinance 2 too large") {
    HIP_CHECK_ERROR(
        hipExtGetLinkTypeAndHopCount(0, HipTest::getDeviceCount(), &link_type, &hop_count),
        hipErrorInvalidDevice);
  }

  SECTION("device ordinances too large") {
    HIP_CHECK_ERROR(hipExtGetLinkTypeAndHopCount(HipTest::getDeviceCount(),
                                                 HipTest::getDeviceCount(), &link_type, &hop_count),
                    hipErrorInvalidDevice);
  }

  SECTION("device 1 < 0") {
    HIP_CHECK_ERROR(hipExtGetLinkTypeAndHopCount(-1, 0, &link_type, &hop_count),
                    hipErrorInvalidDevice);
  }

  SECTION("device 2 < 0") {
    HIP_CHECK_ERROR(hipExtGetLinkTypeAndHopCount(0, -1, &link_type, &hop_count),
                    hipErrorInvalidDevice);
  }

  SECTION("both devices < 0") {
    HIP_CHECK_ERROR(hipExtGetLinkTypeAndHopCount(-1, -1, &link_type, &hop_count),
                    hipErrorInvalidDevice);
  }

  SECTION("linktype == nullptr") {
    HIP_CHECK_ERROR(hipExtGetLinkTypeAndHopCount(0, 0, nullptr, &hop_count), hipErrorInvalidValue);
  }

  SECTION("hopcount == nullptr") {
    HIP_CHECK_ERROR(hipExtGetLinkTypeAndHopCount(0, 0, &link_type, nullptr), hipErrorInvalidValue);
  }

  SECTION("linktype and hopcount == nullptr") {
    HIP_CHECK_ERROR(hipExtGetLinkTypeAndHopCount(0, 0, nullptr, nullptr), hipErrorInvalidValue);
  }
}
#endif