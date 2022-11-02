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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

#define R 8  // rows, height
#define C 8  // columns, width


TEST_CASE("Unit_hipGetChannelDesc_CreateAndGet") {
  CHECK_IMAGE_SUPPORT

  hipChannelFormatDesc chan_test, chan_desc;
  hipArray* hipArray;

  chan_desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindSigned);
  HIP_CHECK(hipMallocArray(&hipArray, &chan_desc, C, R, 0));
  HIP_CHECK(hipGetChannelDesc(&chan_test, hipArray));

  if ((chan_test.x != 32) || (chan_test.y != 0) || (chan_test.z != 0) || (chan_test.f != 0)) {
    INFO("Mismatch observed : " << chan_test.x << chan_test.y << chan_test.z << chan_test.f);
    REQUIRE(false);
  }

  HIP_CHECK(hipFreeArray(hipArray));
}


TEST_CASE("Unit_hipGetChannelDesc_negative") {
  CHECK_IMAGE_SUPPORT

  hipChannelFormatDesc chan_test, chan_desc;
  hipArray* hipArray;

  chan_desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindSigned);
  HIP_CHECK(hipMallocArray(&hipArray, &chan_desc, C, R, 0));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipGetChannelDesc(nullptr, hipArray), hipErrorInvalidValue);
  }

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipGetChannelDesc(&chan_test, nullptr), hipErrorInvalidHandle);
  }

  HIP_CHECK(hipFreeArray(hipArray));
}
