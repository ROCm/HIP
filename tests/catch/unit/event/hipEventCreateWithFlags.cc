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
/*
Testcase Scenarios :
Unit_hipEventCreateWithFlags_Positive - Test simple event creation with hipEventCreateWithFlags api for each flag
*/

#include <hip_test_common.hh>

TEST_CASE("Unit_hipEventCreateWithFlags_Positive") {

#if HT_AMD
  // On AMD platform, hipEventInterprocess support is under development. Use of this flag will return an error. Omitted
  const unsigned int flagUnderTest = GENERATE(hipEventDefault, hipEventBlockingSync, hipEventDisableTiming, hipEventReleaseToDevice, hipEventReleaseToSystem);
#else
  // On Non-AMD platforms hipEventReleaseToDevice / hipEventReleaseToSystem are not defined
  const unsigned int flagUnderTest = GENERATE(hipEventDefault, hipEventBlockingSync, hipEventDisableTiming, hipEventInterprocess | hipEventDisableTiming);
#endif

  hipEvent_t event;
  HIP_CHECK(hipEventCreateWithFlags(&event, flagUnderTest));
  REQUIRE(event != nullptr);

  HIP_CHECK(hipEventDestroy(event));
}
