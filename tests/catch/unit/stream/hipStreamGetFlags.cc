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
/**
Testcase Scenarios :
1) Test flag value of stream created with hipStreamCreateWithFlags/
 /hipStreamCreate/hipStreamCreateWithPriority.
2) Negative tests for hipStreamGetFlags api.
3) Test flag value when streams created with CUMask.
*/

#include <hip_test_common.hh>

/**
 * @brief Check that hipStreamGetFlags returns the same flags that were used to create the stream.
 *
 */
TEST_CASE("Unit_hipStreamGetFlags_Basic") {
  unsigned int expectedFlag = GENERATE(hipStreamDefault, hipStreamNonBlocking);
  unsigned int returnedFlags;
  hipStream_t stream;

  HIP_CHECK(hipStreamCreateWithFlags(&stream, expectedFlag));
  HIP_CHECK(hipStreamGetFlags(stream, &returnedFlags));
  REQUIRE((returnedFlags & expectedFlag) == expectedFlag);
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * @brief Negative scenarios for hipStreamGetFlags.
 *
 */
TEST_CASE("Unit_hipStreamGetFlags_Negative") {
  hipStream_t validStream;
  unsigned int flags;

  HIP_CHECK(hipStreamCreate(&validStream));

  SECTION("Nullptr Stream && Valid Flags") { /* EXSWCPHIPT-17 */
#if HT_AMD
    HIP_CHECK_ERROR(hipStreamGetFlags(nullptr, &flags), hipErrorInvalidValue);
#elif HT_NVIDIA
    HIP_CHECK(hipStreamGetFlags(nullptr, &flags));
#endif
  }

  SECTION("Valid Stream && Nullptr Flags") {
    HIP_CHECK_ERROR(hipStreamGetFlags(validStream, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipStreamDestroy(validStream));
}

#if HT_AMD
/**
 * Test flag value when streams created with CUMask.
 */
TEST_CASE("Unit_hipStreamGetFlags_StreamsCreatedWithCUMask") {
  hipStream_t stream;
  unsigned int flags;
  const uint32_t cuMask = 0xffffffff;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, 1, &cuMask));
  HIP_CHECK(hipStreamGetFlags(stream, &flags));
  REQUIRE(flags == hipStreamDefault);
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif
