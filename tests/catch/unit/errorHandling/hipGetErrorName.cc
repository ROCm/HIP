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
#include "errorEnumerators.h"

TEST_CASE("Unit_hipGetErrorName_Positive_Basic") {
  const char* error_string = nullptr;
  const auto enumerator =
      GENERATE(from_range(std::begin(kErrorEnumerators), std::end(kErrorEnumerators)));

  error_string = hipGetErrorName(enumerator);

  REQUIRE(error_string != nullptr);
  REQUIRE(strlen(error_string) > 0);
}

TEST_CASE("Unit_hipGetErrorName_Negative_Parameters") {
  const char* error_string = hipGetErrorName(static_cast<hipError_t>(-1));
  REQUIRE(error_string != nullptr);
#if HT_NVIDIA
  REQUIRE_THAT(error_string, Catch::Equals("cudaErrorUnknown"));
#elif HT_AMD
  // TODO
  REQUIRE_THAT(error_string, Catch::Equals("TODO"));
#endif
}