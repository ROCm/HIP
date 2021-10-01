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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios :

 1) Test hipHostMalloc() api with ptr as nullptr and check for return value.
 2) Test hipHostMalloc() api with size as max(size_t) and check for OOM error.
 3) Test hipHostMalloc() api with flags as max(unsigned int) and validate
 return value.
 4) Pass size as zero for hipHostMalloc() api and check ptr is reset with
 with return value success.
*/

#include <hip_test_common.hh>

/**
 * Performs argument validation of hipHostMalloc api.
 */
TEST_CASE("Unit_hipHostMalloc_ArgValidation") {
  hipError_t ret;
  constexpr size_t allocSize = 1000;
  char *ptr;

  SECTION("Pass ptr as nullptr") {
    ret = hipHostMalloc(static_cast<void **>(nullptr), allocSize);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Size as max(size_t)") {
    ret = hipHostMalloc(&ptr, std::numeric_limits<std::size_t>::max());
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Flags as max(uint)") {
    ret = hipHostMalloc(&ptr, allocSize,
                             std::numeric_limits<unsigned int>::max());
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Pass size as zero and check ptr reset") {
    HIP_CHECK(hipHostMalloc(&ptr, 0));
    REQUIRE(ptr == nullptr);
  }
}
