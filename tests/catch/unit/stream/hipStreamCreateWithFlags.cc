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
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios :
1) Validates functionality of hipStreamCreateWithFlags when stream = nullptr.
2) Validates functionality of hipStreamCreateWithFlags when flag = 0xffffffff.
*/

#include <hip_test_common.hh>


TEST_CASE("Unit_hipStreamCreateWithFlags_ArgValidation") {
  // stream = nullptr test
  SECTION("stream is nullptr") {
    REQUIRE(hipStreamCreateWithFlags(nullptr, hipStreamDefault) != hipSuccess);
  }
  // flag value invalid test
  SECTION("flag value invalid") {
    hipStream_t stream;
    REQUIRE(hipStreamCreateWithFlags(&stream, 0xffffffff) != hipSuccess);
  }
}
