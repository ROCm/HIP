/*
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
*/

/**
Testcase Scenarios :
 1) Test hipMemset apis with invalid pointer and invalid 2D pitch.
 2) Test hipMemsetAsync apis with invalid pointer and invalid 2D pitch.
*/


#include <hip_test_common.hh>

#define N 50
#define MEMSETVAL 0x42

/**
 * Testcase validates hipMemset apis behavior with
 * invalid pointer and invalid 2D pitch value.
 */
TEST_CASE("Unit_hipMemset_InvalidPtrTests") {
  hipError_t ret;
  constexpr int Nbytes = N*sizeof(char);
  char *A_d;

  SECTION("hipMemset with null") {
    ret = hipMemset(NULL, MEMSETVAL , Nbytes);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipMemset with hostptr") {
    char *A_h;
    A_h = reinterpret_cast<char *>(malloc(Nbytes));

    ret = hipMemset(A_h, MEMSETVAL, Nbytes);
    REQUIRE(ret != hipSuccess);

    free(A_h);
  }

  SECTION("hipMemsetD32 with null") {
    ret = hipMemsetD32(NULL, MEMSETVAL , Nbytes);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipMemsetD16 with null") {
    ret = hipMemsetD16(NULL, MEMSETVAL , Nbytes);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipMemsetD8 with null") {
    ret = hipMemsetD8(NULL, MEMSETVAL , Nbytes);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipMemset2D with null") {
    constexpr size_t NUM_H = 256, NUM_W = 256;
    size_t pitch_A;
    size_t width = NUM_W * sizeof(char);

    HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A,
                                                             width , NUM_H));
    ret = hipMemset2D(NULL, pitch_A, MEMSETVAL, NUM_W, NUM_H);
    REQUIRE(ret != hipSuccess);

    hipFree(A_d);
  }
}


/**
 * Testcase validates hipMemsetAsync apis behavior with
 * invalid pointer and invalid 2D pitch value.
 */
TEST_CASE("Unit_hipMemsetAsync_InvalidPtrTests") {
  hipError_t ret;
  constexpr int Nbytes = N*sizeof(char);
  char *A_d;

  SECTION("hipMemsetAsync with null") {
    ret = hipMemsetAsync(NULL, MEMSETVAL, Nbytes , 0);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipMemsetD32Async with null") {
    ret = hipMemsetD32Async(NULL, MEMSETVAL , Nbytes, 0);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipMemsetD16Async with null") {
    ret = hipMemsetD16Async(NULL, MEMSETVAL , Nbytes, 0);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipMemsetD8Async with null") {
    ret = hipMemsetD8Async(NULL, MEMSETVAL , Nbytes, 0);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipMemset2DAsync with null") {
    constexpr size_t NUM_H = 256, NUM_W = 256;
    size_t pitch_A;
    size_t width = NUM_W * sizeof(char);

    HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A,
                                                             width , NUM_H));
    ret = hipMemset2DAsync(NULL, pitch_A, MEMSETVAL, NUM_W, NUM_H, 0);
    REQUIRE(ret != hipSuccess);

    hipFree(A_d);
  }
}
