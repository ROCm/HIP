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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 Testcase Scenarios :
 1) hipMemset2D api with basic functionality.
 2) hipMemset2DAsync api with basic functionality.
 3) hipMemset2D api with partial memset and unique width/height.
*/


#include <hip_test_common.hh>


// Table with unique width/height and memset values.
// (width2D, height2D, memsetWidth, memsetHeight)
typedef std::tuple<int, int, int, int> tupletype;

static constexpr std::initializer_list<tupletype> tableItems {
               std::make_tuple(20,   20, 20, 20),
               std::make_tuple(10,   10,  4,  4),
               std::make_tuple(100, 100, 20, 40),
               std::make_tuple(256, 256, 39, 19),
               std::make_tuple(100, 100, 20,  0),
               std::make_tuple(100, 100,  0, 20),
               std::make_tuple(100, 100,  0,  0),
               };



/**
 * Basic Functionality of hipMemset2D
 */
TEST_CASE("Unit_hipMemset2D_BasicFunctional") {
  constexpr int memsetval = 0x24;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t pitch_A;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH;
  size_t elements = numW * numH;
  char *A_d, *A_h;

  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width,
                          numH));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);

  for (size_t i = 0; i < elements; i++) {
    A_h[i] = 1;
  }

  HIP_CHECK(hipMemset2D(A_d, pitch_A, memsetval, numW, numH));
  HIP_CHECK(hipMemcpy2D(A_h, width, A_d, pitch_A, numW, numH,
                       hipMemcpyDeviceToHost));

  for (size_t i = 0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      INFO("Memset2D mismatch at index:" << i << " computed:"
                                     << A_h[i] << " memsetval:" << memsetval);
      REQUIRE(false);
    }
  }

  hipFree(A_d);
  free(A_h);
}


/**
 * Basic Functionality of hipMemset2DAsync
 */
TEST_CASE("Unit_hipMemset2DAsync_BasicFunctional") {
  constexpr int memsetval = 0x26;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t pitch_A;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH;
  size_t elements = numW * numH;
  char *A_d, *A_h;

  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A,
                          width, numH));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);

  for (size_t i = 0; i < elements; i++) {
      A_h[i] = 1;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMemset2DAsync(A_d, pitch_A, memsetval, numW, numH, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy2D(A_h, width, A_d, pitch_A, numW, numH,
                       hipMemcpyDeviceToHost));

  for (size_t i=0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      INFO("Memset2DAsync mismatch at index:" << i << " computed:"
                                     << A_h[i] << " memsetval:" << memsetval);
      REQUIRE(false);
    }
  }

  hipFree(A_d);
  HIP_CHECK(hipStreamDestroy(stream));
  free(A_h);
}


/**
 * Memset partial buffer with unique Width and Height
 */
TEST_CASE("Unit_hipMemset2D_UniqueWidthHeight") {
  int width2D, height2D;
  int memsetWidth, memsetHeight;
  char *A_d, *A_h;
  size_t pitch_A;
  constexpr int memsetval = 0x26;

  std::tie(width2D, height2D, memsetWidth, memsetHeight) =
                 GENERATE(table<int, int, int, int>(tableItems));

  size_t width = width2D * sizeof(char);
  size_t sizeElements = width * height2D;

  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A,
                          width, height2D));

  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);

  for (size_t index = 0; index < sizeElements; index++) {
    A_h[index] = 'c';
  }

  INFO("2D Dimension: Width:" << width2D << " Height:" << height2D <<
           " MemsetWidth:" << memsetWidth << " MemsetHeight:" << memsetHeight);

  HIP_CHECK(hipMemset2D(A_d, pitch_A, memsetval, memsetWidth, memsetHeight));
  HIP_CHECK(hipMemcpy2D(A_h, width, A_d, pitch_A, width2D, height2D,
                       hipMemcpyDeviceToHost));

  for (int row = 0; row < memsetHeight; row++) {
    for (int column = 0; column < memsetWidth; column++) {
      if (A_h[(row * width) + column] != memsetval) {
        INFO("A_h[" << row << "][" << column << "]" <<
                                         " didnot match " << memsetval);
        REQUIRE(false);
      }
    }
  }

  hipFree(A_d);
  free(A_h);
}

