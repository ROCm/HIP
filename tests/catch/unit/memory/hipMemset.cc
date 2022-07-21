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
 1) Test hipMemset small size buffers with unique memset values.
 2) Test hipMemset, hipMemsetD8, hipMemsetD16, hipMemsetD32 apis with unique
    number of elements and memset values.
 3) Test hipMemsetAsync, hipMemsetD8Async, hipMemsetD16Async, hipMemsetD32Async
    apis with unique number of elements and memset values.
 4) Test two memset async operations at the same time.
*/


#include <hip_test_common.hh>


// Table with unique number of elements and memset values.
// (N, memsetval, memsetD32val, memsetD16val, memsetD8val)
typedef std::tuple<size_t, char, int, int16_t, char> tupletype;
static constexpr std::initializer_list<tupletype> tableItems {
               std::make_tuple((4*1024*1024), 0x42, 0xDEADBEEF, 0xDEAD, 0xDE),
               std::make_tuple((10)         , 0x42, 0x101     , 0x10,   0x1),
               std::make_tuple((10013)      , 0x5a, 0xDEADBEEF, 0xDEAD, 0xDE),
               std::make_tuple((256*1024*1024), 0xa6, 0xCAFEBABE, 0xCAFE, 0xCA)
               };

enum MemsetType {
  hipMemsetTypeDefault,
  hipMemsetTypeD8,
  hipMemsetTypeD16,
  hipMemsetTypeD32
};

template<typename T>
static bool testhipMemset(T *A_h, T *A_d, T memsetval, enum MemsetType type,
                  size_t numElements) {
  size_t Nbytes = numElements * sizeof(T);
  bool testResult = true;
  constexpr auto MAX_OFFSET = 3;  // To memset on unaligned ptr.

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  A_h = reinterpret_cast<T*> (malloc(Nbytes));
  REQUIRE(A_h != nullptr);

  for (int offset = MAX_OFFSET; offset >= 0; offset --) {
    if (type == hipMemsetTypeDefault) {
      HIP_CHECK(hipMemset(A_d + offset, memsetval, numElements - offset));

    } else if (type == hipMemsetTypeD8) {
      HIP_CHECK(hipMemsetD8((hipDeviceptr_t)(A_d + offset), memsetval,
                                                    numElements - offset));

    } else if (type == hipMemsetTypeD16) {
      HIP_CHECK(hipMemsetD16((hipDeviceptr_t)(A_d + offset), memsetval,
                                                    numElements - offset));

    } else if (type == hipMemsetTypeD32) {
      HIP_CHECK(hipMemsetD32((hipDeviceptr_t)(A_d + offset), memsetval,
                                                    numElements - offset));
    }

    HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
    for (size_t i = offset; i < numElements; i++) {
      if (A_h[i] != memsetval) {
        testResult = false;
        CAPTURE(i, A_h[i], memsetval);
        break;
      }
    }
  }

  HIP_CHECK(hipFree(A_d));
  free(A_h);
  return testResult;
}


template<typename T>
static bool testhipMemsetAsync(T *A_h, T *A_d, T memsetval,
                                 enum MemsetType type, size_t numElements) {
  size_t Nbytes = numElements * sizeof(T);
  bool testResult = true;
  constexpr auto MAX_OFFSET = 3;  // To memset on unaligned ptr.
  hipStream_t stream;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  A_h = reinterpret_cast<T*> (malloc(Nbytes));
  REQUIRE(A_h != nullptr);

  for (int offset = MAX_OFFSET; offset >= 0; offset --) {
    if (type == hipMemsetTypeDefault) {
      HIP_CHECK(hipMemsetAsync(A_d + offset, memsetval, numElements - offset,
                                                                      stream));

    } else if (type == hipMemsetTypeD8) {
      HIP_CHECK(hipMemsetD8Async((hipDeviceptr_t)(A_d + offset), memsetval,
                                                numElements - offset, stream));

    } else if (type == hipMemsetTypeD16) {
      HIP_CHECK(hipMemsetD16Async((hipDeviceptr_t)(A_d + offset), memsetval,
                                                numElements - offset, stream));

    } else if (type == hipMemsetTypeD32) {
      HIP_CHECK(hipMemsetD32Async((hipDeviceptr_t)(A_d + offset), memsetval,
                                                numElements - offset, stream));
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
    for (size_t i = offset; i < numElements; i++) {
      if (A_h[i] != memsetval) {
        testResult = false;
        CAPTURE(i, A_h[i], memsetval);
        break;
      }
    }
  }

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipStreamDestroy(stream));
  free(A_h);
  return testResult;
}


/**
 * Test hipMemset, hipMemsetD8, hipMemsetD16, hipMemsetD32 apis with unique
 * number of elements and memset values.
 */
TEST_CASE("Unit_hipMemset_SetMemoryWithOffset") {
  char memsetval;
  int memsetD32val;
  int16_t memsetD16val;
  char memsetD8val;
  size_t N;
  bool ret;

  std::tie(N, memsetval, memsetD32val, memsetD16val, memsetD8val) =
                 GENERATE(table<size_t, char, int, int16_t, char>(tableItems));


  SECTION("Memset with hipMemsetTypeDefault") {
    char *cA_d{nullptr}, *cA_h{nullptr};
    ret = testhipMemset(cA_h, cA_d, memsetval, hipMemsetTypeDefault, N);
    REQUIRE(ret == true);
  }

  SECTION("Memset with hipMemsetTypeD32") {
    int32_t *iA_d{nullptr}, *iA_h{nullptr};
    ret = testhipMemset(iA_h, iA_d, memsetD32val, hipMemsetTypeD32, N);
    REQUIRE(ret == true);
  }

  SECTION("Memset with hipMemsetTypeD16") {
    int16_t *siA_d{nullptr}, *siA_h{nullptr};
    ret = testhipMemset(siA_h, siA_d, memsetD16val, hipMemsetTypeD16, N);
    REQUIRE(ret == true);
  }

  SECTION("Memset with hipMemsetTypeD8") {
    char *cA_d{nullptr}, *cA_h{nullptr};
    ret = testhipMemset(cA_h, cA_d, memsetD8val, hipMemsetTypeD8, N);
    REQUIRE(ret == true);
  }
}


/**
 * Test hipMemsetAsync, hipMemsetD8Async, hipMemsetD16Async, hipMemsetD32Async
 * apis with unique number of elements and memset values.
 */
TEST_CASE("Unit_hipMemsetAsync_SetMemoryWithOffset") {
  char memsetval;
  int memsetD32val;
  int16_t memsetD16val;
  char memsetD8val;
  size_t N;
  bool ret;

  std::tie(N, memsetval, memsetD32val, memsetD16val, memsetD8val) =
                 GENERATE(table<size_t, char, int, int16_t, char>(tableItems));


  SECTION("Memset with hipMemsetTypeDefault") {
    char *cA_d{nullptr}, *cA_h{nullptr};
    ret = testhipMemsetAsync(cA_h, cA_d, memsetval, hipMemsetTypeDefault, N);
    REQUIRE(ret == true);
  }

  SECTION("Memset with hipMemsetTypeD32") {
    int32_t *iA_d{nullptr}, *iA_h{nullptr};
    ret = testhipMemsetAsync(iA_h, iA_d, memsetD32val, hipMemsetTypeD32, N);
    REQUIRE(ret == true);
  }

  SECTION("Memset with hipMemsetTypeD16") {
    int16_t *siA_d{nullptr}, *siA_h{nullptr};
    ret = testhipMemsetAsync(siA_h, siA_d, memsetD16val, hipMemsetTypeD16, N);
    REQUIRE(ret == true);
  }

  SECTION("Memset with hipMemsetTypeD8") {
    char *cA_d{nullptr}, *cA_h{nullptr};
    ret = testhipMemsetAsync(cA_h, cA_d, memsetD8val, hipMemsetTypeD8, N);
    REQUIRE(ret == true);
  }
}

/**
 * Test hipMemset small size buffers with unique memset values.
 */
TEST_CASE("Unit_hipMemset_SmallBufferSizes") {
  char *A_d, *A_h;
  constexpr int memsetval = 0x24;

  auto numElements = GENERATE(range(1, 4));
  int numBytes = numElements * sizeof(char);

  HIP_CHECK(hipMalloc(&A_d, numBytes));
  A_h = reinterpret_cast<char*> (malloc(numBytes));

  HIP_CHECK(hipMemset(A_d, memsetval, numBytes));
  HIP_CHECK(hipMemcpy(A_h, A_d, numBytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < numBytes; i++) {
    if (A_h[i] != memsetval) {
      INFO("Mismatch at index:" << i << " computed:" << A_h[i]
                                          << " memsetval:" << memsetval);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipFree(A_d));
  free(A_h);
}


/**
 * Test two memset async operations at the same time.
 */
TEST_CASE("Unit_hipMemset_2AsyncOperations") {
  std::vector<float> v;
  v.resize(2048);
  float* p2, *p3;
  hipMalloc(reinterpret_cast<void**>(&p2), 4096 + 4096*2);
  p3 = p2+2048;
  hipStream_t s;
  hipStreamCreate(&s);
  hipMemsetAsync(p2, 0, 32*32*4, s);
  hipMemsetD32Async((hipDeviceptr_t)p3, 0x3fe00000, 32*32, s);
  hipStreamSynchronize(s);
  for (int i = 0; i < 256; ++i) {
    hipMemsetAsync(p2, 0, 32*32*4, s);
    hipMemsetD32Async((hipDeviceptr_t)p3, 0x3fe00000, 32*32, s);
  }
  hipStreamSynchronize(s);
  hipDeviceSynchronize();
  hipMemcpy(&v[0], p2, 1024, hipMemcpyDeviceToHost);
  hipMemcpy(&v[1024], p3, 1024, hipMemcpyDeviceToHost);

  REQUIRE(v[0] == 0);
  REQUIRE(v[1024] == 1.75f);
}
