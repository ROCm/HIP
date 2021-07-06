/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <utility>
#include <vector>
/*
This testfile verifies the following scenarios of all hipMemcpy API
1. Negative Scenarios
2. Half Memory copy scenarios
3. Null check scenario
*/

static constexpr auto NUM_ELM{1024*1024};


/*This testcase verifies the negative scenarios of hipMemcpy APIs
*/
TEST_CASE("Unit_hipMemcpy_Negative") {
  // Initialization of variables
  float *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  float *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HIP_CHECK(hipSetDevice(0));
  HipTest::initArrays<float>(&A_d, &B_d, &C_d,
                             &A_h, &B_h, &C_h,
                             NUM_ELM*sizeof(float));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  SECTION("Pass nullptr to destination pointer for all Memcpy APIs") {
    REQUIRE(hipMemcpy(nullptr, A_d, NUM_ELM * sizeof(float),
                      hipMemcpyDefault) != hipSuccess);
    REQUIRE(hipMemcpyAsync(nullptr, A_h, NUM_ELM * sizeof(float),
                           hipMemcpyDefault, stream) != hipSuccess);
    REQUIRE(hipMemcpyHtoD(hipDeviceptr_t(nullptr), A_h,
                          NUM_ELM * sizeof(float)) != hipSuccess);
    REQUIRE(hipMemcpyHtoDAsync(hipDeviceptr_t(nullptr), A_h,
                               NUM_ELM * sizeof(float),
                               stream) != hipSuccess);
    REQUIRE(hipMemcpyDtoH(nullptr, hipDeviceptr_t(A_d),
                          NUM_ELM * sizeof(float)) != hipSuccess);
    REQUIRE(hipMemcpyDtoHAsync(nullptr, hipDeviceptr_t(A_d),
                               NUM_ELM * sizeof(float),
                               stream) != hipSuccess);
    REQUIRE(hipMemcpyDtoD(hipDeviceptr_t(nullptr),
                          hipDeviceptr_t(A_d), NUM_ELM * sizeof(float))
                          != hipSuccess);
    REQUIRE(hipMemcpyDtoDAsync(hipDeviceptr_t(nullptr),
                               hipDeviceptr_t(A_d),
                               NUM_ELM * sizeof(float), stream)
                               != hipSuccess);
  }

  SECTION("Passing nullptr to source pointer") {
    REQUIRE(hipMemcpy(A_h, nullptr, NUM_ELM * sizeof(float),
                      hipMemcpyDefault) != hipSuccess);
    REQUIRE(hipMemcpyAsync(A_d, nullptr,
                           NUM_ELM * sizeof(float),
                           hipMemcpyDefault, stream) != hipSuccess);
    REQUIRE(hipMemcpyHtoD(hipDeviceptr_t(A_d), nullptr,
                          NUM_ELM * sizeof(float)) != hipSuccess);
    REQUIRE(hipMemcpyHtoDAsync(hipDeviceptr_t(A_d), nullptr,
                               NUM_ELM * sizeof(float),
                               stream) != hipSuccess);
    REQUIRE(hipMemcpyDtoH(A_h, hipDeviceptr_t(nullptr),
                          NUM_ELM * sizeof(float)) != hipSuccess);
    REQUIRE(hipMemcpyDtoHAsync(A_h, hipDeviceptr_t(nullptr),
                               NUM_ELM * sizeof(float),
                               stream) != hipSuccess);
    REQUIRE(hipMemcpyDtoD(hipDeviceptr_t(A_d),
                          hipDeviceptr_t(nullptr), NUM_ELM * sizeof(float))
                          != hipSuccess);
    REQUIRE(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d),
                               hipDeviceptr_t(nullptr),
                               NUM_ELM * sizeof(float), stream)
                               != hipSuccess);
  }

  SECTION("Passing nullptr to both source and dest pointer") {
    REQUIRE(hipMemcpy(nullptr, nullptr, NUM_ELM * sizeof(float),
                      hipMemcpyDefault) != hipSuccess);
    REQUIRE(hipMemcpyAsync(nullptr, nullptr, NUM_ELM * sizeof(float),
                           hipMemcpyDefault, stream) != hipSuccess);
    REQUIRE(hipMemcpyHtoD(hipDeviceptr_t(nullptr), nullptr,
                          NUM_ELM * sizeof(float)) != hipSuccess);
    REQUIRE(hipMemcpyHtoDAsync(hipDeviceptr_t(nullptr), nullptr,
                               NUM_ELM * sizeof(float),
                               stream) != hipSuccess);
    REQUIRE(hipMemcpyDtoH(nullptr, hipDeviceptr_t(nullptr),
                          NUM_ELM * sizeof(float)) != hipSuccess);
    REQUIRE(hipMemcpyDtoHAsync(nullptr, hipDeviceptr_t(nullptr),
                               NUM_ELM * sizeof(float),
                               stream) != hipSuccess);
    REQUIRE(hipMemcpyDtoD(hipDeviceptr_t(nullptr),
                          hipDeviceptr_t(nullptr), NUM_ELM * sizeof(float))
                          != hipSuccess);
    REQUIRE(hipMemcpyDtoDAsync(hipDeviceptr_t(nullptr),
                               hipDeviceptr_t(nullptr),
                               NUM_ELM * sizeof(float), stream)
                               != hipSuccess);
  }

  SECTION("Passing same pointers") {
    HIP_CHECK(hipMemcpy(A_d, A_d, (NUM_ELM/2) * sizeof(float),
                        hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(A_h, A_h, (NUM_ELM/2) * sizeof(float),
                        hipMemcpyDefault));
    HIP_CHECK(hipMemcpyAsync(A_d, A_d, (NUM_ELM/2) * sizeof(float),
                            hipMemcpyDefault, stream));
    HIP_CHECK(hipMemcpyAsync(A_h, A_h, (NUM_ELM/2) * sizeof(float),
                            hipMemcpyDefault, stream));
    HIP_CHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d),
                            hipDeviceptr_t(A_d),
                            NUM_ELM * sizeof(float)));
    HIP_CHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d),
                                 hipDeviceptr_t(A_d),
                                 NUM_ELM * sizeof(float), stream));
  }

  HipTest::freeArrays<float>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipStreamDestroy(stream));
}

/*
This testcase verifies the Nullcheck for all the 8 Memcpy APIs
*/
TEST_CASE("Unit_hipMemcpy_NullCheck") {
  // Initialization of variables
  float *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  float *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HIP_CHECK(hipSetDevice(0));
  HipTest::initArrays<float>(&A_d, &B_d, &C_d,
                             &A_h, &B_h, &C_h,
                             NUM_ELM*sizeof(float));
  hipStream_t stream;
  hipStreamCreate(&stream);
  HIP_CHECK(hipMemcpy(A_d, C_h,
                      NUM_ELM*sizeof(float),
                      hipMemcpyHostToDevice));

  SECTION("hipMemcpyHtoD API null size check") {
    REQUIRE(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h,
                          0) == hipSuccess);
    HIP_CHECK(hipMemcpy(B_h, A_d,
                        NUM_ELM*sizeof(float),
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(C_h, B_h, NUM_ELM);
  }

  SECTION("hipMemcpyHtoDAsync API null size check") {
    HIP_CHECK(hipMemcpyHtoDAsync(hipDeviceptr_t(A_d), A_h,
                          0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(B_h, A_d,
                        NUM_ELM*sizeof(float),
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(C_h, B_h, NUM_ELM);
  }
  SECTION("hipMemcpy API null size check") {
    HIP_CHECK(hipMemcpy(A_d, B_h, 0, hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(B_h, A_d,
                        NUM_ELM*sizeof(float),
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(C_h, B_h, NUM_ELM);
  }
  SECTION("hipMemcpyAsync API null size check") {
    HIP_CHECK(hipMemcpyAsync(A_d, B_h, 0, hipMemcpyDefault, stream));
    HIP_CHECK(hipMemcpy(B_h, A_d,
                        NUM_ELM*sizeof(float),
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(C_h, B_h, NUM_ELM);
  }
  SECTION("hipMemcpyDtoH API null size check") {
    HIP_CHECK(hipMemcpyDtoH(C_h, hipDeviceptr_t(A_d), 0));
    HIP_CHECK(hipMemcpy(B_h, A_d,
                        NUM_ELM*sizeof(float),
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(C_h, B_h, NUM_ELM);
  }
  SECTION("hipMemcpyDtoHAsync API null size check") {
    HIP_CHECK(hipMemcpyDtoHAsync(C_h, hipDeviceptr_t(A_d), 0, stream));
    HIP_CHECK(hipMemcpy(B_h, A_d,
                        NUM_ELM*sizeof(float),
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(C_h, B_h, NUM_ELM);
  }
  SECTION("hipMemcpyDtoD API null size check") {
    HIP_CHECK(hipMemcpy(C_d, A_h,
                        NUM_ELM*sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyDtoD(hipDeviceptr_t(C_d), hipDeviceptr_t(A_d), 0));
    HIP_CHECK(hipMemcpy(B_h, C_d,
                        NUM_ELM*sizeof(float),
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM);
  }
  SECTION("hipMemcpyDtoDAsync API null size check") {
    HIP_CHECK(hipMemcpy(C_d, A_h,
                        NUM_ELM*sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(C_d), hipDeviceptr_t(A_d),
                                 0, stream));
    HIP_CHECK(hipMemcpy(B_h, C_d,
                        NUM_ELM*sizeof(float),
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM);
  }

  HipTest::freeArrays<float>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipStreamDestroy(stream));
}

/*
This testcase verifies all the hipMemcpy APIs by
copying half the memory.
*/
TEST_CASE("Unit_hipMemcpy_HalfMemCopy") {
  // Initialization of variables
  float *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  float *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HIP_CHECK(hipSetDevice(0));
  HipTest::initArrays<float>(&A_d, &B_d, &C_d,
                             &A_h, &B_h, &C_h,
                             NUM_ELM*sizeof(float));
  hipStream_t stream;
  hipStreamCreate(&stream);

  SECTION("hipMemcpyHtoD half memory copy") {
    HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h,
                            (NUM_ELM * sizeof(float))/2));
    HIP_CHECK(hipMemcpy(B_h, A_d,
                        (NUM_ELM*sizeof(float))/2,
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM/2);
  }

  SECTION("hipMemcpyHtoDAsync half memory copy") {
    HIP_CHECK(hipMemcpyHtoDAsync(hipDeviceptr_t(A_d), A_h,
                                 (NUM_ELM * sizeof(float))/2, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(B_h, A_d,
                        (NUM_ELM*sizeof(float))/2,
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM/2);
  }

  SECTION("hipMemcpyDtoH half memory copy") {
    HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h,
                            (NUM_ELM * sizeof(float))));
    HIP_CHECK(hipMemcpyDtoH(B_h, hipDeviceptr_t(A_d),
                            (NUM_ELM * sizeof(float))/2));
    HipTest::checkTest(A_h, B_h, NUM_ELM/2);
  }

  SECTION("hipMemcpyDtoHAsync half memory copy") {
    HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h,
                            (NUM_ELM * sizeof(float))));
    HIP_CHECK(hipMemcpyDtoHAsync(B_h, hipDeviceptr_t(A_d),
                                 (NUM_ELM * sizeof(float))/2,
                                 stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HipTest::checkTest(A_h, B_h, NUM_ELM/2);
  }

  SECTION("hipMemcpyDtoD half memory copy") {
    HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h,
                            (NUM_ELM * sizeof(float))/2));
    HIP_CHECK(hipMemcpyDtoD(hipDeviceptr_t(B_d), hipDeviceptr_t(A_d),
                            (NUM_ELM*sizeof(float))/2));
    HIP_CHECK(hipMemcpy(B_h, B_d,
                        (NUM_ELM*sizeof(float))/2,
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM/2);
  }

  SECTION("hipMemcpyDtoDAsync half memory copy") {
    HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h,
                            (NUM_ELM * sizeof(float))/2));
    HIP_CHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(B_d), hipDeviceptr_t(A_d),
                            (NUM_ELM*sizeof(float))/2,
                            stream));
    HIP_CHECK(hipMemcpy(B_h, B_d,
                        (NUM_ELM*sizeof(float))/2,
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM/2);
  }

  SECTION("hipMemcpy half memory copy") {
    HIP_CHECK(hipMemcpy(A_d, A_h
                        , (NUM_ELM*sizeof(float)),
                        hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(B_h, A_d,
                        (NUM_ELM/2)*sizeof(float),
                        hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM/2);
  }

  SECTION("hipMemcpyAsync half memory copy") {
    HIP_CHECK(hipMemcpy(A_d, A_h,
                        (NUM_ELM*sizeof(float)),
                        hipMemcpyDefault));
    HIP_CHECK(hipMemcpyAsync(B_h, A_d,
                             (NUM_ELM/2)*sizeof(float),
                             hipMemcpyDeviceToHost, stream));
    HipTest::checkTest(A_h, B_h, NUM_ELM/2);
  }
  HipTest::freeArrays<float>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipStreamDestroy(stream));
}
