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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
 * Test Scenarios:
 * 1. Perform simple and pinned host memory of  hipMemcpyHtoA API
 * 2. Allocate Memory from one GPU device and call hipMemcpyHtoA from Peer
 *    GPU device
 * 3. Perform hipMemcpyHtoA Negative Scenarios
 * 4. Perform bytecount 0  validation for hipMemcpyHtoA API
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>


static constexpr auto NUM_W{10};
static constexpr auto NUM_H{1};
static constexpr auto copy_bytes{2};

/*
This testcase performs the basic and pinned host memory scenarios
of hipMemcpyHtoA API
Input: "B_h" which is initialized with 1.6
Output: "A_d" output of hipMemcpyHtoA is copied to "hData" host variable
        validated the result with "B_h"

The same scenario is then verified with pinned host memory
*/
TEMPLATE_TEST_CASE("Unit_hipMemcpyHtoA_Basic", "[hipMemcpyHtoA]",
                   char, int, float) {
  HIP_CHECK(hipSetDevice(0));
  auto memtype_check =  GENERATE(0, 1);
  hipArray *A_d;
  TestType *hData{nullptr}, *B_h{nullptr};
  size_t width{NUM_W * sizeof(TestType)};

  // Initialization of data
  if (memtype_check) {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
        &hData, &B_h, nullptr, NUM_W, true);
  } else {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
        &hData, &B_h, nullptr, NUM_W);
  }
  HipTest::setDefaultData<TestType>(NUM_W, hData, B_h, nullptr);
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();
  HIP_CHECK(hipMallocArray(&A_d, &desc, NUM_W, NUM_H, hipArrayDefault));
  HIP_CHECK(hipMemcpy2DToArray(A_d, 0, 0, hData, width,
                              width, NUM_H, hipMemcpyHostToDevice));

  // Performing API call
  HIP_CHECK(hipMemcpyHtoA(A_d, 0, B_h, copy_bytes*sizeof(TestType)));
  HIP_CHECK(hipMemcpy2DFromArray(hData, sizeof(TestType)*NUM_W, A_d,
           0, 0, sizeof(TestType)*NUM_W, 1, hipMemcpyDeviceToHost));


  // Validating the result
  REQUIRE(HipTest::checkArray(B_h, hData, copy_bytes, NUM_H) == true);

  // DeAllocating the memory
  HIP_CHECK(hipFreeArray(A_d));
  if (memtype_check) {
    REQUIRE(HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, hData, B_h,
                                           nullptr, true) == true);
  } else {
    REQUIRE(HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, hData, B_h,
                                           nullptr, false) == true);
  }
}


/*
This testcase performs the peer device context scenario
of hipMemcpyHtoA API
Memory is allocated in GPU-0 and the API is triggered from GPU-1
Input: "B_h" which is initialized with 1.6
Output: "A_d" output of hipMemcpyHtoA is copied to "hData" host variable
        validated the result with "B_h"
*/
#if HT_AMD
TEMPLATE_TEST_CASE("Unit_hipMemcpyHtoA_multiDevice-PeerDeviceContext",
                   "[hipMemcpyHtoA]",
                   char, int, float) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    int peerAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&peerAccess, 1, 0));
    if (!peerAccess) {
      SUCCEED("Skipped the test as there is no peer access");
    } else {
      HIP_CHECK(hipSetDevice(0));
      hipArray *A_d;
      TestType *hData{nullptr}, *B_h{nullptr};
      size_t width{NUM_W * sizeof(TestType)};

      // Initialization of data
      HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
          &hData, &B_h, nullptr, NUM_W);
      HipTest::setDefaultData<TestType>(NUM_W, hData, B_h, nullptr);
      hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();
      HIP_CHECK(hipMallocArray(&A_d, &desc, NUM_W, NUM_H, hipArrayDefault));
      HIP_CHECK(hipMemcpy2DToArray(A_d, 0, 0, hData, width,
            width, NUM_H, hipMemcpyHostToDevice));

      // Changing the device context
      HIP_CHECK(hipSetDevice(1));

      // Performing API call
      HIP_CHECK(hipMemcpyHtoA(A_d, 0, B_h, copy_bytes*sizeof(TestType)));
      HIP_CHECK(hipMemcpy2DFromArray(hData, sizeof(TestType)*NUM_W, A_d,
            0, 0, sizeof(TestType)*NUM_W, 1,
            hipMemcpyDeviceToHost));

      // Validating the result
      REQUIRE(HipTest::checkArray(B_h, hData, copy_bytes, NUM_H) == true);

      // DeAllocating the memory
      HIP_CHECK(hipFreeArray(A_d));
      REQUIRE(HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
                                            hData, B_h,
                                            nullptr, false) == true);
    }
  } else {
    SUCCEED("skipping the testcases as numDevices < 2");
  }
}
#endif


/*
This testcase verifies the negative scenarios of hipMemcpyHtoA API
*/
TEST_CASE("Unit_hipMemcpyHtoA_Negative") {
  HIP_CHECK(hipSetDevice(0));
  hipArray *A_d;
  float *hData{nullptr}, *B_h{nullptr};
  size_t width{NUM_W * sizeof(float)};

  // Initialization of data
  HipTest::initArrays<float>(nullptr, nullptr, nullptr,
                             &hData, &B_h, nullptr, NUM_W);
  HipTest::setDefaultData<float>(NUM_W, hData, B_h, nullptr);
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
  HIP_CHECK(hipMallocArray(&A_d, &desc, NUM_W, NUM_H, hipArrayDefault));
  HIP_CHECK(hipMemcpy2DToArray(A_d, 0, 0, hData, width,
                              width, NUM_H, hipMemcpyHostToDevice));

  SECTION("Source pointer is nullptr") {
    REQUIRE(hipMemcpyHtoA(A_d, 0, nullptr, copy_bytes*sizeof(float))
                          != hipSuccess);
  }

  SECTION("Source offset is more than allocated size") {
    REQUIRE(hipMemcpyHtoA(A_d, 100, B_h, copy_bytes*sizeof(float))
                          != hipSuccess);
  }

  SECTION("ByteCount is greater than allocated size") {
    REQUIRE(hipMemcpyHtoA(A_d, 0, B_h, 12*sizeof(float)) != hipSuccess);
  }

  // DeAllocating the memory
  HIP_CHECK(hipFreeArray(A_d));
  REQUIRE(HipTest::freeArrays<float>(nullptr, nullptr, nullptr, hData, B_h,
                                      nullptr, false) == true);
}

/*
This testcase verifies the size 0 check of hipMemcpyHtoA API
This is excluded for AMD as we have a bug already raised
SWDEV-274683
*/
#if HT_NVIDIA
TEST_CASE("Unit_hipMemcpyHtoA_SizeCheck") {
  HIP_CHECK(hipSetDevice(0));
  hipArray *A_d;
  float *hData{nullptr}, *B_h{nullptr}, *def_data{nullptr};
  size_t width{NUM_W * sizeof(float)};

  // Initialization of data
  HipTest::initArrays<float>(nullptr, nullptr, nullptr,
      nullptr, &def_data, nullptr, NUM_W);
  HipTest::initArrays<float>(nullptr, nullptr, nullptr,
      &hData, &B_h, nullptr, NUM_W);
  HipTest::setDefaultData<float>(NUM_W, hData, B_h, nullptr);
  HipTest::setDefaultData<float>(NUM_W, nullptr, def_data, nullptr);
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
  HIP_CHECK(hipMallocArray(&A_d, &desc, NUM_W, NUM_H, hipArrayDefault));
  HIP_CHECK(hipMemcpy2DToArray(A_d, 0, 0, hData, width,
        width, NUM_H, hipMemcpyHostToDevice));

  SECTION("Passing 0 to copy bytes") {
    REQUIRE(hipMemcpyHtoA(A_d, 0, B_h, 0) == hipSuccess);
    HIP_CHECK(hipMemcpy2DFromArray(def_data, sizeof(float)*NUM_W, A_d,
                                  0, 0, sizeof(float)*NUM_W, 1,
                                  hipMemcpyDeviceToHost));

    REQUIRE(HipTest::checkArray(hData, def_data, NUM_W, NUM_H) == true);
  }

  SECTION(" Source Array is nullptr") {
    REQUIRE(hipMemcpyHtoA(nullptr, 0, B_h, copy_bytes*sizeof(float))
                          != hipSuccess);
  }

  // DeAllocating the memory
  HIP_CHECK(hipFreeArray(A_d));
  REQUIRE(HipTest::freeArrays<float>(nullptr, nullptr, nullptr, hData, B_h,
                                      def_data, false) == true);
}
#endif
