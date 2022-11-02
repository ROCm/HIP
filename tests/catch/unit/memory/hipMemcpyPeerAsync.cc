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
This testfile verifies the following scenarios of hipMemcpyPeerAsync API
1. Negative Scenarios
2. Memory on one GPU and stream created on another GPU
3. Basic scenario of hipMemcpyPeerAsync API
*/


#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <iostream>

/*This testcase verifies the negative scenarios of hipmemcpypeerAsync
*/
TEST_CASE("Unit_hipMemcpyPeerAsync_Negative") {
  constexpr auto numElements{10};
  constexpr auto copy_bytes{numElements*sizeof(int)};
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    int canAccessPeer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      // Initialization of variables
      int *A_d{nullptr}, *B_d{nullptr};
      int *A_h{nullptr}, *B_h{nullptr};
      hipStream_t stream;
      HIP_CHECK(hipSetDevice(0));
      HIP_CHECK(hipStreamCreate(&stream));
      HipTest::initArrays<int>(&A_d, nullptr, nullptr,
          &A_h, &B_h, nullptr, numElements*sizeof(int));
      HipTest::setDefaultData<int>(numElements, A_h, B_h, nullptr);
      HIP_CHECK(hipSetDevice(1));
      HipTest::initArrays<int>(nullptr, &B_d, nullptr,
          nullptr, nullptr, nullptr, numElements*sizeof(int));
      HIP_CHECK(hipMemcpy(B_d, B_h, numElements*sizeof(int),
                          hipMemcpyHostToDevice));

      SECTION("Nullptr to Destination Pointer") {
        REQUIRE(hipMemcpyPeerAsync(nullptr, 1, A_d, 0, copy_bytes,
              stream) != hipSuccess);
      }

      SECTION("Nullptr to Source Pointer") {
        REQUIRE(hipMemcpyPeerAsync(B_d, 1, nullptr, 0, copy_bytes,
              stream) != hipSuccess);
      }

      SECTION("Pass NumElements as 0") {
        HIP_CHECK(hipMemcpy(A_d, A_h, numElements*sizeof(int),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpyPeerAsync(B_d, 1, A_d, 0, 0,
                                     stream));
        HIP_CHECK(hipMemcpy(A_h, B_d, numElements*sizeof(int),
                            hipMemcpyDeviceToHost));
        HipTest::checkTest<int>(A_h, B_h, numElements);
      }

      SECTION("Passing more than allocated size") {
        REQUIRE(hipMemcpyPeerAsync(B_d, 1, A_d, 0,
              100*sizeof(int), stream) != hipSuccess);
      }

      SECTION("Passing invalid Destination device ID") {
        REQUIRE(hipMemcpyPeerAsync(B_d, numDevices, A_d, 0, copy_bytes,
              stream) != hipSuccess);
      }

      SECTION("Passing invalid Source device ID") {
        REQUIRE(hipMemcpyPeerAsync(B_d, 0, A_d, numDevices, copy_bytes,
              stream) != hipSuccess);
      }

      HipTest::freeArrays<int>(A_d, B_d, nullptr, A_h, B_h, nullptr, false);
      HIP_CHECK(hipStreamDestroy(stream));
    } else {
      SUCCEED("Machine Does not have P2P capability");
    }
  } else {
    SUCCEED("Number of devices are < 2");
  }
}
/*
 * This test case verifies the basic scenario of hipMemcpyPeer API
 * Initializes data in GPU-0
 * Launches the kernel and performs addition in GPU-0
 * Copies the data from GPU-0 to GPU-1 using hipMemcpyPeerAsync API
 * Then performs the addition and validates the sum
 */

TEST_CASE("Unit_hipMemcpyPeerAsync_Basic") {
  constexpr auto numElements{10};
  constexpr auto copy_bytes{numElements*sizeof(int)};

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    int canAccessPeer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      // Initialization of Variables on GPU-0
      int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
      int *X_d{nullptr}, *Y_d{nullptr}, *Z_d{nullptr};
      int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
      hipStream_t stream;
      HIP_CHECK(hipSetDevice(0));
      HipTest::initArrays<int>(&A_d, &B_d, &C_d,
          &A_h, &B_h, &C_h, numElements*sizeof(int));
      HipTest::setDefaultData<int>(numElements, A_h, B_h, nullptr);
      HIP_CHECK(hipMemcpy(A_d, A_h, numElements*sizeof(int),
                          hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(B_d, B_h, numElements*sizeof(int),
                          hipMemcpyHostToDevice));
      HIP_CHECK(hipStreamCreate(&stream));

      // Initialization of Variables in GPU-1
      HIP_CHECK(hipSetDevice(1));
      HipTest::initArrays<int>(&X_d, &Y_d, &Z_d, nullptr,
          nullptr, nullptr, numElements*sizeof(int));

      // Launching kernel and performing vector addition in GPU-0
      HIP_CHECK(hipSetDevice(0));
      hipLaunchKernelGGL(HipTest::vectorADD, dim3(1), dim3(1),
          0, 0, static_cast<const int*>(A_d),
          static_cast<const int*>(B_d), C_d, numElements*sizeof(int));
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipMemcpy(C_h, C_d, numElements*sizeof(int),
               hipMemcpyDeviceToHost));
      HipTest::checkVectorADD<int>(A_h, B_h, C_h, numElements);

      // Copying data from GPU-0 to GPU-1 and performing vector addition
      HIP_CHECK(hipSetDevice(1));
      SECTION("Calling hipMemcpyPerAsync() using user defined stream obj") {
        HIP_CHECK(hipMemcpyPeerAsync(X_d, 1, A_d, 0, copy_bytes,
                                     stream));
        HIP_CHECK(hipMemcpyPeerAsync(Y_d, 1, B_d, 0, copy_bytes,
                                     stream));
        HIP_CHECK(hipStreamSynchronize(stream));
      }
      SECTION("Calling hipMemcpyPerAsync() using hipStreamPerThread") {
        HIP_CHECK(hipMemcpyPeerAsync(X_d, 1, A_d, 0, copy_bytes,
                                     hipStreamPerThread));
        HIP_CHECK(hipMemcpyPeerAsync(Y_d, 1, B_d, 0, copy_bytes,
                                     hipStreamPerThread));
        HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
      }
      hipLaunchKernelGGL(HipTest::vectorADD, dim3(1), dim3(1),
          0, 0, static_cast<const int*>(X_d),
          static_cast<const int*>(Y_d), Z_d, numElements*sizeof(int));
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipMemcpy(C_h, Z_d, numElements*sizeof(int),
               hipMemcpyDeviceToHost));
      HipTest::checkVectorADD<int>(A_h, B_h, C_h, numElements);

      // Cleaning the Memory
      HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
      HipTest::freeArrays<int>(X_d, Y_d, Z_d, nullptr, nullptr, nullptr, false);
      HIP_CHECK(hipStreamDestroy(stream));
    } else {
      SUCCEED("Machine Does not have P2P capability");
    }
  } else {
    SUCCEED("Number of devices are < 2");
  }
}

/*
 * This test case verifies the following functionality where
   Memory is allocated in One GPU and
   stream created on another GPU
 * Initializes all the data in GPU-0
 * Creating stream in GPU-1
 * Launches the kernel and performs addition in GPU-0
 * Copies the data from GPU-0 to GPU-1 using hipMemcpyPeerAsync API
 * where stream is created in GPU-1
 * Then performs the addition and validates the sum
 */
TEST_CASE("Unit_hipMemcpyPeerAsync_StreamOnDiffDevice") {
  constexpr auto numElements{10};
  constexpr auto copy_bytes{numElements*sizeof(int)};
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    int canAccessPeer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
      int *X_d{nullptr}, *Y_d{nullptr}, *Z_d{nullptr};
      int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
      hipStream_t stream;
      HIP_CHECK(hipSetDevice(0));

      // Initialization of all variables in GPU-0
      HipTest::initArrays<int>(&A_d, &B_d, &C_d,
          &A_h, &B_h, &C_h, numElements*sizeof(int));
      HIP_CHECK(hipMemcpy(A_d, A_h, numElements*sizeof(int),
               hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(B_d, B_h, numElements*sizeof(int),
            hipMemcpyHostToDevice));
      HipTest::initArrays<int>(&X_d, &Y_d, &Z_d, nullptr,
          nullptr, nullptr, numElements*sizeof(int));

      // Stream created in GPU-1
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipStreamCreate(&stream));

      // Performing vector addition and validate the data
      HIP_CHECK(hipSetDevice(0));
      hipLaunchKernelGGL(HipTest::vectorADD, dim3(1), dim3(1),
          0, 0, static_cast<const int*>(A_d),
          static_cast<const int*>(B_d), C_d, numElements*sizeof(int));
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipMemcpy(C_h, C_d, numElements*sizeof(int),
               hipMemcpyDeviceToHost));
      HipTest::checkVectorADD<int>(A_h, B_h, C_h, numElements);

      // Copying the data from GPU-0 to GPU-1 where stream is from diff device
      HIP_CHECK(hipMemcpyPeerAsync(X_d, 1, A_d, 0, copy_bytes,
               stream));
      HIP_CHECK(hipMemcpyPeerAsync(Y_d, 1, B_d, 0, copy_bytes,
               stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      hipLaunchKernelGGL(HipTest::vectorADD, dim3(1), dim3(1),
          0, 0, static_cast<const int*>(X_d),
          static_cast<const int*>(Y_d), Z_d, numElements*sizeof(int));
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipMemcpy(C_h, Z_d, numElements*sizeof(int),
               hipMemcpyDeviceToHost));

      // Cleaning the data
      HipTest::checkVectorADD<int>(A_h, B_h, C_h, numElements);
      HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
      HipTest::freeArrays<int>(X_d, Y_d, Z_d, nullptr, nullptr, nullptr, false);
      HIP_CHECK(hipStreamDestroy(stream));
    } else {
      SUCCEED("Machine Does not have P2P capability");
    }
  } else {
    SUCCEED("Number of devices are < 2");
  }
}

