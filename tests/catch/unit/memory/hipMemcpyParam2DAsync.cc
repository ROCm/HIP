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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
This testfile verifies the following scenarios of hipMemcpyParam2DAsync API
1. Negative Scenarios
2. Extent Validation Scenarios
3. D2D copy for different datatypes
4. H2D and D2H copy for different datatypes
5. Device context change scenario where memory allocated in one GPU
   stream created in another GPU
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

static constexpr size_t NUM_W{10};
static constexpr size_t NUM_H{10};
/*
 * This testcase verifies D2D functionality of hipMemcpyParam2DAsync API
 * Where Memory is allocated in GPU-0 and stream is created in GPU-1
 *
 * Input: Intializing "A_d" device variable with "C_h" host variable
 * Output: "A_d" device variable to "E_d" device variable
 *
 * Validating the result by copying "E_d" to "A_h" and checking
 * it with the initalized data "C_h".
 *
 */
TEMPLATE_TEST_CASE("Unit_hipMemcpyParam2DAsync_multiDevice-StreamOnDiffDevice",
                   "[hipMemcpyParam2DAsync]", char, float, int,
                   double, long double) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    // Allocating and Initializing the data
    HIP_CHECK(hipSetDevice(0));
    TestType* A_h{nullptr}, *C_h{nullptr}, *A_d{nullptr};
    size_t pitch_A;
    size_t width{NUM_W * sizeof(TestType)};
    HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                            &pitch_A, width, NUM_H));
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                  &A_h, nullptr, &C_h,
                                  width*NUM_H, false);
    HipTest::setDefaultData<TestType>(NUM_W*NUM_H, A_h, nullptr, C_h);
    int peerAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&peerAccess, 1, 0));
    if (!peerAccess) {
      SUCCEED("Skipped the test as there is no peer access");
    } else {
      TestType *E_d{nullptr};
      size_t pitch_E;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&E_d),
                              &pitch_E, width, NUM_H));

      // Initalizing A_d with C_h
      HIP_CHECK(hipSetDevice(1));
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(hipMemcpy2DAsync(A_d, pitch_A, C_h, width,
                           NUM_W*sizeof(TestType), NUM_H,
                           hipMemcpyHostToDevice, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      // Device to Device
      hip_Memcpy2D desc = {};
#ifdef __HIP_PLATFORM_NVCC__
      desc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
#else
      desc.srcMemoryType = hipMemoryTypeDevice;
#endif
      desc.srcHost = A_d;
      desc.srcDevice = hipDeviceptr_t(A_d);
      desc.srcPitch = pitch_A;
#ifdef __HIP_PLATFORM_NVCC__
      desc.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
      desc.dstMemoryType = hipMemoryTypeDevice;
#endif
      desc.dstHost = E_d;
      desc.dstDevice = hipDeviceptr_t(E_d);
      desc.dstPitch = pitch_E;
      desc.WidthInBytes = NUM_W*sizeof(TestType);
      desc.Height = NUM_H;
      REQUIRE(hipMemcpyParam2DAsync(&desc, stream) == hipSuccess);
      HIP_CHECK(hipStreamSynchronize(stream));

      // Copying the result E_d to A_h host variable
      HIP_CHECK(hipMemcpy2D(A_h, width, E_d, pitch_E,
            NUM_W*sizeof(TestType), NUM_H,
            hipMemcpyDeviceToHost));
      HIP_CHECK(hipDeviceSynchronize());
      // Validating the result
      REQUIRE(HipTest::checkArray<TestType>(A_h, C_h, NUM_W, NUM_H) == true);

      // DeAllocating the memory
      HIP_CHECK(hipFree(E_d));
      HIP_CHECK(hipFree(A_d));
      HIP_CHECK(hipStreamDestroy(stream));
      HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
          A_h, nullptr, C_h, false);
    }
  } else {
    SUCCEED("skipping the testcases as numDevices < 2");
  }
}

/*
 * This testcase verifies D2D functionality of hipMemcpyParam2DAsync API
 * Input: Intializing "A_d" device variable with "C_h" host variable
 * Output: "A_d" device variable to "E_d" device variable
 *
 * Validating the result by copying "E_d" to "A_h" and checking
 * it with the initalized data "C_h".
 *
 */
TEMPLATE_TEST_CASE("Unit_hipMemcpyParam2DAsync_multiDevice-D2D",
                   "[hipMemcpyParam2DAsync]", char,
                   int, float, double, long double) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    // Allocating and Initializing the data
    HIP_CHECK(hipSetDevice(0));
    TestType* A_h{nullptr}, *C_h{nullptr}, *A_d{nullptr};
    size_t pitch_A;
    size_t width{NUM_W * sizeof(TestType)};
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                            &pitch_A, width, NUM_H));
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                  &A_h, nullptr, &C_h,
                                  width*NUM_H, false);
    HipTest::setDefaultData<TestType>(NUM_W*NUM_H, A_h, nullptr, C_h);

    int peerAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&peerAccess, 1, 0));
    if (!peerAccess) {
      SUCCEED("Skipped the test as there is no peer access");
    } else {
      HIP_CHECK(hipSetDevice(1));
      TestType *E_d;
      size_t pitch_E;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&E_d),
            &pitch_E, width, NUM_H));

      // Initializing A_d  with C_h
      HIP_CHECK(hipMemcpy2D(A_d, pitch_A, C_h, width,
            NUM_W*sizeof(TestType), NUM_H, hipMemcpyHostToDevice));

      // Device to Device
      hip_Memcpy2D desc = {};
#ifdef __HIP_PLATFORM_NVCC__
      desc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
#else
      desc.srcMemoryType = hipMemoryTypeDevice;
#endif
      desc.srcHost = A_d;
      desc.srcDevice = hipDeviceptr_t(A_d);
      desc.srcPitch = pitch_A;
#ifdef __HIP_PLATFORM_NVCC__
      desc.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
      desc.dstMemoryType = hipMemoryTypeDevice;
#endif
      desc.dstHost = E_d;
      desc.dstDevice = hipDeviceptr_t(E_d);
      desc.dstPitch = pitch_E;
      desc.WidthInBytes = NUM_W*sizeof(TestType);
      desc.Height = NUM_H;
      REQUIRE(hipMemcpyParam2DAsync(&desc, stream) == hipSuccess);
      HIP_CHECK(hipStreamSynchronize(stream));

      // Copying the result E_d to A_h host variable
      HIP_CHECK(hipMemcpy2D(A_h, width, E_d, pitch_E,
            NUM_W*sizeof(TestType), NUM_H, hipMemcpyDeviceToHost));

      // Validating the result
      REQUIRE(HipTest::checkArray<TestType>(A_h, C_h, NUM_W, NUM_H) == true);

      // DeAllocating the memory
      HIP_CHECK(hipFree(A_d));
      HIP_CHECK(hipStreamDestroy(stream));
      HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
          A_h, nullptr, C_h, false);
    }
  } else {
    SUCCEED("skipping the testcases as numDevices < 2");
  }
}

/*
 * This testcase verifies H2D & D2H functionality of hipMemcpyParam2DAsync API
 * H2D case:
 * Input: "C_h" host variable initialized with default data
 * Output: "A_d" device variable
 *
 * D2H case:
 * Input: "A_d" device variable from the previous output
 * OutPut: "A_h" variable
 *
 * Validating the result by comparing "A_h" to "C_h"
 */
TEMPLATE_TEST_CASE("Unit_hipMemcpyParam2DAsync_multiDevice-H2D-D2H",
                   "[hipMemcpyParam2DAsync]", char,
                   int, float, double, long double) {
  // 1 refers to pinned host memory and 0 refers
  // to unpinned memory
  auto memory_type = GENERATE(0, 1);
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    // Allocating and Initializing the data
    HIP_CHECK(hipSetDevice(0));
    TestType* A_h{nullptr}, *C_h{nullptr},
      *A_d{nullptr};
    size_t pitch_A;
    size_t width{NUM_W * sizeof(TestType)};
    hipStream_t stream;

    HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                            &pitch_A, width, NUM_H));

    // Based on memory type (pinned/unpinned) allocating memory
    if (memory_type) {
      HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                    &A_h, nullptr, &C_h,
                                    width*NUM_H, true);
    } else {
      HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                    &A_h, nullptr, &C_h,
                                    width*NUM_H, false);
    }
    HipTest::setDefaultData<TestType>(NUM_W*NUM_H, A_h, nullptr, C_h);
    int peerAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&peerAccess, 1, 0));
    if (!peerAccess) {
      SUCCEED("Skipped the test as there is no peer access");
    } else {
      // Host to Device
      hip_Memcpy2D desc = {};
      HIP_CHECK(hipStreamCreate(&stream));
#ifdef __HIP_PLATFORM_NVCC__
      desc.srcMemoryType = CU_MEMORYTYPE_HOST;
#else
      desc.srcMemoryType = hipMemoryTypeHost;
#endif
      desc.srcHost = C_h;
      desc.srcDevice = hipDeviceptr_t(C_h);
      desc.srcPitch = width;
#ifdef __HIP_PLATFORM_NVCC__
      desc.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
      desc.dstMemoryType = hipMemoryTypeDevice;
#endif
      desc.dstHost = A_d;
      desc.dstDevice = hipDeviceptr_t(A_d);
      desc.dstPitch = pitch_A;
      desc.WidthInBytes = NUM_W*sizeof(TestType);
      desc.Height = NUM_H;
      REQUIRE(hipMemcpyParam2DAsync(&desc, stream) == hipSuccess);
      HIP_CHECK(hipStreamSynchronize(stream));

      // Device to Host
      memset(&desc, 0x0, sizeof(hip_Memcpy2D));
#ifdef __HIP_PLATFORM_NVCC__
      desc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
#else
      desc.srcMemoryType = hipMemoryTypeDevice;
#endif
      desc.srcHost = A_d;
      desc.srcDevice = hipDeviceptr_t(A_d);
      desc.srcPitch = pitch_A;
#ifdef __HIP_PLATFORM_NVCC__
      desc.dstMemoryType = CU_MEMORYTYPE_HOST;
#else
      desc.dstMemoryType = hipMemoryTypeHost;
#endif
      desc.dstHost = A_h;
      desc.dstDevice = hipDeviceptr_t(A_h);
      desc.dstPitch = width;
      desc.WidthInBytes = NUM_W*sizeof(TestType);
      desc.Height = NUM_H;
      REQUIRE(hipMemcpyParam2DAsync(&desc, stream) == hipSuccess);
      HIP_CHECK(hipStreamSynchronize(stream));

      // Validating the result
      REQUIRE(HipTest::checkArray<TestType>(A_h, C_h, NUM_W, NUM_H) == true);

      // DeAllocating the memory
      HIP_CHECK(hipFree(A_d));
      HIP_CHECK(hipStreamDestroy(stream));
      if (memory_type) {
        HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
                                      A_h, nullptr, C_h, true);
      } else {
        HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
                                      A_h, nullptr, C_h, false);
      }
    }
  } else {
      SUCCEED("skipping the testcases as numDevices < 2");
  }
}
/*
 * This testcase verifies the extent validation scenarios
 */
TEST_CASE("Unit_hipMemcpyParam2DAsync_ExtentValidation") {
  HIP_CHECK(hipSetDevice(0));
  char* A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr},
      * A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(char)};
  constexpr auto memsetval{100};
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocating and Initializing the data
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                          &pitch_A, width, NUM_H));
  HipTest::initArrays<char>(nullptr, nullptr, nullptr,
                            &A_h, nullptr, &C_h,
                            width*NUM_H, false);
  HipTest::initArrays<char>(nullptr, nullptr, nullptr,
                            &B_h, nullptr, nullptr,
                            width*NUM_H, false);
  HipTest::setDefaultData<char>(NUM_W*NUM_H, A_h, nullptr, C_h);
  HipTest::setDefaultData<char>(NUM_W*NUM_H, B_h, nullptr, nullptr);
  HIP_CHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));

  // Device to Host
  hip_Memcpy2D desc = {};
#ifdef __HIP_PLATFORM_NVCC__
  desc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
#else
  desc.srcMemoryType = hipMemoryTypeDevice;
#endif
  desc.srcHost = A_d;
  desc.srcDevice = hipDeviceptr_t(A_d);
  desc.srcPitch = pitch_A;
#ifdef __HIP_PLATFORM_NVCC__
  desc.dstMemoryType = CU_MEMORYTYPE_HOST;
#else
  desc.dstMemoryType = hipMemoryTypeHost;
#endif
  desc.dstHost = A_h;
  desc.dstDevice = hipDeviceptr_t(A_h);
  desc.dstPitch = width;
  desc.WidthInBytes = NUM_W;
  desc.Height = NUM_H;

  SECTION("Destination Pitch is 0") {
    desc.dstPitch = 0;
    REQUIRE(hipMemcpyParam2DAsync(&desc, stream) == hipSuccess);
  }

  SECTION("Source Pitch is 0") {
    desc.srcPitch = 0;
    REQUIRE(hipMemcpyParam2DAsync(&desc, stream) == hipSuccess);
  }

  SECTION("Height is 0") {
    desc.Height = 0;
    REQUIRE(hipMemcpyParam2DAsync(&desc, stream) == hipSuccess);
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(HipTest::checkArray<char>(A_h, B_h, NUM_W, NUM_H) == true);
  }

  SECTION("Width is 0") {
    desc.Height = 0;
    REQUIRE(hipMemcpyParam2DAsync(&desc, stream) == hipSuccess);
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(HipTest::checkArray<char>(A_h, B_h, NUM_W, NUM_H) == true);
  }

  // DeAllocating the Memory
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipStreamDestroy(stream));
  HipTest::freeArrays<char>(nullptr, nullptr, nullptr,
                                A_h, B_h, C_h, false);
}

/*
 * This testcase verifies the negative scenarios
 */
TEST_CASE("Unit_hipMemcpyParam2DAsync_Negative") {
  HIP_CHECK(hipSetDevice(0));
  float* A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr},
       * A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(float)};
  constexpr auto memsetval{100};
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocating and Initializing the data
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                          &pitch_A, width, NUM_H));
  HipTest::initArrays<float>(nullptr, nullptr, nullptr,
                                &A_h, &B_h, &C_h,
                                width*NUM_H, false);
  HipTest::setDefaultData<float>(NUM_W*NUM_H, A_h, B_h, C_h);
  HIP_CHECK(hipMemset2D(A_d, pitch_A, memsetval, NUM_W, NUM_H));

  // Device to Host
  hip_Memcpy2D desc = {};
#ifdef __HIP_PLATFORM_NVCC__
  desc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
#else
  desc.srcMemoryType = hipMemoryTypeDevice;
#endif
  desc.srcHost = A_d;
  desc.srcDevice = hipDeviceptr_t(A_d);
  desc.srcPitch = pitch_A;
#ifdef __HIP_PLATFORM_NVCC__
  desc.dstMemoryType = CU_MEMORYTYPE_HOST;
#else
  desc.dstMemoryType = hipMemoryTypeHost;
#endif
  desc.dstHost = A_h;
  desc.dstDevice = hipDeviceptr_t(A_h);
  desc.dstPitch = width;
  desc.WidthInBytes = NUM_W;
  desc.Height = NUM_H;

  SECTION("Null Pointer to Source Device Pointer") {
    desc.srcDevice = hipDeviceptr_t(nullptr);
    REQUIRE(hipMemcpyParam2DAsync(&desc, stream) != hipSuccess);
  }

  SECTION("Null Pointer to Destination Device Pointer") {
    memset(&desc, 0x0, sizeof(hip_Memcpy2D));
#ifdef __HIP_PLATFORM_NVCC__
    desc.srcMemoryType = CU_MEMORYTYPE_HOST;
#else
    desc.srcMemoryType = hipMemoryTypeHost;
#endif
    desc.srcHost = A_h;
    desc.srcDevice = hipDeviceptr_t(A_h);
    desc.srcPitch = width;
#ifdef __HIP_PLATFORM_NVCC__
    desc.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
    desc.dstMemoryType = hipMemoryTypeDevice;
#endif
    desc.dstHost = A_d;
    desc.dstDevice = hipDeviceptr_t(nullptr);
    desc.dstPitch = pitch_A;
    desc.WidthInBytes = NUM_W;
    desc.Height = NUM_H;

    REQUIRE(hipMemcpyParam2DAsync(&desc, stream) != hipSuccess);
  }

  SECTION("Null Pointer to both Src & Dst Device Pointer") {
    desc.srcDevice = hipDeviceptr_t(nullptr);
    desc.dstDevice = hipDeviceptr_t(nullptr);
    REQUIRE(hipMemcpyParam2DAsync(&desc, stream) != hipSuccess);
  }

  SECTION("Width > src/dest pitches") {
    desc.WidthInBytes = pitch_A+1;
    REQUIRE(hipMemcpyParam2DAsync(&desc, stream) != hipSuccess);
  }

  // DeAllocating the memory
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HipTest::freeArrays<float>(nullptr, nullptr, nullptr,
                                A_h, B_h, C_h, false);
}
