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

/*
Test Scenarios of hipMallocPitch API
1. Negative Scenarios
2. Basic Functionality Scenario
3. Allocate memory using hipMallocPitch API, Launch Kernel validate result.
4. Allocate Memory in small chunks and large chunks and check for possible memory leaks
5. Allocate Memory using hipMallocPitch API, Memcpy2D on the allocated variables.
6. Multithreaded scenario
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

static constexpr auto SMALLCHUNK_NUMW{4};
static constexpr auto SMALLCHUNK_NUMH{4};
static constexpr auto LARGECHUNK_NUMW{1025};
static constexpr auto LARGECHUNK_NUMH{1000};
static constexpr auto NUM_W{10};
static constexpr auto NUM_H{10};
static constexpr auto COLUMNS{8};
static constexpr auto ROWS{8};
static constexpr auto CHUNK_LOOP{100};


template<typename T>
__global__ void copy_var(T* A, T* B,
                              size_t ROWS, size_t pitch_A) {
  for (uint64_t i = 0; i< ROWS*pitch_A; i= i+pitch_A) {
    A[i] = B[i];
  }
}
template<typename T>
static bool validateResult(T* A, T* B, size_t pitch_A) {
  bool testResult = true;
  for (uint64_t i=0; i < pitch_A*ROWS; i=i+pitch_A) {
    if (A[i] != B[i]) {
      testResult = false;
      break;
    }
  }
  return testResult;
}
/*
 * This API verifies  memory allocations for small and
 * bigger chunks of data.
 * Two scenarios are verified in this API
 * 1. SmallChunk: Allocates SMALLCHUNK_NUMW in a loop and
 *    releases the memory and verifies the meminfo.
 * 2. LargeChunk: Allocates LARGECHUNK_NUMW in a loop and
 *    releases the memory and verifies the meminfo
 *
 * In both cases, the memory info before allocation and
 * after releasing the memory should be the same
 *
 */
template<typename T>
static void MemoryAllocDiffSizes(int gpu) {
  HIP_CHECK(hipSetDevice(gpu));
  std::vector<size_t> array_size;
  array_size.push_back(SMALLCHUNK_NUMH);
  array_size.push_back(LARGECHUNK_NUMH);
  for (auto &sizes : array_size) {
    T* A_d[CHUNK_LOOP];
    size_t pitch_A;
    size_t width;
    if (sizes == SMALLCHUNK_NUMH) {
      width = SMALLCHUNK_NUMW * sizeof(T);
    } else {
      width = LARGECHUNK_NUMW * sizeof(T);
    }
    size_t tot, avail, ptot, pavail;
    HIP_CHECK(hipMemGetInfo(&pavail, &ptot));
    for (int i = 0; i < CHUNK_LOOP; i++) {
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d[i]),
            &pitch_A, width, sizes));
    }
    for (int i = 0; i < CHUNK_LOOP; i++) {
      HIP_CHECK(hipFree(A_d[i]));
    }
    HIP_CHECK(hipMemGetInfo(&avail, &tot));
    if (pavail != avail) {
      HIPASSERT(false);
    }
  }
}

/*Thread Function */
static void threadFunc(int gpu) {
  MemoryAllocDiffSizes<float>(gpu);
}
/*
 * This testcase verifies the negative scenarios of hipMallocPitch API
 */
#if 0 //TODO: Review, fix and re-enable test
TEST_CASE("Unit_hipMallocPitch_Negative") {
  float* A_d;
  size_t pitch_A;
  size_t width{NUM_W * sizeof(float)};
#if HT_NVIDIA
  SECTION("NullPtr to Pitched Ptr") {
    REQUIRE(hipMallocPitch(nullptr,
            &pitch_A, width, NUM_H) != hipSuccess);
  }

  SECTION("nullptr to pitch") {
    REQUIRE(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                           nullptr, width, NUM_H) != hipSuccess);
  }
#endif
  SECTION("Width 0 in hipMallocPitch") {
    REQUIRE(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                           &pitch_A, 0, NUM_H) == hipSuccess);
  }

  SECTION("Height 0 in hipMallocPitch") {
    REQUIRE(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                           &pitch_A, width, 0) == hipSuccess);
  }

  SECTION("Max int values") {
    REQUIRE(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                           &pitch_A, std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max()) != hipSuccess);
  }
}
#endif
/*
 * This testcase verifies the basic scenario of
 * hipMallocPitch API for different datatypes
 *
 */
TEMPLATE_TEST_CASE("Unit_hipMallocPitch_Basic",
    "[hipMallocPitch]", int, unsigned int, float) {
  TestType* A_d;
  size_t pitch_A;
  size_t width{NUM_W * sizeof(TestType)};
  REQUIRE(hipMallocPitch(reinterpret_cast<void**>(&A_d),
          &pitch_A, width, NUM_H) == hipSuccess);
  HIP_CHECK(hipFree(A_d));
}

/*
 * This testcase verifies hipMallocPitch API for small
 * and big chunks of data.
 */
TEMPLATE_TEST_CASE("Unit_hipMallocPitch_SmallandBigChunks",
    "[hipMallocPitch]", int, unsigned int, float) {
  MemoryAllocDiffSizes<TestType>(0);
}

/*
 * This testcase verifies the memory allocated by hipMallocPitch API
 * by performing Memcpy2D on the allocated memory.
 */
TEMPLATE_TEST_CASE("Unit_hipMallocPitch_Memcpy2D", ""
                   , int, float, double) {
  HIP_CHECK(hipSetDevice(0));
  TestType  *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr},
            *B_d{nullptr};
  size_t pitch_A, pitch_B;
  size_t width{NUM_W * sizeof(TestType)};

  // Allocating memory
  HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                  &A_h, &B_h, &C_h, NUM_W*NUM_H, false);
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                          &pitch_A, width, NUM_H));
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&B_d),
                          &pitch_B, width, NUM_H));

  // Initialize the data
  HipTest::setDefaultData<TestType>(NUM_W*NUM_H, A_h, B_h, C_h);

  // Host to Device
  HIP_CHECK(hipMemcpy2D(A_d, pitch_A, A_h, COLUMNS*sizeof(TestType),
                        COLUMNS*sizeof(TestType), ROWS, hipMemcpyHostToDevice));

  // Performs D2D on same GPU device
  HIP_CHECK(hipMemcpy2D(B_d, pitch_B, A_d,
                        pitch_A, COLUMNS*sizeof(TestType),
                        ROWS, hipMemcpyDeviceToDevice));

  // hipMemcpy2D Device to Host
  HIP_CHECK(hipMemcpy2D(B_h, COLUMNS*sizeof(TestType), B_d, pitch_B,
                        COLUMNS*sizeof(TestType), ROWS,
                        hipMemcpyDeviceToHost));

  // Validating the result
  REQUIRE(HipTest::checkArray<TestType>(A_h, B_h, COLUMNS, ROWS) == true);


  // DeAllocating the memory
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
                                A_h, B_h, C_h, false);
}



/*
This testcase verifies the hipMallocPitch API in multithreaded
scenario by launching threads in parallel on multiple GPUs
and verifies the hipMallocPitch API with small and big chunks data
*/

TEST_CASE("Unit_hipMallocPitch_MultiThread", "") {
  std::vector<std::thread> threadlist;
  int devCnt = 0;

  devCnt = HipTest::getDeviceCount();

  size_t tot, avail, ptot, pavail;
  HIP_CHECK(hipMemGetInfo(&pavail, &ptot));
  for (int i = 0; i < devCnt; i++) {
    threadlist.push_back(std::thread(threadFunc, i));
  }

  for (auto &t : threadlist) {
    t.join();
  }
  HIP_CHECK(hipMemGetInfo(&avail, &tot));

  if (pavail != avail) {
    WARN("Memory leak of hipMallocPitch API in multithreaded scenario");
    REQUIRE(false);
  }
}

/*
 * This testcase verifies hipMallocPitch API by
 *  1. Allocating Memory using hipMallocPitch API
 *  2. Launching the kernel and copying the data from the allocated kernel
 *     variable to another kernel variable.
 *  3. Validating the result
 */
TEMPLATE_TEST_CASE("Unit_hipMallocPitch_KernelLaunch", ""
                   , int, float, double) {
  HIP_CHECK(hipSetDevice(0));
  TestType  *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr},
            *B_d{nullptr};
  size_t pitch_A, pitch_B;
  size_t width{NUM_W * sizeof(TestType)};

  // Allocating memory
  HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                  &A_h, &B_h, &C_h, NUM_W*NUM_H, false);
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                          &pitch_A, width, NUM_H));
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&B_d),
                          &pitch_B, width, NUM_H));

  // Host to Device
  HIP_CHECK(hipMemcpy2D(A_d, pitch_A, A_h, COLUMNS*sizeof(TestType),
                        COLUMNS*sizeof(TestType), ROWS, hipMemcpyHostToDevice));


  hipLaunchKernelGGL(copy_var<TestType>, dim3(1), dim3(1),
          0, 0, static_cast<TestType*>(A_d),
          static_cast<TestType*>(B_d), ROWS, pitch_A);


  // hipMemcpy2D Device to Host
  HIP_CHECK(hipMemcpy2D(B_h, COLUMNS*sizeof(TestType), B_d, pitch_B,
                        COLUMNS*sizeof(TestType), ROWS,
                        hipMemcpyDeviceToHost));

  // Validating the result
  validateResult(A_h, B_h, pitch_A);

  // DeAllocating the memory
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
                                A_h, B_h, C_h, false);
}


