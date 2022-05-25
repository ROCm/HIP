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

/**
Testcase Scenarios :

 1) Test hipMalloc() api passing zero size and confirming *ptr returning
 nullptr. Also pass nullptr to hipFree() api.

 2) Pass maximum value of size_t for hipMalloc() api and make sure appropriate
 error is returned.

 3) Check for hipMalloc() error code, passing invalid/null pointer.

 4) Regress hipMalloc()/hipFree() in loop for bigger chunk of allocation
 with adequate number of iterations and later test for kernel execution on
 default gpu.

 5) Regress hipMalloc()/hipFree() in loop while allocating smaller chunks
 keeping maximum number of iterations and then run kernel code on default
 gpu, perfom data validation.

 6) Check hipMalloc() api adaptability when app creates small chunks of memory
 continuously, stores it for later use and then frees it at later point
 of time.

 7) Multithread Scenario : Exercise hipMalloc() api parellely on all gpus from
 multiple threads and regress the api.

 8) Validate memory usage with hipMemGetInfo() while regressing hipMalloc()
 api. Check for any possible memory leaks.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#include <atomic>
#include <limits>
#include <vector>


/* Buffer size for bigger chunks in alloc/free cycles */
static constexpr auto BuffSizeBC = 5 * 1024 * 1024;

/* Buffer size for smaller chunks in alloc/free cycles */
static constexpr auto BuffSizeSC = 16;

/* You may change it for individual test.
 * But default 100 is for quick return in Jenkin Build */
static constexpr auto NumDiv = 100;

/* Max alloc/free iterations for smaller chunks */
static constexpr auto MaxAllocFree_SmallChunks = (5000000 / NumDiv);

/* Max alloc/free iterations for bigger chunks */
static constexpr auto MaxAllocFree_BigChunks = 10000;

/* Max alloc and pool iterations */
static constexpr auto MaxAllocPoolIter = (2000000 / NumDiv);

/* Test status shared across threads */
static std::atomic<bool> g_thTestPassed{true};


/**
 * Validates data consistency on supplied gpu
 */
static bool validateMemoryOnGPU(int gpu, bool concurOnOneGPU = false) {
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t prevAvl, prevTot, curAvl, curTot;
  bool TestPassed = true;
  constexpr auto N = 4 * 1024 * 1024;
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  size_t Nbytes = N * sizeof(int);

  HIP_CHECK(hipSetDevice(gpu));
  HIP_CHECK(hipMemGetInfo(&prevAvl, &prevTot));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                     static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d, N);

  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  if (!HipTest::checkVectorADD(A_h, B_h, C_h, N)) {
    UNSCOPED_INFO("Validation PASSED for gpu " << gpu);
  } else {
    UNSCOPED_INFO("Validation FAILED for gpu " << gpu);
    TestPassed = false;
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipMemGetInfo(&curAvl, &curTot));

  if (!concurOnOneGPU && (prevAvl != curAvl || prevTot != curTot)) {
    // In concurrent calls on one GPU, we cannot verify leaking in this way
    UNSCOPED_INFO("validateMemoryOnGPU : Memory allocation mismatch observed."
                  << "Possible memory leak.");
    TestPassed = false;
  }

  return TestPassed;
}


/**
 * Regress memory allocation and free in loop
 */
static bool regressAllocInLoop(int gpu) {
  bool TestPassed = true;
  size_t tot, avail, ptot, pavail, numBytes;
  int i = 0;
  int* ptr;

  HIP_CHECK(hipSetDevice(gpu));
  numBytes = BuffSizeBC;

  // Exercise allocation in loop with bigger chunks
  for (i = 0; i < MaxAllocFree_BigChunks; i++) {
    HIP_CHECK(hipMemGetInfo(&pavail, &ptot));
    HIP_CHECK(hipMalloc(&ptr, numBytes));
    HIP_CHECK(hipMemGetInfo(&avail, &tot));
    HIP_CHECK(hipFree(ptr));

    if (pavail - avail < numBytes) {  // We expect pavail-avail >= numBytes
      UNSCOPED_INFO("LoopAllocation " << i << " : Memory allocation of " << numBytes
                                      << " not matching with hipMemGetInfo - FAIL."
                                      << "pavail=" << pavail << ", ptot=" << ptot
                                      << ", avail=" << avail << ", tot=" << tot
                                      << ", pavail-avail=" << pavail - avail);
      TestPassed = false;
      break;
    }
  }

  // Exercise allocation in loop with smaller chunks and maximum iters
  HIP_CHECK(hipMemGetInfo(&pavail, &ptot));
  numBytes = BuffSizeSC;

  for (i = 0; i < MaxAllocFree_SmallChunks; i++) {
    HIP_CHECK(hipMalloc(&ptr, numBytes));

    HIP_CHECK(hipFree(ptr));
  }

  HIP_CHECK(hipMemGetInfo(&avail, &tot));

  if ((pavail != avail) || (ptot != tot)) {
    UNSCOPED_INFO("LoopAllocation : Memory allocation mismatch observed."
                  << "Possible memory leak.");
    TestPassed &= false;
  }

  return TestPassed;
}

/**
 * Validates data consistency on supplied gpu
 * In Multithreaded Environment
 */
static bool validateMemoryOnGpuMThread(int gpu, bool concurOnOneGPU = false) {
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t prevAvl, prevTot, curAvl, curTot;
  bool TestPassed = true;
  constexpr auto N = 4 * 1024 * 1024;
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  size_t Nbytes = N * sizeof(int);
  HIPCHECK(hipSetDevice(gpu));
  HIPCHECK(hipMemGetInfo(&prevAvl, &prevTot));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                     static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d, N);

  HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  if (!HipTest::checkVectorADD(A_h, B_h, C_h, N)) {
    UNSCOPED_INFO("Validation PASSED for gpu " << gpu);
  } else {
    UNSCOPED_INFO("Validation FAILED for gpu " << gpu);
    TestPassed = false;
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIPCHECK(hipMemGetInfo(&curAvl, &curTot));

  if (!concurOnOneGPU && (prevAvl != curAvl || prevTot != curTot)) {
    // In concurrent calls on one GPU, we cannot verify leaking in this way
    UNSCOPED_INFO(
        "validateMemoryOnGpuMThread : Memory allocation mismatch observed."
        "Possible memory leak.");
    TestPassed = false;
  }

  return TestPassed;
}

/**
 * Regress memory allocation and free in loop
 * In Multithreaded Environment
 */
static bool regressAllocInLoopMthread(int gpu) {
  bool TestPassed = true;
  size_t tot, avail, ptot, pavail, numBytes;
  int i = 0;
  int* ptr;

  HIPCHECK(hipSetDevice(gpu));
  numBytes = BuffSizeBC;

  // Exercise allocation in loop with bigger chunks
  for (i = 0; i < MaxAllocFree_BigChunks; i++) {
    HIPCHECK(hipMemGetInfo(&pavail, &ptot));
    HIPCHECK(hipMalloc(&ptr, numBytes));
    HIPCHECK(hipMemGetInfo(&avail, &tot));
    HIPCHECK(hipFree(ptr));

    if (pavail - avail < numBytes) {  // We expect pavail-avail >= numBytes
      UNSCOPED_INFO("LoopAllocation " << i << " : Memory allocation of " << numBytes
                                      << " not matching with hipMemGetInfo - FAIL."
                                      << "pavail=" << pavail << ", ptot=" << ptot
                                      << ", avail=" << avail << ", tot=" << tot
                                      << ", pavail-avail=" << pavail - avail);
      TestPassed = false;
      break;
    }
  }

  // Exercise allocation in loop with smaller chunks and maximum iters
  HIPCHECK(hipMemGetInfo(&pavail, &ptot));
  numBytes = BuffSizeSC;

  for (i = 0; i < MaxAllocFree_SmallChunks; i++) {
    HIPCHECK(hipMalloc(&ptr, numBytes));

    HIPCHECK(hipFree(ptr));
  }

  HIPCHECK(hipMemGetInfo(&avail, &tot));

  if ((pavail != avail) || (ptot != tot)) {
    UNSCOPED_INFO("LoopAllocation : Memory allocation mismatch observed."
                  << "Possible memory leak.");
    TestPassed &= false;
  }

  return TestPassed;
}

/*
 * Thread func to regress alloc and check data consistency
 */
static void threadFunc(int gpu) {
  g_thTestPassed = regressAllocInLoopMthread(gpu) && validateMemoryOnGpuMThread(gpu);

  UNSCOPED_INFO("thread execution status on gpu" << gpu << ":" << g_thTestPassed.load());
}


/* Performs Argument Validation of api */
TEST_CASE("Unit_hipMalloc_ArgumentValidation") {
  int* ptr{nullptr};

  SECTION("hipMalloc() when size(0)") {
    HIP_CHECK(hipMalloc(&ptr, 0));
    // ptr expected to be reset to null ptr
    REQUIRE(ptr == nullptr);
  }

  SECTION("hipFree() when freeing nullptr") {
    HIP_CHECK(hipFree(ptr));
  }

  SECTION("hipMalloc() with invalid argument") {
    HIP_CHECK_ERROR(hipMalloc(nullptr, 100), hipErrorInvalidValue);
  }

  SECTION("hipMalloc() with max size_t") {
    HIP_CHECK_ERROR(hipMalloc(&ptr, std::numeric_limits<std::size_t>::max()),
                    hipErrorMemoryAllocation);
  }
}

/**
 * Regress hipMalloc()/hipFree() in loop for bigger chunks and
 * smaller chunks of memory allocation
 */
TEST_CASE("Unit_hipMalloc_LoopRegressionAllocFreeCycles") {
  int devCnt = 0;

  // Get GPU count
  HIP_CHECK(hipGetDeviceCount(&devCnt));
  REQUIRE(devCnt > 0);

  CHECK(regressAllocInLoop(0) == true);
  CHECK(validateMemoryOnGPU(0) == true);
}

/**
 * Application Behavior Modelling.
 * Check hipMalloc() api adaptability when app creates small chunks of memory
 * continuously, stores it for later use and then frees it at later point
 * of time.
 */
TEST_CASE("Unit_hipMalloc_AllocateAndPoolBuffers") {
  size_t avail{0}, tot{0}, pavail{0}, ptot{0};
  bool ret{false};
  hipError_t err{};
  std::vector<int*> ptrlist{};
  constexpr auto BuffSize = 10;
  int devCnt{0}, *ptr{nullptr};

  // Get GPU count
  HIP_CHECK(hipGetDeviceCount(&devCnt));
  REQUIRE(devCnt > 0);

  HIP_CHECK(hipMemGetInfo(&pavail, &ptot));

  // Allocate small chunks of memory million times
  for (int i = 0; i < MaxAllocPoolIter; i++) {
    if ((err = hipMalloc(&ptr, BuffSize)) != hipSuccess) {
      HIP_CHECK(hipMemGetInfo(&avail, &tot));

      INFO("Loop regression pool allocation failure. "
           << "Total gpu memory " << tot / (1024.0 * 1024.0) << ", Free memory "
           << avail / (1024.0 * 1024.0) << " iter " << i << " error " << hipGetErrorString(err));

      REQUIRE(false);
    }

    // Store pointers allocated to emulate memory pool of app
    ptrlist.push_back(ptr);
  }

  // Free ptrs at later point of time
  for (auto& t : ptrlist) {
    HIP_CHECK(hipFree(t));
  }

  HIP_CHECK(hipMemGetInfo(&avail, &tot));

  ret = validateMemoryOnGPU(0);
  REQUIRE(ret == true);
  REQUIRE(pavail == avail);
  REQUIRE(ptot == tot);
}


/**
 * Exercise hipMalloc() api parellely on all gpus from
 * multiple threads and regress the api.
 */
TEST_CASE("Unit_hipMalloc_Multithreaded_MultiGPU") {
  std::vector<std::thread> threadlist;
  int devCnt;

  // Get GPU count
  HIP_CHECK(hipGetDeviceCount(&devCnt));
  REQUIRE(devCnt > 0);

  for (int i = 0; i < devCnt; i++) {
    threadlist.push_back(std::thread(threadFunc, i));
  }

  for (auto& t : threadlist) {
    t.join();
  }

  REQUIRE(g_thTestPassed == true);
}
