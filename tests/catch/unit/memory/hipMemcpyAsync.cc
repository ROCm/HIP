/*
Copyright (c) 2022 - present Advanced Micro Devices, Inc. All rights reserved.
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
This testcase verifies the following scenarios
1. hipMemcpyAsync with kernel launch
2. H2D-D2D-D2H-H2PinnMem and device context change scenarios
3. This test launches multiple threads which uses same stream to deploy kernel
   and also launch hipMemcpyAsync() api. This test case is simulate the scenario
   reported in SWDEV-181598.
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <atomic>

#define NUM_THREADS 16

static constexpr auto NUM_ELM{1024 * 1024};



static constexpr size_t N_ELMTS{32 * 1024};
std::atomic<size_t> Thread_count { 0 };
static unsigned blocksPerCU{6};  // to hide latency
static unsigned threadsPerBlock{256};

template<typename T>
void Thread_func(T *A_d, T *B_d, T* C_d, T* C_h, size_t Nbytes,
                 hipStream_t mystream) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU,
                                          threadsPerBlock, N_ELMTS);
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                     dim3(threadsPerBlock), 0,
                     mystream, A_d, C_d, N_ELMTS);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, mystream));
  // The following two MemcpyAsync calls are for sole
  // purpose of loading stream with multiple async calls
  HIP_CHECK(hipMemcpyAsync(B_d, A_d, Nbytes,
                           hipMemcpyDeviceToDevice, mystream));
  HIP_CHECK(hipMemcpyAsync(B_d, A_d, Nbytes,
                           hipMemcpyDeviceToDevice, mystream));
  Thread_count++;
}

template<typename T>
void Thread_func_MultiStream() {
  int Data_mismatch = 0;
  T *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  T *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  size_t Nbytes = N_ELMTS * sizeof(T);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU,
                                          threadsPerBlock, N_ELMTS);

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N_ELMTS, false);
  hipStream_t mystream;
  HIP_CHECK(hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, mystream));
  hipLaunchKernelGGL((HipTest::vector_square), dim3(blocks),
                     dim3(threadsPerBlock), 0,
                     mystream, A_d, C_d, N_ELMTS);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, mystream));
  // The following hipMemcpyAsync() is called only to
  // load stream with multiple Async calls
  HIP_CHECK(hipMemcpyAsync(B_d, A_d, Nbytes,
                           hipMemcpyDeviceToDevice, mystream));
  Thread_count++;

  HIP_CHECK(hipStreamSynchronize(mystream));
  HIP_CHECK(hipStreamDestroy(mystream));
  // Verifying result of the kernel computation
  for (size_t i = 0; i < N_ELMTS; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      Data_mismatch++;
    }
  }
  // Releasing resources
  HipTest::freeArrays<T>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  REQUIRE(Data_mismatch == 0);
}

/*
This testcase verifies hipMemcpyAsync API
Initializes device variables
Launches kernel and performs the sum of device variables
copies the result to host variable and validates the result.
*/
TEMPLATE_TEST_CASE("Unit_hipMemcpyAsync_KernelLaunch", "", int, float,
                   double) {
  size_t Nbytes = NUM_ELM * sizeof(TestType);

  TestType *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HIP_CHECK(hipSetDevice(0));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, NUM_ELM, false);

  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(1), dim3(1), 0, 0,
                     static_cast<const TestType*>(A_d),
                     static_cast<const TestType*>(B_d), C_d, NUM_ELM);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));

  HipTest::checkVectorADD(A_h, B_h, C_h, NUM_ELM);

  HipTest::freeArrays<TestType>(A_d, B_d, C_d, A_h, B_h, C_h, false);
}
/*
This testcase verifies the following scenarios
1. H2H,H2PinMem and PinnedMem2Host
2. H2D-D2D-D2H in same GPU
3. Pinned Host Memory to device variables in same GPU
4. Device context change
5. H2D-D2D-D2H peer GPU
*/
TEMPLATE_TEST_CASE("Unit_hipMemcpyAsync_H2H-H2D-D2H-H2PinMem", "", char, int,
                   float, double) {
  TestType *A_d{nullptr}, *B_d{nullptr};
  TestType *A_h{nullptr}, *B_h{nullptr};
  TestType *A_Ph{nullptr}, *B_Ph{nullptr};
  HIP_CHECK(hipSetDevice(0));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HipTest::initArrays<TestType>(&A_d, &B_d, nullptr,
                                &A_h, &B_h, nullptr,
                                NUM_ELM*sizeof(TestType));
  HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                &A_Ph, &B_Ph, nullptr,
                                NUM_ELM*sizeof(TestType), true);

  SECTION("H2H, H2PinMem and PinMem2H") {
    HIP_CHECK(hipMemcpyAsync(B_h, A_h, NUM_ELM*sizeof(TestType),
                             hipMemcpyHostToHost, stream));
    HIP_CHECK(hipMemcpyAsync(A_Ph, B_h, NUM_ELM*sizeof(TestType),
                             hipMemcpyHostToHost, stream));
    HIP_CHECK(hipMemcpyAsync(B_Ph, A_Ph, NUM_ELM*sizeof(TestType),
                             hipMemcpyHostToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HipTest::checkTest(A_h, B_Ph, NUM_ELM);
  }

  SECTION("H2D-D2D-D2H-SameGPU") {
    HIP_CHECK(hipMemcpyAsync(A_d, A_h, NUM_ELM*sizeof(TestType),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(B_d, A_d, NUM_ELM*sizeof(TestType),
                             hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(B_h, B_d, NUM_ELM*sizeof(TestType),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HipTest::checkTest(A_h, B_h, NUM_ELM);
  }

  SECTION("pH2D-D2D-D2pH-SameGPU") {
    HIP_CHECK(hipMemcpyAsync(A_d, A_Ph, NUM_ELM*sizeof(TestType),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(B_d, A_d, NUM_ELM*sizeof(TestType),
                             hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(B_Ph, B_d, NUM_ELM*sizeof(TestType),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HipTest::checkTest(A_Ph, B_Ph, NUM_ELM);
  }
  SECTION("H2D-D2D-D2H-DeviceContextChange") {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
      SUCCEED("deviceCount less then 2");
    } else {
      int canAccessPeer = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
      if (canAccessPeer) {
        HIP_CHECK(hipSetDevice(1));
        HIP_CHECK(hipMemcpyAsync(A_d, A_h, NUM_ELM*sizeof(TestType),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(B_d, A_d, NUM_ELM*sizeof(TestType),
                                 hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(B_h, B_d, NUM_ELM*sizeof(TestType),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HipTest::checkTest(A_h, B_h, NUM_ELM);

      } else {
        SUCCEED("P2P capability is not present");
      }
    }
  }

  SECTION("H2D-D2D-D2H-PeerGPU") {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
      SUCCEED("deviceCount less then 2");
    } else {
      int canAccessPeer = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
      if (canAccessPeer) {
        HIP_CHECK(hipSetDevice(1));
        TestType *C_d{nullptr};
        HipTest::initArrays<TestType>(nullptr, nullptr, &C_d,
                                      nullptr, nullptr, nullptr,
                                      NUM_ELM*sizeof(TestType));
        HIP_CHECK(hipMemcpyAsync(A_d, A_h, NUM_ELM*sizeof(TestType),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(C_d, A_d, NUM_ELM*sizeof(TestType),
                                 hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(B_h, C_d, NUM_ELM*sizeof(TestType),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HipTest::checkTest(A_h, B_h, NUM_ELM);
        HIP_CHECK(hipFree(C_d));

      } else {
        SUCCEED("P2P capability is not present");
      }
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));

  HipTest::freeArrays<TestType>(A_d, B_d, nullptr, A_h, B_h, nullptr, false);
  HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, A_Ph,
                                B_Ph, nullptr, true);
}

// This test launches multiple threads which uses same stream to deploy kernel
// and also launch hipMemcpyAsync() api. This test case is simulate the scenario
// reported in SWDEV-181598

TEMPLATE_TEST_CASE("Unit_hipMemcpyAsync_hipMultiMemcpyMultiThread", "",
                   int, float, double) {
  size_t Nbytes = N_ELMTS * sizeof(TestType);

  int Data_mismatch = 0;
  hipStream_t mystream;
  TestType *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N_ELMTS, false);

  HIP_CHECK(hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, mystream));

  std::thread T[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++) {
    T[i] = std::thread(Thread_func<TestType>, A_d, B_d, C_d,
                       C_h, Nbytes, mystream);
  }

  // Wait until all the threads finish their execution
  for (int i = 0; i < NUM_THREADS; i++) {
    T[i].join();
  }

  HIP_CHECK(hipStreamSynchronize(mystream));
  HIP_CHECK(hipStreamDestroy(mystream));

  // Verifying the result of the kernel computation
  for (size_t i = 0; i < N_ELMTS; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      Data_mismatch++;
    }
  }
  REQUIRE(Thread_count.load() == NUM_THREADS);
  REQUIRE(Data_mismatch == 0);
  HipTest::freeArrays<TestType>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  Thread_count.exchange(0);
}

TEMPLATE_TEST_CASE("Unit_hipMemcpyAsync_hipMultiMemcpyMultiThreadMultiStream",
                   "", int, float, double) {
  std::thread T[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++) {
    T[i] = std::thread(Thread_func_MultiStream<TestType>);
  }

  // Wait until all the threads finish their execution
  for (int i = 0; i < NUM_THREADS; i++) {
    T[i].join();
  }

  REQUIRE(Thread_count.load() == NUM_THREADS);
  Thread_count.exchange(0);
}

/*
This testcase verifies hipMemcpy API with pinnedMemory and hostRegister
along with kernel launches
*/

TEMPLATE_TEST_CASE("Unit_hipMemcpyAsync_PinnedRegMemWithKernelLaunch",
                   "", int, float, double) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    SUCCEED("No of devices are less than 2");
  } else {
    // 1 refers to pinned Memory
    // 2 refers to register Memory
    int MallocPinType = GENERATE(0, 1);
    size_t Nbytes = NUM_ELM * sizeof(TestType);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU,
                                            threadsPerBlock, NUM_ELM);

    TestType *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
    TestType *X_d{nullptr}, *Y_d{nullptr}, *Z_d{nullptr};
    TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
    if (MallocPinType) {
      HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, NUM_ELM, true);
    } else {
      A_h = reinterpret_cast<TestType*>(malloc(Nbytes));
      HIP_CHECK(hipHostRegister(A_h, Nbytes, hipHostRegisterDefault));
      B_h = reinterpret_cast<TestType*>(malloc(Nbytes));
      HIP_CHECK(hipHostRegister(B_h, Nbytes, hipHostRegisterDefault));
      C_h = reinterpret_cast<TestType*>(malloc(Nbytes));
      HIP_CHECK(hipHostRegister(C_h, Nbytes, hipHostRegisterDefault));
      HipTest::initArrays<TestType>(&A_d, &B_d, &C_d, nullptr, nullptr,
                                    nullptr, NUM_ELM, false);
      HipTest::setDefaultData<TestType>(NUM_ELM, A_h, B_h, C_h);
    }
    HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, 0, static_cast<const TestType*>(A_d),
                       static_cast<const TestType*>(B_d), C_d, NUM_ELM);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
    HipTest::checkVectorADD(A_h, B_h, C_h, NUM_ELM);

    unsigned int seed = time(0);
    HIP_CHECK(hipSetDevice(HipTest::RAND_R(&seed) % (numDevices-1)+1));

    int device;
    HIP_CHECK(hipGetDevice(&device));
    INFO("hipMemcpy is set to happen between device 0 and device "
          << device);
    HipTest::initArrays<TestType>(&X_d, &Y_d, &Z_d, nullptr,
                                  nullptr, nullptr, NUM_ELM, false);

    hipStream_t gpu1Stream;
    HIP_CHECK(hipStreamCreate(&gpu1Stream));

    for (int j = 0; j < NUM_ELM; j++) {
      A_h[j] = 0;
      B_h[j] = 0;
      C_h[j] = 0;
    }

    HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpyAsync(X_d, A_h, Nbytes, hipMemcpyHostToDevice, gpu1Stream));
    HIP_CHECK(hipMemcpy(B_h, B_d, Nbytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpyAsync(Y_d, B_h, Nbytes, hipMemcpyHostToDevice, gpu1Stream));

    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock),
                       0, 0, static_cast<const TestType*>(X_d),
                       static_cast<const TestType*>(Y_d), Z_d, NUM_ELM);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpyAsync(C_h, Z_d, Nbytes,
                             hipMemcpyDeviceToHost, gpu1Stream));
    HIP_CHECK(hipStreamSynchronize(gpu1Stream));

    HipTest::checkVectorADD(A_h, B_h, C_h, NUM_ELM);

    if (MallocPinType) {
      HipTest::freeArrays<TestType>(A_d, B_d, C_d, A_h, B_h, C_h, true);
    } else {
      HIP_CHECK(hipHostUnregister(A_h));
      free(A_h);
      HIP_CHECK(hipHostUnregister(B_h));
      free(B_h);
      HIP_CHECK(hipHostUnregister(C_h));
      free(C_h);
      HipTest::freeArrays<TestType>(A_d, B_d, C_d, nullptr,
                                    nullptr, nullptr, false);
    }
      HipTest::freeArrays<TestType>(X_d, Y_d, Z_d, nullptr,
                                    nullptr, nullptr, false);
      HIP_CHECK(hipStreamDestroy(gpu1Stream));
  }
}

