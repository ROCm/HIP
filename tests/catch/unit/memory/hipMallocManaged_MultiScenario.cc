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
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

/*
   This testcase verifies the hipMallocManaged API in the following scenarios
   1. MultiChunkSingleDevice Scenario
   2. MultiChunkMultiDevice Scenario
   3. Negative Scenarios
   4. OverSubscription scenario
   5. Device context change
   6. Multiple Pointers
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <atomic>

const size_t MAX_GPU{256};
static size_t N{4*1024*1024};
#define INIT_VAL 123


/*
 * Kernel function to perform addition operation.
 */
template <typename T>
__global__ void
vector_sum(T *Ad1, T *Ad2, size_t NUM_ELMTS) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = offset; i < NUM_ELMTS; i += stride) {
        Ad2[i] = Ad1[i] + Ad1[i];
    }
}

// The following Test case tests the following scenario:
// A large chunk of hipMallocManaged() memory(Hmm) is created
// Equal parts of Hmm is accessed and
// kernel is launched on acessed chunk of hmm memory
// and checks if there are any inconsistencies or access issues
TEST_CASE("Unit_hipMallocManaged_MultiChunkSingleDevice") {
std::atomic<int> DataMismatch{0};
  constexpr int Chunks = 4;
  int Counter = 0;
  int NUM_ELMS = (1024 * 1024);
  float *Ad[Chunks], *Hmm = nullptr, *Ah = new float[NUM_ELMS];
  hipStream_t stream[Chunks];
  for (int i = 0; i < Chunks; ++i) {
    HIP_CHECK(hipMalloc(&Ad[i], NUM_ELMS * sizeof(float)));
    HIP_CHECK(hipMemset(Ad[i], 0, NUM_ELMS * sizeof(float)));
    HIP_CHECK(hipStreamCreate(&stream[i]));
  }
  HIP_CHECK(hipMallocManaged(&Hmm, (Chunks * NUM_ELMS * sizeof(float))));
  for (int i = 0; i < Chunks; ++i) {
    for (; Counter < ((i + 1) * NUM_ELMS); ++Counter) {
      Hmm[Counter] = (INIT_VAL + i);
    }
  }
  const unsigned threadsPerBlock = 256;
  const unsigned blocks = (NUM_ELMS + 255)/256;
  for (int k = 0; k < Chunks; ++k) {
    vector_sum<float> <<<blocks, threadsPerBlock, 0, stream[k]>>>
                      (&Hmm[k * NUM_ELMS], Ad[k], NUM_ELMS);
  }
  HIP_CHECK(hipDeviceSynchronize());
  for (int m = 0; m < Chunks; ++m) {
    HIP_CHECK(hipMemcpy(Ah, Ad[m], NUM_ELMS * sizeof(float),
                       hipMemcpyDeviceToHost));
    for (int n = 0; n < NUM_ELMS; ++n) {
      if (Ah[n] != ((INIT_VAL + m) * 2)) {
        DataMismatch++;
      }
    }
  }
  REQUIRE(DataMismatch.load() == 0);
  for (int i = 0; i < Chunks; ++i) {
    HIP_CHECK(hipFree(Ad[i]));
    HIP_CHECK(hipStreamDestroy(stream[i]));
  }
  HIP_CHECK(hipFree(Hmm));
  delete [] Ah;
}

// The following Test case tests the following scenario:
// A large chunk of hipMallocManaged() memory(Hmm) is created
// Equal parts of Hmm is accessed on available gpus and
// kernel is launched on acessed chunk of hmm memory
// and checks if there are any inconsistencies or access issues
TEST_CASE("Unit_hipMallocManaged_MultiChunkMultiDevice") {
  std::atomic<int> DataMismatch{0};
  int Counter = 0;
  int NumDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&NumDevices));
  unsigned int NUM_ELMS = (1024 * 1024);
  float *Ad[MAX_GPU], *Hmm = NULL, *Ah = new float[NUM_ELMS];
  hipStream_t stream[MAX_GPU];
  for (int Oloop = 0; Oloop < NumDevices; ++Oloop) {
    HIP_CHECK(hipSetDevice(Oloop));
    HIP_CHECK(hipMalloc(&Ad[Oloop], NUM_ELMS * sizeof(float)));
    HIP_CHECK(hipMemset(Ad[Oloop], 0, NUM_ELMS * sizeof(float)));
    HIP_CHECK(hipStreamCreate(&stream[Oloop]));
  }
  HIP_CHECK(hipMallocManaged(&Hmm, (NumDevices * NUM_ELMS * sizeof(float))));
  for (int i = 0; i < NumDevices; ++i) {
    for (; Counter < static_cast<int>((i + 1) * NUM_ELMS); ++Counter) {
      Hmm[Counter] = INIT_VAL + i;
    }
  }
  const unsigned threadsPerBlock = 256;
  const unsigned blocks = (NUM_ELMS + 255)/256;
  for (int Klaunch = 0; Klaunch < NumDevices; ++Klaunch) {
    HIP_CHECK(hipSetDevice(Klaunch));
    vector_sum<float> <<<blocks, threadsPerBlock, 0, stream[Klaunch]>>>
                      (&Hmm[Klaunch * NUM_ELMS], Ad[Klaunch], NUM_ELMS);
  }
  HIP_CHECK(hipDeviceSynchronize());
  for (int m = 0; m < NumDevices; ++m) {
    HIP_CHECK(hipMemcpy(Ah, Ad[m], NUM_ELMS * sizeof(float),
                       hipMemcpyDeviceToHost));
    for (size_t n = 0; n < NUM_ELMS; ++n) {
      if (Ah[n] != ((INIT_VAL + m) * 2)) {
        DataMismatch++;
      }
    }
    memset(reinterpret_cast<void*>(Ah), 0, NUM_ELMS * sizeof(float));
  }
  REQUIRE(DataMismatch.load() == 0);
  for (int i = 0; i < NumDevices; ++i) {
    HIP_CHECK(hipFree(Ad[i]));
    HIP_CHECK(hipStreamDestroy(stream[i]));
  }
  HIP_CHECK(hipFree(Hmm));
  delete [] Ah;
}

// The following tests oversubscription hipMallocManaged() api
// Currently disabled.
TEST_CASE("Unit_hipMallocManaged_OverSubscription") {
  void *A = nullptr;
  size_t total = 0, free = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  // ToDo: In case of HMM, memory over-subscription is allowed.  Hence, relook
  // into how out of memory can be tested.
  // Demanding more mem size than available
#if HT_AMD
  REQUIRE(hipMallocManaged(&A, (free +1), hipMemAttachGlobal) != hipSuccess);
#endif
}

// The following test does negative testing of hipMallocManaged() api
// by passing invalid values and check if the behavior is as expected
TEST_CASE("Unit_hipMallocManaged_Negative") {
  void *A;
  size_t total = 0, free = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));

  SECTION("Nullptr to devPtr") {
    REQUIRE(hipMallocManaged(NULL, 1024, hipMemAttachGlobal) != hipSuccess);
  }

  // cuda api doc says : If size is 0, cudaMallocManaged returns
  // cudaErrorInvalidValue. However, it is observed that cuda 11.2 api returns
  // success and contradicts with api doc.

  // With size(0), api expected to return error code (or)
  // reset ptr while returning success (to accommodate cuda 11.2 api behavior).
  SECTION("size 0 with flag hipMemAttachGlobal") {
#if HT_AMD
    REQUIRE(hipMallocManaged(&A, 0, hipMemAttachGlobal) != hipSuccess);
#else
    REQUIRE(hipMallocManaged(&A, 0, hipMemAttachHost) == hipSuccess);
#endif
  }

  SECTION("devptr is nullptr with flag hipMemAttachHost") {
    REQUIRE(hipMallocManaged(NULL, 1024, hipMemAttachHost) != hipSuccess);
  }

  // cuda api doc says : If size is 0, cudaMallocManaged returns
  // cudaErrorInvalidValue. However, it is observed that cuda 11.2 api returns
  // success and contradicts with api doc.

  // With size(0), api expected to return error code (or)
  // reset ptr while returning success (to accommodate cuda 11.2 api behavior).
  SECTION("size 0 with flag hipMemAttachHost") {
#if HT_AMD
    REQUIRE(hipMallocManaged(&A, 0, hipMemAttachHost) != hipSuccess);
#else
    REQUIRE(hipMallocManaged(&A, 0, hipMemAttachHost) == hipSuccess);
#endif
  }
  SECTION("nullptr to devptr, size 0 and flag 0") {
    REQUIRE(hipMallocManaged(NULL, 0, 0) != hipSuccess);
  }

  SECTION("Numeric value to flag parameter") {
    REQUIRE(hipMallocManaged(&A, 1024, 145) != hipSuccess);
  }

  SECTION("Negative value to size") {
    REQUIRE(hipMallocManaged(&A, -10, hipMemAttachGlobal));
  }
}

// Allocate two pointers using hipMallocManaged(), initialize,
// then launch kernel using these pointers directly and
// later validate the content without using any Memcpy.
TEMPLATE_TEST_CASE("Unit_hipMallocManaged_TwoPointers", "",
                    int, float, double) {
  int NumDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&NumDevices));
  TestType *Hmm1 = nullptr, *Hmm2 = nullptr;

  for (int i = 0; i < NumDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    std::atomic<int> DataMismatch{0};
    HIP_CHECK(hipMallocManaged(&Hmm1, N * sizeof(TestType)));
    HIP_CHECK(hipMallocManaged(&Hmm2, N * sizeof(TestType)));
    for (size_t m = 0; m < N; ++m) {
      Hmm1[m] = m;
      Hmm2[m] = 0;
    }
    const unsigned threadsPerBlock = 256;
    const unsigned blocks = (N + 255)/256;
    // Kernel launch
    vector_sum <<<blocks, threadsPerBlock>>> (Hmm1, Hmm2, N);
    HIP_CHECK(hipDeviceSynchronize());
    for (size_t v = 0; v < N; ++v) {
      if (Hmm2[v] != static_cast<TestType>(v + v)) {
        DataMismatch++;
      }
    }
    REQUIRE(DataMismatch.load() == 0);
    HIP_CHECK(hipFree(Hmm1));
    HIP_CHECK(hipFree(Hmm2));
  }
}

// In the following test, a memory is created using hipMallocManaged() by
// setting a device and verified if it is accessible when the context is set
// to all other devices. This include verification and Device two Device
// transfers and kernel launch o discover if there any access issues.

TEMPLATE_TEST_CASE("Unit_hipMallocManaged_DeviceContextChange", "",
    unsigned char, int, float, double) {
  std::atomic<unsigned int> DataMismatch;
  TestType *Ah1 = new TestType[N], *Ah2 = new TestType[N], *Ad = nullptr,
           *Hmm = nullptr;
  int NumDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&NumDevices));

  for (size_t i =0; i < N; ++i) {
    Ah1[i] = INIT_VAL;
    Ah2[i] = 0;
  }
  for (int Oloop = 0; Oloop < NumDevices; ++Oloop) {
    DataMismatch = 0;
    HIP_CHECK(hipSetDevice(Oloop));
    HIP_CHECK(hipMallocManaged(&Hmm, N * sizeof(TestType)));
    for (int Iloop = 0; Iloop < NumDevices; ++Iloop) {
      HIP_CHECK(hipSetDevice(Iloop));
      HIP_CHECK(hipMalloc(&Ad, N * sizeof(TestType)));
      // Copy data from host to hipMallocMananged memory and verify
      HIP_CHECK(hipMemcpy(Hmm, Ah1, N * sizeof(TestType),
            hipMemcpyHostToDevice));
      for (size_t v = 0; v < N; ++v) {
        if (Hmm[v] != INIT_VAL) {
          DataMismatch++;
        }
      }
      REQUIRE(DataMismatch.load() == 0);

      // Executing D2D transfer with hipMallocManaged memory and verify
      HIP_CHECK(hipMemcpy(Ad, Hmm, N * sizeof(TestType),
                          hipMemcpyDeviceToDevice));
      HIP_CHECK(hipMemcpy(Ah2, Ad, N * sizeof(TestType),
                          hipMemcpyDeviceToHost));
      for (size_t k = 0; k < N; ++k) {
        if (Ah2[k] != INIT_VAL) {
          DataMismatch++;
        }
      }
      REQUIRE(DataMismatch.load() == 0);
      HIP_CHECK(hipMemset(Ad, 0, N * sizeof(TestType)));
      const unsigned threadsPerBlock = 256;
      const unsigned blocks = (N + 255)/256;
      // Launching the kernel to check if there is any access issue with
      // hipMallocManaged memory and local device's memory
      vector_sum <<<blocks, threadsPerBlock>>> (Hmm, Ad, N);
      hipDeviceSynchronize();
      HIP_CHECK(hipMemcpy(Ah2, Ad, N * sizeof(TestType),
                          hipMemcpyDeviceToHost));
      for (size_t m = 0; m < N; ++m) {
        if (Ah2[m] != 246) {
          DataMismatch++;
        }
      }
      REQUIRE(DataMismatch.load() == 0);
      HIP_CHECK(hipFree(Ad));
    }
    HIP_CHECK(hipFree(Hmm));
  }
  free(Ah1);
  free(Ah2);
}
