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

#include "hipMallocManagedCommon.hh"
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <atomic>

const size_t MAX_GPU{256};
static size_t N{4 * 1024 * 1024};
static unsigned blocksPerCU{6};
static unsigned threadsPerBlock{256};
#define INIT_VAL 123


/*
 * Kernel function to perform addition operation.
 */
template <typename T> __global__ void vector_sum(T* Ad1, T* Ad2, size_t NUM_ELMTS) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < NUM_ELMTS; i += stride) {
    Ad2[i] = Ad1[i] + Ad1[i];
  }
}

/*
 * Kernel function to perform multiplication
 */
__global__ void KernelDouble(float* Hmm, float* dPtr, size_t n) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    dPtr[index] = 2 * Hmm[index];
  }
}

/*
 * Host function to perform multiplication
 */
void HostKernelDouble(float* Hmm, float* hPtr, size_t n) {
  for (size_t i = 0; i < n; i++) {
    hPtr[i] = 2 * Hmm[i];
  }
}

/*
   This testcase verifies the concurrent access of hipMallocManaged Memory on host and device.
 */
TEST_CASE("Unit_hipMallocManaged_HostDeviceConcurrent") {
  auto managed = HmmAttrPrint();
  if (managed != 1) {
    HipTest::HIP_SKIP_TEST("GPU doesn't support managed memory so skipping test.");
    return;
  }

  float *Hmm = nullptr, *hPtr = nullptr, *dPtr = nullptr, *resPtr = nullptr;

  hPtr = reinterpret_cast<float*>(malloc(N * sizeof(float)));
  resPtr = reinterpret_cast<float*>(malloc(N * sizeof(float)));

  HIP_CHECK(hipMalloc(&dPtr, N * sizeof(float)));
  HIP_CHECK(hipMallocManaged(&Hmm, N * sizeof(float)));
  memset(Hmm, 2.0, N * sizeof(float));

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  std::thread host_thread(HostKernelDouble, Hmm, hPtr, N);
  KernelDouble<<<dim3(blocks), dim3(threadsPerBlock), 0, 0>>>(Hmm, dPtr, N);
  host_thread.join();
  HIP_CHECK(hipMemcpy(resPtr, dPtr, N * sizeof(float), hipMemcpyDeviceToHost));

  for (size_t i = 0; i < N; i++) {
    REQUIRE(hPtr[i] == resPtr[i]);
  }

  free(hPtr);
  HIP_CHECK(hipFree(dPtr));
  HIP_CHECK(hipFree(Hmm));
}

// The following Test case tests the following scenario:
// A large chunk of hipMallocManaged() memory(Hmm) is created
// Equal parts of Hmm is accessed and
// kernel is launched on acessed chunk of hmm memory
// and checks if there are any inconsistencies or access issues
TEST_CASE("Unit_hipMallocManaged_MultiChunkSingleDevice") {
  auto managed = HmmAttrPrint();
  if (managed != 1) {
    HipTest::HIP_SKIP_TEST("GPU doesn't support managed memory so skipping test.");
    return;
  }

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
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  for (int k = 0; k < Chunks; ++k) {
    vector_sum<float>
        <<<blocks, threadsPerBlock, 0, stream[k]>>>(&Hmm[k * NUM_ELMS], Ad[k], NUM_ELMS);
  }
  HIP_CHECK(hipDeviceSynchronize());
  for (int m = 0; m < Chunks; ++m) {
    HIP_CHECK(hipMemcpy(Ah, Ad[m], NUM_ELMS * sizeof(float), hipMemcpyDeviceToHost));
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
  delete[] Ah;
}

// The following Test case tests the following scenario:
// A large chunk of hipMallocManaged() memory(Hmm) is created
// Equal parts of Hmm is accessed on available gpus and
// kernel is launched on acessed chunk of hmm memory
// and checks if there are any inconsistencies or access issues
TEST_CASE("Unit_hipMallocManaged_MultiChunkMultiDevice") {
  auto managed = HmmAttrPrint();
  if (managed != 1) {
    HipTest::HIP_SKIP_TEST("GPU doesn't support managed memory so skipping test.");
    return;
  }

  std::atomic<int> DataMismatch{0};
  int Counter = 0;
  int NumDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&NumDevices));
  if (NumDevices < 2) {
    HipTest::HIP_SKIP_TEST("Skipping test because more than one device was not found.");
    return;
  }
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
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  for (int Klaunch = 0; Klaunch < NumDevices; ++Klaunch) {
    HIP_CHECK(hipSetDevice(Klaunch));
    vector_sum<float><<<blocks, threadsPerBlock, 0, stream[Klaunch]>>>(&Hmm[Klaunch * NUM_ELMS],
                                                                       Ad[Klaunch], NUM_ELMS);
  }
  HIP_CHECK(hipDeviceSynchronize());
  for (int m = 0; m < NumDevices; ++m) {
    HIP_CHECK(hipMemcpy(Ah, Ad[m], NUM_ELMS * sizeof(float), hipMemcpyDeviceToHost));
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
  delete[] Ah;
}

// The following tests oversubscription hipMallocManaged() api
// Currently disabled.
TEST_CASE("Unit_hipMallocManaged_OverSubscription") {
  auto managed = HmmAttrPrint();
  if (managed != 1) {
    HipTest::HIP_SKIP_TEST("GPU doesn't support managed memory so skipping test.");
    return;
  }

  void* A = nullptr;
  size_t total = 0, free = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  // ToDo: In case of HMM, memory over-subscription is allowed.  Hence, relook
  // into how out of memory can be tested.
  // Demanding more mem size than available
#if HT_AMD
  HIP_CHECK_ERROR(hipMallocManaged(&A, (free + 1), hipMemAttachGlobal), hipErrorOutOfMemory);
#endif
}

// The following test does negative testing of hipMallocManaged() api
// by passing invalid values and check if the behavior is as expected
TEST_CASE("Unit_hipMallocManaged_Negative") {
  void* A;
  size_t total = 0, free = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));

  SECTION("Nullptr to devPtr") {
    HIP_CHECK_ERROR(hipMallocManaged(NULL, 1024, hipMemAttachGlobal), hipErrorInvalidValue);
  }

  // cuda api doc says : If size is 0, cudaMallocManaged returns
  // cudaErrorInvalidValue. However, it is observed that cuda 11.2 api returns
  // success and contradicts with api doc.

  // With size(0), api expected to return error code (or)
  // reset ptr while returning success (to accommodate cuda 11.2 api behavior).
  SECTION("size 0 with flag hipMemAttachGlobal") {
#if HT_AMD
    HIP_CHECK_ERROR(hipMallocManaged(&A, 0, hipMemAttachGlobal), hipErrorInvalidValue);
#else
    HIP_CHECK(hipMallocManaged(&A, 0, hipMemAttachGlobal));
#endif
  }

  SECTION("devptr is nullptr with flag hipMemAttachHost") {
    HIP_CHECK_ERROR(hipMallocManaged(NULL, 1024, hipMemAttachHost), hipErrorInvalidValue);
  }

  // cuda api doc says : If size is 0, cudaMallocManaged returns
  // cudaErrorInvalidValue. However, it is observed that cuda 11.2 api returns
  // success and contradicts with api doc.

  // With size(0), api expected to return error code (or)
  // reset ptr while returning success (to accommodate cuda 11.2 api behavior).
  SECTION("size 0 with flag hipMemAttachHost") {
#if HT_AMD
    HIP_CHECK_ERROR(hipMallocManaged(&A, 0, hipMemAttachHost), hipErrorInvalidValue);
#else
    HIP_CHECK(hipMallocManaged(&A, 0, hipMemAttachHost));
#endif
  }

  SECTION("nullptr to devptr, size 0 and flag 0") {
    HIP_CHECK_ERROR(hipMallocManaged(NULL, 0, 0), hipErrorInvalidValue);
  }

  SECTION("Invalid flag parameter") {
    HIP_CHECK_ERROR(hipMallocManaged(&A, 1024, 145), hipErrorInvalidValue);
  }
  SECTION("Invalid flag parameter- flag set to 0") {
    HIP_CHECK_ERROR(hipMallocManaged(&A, 1024, 0), hipErrorInvalidValue);
  }
  SECTION("Invalid flag parameter- Both flags set") {
    HIP_CHECK_ERROR(hipMallocManaged(&A, 1024, hipMemAttachGlobal | hipMemAttachHost),
                    hipErrorInvalidValue);
  }
}

// Allocate two pointers using hipMallocManaged(), initialize,
// then launch kernel using these pointers directly and
// later validate the content without using any Memcpy.
TEMPLATE_TEST_CASE("Unit_hipMallocManaged_TwoPointers", "", int, float, double) {
  auto managed = HmmAttrPrint();
  if (managed != 1) {
    HipTest::HIP_SKIP_TEST("GPU doesn't support managed memory so skipping test.");
    return;
  }

  int NumDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&NumDevices));
  TestType *Hmm1 = nullptr, *Hmm2 = nullptr;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  for (int i = 0; i < NumDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    std::atomic<int> DataMismatch{0};
    HIP_CHECK(hipMallocManaged(&Hmm1, N * sizeof(TestType)));
    HIP_CHECK(hipMallocManaged(&Hmm2, N * sizeof(TestType)));
    for (size_t m = 0; m < N; ++m) {
      Hmm1[m] = m;
      Hmm2[m] = 0;
    }
    // Kernel launch
    vector_sum<<<blocks, threadsPerBlock>>>(Hmm1, Hmm2, N);
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

TEMPLATE_TEST_CASE("Unit_hipMallocManaged_DeviceContextChange", "", unsigned char, int, float,
                   double) {
  auto managed = HmmAttrPrint();
  if (managed != 1) {
    HipTest::HIP_SKIP_TEST("GPU doesn't support managed memory so skipping test.");
    return;
  }

  std::atomic<unsigned int> DataMismatch;
  TestType *Ah1 = new TestType[N], *Ah2 = new TestType[N], *Ad = nullptr, *Hmm = nullptr;
  int NumDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&NumDevices));
  if (NumDevices < 2) {
    HipTest::HIP_SKIP_TEST("Skipping test because more than one device was not found.");
    return;
  }

  for (size_t i = 0; i < N; ++i) {
    Ah1[i] = INIT_VAL;
    Ah2[i] = 0;
  }
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  for (int Oloop = 0; Oloop < NumDevices; ++Oloop) {
    DataMismatch = 0;
    HIP_CHECK(hipSetDevice(Oloop));
    HIP_CHECK(hipMallocManaged(&Hmm, N * sizeof(TestType)));
    for (int Iloop = 0; Iloop < NumDevices; ++Iloop) {
      HIP_CHECK(hipSetDevice(Iloop));
      HIP_CHECK(hipMalloc(&Ad, N * sizeof(TestType)));
      // Copy data from host to hipMallocMananged memory and verify
      HIP_CHECK(hipMemcpy(Hmm, Ah1, N * sizeof(TestType), hipMemcpyHostToDevice));
      for (size_t v = 0; v < N; ++v) {
        if (Hmm[v] != INIT_VAL) {
          DataMismatch++;
        }
      }
      REQUIRE(DataMismatch.load() == 0);

      // Executing D2D transfer with hipMallocManaged memory and verify
      HIP_CHECK(hipMemcpy(Ad, Hmm, N * sizeof(TestType), hipMemcpyDeviceToDevice));
      HIP_CHECK(hipMemcpy(Ah2, Ad, N * sizeof(TestType), hipMemcpyDeviceToHost));
      for (size_t k = 0; k < N; ++k) {
        if (Ah2[k] != INIT_VAL) {
          DataMismatch++;
        }
      }
      REQUIRE(DataMismatch.load() == 0);
      HIP_CHECK(hipMemset(Ad, 0, N * sizeof(TestType)));
      // Launching the kernel to check if there is any access issue with
      // hipMallocManaged memory and local device's memory
      vector_sum<<<blocks, threadsPerBlock>>>(Hmm, Ad, N);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(Ah2, Ad, N * sizeof(TestType), hipMemcpyDeviceToHost));
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
