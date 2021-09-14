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

/**
Testcase Scenarios :

1)Validates functionality of hipStreamAddCallback with created stream.

2)Validates functionality of stream with cu mask.

3)Create a stream with all CU masks disabled (0x00000000).
Verify that default CU mask is set for the stream.

4)Size is greater than physical CU number. In this case the extra elements
are ignored and hipExtStreamCreateWithCUMask must return hipSuccess.

5)Negative Testing of hipExtStreamCreateWithCUMask.
*/


#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <unistd.h>
#include <iostream>
#include <vector>

#define NUM_CU_PARTITIONS 4
#define CONSTANT 1.618f
#define SIZE_INBYTES_OF_MB (1024*1024)
#define GRIDSIZE 512
#define BLOCKSIZE 256
#define ZERO_MASK 0x00000000


namespace hipExtStreamCreateWithCUMaskTest {

float *A_h, *C_h;
bool cbDone = false;
bool isPassed = true;
size_t N = 4 * SIZE_INBYTES_OF_MB;


// Make a default CU mask bit-array where all CUs are active
// this default mask is expected to be returned when there is no
// custom or global CU mask defined.
void createDefaultCUMask(std::vector<uint32_t> *pdefaultCUMask,
                         int numOfCUs) {
  uint32_t temp = 0;
  uint32_t bit_index = 0;
  for (int i = 0; i < numOfCUs; i++) {
    temp |= 1UL << bit_index;
    if (bit_index >= 32) {
      (*pdefaultCUMask).push_back(temp);
      temp = 0;
      bit_index = 0;
      temp |= 1UL << bit_index;
    }
    bit_index += 1;
  }
  if (bit_index != 0) {
    (*pdefaultCUMask).push_back(temp);
  }
}
// Create masks of disabled CU masks.
void createDisabledCUMask(std::vector<uint32_t> *pdisabledCUMask,
                         int numOfCUs) {
  uint32_t temp = ZERO_MASK;
  uint32_t bit_index = 0;
  for (int i = 0; i < numOfCUs; i++) {
    if (bit_index >= 32) {
      (*pdisabledCUMask).push_back(temp);
      temp = ZERO_MASK;
      bit_index = 0;
    }
    bit_index += 1;
  }
  if (bit_index != 0) {
    (*pdisabledCUMask).push_back(temp);
  }
}

void Callback(hipStream_t stream, hipError_t status,
              void* userData) {
  isPassed = true;
  stream = 0;
  HIP_CHECK(status);
  REQUIRE(userData == nullptr);
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      UNSCOPED_INFO("Data mismatch at index: " << i);
      isPassed = false;
      break;
    }
  }
  cbDone = true;
}
}  // namespace hipExtStreamCreateWithCUMaskTest

using hipExtStreamCreateWithCUMaskTest::A_h;
using hipExtStreamCreateWithCUMaskTest::C_h;
using hipExtStreamCreateWithCUMaskTest::cbDone;
using hipExtStreamCreateWithCUMaskTest::isPassed;
using hipExtStreamCreateWithCUMaskTest::N;
using hipExtStreamCreateWithCUMaskTest::createDefaultCUMask;
using hipExtStreamCreateWithCUMaskTest::createDisabledCUMask;
using hipExtStreamCreateWithCUMaskTest::Callback;


/**
 * Scenario: Validates functionality of hipStreamAddCallback with created stream.
 */
TEST_CASE("Unit_hipExtStreamCreateWithCUMask_ValidateCallbackFunc") {
  float *A_d, *C_d;
  size_t Nbytes = N * sizeof(float);
  cbDone = false;
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(C_h != nullptr);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
    A_h[i] = CONSTANT + i;
  }

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));

  hipStream_t mystream;
  std::vector<uint32_t> defaultCUMask;
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  createDefaultCUMask(&defaultCUMask, props.multiProcessorCount);

  hipExtStreamCreateWithCUMask(&mystream, defaultCUMask.size(),
                               defaultCUMask.data());
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice,
          mystream));
  const unsigned blocks = GRIDSIZE;
  const unsigned threadsPerBlock = BLOCKSIZE;
  hipLaunchKernelGGL((HipTest::vector_square), dim3(blocks),
                     dim3(threadsPerBlock), 0, mystream, A_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost,
          mystream));
  HIP_CHECK(hipStreamAddCallback(mystream, Callback, nullptr, 0));
  while (!cbDone) usleep(100000);  // Sleep for 100 ms
  HIP_CHECK(hipStreamDestroy(mystream));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(C_d)));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
  free(C_h);
  free(A_h);
  REQUIRE(isPassed == true);
}

/**
 * Scenario: Validates functionality of stream with cu mask.
 */
TEST_CASE("Unit_hipExtStreamCreateWithCUMask_Functionality") {
  const int KNumPartition = NUM_CU_PARTITIONS;
  float *dA[KNumPartition], *dC[KNumPartition];
  float *hA, *hC;
  size_t N = 25*SIZE_INBYTES_OF_MB;
  size_t Nbytes = N * sizeof(float);
  std::vector<hipStream_t> streams(KNumPartition);
  std::vector<std::vector<uint32_t>> cuMasks(KNumPartition);
  std::stringstream ss[KNumPartition];

  int nGpu = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    WARN("Didn't find any GPU! skipping the test!");
    return;
  }

  static int device = 0;
  HIP_CHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  INFO("info: running on bus " << "0x" << props.pciBusID << " "
  << props.name << " with " << props.multiProcessorCount << " CUs");

  hA =  new float[Nbytes];
  REQUIRE(hA != nullptr);
  hC =  new float[Nbytes];
  REQUIRE(hC != nullptr);
  for (size_t i = 0; i < N; i++) {
    hA[i] = CONSTANT + i;
  }

  for (int np = 0; np < KNumPartition; np++) {
    HIP_CHECK(hipMalloc(&dA[np], Nbytes));
    HIP_CHECK(hipMalloc(&dC[np], Nbytes));
    // make unique CU masks in the multiple of dwords for each stream
    uint32_t temp = 0;
    uint32_t bit_index = np;
    for (int i = np; i < props.multiProcessorCount; i = i + 4) {
      temp |= 1UL << bit_index;
      if (bit_index >= 32) {
        cuMasks[np].push_back(temp);
        temp = 0;
        bit_index = np;
        temp |= 1UL << bit_index;
      }
      bit_index += 4;
    }
    if (bit_index != 0) {
      cuMasks[np].push_back(temp);
    }

    HIP_CHECK(hipExtStreamCreateWithCUMask(&streams[np], cuMasks[np].size(),
            cuMasks[np].data()));

    HIP_CHECK(hipMemcpy(dA[np], hA, Nbytes, hipMemcpyHostToDevice));

    ss[np] << std::hex;
    for (int i = cuMasks[np].size() - 1; i >= 0; i--) {
      ss[np] << cuMasks[np][i];
    }
  }

  const unsigned blocks = GRIDSIZE;
  const unsigned threadsPerBlock = BLOCKSIZE;

  auto single_start = std::chrono::steady_clock::now();
  INFO("info: launch 'vector_square' kernel on one stream " <<
  streams[0] << " with CU mask: 0x" << ss[0].str().c_str());

  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                     dim3(threadsPerBlock), 0, streams[0], dA[0], dC[0], N);
  hipDeviceSynchronize();

  auto single_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> single_kernel_time = single_end - single_start;

  HIP_CHECK(hipMemcpy(hC, dC[0], Nbytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < N; i++) {
    REQUIRE(hC[i] == (hA[i] * hA[i]));
  }

  INFO("info: launch 'vector_square' kernel on "
  << KNumPartition << " streams:");
  auto all_start = std::chrono::steady_clock::now();
  for (int np = 0; np < KNumPartition; np++) {
    INFO("info: launch 'vector_square' kernel on the stream "
    << streams[np] << " with CU mask: 0x" << ss[np].str().c_str());
    hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
    dim3(threadsPerBlock), 0, streams[np], dA[np], dC[np], N);
  }
  hipDeviceSynchronize();

  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> all_kernel_time = all_end - all_start;

  for (int np = 0; np < KNumPartition; np++) {
    HIP_CHECK(hipMemcpy(hC, dC[np], Nbytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < N; i++) {
      REQUIRE(hC[i] == (hA[i] * hA[i]));
    }
  }

  INFO("info: kernel launched on one stream took: " <<
  single_kernel_time.count() << " seconds");
  INFO("info: kernels launched on " << KNumPartition <<
  " streams took: " << all_kernel_time.count() << " seconds");
  INFO("info: launching kernels on " << KNumPartition <<
  " streams asynchronously is " <<
  single_kernel_time.count() / (all_kernel_time.count() / KNumPartition)
  << " times faster per stream than launching on one stream alone");

  delete [] hA;
  delete [] hC;
  for (int np = 0; np < KNumPartition; np++) {
    hipFree(dC[np]);
    hipFree(dA[np]);
    HIP_CHECK(hipStreamDestroy(streams[np]));
  }
}

/**
 * Scenario: Create a stream with all CU masks disabled (0x00000000).
 * Verify that default CU mask is set for the stream.
 */
TEST_CASE("Unit_hipExtStreamCreateWithCUMask_AllCUsMasked") {
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  // make a CU mask with all CUs disabled.
  std::vector<uint32_t> allCUDisabled;
  createDisabledCUMask(&allCUDisabled, props.multiProcessorCount);
  hipStream_t stream;
  HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, allCUDisabled.size(),
          allCUDisabled.data()));
  // Verify whether default CU mask is set for the stream.
  uint32_t size = (props.multiProcessorCount / 32) + 1;
  std::vector<uint32_t> cuMask(size);
  std::vector<uint32_t> defaultCUMask;
  createDefaultCUMask(&defaultCUMask, props.multiProcessorCount);
  HIP_CHECK(hipExtStreamGetCUMask(stream, cuMask.size(), &cuMask[0]));
  for (int i = 0; i < static_cast<int>(defaultCUMask.size()); i++) {
    REQUIRE(defaultCUMask[i] == cuMask[i]);
  }
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Scenario: Negative Testing of hipExtStreamCreateWithCUMask.
 */
TEST_CASE("Unit_hipExtStreamCreateWithCUMask_NegTst") {
  std::vector<uint32_t> defaultCUMask;
  REQUIRE(hipSuccess == hipSetDevice(0));
  hipDeviceProp_t props;
  REQUIRE(hipSuccess == hipGetDeviceProperties(&props, 0));
  createDefaultCUMask(&defaultCUMask, props.multiProcessorCount);
  hipStream_t stream;
  // Negative Scenario 1: stream = nullptr
  SECTION("stream is nullptr") {
    REQUIRE_FALSE(hipExtStreamCreateWithCUMask(nullptr,
                  defaultCUMask.size(),
                  defaultCUMask.data()) == hipSuccess);
  }
  // Negative Scenario 2: cuMaskSize = 0
  SECTION("cuMaskSize is 0") {
    REQUIRE_FALSE(hipExtStreamCreateWithCUMask(&stream, 0,
                  defaultCUMask.data()) == hipSuccess);
  }
  // Negative Scenario 3: cuMask = nullptr
  SECTION("cuMask is nullptr") {
    REQUIRE_FALSE(hipExtStreamCreateWithCUMask(&stream,
                  defaultCUMask.size(),
                  nullptr) == hipSuccess);
  }
}
