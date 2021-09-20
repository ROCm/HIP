/*
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
*/

/**
 Testcase Scenarios :
 1) Order of execution of device kernel and hipMemset2DAsync api
 2) hipMemSet2DAsync execution in multiple threads
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_helper.hh>
#include <hip_test_kernels.hh>


/* Defines */
#define NUM_THREADS 1000
#define ITER 100
#define NUM_H 256
#define NUM_W 256



void queueJobsForhipMemset2DAsync(char* A_d, char* A_h, size_t pitch,
                                  size_t width, hipStream_t stream) {
  constexpr int memsetval = 0x22;
  HIPCHECK(hipMemset2DAsync(A_d, pitch, memsetval, NUM_W, NUM_H, stream));
  HIPCHECK(hipMemcpy2DAsync(A_h, width, A_d, pitch, NUM_W, NUM_H,
                            hipMemcpyDeviceToHost, stream));
}


/**
 * Order of execution of device kernel and hipMemset2DAsync api.
 */
TEST_CASE("Unit_hipMemset2DAsync_WithKernel") {
  constexpr auto N = 4 * 1024 * 1024;
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  constexpr int memsetval = 0x22;
  char *A_d, *A_h, *B_d, *B_h, *C_d;
  size_t pitch_A, pitch_B, pitch_C;
  size_t width = NUM_W * sizeof(char);
  size_t sizeElements = width * NUM_H;
  size_t elements = NUM_W * NUM_H;
  unsigned blocks{};
  int validateCount{};
  hipStream_t stream;

  blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A,
                                                               width, NUM_H));
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&B_d), &pitch_B,
                                                               width, NUM_H));

  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);
  B_h = reinterpret_cast<char*>(malloc(sizeElements));
  REQUIRE(B_h != nullptr);
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&C_d), &pitch_C,
                                                               width, NUM_H));

  for (size_t i = 0; i < elements; i++) {
    B_h[i] = i;
  }
  HIP_CHECK(hipMemcpy2D(B_d, width, B_h, pitch_B, NUM_W, NUM_H,
                       hipMemcpyHostToDevice));
  HIP_CHECK(hipStreamCreate(&stream));


  for (size_t k = 0; k < ITER; k++) {
    hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                   dim3(threadsPerBlock), 0, stream, B_d, C_d, elements);

    HIP_CHECK(hipMemset2DAsync(C_d, pitch_C, memsetval, NUM_W, NUM_H, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy2D(A_h, width, C_d, pitch_C, NUM_W, NUM_H,
                         hipMemcpyDeviceToHost));

    for (size_t p = 0 ; p < elements ; p++) {
      if (A_h[p] == memsetval) {
        validateCount+= 1;
      }
    }
  }

  REQUIRE(static_cast<size_t>(validateCount) == (ITER * elements));

  HIP_CHECK(hipFree(A_d)); HIP_CHECK(hipFree(B_d)); HIP_CHECK(hipFree(C_d));
  free(A_h); free(B_h);
  HIP_CHECK(hipStreamDestroy(stream));
}


/**
 * hipMemSet2DAsync execution in multiple threads.
 */
TEST_CASE("Unit_hipMemset2DAsync_MultiThread") {
  constexpr auto N = 4 * 1024 * 1024;
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  constexpr auto memPerThread = 200;
  constexpr int memsetval = 0x22;
  char *A_d, *A_h, *B_d, *B_h, *C_d;
  size_t pitch_A, pitch_B, pitch_C;
  size_t width = NUM_W * sizeof(char);
  size_t sizeElements = width * NUM_H;
  size_t elements = NUM_W * NUM_H;
  unsigned blocks{};
  int validateCount{};
  hipStream_t stream;

  blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  auto thread_count = HipTest::getHostThreadCount(memPerThread, NUM_THREADS);
  if (thread_count == 0) {
    WARN("Resources not available for thread creation");
    return;
  }

  std::thread *t = new std::thread[thread_count];

  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A,
                                                               width, NUM_H));
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&B_d), &pitch_B,
                                                               width, NUM_H));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);
  B_h = reinterpret_cast<char*>(malloc(sizeElements));
  REQUIRE(B_h != nullptr);
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&C_d), &pitch_C,
                                                               width, NUM_H));

  for (size_t i = 0 ; i < elements ; i++) {
    B_h[i] = i;
  }
  HIP_CHECK(hipMemcpy2D(B_d, width, B_h, pitch_B, NUM_W, NUM_H,
                       hipMemcpyHostToDevice));
  HIP_CHECK(hipStreamCreate(&stream));

  for (int i = 0 ; i < ITER ; i++) {
    for (size_t k = 0 ; k < thread_count; k++) {
      if (k%2) {
        t[k] = std::thread(queueJobsForhipMemset2DAsync, A_d, A_h, pitch_A,
                           width, stream);
      } else {
        t[k] = std::thread(queueJobsForhipMemset2DAsync, A_d, B_h, pitch_A,
                           width, stream);
      }
    }
    for (size_t j = 0 ; j < thread_count; j++) {
      t[j].join();
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    for (size_t k = 0 ; k < elements ; k++) {
      if ((A_h[k] == memsetval) && (B_h[k] == memsetval)) {
        validateCount+= 1;
      }
    }
  }

  REQUIRE(static_cast<size_t>(validateCount) == (ITER * elements));

  HIP_CHECK(hipFree(A_d)); HIP_CHECK(hipFree(B_d)); HIP_CHECK(hipFree(C_d));
  free(A_h); free(B_h);
  HIP_CHECK(hipStreamDestroy(stream));

  delete[] t;
}
