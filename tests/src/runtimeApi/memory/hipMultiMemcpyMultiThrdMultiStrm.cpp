/*
 Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
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

// Testcase Description: This test launches multiple threads which creates a stream to deploy kernel
// and also launch hipMemcpyAsync() api on the same stream. This test case is simulate the scenario
// reported in SWDEV-181598.
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11 EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include <stdio.h>
#include <thread>
#include <atomic>
#include "hip/hip_runtime.h"
#include "test_common.h"

#define NUM_THREADS 16

size_t N_ELMTS = 1024;
size_t Nbytes = N_ELMTS * sizeof(float);
std::atomic<size_t> Thread_count { 0 };

const unsigned ThreadsPerBlock = 256;
const unsigned blocks = (N_ELMTS + 255) / ThreadsPerBlock;

__global__ void vector_square(float* C_d, float* A_d, size_t N_ELMTS) {
  size_t gputhread = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = gputhread; i < N_ELMTS; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
  }
}

void Thread_func() {
  int Data_mismatch = 0;
  float *A_h, *C_h, *A_d, *C_d, *B_d;
  A_h = (float*) malloc(Nbytes);
  HIPCHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  C_h = (float*) malloc(Nbytes);
  HIPCHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);

  // Fill with Phi + i
  for (size_t i = 0; i < N_ELMTS; i++) {
    A_h[i] = 1.618f + i;
  }

  HIPCHECK(hipMalloc(&A_d, Nbytes));
  HIPCHECK(hipMalloc(&C_d, Nbytes));
  HIPCHECK(hipMalloc(&B_d, Nbytes));
  hipStream_t mystream;
  HIPCHECK(hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));
  HIPCHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, mystream));
  hipLaunchKernelGGL((vector_square), dim3(blocks), dim3(ThreadsPerBlock), 0,
      mystream, C_d, A_d, N_ELMTS);
  HIPCHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, mystream));
  // The following hipMemcpyAsync() is called only to load stream with multiple Async calls
  HIPCHECK(hipMemcpyAsync(B_d, A_d, Nbytes, hipMemcpyDeviceToDevice, mystream));
  Thread_count++;

  HIPCHECK(hipStreamSynchronize(mystream));
  HIPCHECK(hipStreamDestroy(mystream));
  // Verifying result of the kernel computation
  for (size_t i = 0; i < N_ELMTS; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      Data_mismatch++;
    }
  }
  // Releasing resources
  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(C_d));
  HIPCHECK(hipFree(B_d));
  free(A_h);
  free(C_h);

  if (Data_mismatch != 0) {
    failed("Mismatch found in the result of the computation!");
  }
}

int main(int argc, char* argv[]) {

  std::thread T[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++) {
    T[i] = std::thread(Thread_func);
  }

  // Wait until all the threads finish their execution
  for (int i = 0; i < NUM_THREADS; i++) {
    T[i].join();
  }

  if (Thread_count.load() != NUM_THREADS) {
    failed(
        "Seems like all the  launched threads didnot complete the execution!");
  }
  passed();
}
