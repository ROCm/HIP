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

// Testcase Description: This test case is used to check the behaviour of HIP
// when multiple hipStreaAddCallback() are called over multiple Threads
// This test case is disabled currently.

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11 EXCLUDE_HIP_PLATFORM all
 * TEST: %t
 * HIT_END
 */



#include <stdio.h>
#include <thread>
#include <chrono>
#include <atomic>
#include "hip/hip_runtime.h"
#include "test_common.h"

#ifdef __HIP_PLATFORM_AMD__
#define HIPRT_CB
#endif

#define NUM_THREADS 2000

size_t Num = 4096;
std::atomic<size_t>Cb_count{0}, Data_mismatch{0};
hipStream_t mystream;
float *A_h, *C_h;

__global__ void vector_square(float* C_d, float* A_d, size_t Num) {
  size_t gputhread = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = gputhread; i < Num; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
  }

  // Delay thread 1 only in the GPU
  if (gputhread == 1) {
    unsigned long long int wait_t = 3200000000, start = clock64(), cur;
    do {
      cur = clock64() - start;
    } while (cur < wait_t);
  }
}


static void HIPRT_CB Thread1_Callback(hipStream_t stream, hipError_t status,
                                      void* userData) {
  for (size_t i = 0; i < Num; i++) {
    // Validate the data and update Data_mismatch
    if (C_h[i] != A_h[i] * A_h[i]) {
      Data_mismatch++;
    }
  }

  // Increment the Cb_count to indicate that the callback is processed.
  ++Cb_count;
}

static void HIPRT_CB Thread2_Callback(hipStream_t stream, hipError_t status,
                                      void* userData) {
  for (size_t i = 0; i < Num; i++) {
    // Validate the data and update Data_mismatch
    if (C_h[i] != A_h[i] * A_h[i]) {
      Data_mismatch++;
    }
  }

  // Increment the Cb_count to indicate that the callback is processed.
  ++Cb_count;
}

void Thread1_func() {
  HIPCHECK(hipStreamAddCallback(mystream, Thread1_Callback, NULL, 0));
}

void Thread2_func() {
  HIPCHECK(hipStreamAddCallback(mystream, Thread2_Callback, NULL, 0));
}


int main(int argc, char* argv[]) {
  float *A_d, *C_d;
  size_t Nbytes = Num * sizeof(float);

  A_h = (float*)malloc(Nbytes);
  HIPCHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  C_h = (float*)malloc(Nbytes);
  HIPCHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);

  // Fill with Phi + i
  for (size_t i = 0; i < Num; i++) {
    A_h[i] = 1.618f + i;
  }

  HIPCHECK(hipMalloc(&A_d, Nbytes));
  HIPCHECK(hipMalloc(&C_d, Nbytes));

  HIPCHECK(hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));

  HIPCHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, mystream));

  const unsigned threadsPerBlock = 256;
  const unsigned blocks = (Num+255)/threadsPerBlock;

  hipLaunchKernelGGL((vector_square), dim3(blocks), dim3(threadsPerBlock), 0,
                      mystream, C_d, A_d, Num);

  HIPCHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, mystream));

  auto thread_count = getHostThreadCount(200, NUM_THREADS);
  if (thread_count == 0) {
    failed("Thread count is 0");
  }
  std::thread *T = new std::thread[thread_count];
  for (int i = 0; i < thread_count; i++) {
    // Use different callback for every even thread
    // The callbacks will be added to same stream from different threads
    if ((i%2) == 0)
      T[i] = std::thread(Thread1_func);
    else
      T[i] = std::thread(Thread2_func);
  }

  // Wait until all the threads finish their execution
  for (int i = 0; i < thread_count; i++) {
    T[i].join();
  }

  HIPCHECK(hipStreamSynchronize(mystream));
  HIPCHECK(hipStreamDestroy(mystream));

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(C_d));

  free(A_h);
  free(C_h);

  // Cb_count should match total number of callbacks added from both threads
  // Data_mismatch will be updated if there is problem in data validation
  if (Cb_count.load() != thread_count) {
     failed("All callbacks for stream did not get called!");
  } else if (Data_mismatch.load() != 0) {
     failed("Mismatch found in the result of the computation!");
  }
  delete[] T;

  passed();
}
