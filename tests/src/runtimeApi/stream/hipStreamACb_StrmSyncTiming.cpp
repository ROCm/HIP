/*
* Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
* IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

// Testcase Description: This test case checks whether hipStreamSynchronize()
// is taking less time than the time taken by Callback() function launched
// by hipStreamAddCallback() api.

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11 EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <atomic>
#include "hip/hip_runtime.h"
#include "test_common.h"

#ifdef __HIP_PLATFORM_AMD__
#define HIPRT_CB
#endif

#define SECONDS_TO_WAIT 5
#define TO_MICROSECONDS 1000000

hipStream_t mystream;
size_t N_elmts = 4096;
bool Init_callback = false;
std::atomic<int> Data_mismatch{0};

__global__ void vector_square(float* C_d, float* A_d, size_t N_elmts) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N_elmts; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
  }

  // Delay the thread 1
  if (offset == 1) {
    unsigned long long int wait_t = 3200000000, start = clock64(), cur;
    do {
      cur = clock64() - start;
    } while (cur < wait_t);
  }
}

float *A_h, *C_h;

static void HIPRT_CB Callback1(hipStream_t stream, hipError_t status,
                               void* userData) {
  // Mark that the callback is entered.  This is checked in main thread.
  Init_callback = true;

  // Validate the data
  for (size_t i = 0; i < N_elmts; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      Data_mismatch++;
    }
  }

  // Delay the callback completion
  sleep(SECONDS_TO_WAIT);
}

bool rangedCompare(long a, long b) {
  auto diff = b - a;
  if (diff < 0) diff *= -1;
  if (diff < 500) return true;
  return false;
}


int main(int argc, char* argv[]) {
  float *A_d, *C_d;
  size_t Nbytes = N_elmts * sizeof(float);
  float tElapsed = 1.0f;

  A_h = (float*)malloc(Nbytes);
  HIPCHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  C_h = (float*)malloc(Nbytes);
  HIPCHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);

  // Fill with Phi + i
  for (size_t i = 0; i < N_elmts; i++) {
    A_h[i] = 1.618f + i;
  }

  HIPCHECK(hipMalloc(&A_d, Nbytes));
  HIPCHECK(hipMalloc(&C_d, Nbytes));

  HIPCHECK(hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));

  HIPCHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, mystream));

  const unsigned threadsPerBlock = 256;
  const unsigned blocks = (N_elmts + 255)/threadsPerBlock;

  hipLaunchKernelGGL((vector_square), dim3(blocks), dim3(threadsPerBlock), 0,
                      mystream, C_d, A_d, N_elmts);
  HIPCHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, mystream));
  HIPCHECK(hipStreamAddCallback(mystream, Callback1, NULL, 0));

  // Wait untill Callback() function changes the Init_callback value to true
  while (!Init_callback) {}

  // Since the callback is supposed to be called only after an implicit stream
  // synchronization, hipStreamSynchronize call shoud not take much time.
  auto start = std::chrono::high_resolution_clock::now();
  HIPCHECK(hipStreamSynchronize(mystream));
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  HIPCHECK(hipStreamDestroy(mystream));
  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(C_d));
  free(A_h);
  free(C_h);

  if (Data_mismatch.load() != 0) {
    failed("Output from kernel execution is not as expected");
  }

  // There is a delay of 5000000 microseconds in the Callback() function, the
  // duration.count() value is expected to less than 5000000 microseconds
  // because it is expected that stream synchronization completed the moment
  // Callback function starts the execution and not untill Callback function
  // completes the execution. Therefore the hipStreamSynchronize() in the
  // main thread should hardly take any time to complete.

  if ((duration.count() < (SECONDS_TO_WAIT * TO_MICROSECONDS)) ||
      (rangedCompare(duration.count(), SECONDS_TO_WAIT * TO_MICROSECONDS))) {
    passed();
  } else {
    failed("hipStreamSynchronize is waiting untill Callback() completes.");
  }
}
