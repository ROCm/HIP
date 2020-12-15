/*
 * Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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
 * */

// Testcase Description:: This test case is used to check if the runtime is ok
// when hipStreamAddCallback() is called back to back multiple calls

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11 EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */


#include <stdio.h>
#include <unistd.h>
#include <mutex>
#include <atomic>
#include "hip/hip_runtime.h"
#include "test_common.h"

#ifdef __HIP_PLATFORM_AMD__
#define HIPRT_CB
#endif

#define NUM_CALLS 1000

hipStream_t mystream;
size_t Num = 4096;
std::atomic<size_t>Cb_count{0}, Data_mismatch{0};
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

static void HIPRT_CB Stream_Callback(hipStream_t stream, hipError_t status,
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

  // Add multiple callbacks to the stream
  for (int i = 0; i< NUM_CALLS; i++) {
    HIPCHECK(hipStreamAddCallback(mystream, Stream_Callback, NULL, 0));
  }

  HIPCHECK(hipStreamSynchronize(mystream));
  HIPCHECK(hipStreamDestroy(mystream));

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(C_d));

  free(A_h);
  free(C_h);

  // Each callback would have validated the data and if any mismatch is found,
  // Data_mismatch will not have proper data.  Validate the same.
  // Cb_count should match the number of callbacks added.
  if (Data_mismatch.load() != 0) {
    failed("Mismatch found in the result of the computation!");
  } else if (Cb_count.load() != NUM_CALLS) {
    failed("All callbacks for stream did not get called!");
  }

  passed();
}
