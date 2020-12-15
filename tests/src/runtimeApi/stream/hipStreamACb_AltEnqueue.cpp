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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11 EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */



// Testcase Description: This test case is used to verify if the callback
// function called through  hipStreamAddCallback() api completes the execution
// in order as hipStreamAddCallback() api queued in their respective streams



#include <stdio.h>
#include <vector>
#include "hip/hip_runtime.h"
#include "test_common.h"


#ifdef __HIP_PLATFORM_AMD__
#define HIPRT_CB
#endif


hipStream_t mystream1, mystream2;
size_t Num = 4096;
std::vector<int> Stream1_Order, Stream2_Order;


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

float *A_h, *C_h, *A_h1, *C_h1;

static void HIPRT_CB Callback_Stream1(hipStream_t stream, hipError_t status,
                                      void* userData) {
  for (size_t i = 0; i < Num; i++) {
      if (C_h[i] != A_h[i] * A_h[i]) {
          std::cout << "Data mismatch in stream1 at: " << i << std::endl;
      }
  }

  // Storing the int passed into this callback into Stream1_Order
  // this will help verify the order in which this Callback function
  // is called.
  Stream1_Order.push_back(*(reinterpret_cast<int*>(userData)));
  delete reinterpret_cast<int*>(userData);
}

static void HIPRT_CB Callback_Stream2(hipStream_t stream, hipError_t status,
                                      void* userData) {
  for (size_t i = 0; i < Num; i++) {
      if (C_h1[i] != A_h1[i] * A_h1[i]) {
          std::cout << "Data mismatch in stream2 at: " << i << std::endl;
      }
  }
  // Storing the int passed into this callback into Stream2_Order
  // this will help verify the order in which this Callback function
  // is called.
  Stream2_Order.push_back(*(reinterpret_cast<int*>(userData)));
  delete reinterpret_cast<int*>(userData);
}

int main(int argc, char* argv[]) {
  float *A_d, *C_d;
  size_t Nbytes = Num * sizeof(float);

  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  A_h1 = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  C_h1 = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);

  // Fill with Phi + i
  for (size_t i = 0; i < Num; i++) {
      A_h[i] = 1.618f + i;
  }
  for (size_t i = 0; i < Num; i++) {
    A_h1[i] = 1.618f + i;
  }

  HIPCHECK(hipMalloc(&A_d, Nbytes));
  HIPCHECK(hipMalloc(&C_d, Nbytes));

  HIPCHECK(hipStreamCreateWithFlags(&mystream1, hipStreamNonBlocking));
  HIPCHECK(hipStreamCreateWithFlags(&mystream2, hipStreamNonBlocking));

  HIPCHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, mystream1));

  const unsigned threadsPerBlock = 256;
  const unsigned blocks = (Num + 255)/threadsPerBlock;
  int *ptr = NULL;
  int *ptr1 = NULL;
  // Queing jobs in both mystream1/2 followed by hipStreamAddCallback
  for (int i = 1; i < 5; ++i) {
    hipLaunchKernelGGL((vector_square), dim3(blocks), dim3(threadsPerBlock),
                       0, mystream1, C_d, A_d, Num);
    HIPCHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost,
                            mystream1));
    ptr = new int;
    *ptr = i;
    HIPCHECK(hipStreamAddCallback(mystream1, Callback_Stream1,
                                  reinterpret_cast<void*>(ptr), 0));

    hipLaunchKernelGGL((vector_square), dim3(blocks), dim3(threadsPerBlock),
                       0, mystream2, C_d, A_d, Num);
    HIPCHECK(hipMemcpyAsync(C_h1, C_d, Nbytes,
                            hipMemcpyDeviceToHost, mystream2));
    ptr1 = new int;
    *ptr1 = i;
    HIPCHECK(hipStreamAddCallback(mystream2, Callback_Stream2,
                                  reinterpret_cast<void*>(ptr1), 0));
  }

  HIPCHECK(hipStreamSynchronize(mystream1));
  HIPCHECK(hipStreamSynchronize(mystream2));

  HIPCHECK(hipStreamDestroy(mystream1));
  HIPCHECK(hipStreamDestroy(mystream2));

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(C_d));
  free(A_h);
  free(C_h);
  free(A_h1);
  free(C_h1);

  // Checking if Stream1_Order has ints in sequencial order or not
  int i = 1;
  for (auto itr=Stream1_Order.begin(); itr != Stream1_Order.end(); ++itr) {
    if (*itr != i) {
      printf("hipStreamAddCallBack() did not execute in sequence");
      printf(" in first stream\n");
      failed("Unexpected behavior!");
    }
    ++i;
  }

  // Checking if Stream2_Order has ints in sequencial order or not
  i = 1;
  for (auto itr=Stream2_Order.begin(); itr != Stream2_Order.end(); ++itr) {
    if (*itr != i) {
      printf("hipStreamAddCallBack() did not execute in sequence");
      printf(" in second stream\n");
      failed("Unexpected behavior!");
    }
    ++i;
  }
  passed();
}
