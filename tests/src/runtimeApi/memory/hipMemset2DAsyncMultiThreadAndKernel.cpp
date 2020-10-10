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
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
*/

//
// Test to verify
// a) Order of execution of device kernel and hipMemset2DAsync api
// b) hipMemSet2DAsync execution in multiple threads
//

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#define NUM_THREADS 1000
#define ITER 100
#define NUM_H 256
#define NUM_W 256

unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
hipStream_t stream;

bool testResult = true;
char *A_d, *A_h, *B_d, *B_h, *C_d;
int validateCount;

size_t pitch_A, pitch_B, pitch_C;
size_t width = NUM_W * sizeof(char);
size_t sizeElements = width * NUM_H;
size_t elements = NUM_W * NUM_H;

/*
 * Square each element in the array B and write to array C.
 */

__global__ void
vector_square(char* B_d, char* C_d, size_t elements) {
  for (int i=0 ; i < elements ; i++) {
    C_d[i] = B_d[i] * B_d[i];
  }
}

void memAllocate() {
  HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, NUM_H));
  HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&B_d), &pitch_B, width, NUM_H));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  HIPASSERT(A_h != NULL);
  B_h = reinterpret_cast<char*>(malloc(sizeElements));
  HIPASSERT(B_h != NULL);
  HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&C_d), &pitch_C, width, NUM_H));

  for (int i = 0 ; i < elements ; i++) {
    B_h[i] = i;
  }
  HIPCHECK(hipMemcpy2D(B_d, width, B_h, pitch_B, NUM_W, NUM_H,
                       hipMemcpyHostToDevice));
  HIPCHECK(hipStreamCreate(&stream));
}

void memDeallocate() {
  HIPCHECK(hipFree(A_d)); HIPCHECK(hipFree(B_d)); HIPCHECK(hipFree(C_d));
  free(A_h); free(B_h);
  HIPCHECK(hipStreamDestroy(stream));
}

void queueJobsForhipMemset2DAsync(char* A_d, char* A_h, size_t pitch,
                                  size_t width) {
  HIPCHECK(hipMemset2DAsync(A_d, pitch, memsetval, NUM_W, NUM_H, stream));
  HIPCHECK(hipMemcpy2DAsync(A_h, width, A_d, pitch, NUM_W, NUM_H,
                            hipMemcpyDeviceToHost, stream));
}

bool testhipMemset2DAsyncWithKernel() {
  validateCount = 0;
  memAllocate();
  printf("info: Launching vector_square kernel and hipMemset2DAsync "
         "simultaneously\n");
  for (int k = 0 ; k < ITER ; k++) {
    hipLaunchKernelGGL(vector_square, dim3(blocks), dim3(threadsPerBlock), 0,
                       stream, B_d, C_d, elements);

    HIPCHECK(hipMemset2DAsync(C_d, pitch_C, memsetval, NUM_W, NUM_H, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipMemcpy2D(A_h, width, C_d, pitch_C, NUM_W, NUM_H,
                         hipMemcpyDeviceToHost));

    for (int p = 0 ; p < elements ; p++) {
      if (A_h[p] == memsetval) {
        validateCount+= 1;
      }
    }
  }

  testResult = (validateCount == (ITER * elements)) ? true : false;
  memDeallocate();
  return testResult;
}

bool testhipMemset2DAsyncMultiThread() {
  validateCount = 0;
  std::thread t[NUM_THREADS];

  memAllocate();

  printf("info: Queueing up hipMemset2DAsync jobs over multiple threads\n");
  for (int i = 0 ; i < ITER ; i++) {
    for (int k = 0 ; k < NUM_THREADS ; k++) {
      if (k%2) {
        t[k] = std::thread(queueJobsForhipMemset2DAsync, A_d, A_h, pitch_A,
                           width);
      } else {
          t[k] = std::thread(queueJobsForhipMemset2DAsync, A_d, B_h, pitch_A,
                             width);
      }
    }
    for (int j = 0 ; j < NUM_THREADS ; j++) {
      t[j].join();
    }

    HIPCHECK(hipStreamSynchronize(stream));
    for (int k = 0 ; k < elements ; k++) {
      if ((A_h[k] == memsetval) && (B_h[k] == memsetval)) {
        validateCount+= 1;
      }
    }
  }
  memDeallocate();
  testResult = (validateCount == (ITER * elements)) ? true : false;
  return testResult;
}

int main() {
  bool testResult = true;

  testResult &= testhipMemset2DAsyncWithKernel();
  if (testResult) {
    printf("Kernel and hipMemset2DAsync executed in correct order!\n");
  } else {
      printf("Kernel and hipMemset2DAsync order of execution failed\n");
  }

  testResult &= testhipMemset2DAsyncMultiThread();
  if (testResult) {
    printf("hipMemset2DAsync jobs on all threads finished successfully!\n");
    passed();
  } else {
      printf("hipMemset2DAsync failed in multi thread scenario\n");
  }

  if (testResult) {
    passed();
  } else {
      failed("One or more tests failed\n");
  }
}
