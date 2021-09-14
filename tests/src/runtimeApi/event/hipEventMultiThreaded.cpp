/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include <thread>
#include <vector>

#define THREADS 2  // threads per core
#define MAX_NUM_THREADS 512
#define ITER 5  // total loop number

// 5 loops and 2 threads per core are enough for function verification.
// You may adjust them for your test purpose.

extern "C" __global__ void WaitKernel(int *Ad, int clockrate) {
  uint64_t wait_t = 500,
  start = clock64()/clockrate, cycles;
  do { cycles = clock64()/clockrate-start;} while (cycles < wait_t);
  *Ad = 1;
}

void t1(hipEvent_t start, hipStream_t stream1, int clkRate, int *A, int *Ad) {
  *A = 0;

  hipLaunchKernelGGL(HIP_KERNEL_NAME(WaitKernel), dim3(1), dim3(1), 0, stream1, Ad, clkRate);

  HIPCHECK(hipEventRecord(start, stream1));

}

int main(int argc, char* argv[]) {

  int NUM_THREADS = min(THREADS * std::thread::hardware_concurrency(), MAX_NUM_THREADS);
  int clkRate = 0;
  std::vector<int *> A, Ad;
  bool TestPassed = true;

  for (int i = 0; i < NUM_THREADS; i++) {
    int *aPtr, *adPtr;
    aPtr = (int *)malloc(sizeof(int));
    A.push_back(aPtr);
    Ad.push_back(adPtr);
    HIPCHECK(hipHostRegister(A[i], sizeof(int), 0));
    HIPCHECK(hipHostGetDevicePointer((void**)&Ad[i], A[i], 0));
  }

  HIPCHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));
  hipStream_t stream1;
  hipStreamCreate(&stream1);
  hipEvent_t start;
  hipEventCreate(&start);
  std::thread t[NUM_THREADS];

  printf("NUM_THREADS=%d\n", NUM_THREADS);
  for (int i = 0; i < ITER; i++) {
    printf("loop %d/%d\n", i, ITER);
    for (int j = 0; j < NUM_THREADS; j++) {
       t[j] = std::thread(t1, start, stream1, clkRate, A[j], Ad[j]);
    }

    for (int j = 0 ; j < NUM_THREADS; j++) {
      t[j].join();
    }

    HIPCHECK(hipStreamWaitEvent(stream1, start, 0));
    hipError_t err = hipEventQuery(start);
    while(err != hipSuccess) {
      err = hipEventQuery(start);
    }

    for (int j = 0; j < NUM_THREADS; j++) {
      if (*A[j] != 1) {
        TestPassed = false;
        break;
      }
    }

    if (!TestPassed) {
      failed("Test Failed due to possible race condition!");
    }
  }

  HIPCHECK(hipStreamDestroy(stream1));
  HIPCHECK(hipEventDestroy(start));

  for (auto ptr: A) {
    free(ptr);
  }

  A.clear();
  Ad.clear();

  passed();
}
