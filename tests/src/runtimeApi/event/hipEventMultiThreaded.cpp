/*
Copyright (c) 2015-2021 Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include <thread>
#include <unistd.h>
#include <atomic>
int *A, *B, *Ad, *Bd;
std::atomic<bool> signal { false };

extern "C" __global__ void WaitKernel(int *Ad, int clockrate) {
  uint64_t wait_t = 500,
  start = clock64()/clockrate, cycles;
  do { cycles = clock64()/clockrate-start;} while (cycles < wait_t);
  *Ad = 1;
}

void thread1(hipEvent_t start, hipStream_t stream1, int clkRate) {
  *B = 0;

  hipLaunchKernelGGL(HIP_KERNEL_NAME(WaitKernel), dim3(1), dim3(1), 0, stream1, Bd, clkRate);

  HIPCHECK(hipEventRecord(start, stream1));

}

void thread2(hipEvent_t start, hipStream_t stream1, int clkRate) {
  *A = 0;

  hipLaunchKernelGGL(HIP_KERNEL_NAME(WaitKernel), dim3(1), dim3(1), 0, stream1, Ad, clkRate);

  HIPCHECK(hipEventRecord(start, stream1));
}

int main(int argc, char* argv[]) {
  int clkRate = 0;
  A = (int *)malloc(sizeof(int));
  Ad = (int *)malloc(sizeof(int));
  B = (int *)malloc(sizeof(int));
  Bd = (int *)malloc(sizeof(int));
  HIPCHECK(hipHostRegister(A, sizeof(int), 0));
  HIPCHECK(hipHostGetDevicePointer((void**)&Ad, A, 0));
  HIPCHECK(hipHostRegister(B, sizeof(int), 0));
  HIPCHECK(hipHostGetDevicePointer((void**)&Bd, B, 0));
  HIPCHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));
  hipStream_t stream1;
  hipStreamCreate(&stream1);
  hipEvent_t start;
  hipEventCreate(&start);

  for (unsigned i = 0; i < 1000; i++) {
    std::thread t1(thread1, start, stream1, clkRate);
    std::thread t2(thread2, start, stream1, clkRate);

    t1.join();
    t2.join();

    HIPCHECK(hipStreamWaitEvent(stream1, start, 0));
    hipError_t err = hipEventQuery(start);
    while(err != hipSuccess) {
      err = hipEventQuery(start);
    }

    if (*A == 1 && *B == 1) {
      continue;
    }
    else {
      failed("Test Failed due to race condition");
    }
  }
  passed();
}
