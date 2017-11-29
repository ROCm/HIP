/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM hcc
 * RUN: %t EXCLUDE_HIP_PLATFORM nvcc
 * HIT_END
 */

 #include "hip/hip_runtime.h"
 #include "test_common.h"

#define LEN 16*1024
#define SIZE LEN*4

__global__ void vectorAdd(hipLaunchParm lp, float *Ad, float *Bd) {
  HIP_DYNAMIC_SHARED(float, sBd);
  int tx = hipThreadIdx_x;
  for(int i=0;i<LEN/64;i++) {
    sBd[tx + i * 64] = Ad[tx + i * 64] + 1.0f;
    Bd[tx + i * 64] = sBd[tx + i * 64];
  }
}

int main() {
  float *A, *B, *Ad, *Bd;
  A = new float[LEN];
  B = new float[LEN];
  for(int i=0;i<LEN;i++) {
    A[i] = 1.0f;
    B[i] = 1.0f;
  }
  hipMalloc(&Ad, SIZE);
  hipMalloc(&Bd, SIZE);
  hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
  hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice);
  hipLaunchKernel(vectorAdd, dim3(1,1,1), dim3(64,1,1), SIZE, 0, Ad, Bd);
  hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost);
  for(int i=0;i<LEN;i++) {
    assert(B[i] > 1.0f && B[i] < 3.0f);
  }
  passed();
}
