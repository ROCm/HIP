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

#include <iostream>
#include <hip/hip_fp16.h>
#include "hip/hip_runtime_api.h"

#define LEN 64
#define HALF_SIZE 64*sizeof(__half)
#define HALF2_SIZE 64*sizeof(__half2)

#if __HIP_ARCH_GFX803__ > 0

__global__ void __halfMath(hipLaunchParm lp, __half *A, __half *B, __half *C) {
  int tx = hipThreadIdx_x;
  __half a = A[tx];
  __half b = B[tx];
  __half c = C[tx];
  c = __hadd(a, c);
  c = __hadd_sat(b, c);
  c = __hfma(a, c, b);
  c = __hfma_sat(b, c, a);
  c = __hsub(a, c);
  c = __hsub_sat(b, c);
  c = __hmul(a, c);
  c = __hmul_sat(b, c);
  c = hdiv(a, c);
}

__global__ void __half2Math(hipLaunchParm lp, __half2 *A, __half2 *B, __half2 *C) {
  int tx = hipThreadIdx_x;
  __half2 a = A[tx];
  __half2 b = B[tx];
  __half2 c = C[tx];
  c = __hadd2(a, c);
  c = __hadd2_sat(b, c);
  c = __hfma2(a, c, b);
  c = __hfma2_sat(b, c, a);
  c = __hsub2(a, c);
  c = __hsub2_sat(b, c);
  c = __hmul2(a, c);
  c = __hmul2_sat(b, c);
}

#endif

int main(){
  __half *A, *B, *C;
  hipMalloc(&A, HALF_SIZE);
  hipMalloc(&B, HALF_SIZE);
  hipMalloc(&C, HALF_SIZE);
  hipLaunchKernel(__halfMath, dim3(1,1,1), dim3(LEN,1,1), 0, 0, A, B, C);
  __half2 *A2, *B2, *C2;
  hipMalloc(&A, HALF2_SIZE);
  hipMalloc(&B, HALF2_SIZE);
  hipMalloc(&C, HALF2_SIZE);
  hipLaunchKernel(__half2Math, dim3(1,1,1), dim3(LEN,1,1), 0, 0, A2, B2, C2);

}
