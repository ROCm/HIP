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
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include <iostream>
#include <hip/hip_fp16.h>
#include "hip/hip_runtime.h"
#include "test_common.h"

#define LEN 64
#define HALF_SIZE 64 * sizeof(__half)
#define HALF2_SIZE 64 * sizeof(__half2)

#if __HIP_ARCH_GFX803__ || __HIP_ARCH_GFX900__

__global__ void __halfMath(hipLaunchParm lp, __half* A, __half* B, __half* C) {
    int tx = threadIdx.x;
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

__global__ void __half2Math(hipLaunchParm lp, __half2* A, __half2* B, __half2* C) {
    int tx = threadIdx.x;
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

__global__ void kernel_hisnan(hipLaunchParm lp, __half* input, int* output) {
    int tx = threadIdx.x;
    output[tx] = __hisnan(input[tx]);
}

__global__ void kernel_hisinf(hipLaunchParm lp, __half* input, int* output) {
    int tx = threadIdx.x;
    output[tx] = __hisinf(input[tx]);
}

#endif


__half host_ushort_as_half(unsigned short s) {
  union {__half h; unsigned short s; } converter;
  converter.s = s;
  return converter.h;
}


void check_hisnan(int NUM_INPUTS, __half* inputCPU, __half* inputGPU) {

  // allocate memory
  auto memsize = NUM_INPUTS * sizeof(int);
  int* outputGPU = nullptr;
  hipMalloc((void**)&outputGPU, memsize);

  // launch the kernel
  hipLaunchKernel(kernel_hisnan, dim3(1), dim3(NUM_INPUTS), 0, 0, inputGPU, outputGPU);

  // copy output from device
  int* outputCPU = (int*) malloc(memsize);
  hipMemcpy(outputCPU, outputGPU, memsize, hipMemcpyDeviceToHost);

  // check output
  for (int i=0; i<NUM_INPUTS; i++) {
    if ((2 <= i) && (i <= 5)) { // inputs are nan, output should be true
      if (outputCPU[i] == 0) {
	failed("__hisnan() returned false for %f (input idx = %d)\n", inputCPU[i], i);
      }
    }
    else { // inputs are NOT nan, output should be false
      if (outputCPU[i] != 0) {
	failed("__hisnan() returned true for %f (input idx = %d)\n", inputCPU[i], i);
      }
    }
  }

  // free memory
  free(outputCPU);
  hipFree(outputGPU);

  // done
  return;
}


void check_hisinf(int NUM_INPUTS, __half* inputCPU, __half* inputGPU) {
  // allocate memory
  auto memsize = NUM_INPUTS * sizeof(int);
  int* outputGPU = nullptr;
  hipMalloc((void**)&outputGPU, memsize);

  // launch the kernel
  hipLaunchKernel(kernel_hisinf, dim3(1), dim3(NUM_INPUTS), 0, 0, inputGPU, outputGPU);

  // copy output from device
  int* outputCPU = (int*) malloc(memsize);
  hipMemcpy(outputCPU, outputGPU, memsize, hipMemcpyDeviceToHost);

  // check output
  for (int i=0; i<NUM_INPUTS; i++) {
    if ((0 <= i) && (i <= 1)) { // inputs are inf, output should be true
      if (outputCPU[i] == 0) {
	failed("__hisinf() returned false for %f (input idx = %d)\n", inputCPU[i], i);
      }
    }
    else { // inputs are NOT inf, output should be false
      if (outputCPU[i] != 0) {
	failed("__hisinf() returned true for %f (input idx = %d)\n", inputCPU[i], i);
      }
    }
  }

  // free memory
  free(outputCPU);
  hipFree(outputGPU);

  // done
  return;
}


void checkFunctional() {

  // allocate memory 
  const int NUM_INPUTS = 16;
  auto memsize = NUM_INPUTS * sizeof(__half);
  __half* inputCPU = (__half*) malloc(memsize);
  
  // populate inputs
  inputCPU[0] = host_ushort_as_half(0x7c00);  //  inf
  inputCPU[1] = host_ushort_as_half(0xfc00);  // -inf
  inputCPU[2] = host_ushort_as_half(0x7c01);  //  nan
  inputCPU[3] = host_ushort_as_half(0x7e00);  //  nan
  inputCPU[4] = host_ushort_as_half(0xfc01);  //  nan
  inputCPU[5] = host_ushort_as_half(0xfe00);  //  nan
  inputCPU[6] = host_ushort_as_half(0x0000);  //  0
  inputCPU[7] = host_ushort_as_half(0x8000);  // -0
  inputCPU[8] = host_ushort_as_half(0x7bff);  // max +ve normal
  inputCPU[9] = host_ushort_as_half(0xfbff);  // max -ve normal
  inputCPU[10] = host_ushort_as_half(0x0400); // min +ve normal
  inputCPU[11] = host_ushort_as_half(0x8400); // min -ve normal
  inputCPU[12] = host_ushort_as_half(0x03ff); // max +ve sub-normal
  inputCPU[13] = host_ushort_as_half(0x83ff); // max -ve sub-normal
  inputCPU[14] = host_ushort_as_half(0x0001); // min +ve sub-normal
  inputCPU[15] = host_ushort_as_half(0x8001); // min -ve sub-normal

  // copy inputs to the GPU
  __half* inputGPU = nullptr;
  hipMalloc((void**)&inputGPU, memsize);
  hipMemcpy(inputGPU, inputCPU, memsize, hipMemcpyHostToDevice);

  // run checks

  check_hisnan(NUM_INPUTS, inputCPU, inputGPU);

  check_hisinf(NUM_INPUTS, inputCPU, inputGPU);

  // free memory
  hipFree(inputGPU);
  free(inputCPU);

  // all done
  return;
}

int main() {
    __half *A, *B, *C;
    hipMalloc(&A, HALF_SIZE);
    hipMalloc(&B, HALF_SIZE);
    hipMalloc(&C, HALF_SIZE);
    hipLaunchKernel(__halfMath, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, A, B, C);
    hipFree(A);
    hipFree(B);
    hipFree(C);
    __half2 *A2, *B2, *C2;
    hipMalloc(&A2, HALF2_SIZE);
    hipMalloc(&B2, HALF2_SIZE);
    hipMalloc(&C2, HALF2_SIZE);
    hipLaunchKernel(__half2Math, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, A2, B2, C2);
    hipFree(A2);
    hipFree(B2);
    hipFree(C2);

    // run some functional checks
    checkFunctional();
    
    passed();
}
