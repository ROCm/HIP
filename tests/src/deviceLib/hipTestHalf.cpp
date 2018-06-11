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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * RUN: %t
 * HIT_END
 */

#include <hip/hip_fp16.h>
#include "hip/hip_runtime.h"

#include "test_common.h"

#if __HIP_ARCH_GFX803__ || __HIP_ARCH_GFX900__ || __HIP_ARCH_GFX906__

__global__
void __halfMath(bool* result, __half a) {
  result[0] = __heq(__hadd(a, __half{1}), __half{2});
  result[0] = __heq(__hadd_sat(a, __half{1}), __half{1}) && result[0];
  result[0] = __heq(__hfma(a, __half{2}, __half{3}), __half{5}) && result[0];
  result[0] =
    __heq(__hfma_sat(a, __half{2}, __half{3}), __half{1}) && result[0];
  result[0] = __heq(__hsub(a, __half{1}), __half{0}) && result[0];
  result[0] = __heq(__hsub_sat(a, __half{2}), __half{0}) && result[0];
  result[0] = __heq(__hmul(a, __half{2}), __half{2}) && result[0];
  result[0] = __heq(__hmul_sat(a, __half{2}), __half{1}) && result[0];
  result[0] = __heq(__hdiv(a, __half{2}), __half{0.5}) && result[0];
}

__device__
bool to_bool(const __half2& x)
{
  auto r = static_cast<const __half2_raw&>(x);

  return r.data.x != 0 && r.data.y != 0;
}

__global__
void __half2Math(bool* result, __half2 a) {
  result[0] =
    to_bool(__heq2(__hadd2(a, __half2{1, 1}), __half2{2, 2}));
  result[0] = to_bool(__heq2(__hadd2_sat(a, __half2{1, 1}), __half2{1, 1})) &&
    result[0];
  result[0] = to_bool(__heq2(
    __hfma2(a, __half2{2, 2}, __half2{3, 3}), __half2{5, 5})) && result[0];
  result[0] = to_bool(__heq2(
    __hfma2_sat(a, __half2{2, 2}, __half2{3, 3}), __half2{1, 1})) && result[0];
  result[0] = to_bool(__heq2(__hsub2(a, __half2{1, 1}), __half2{0, 0})) &&
    result[0];
  result[0] = to_bool(__heq2(__hsub2_sat(a, __half2{2, 2}), __half2{0, 0})) &&
    result[0];
  result[0] = to_bool(__heq2(__hmul2(a, __half2{2, 2}), __half2{2, 2})) &&
    result[0];
  result[0] = to_bool(__heq2(__hmul2_sat(a, __half2{2, 2}), __half2{1, 1})) &&
    result[0];
  result[0] = to_bool(__heq2(__h2div(a, __half2{2, 2}), __half2{0.5, 0.5})) &&
    result[0];
}

__global__
void kernel_hisnan(__half* input, int* output) {
  int tx = threadIdx.x;
  output[tx] = __hisnan(input[tx]);
}

__global__
void kernel_hisinf(__half* input, int* output) { 
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
  hipLaunchKernelGGL(
    kernel_hisnan, dim3(1), dim3(NUM_INPUTS), 0, 0, inputGPU, outputGPU);

  // copy output from device
  int* outputCPU = (int*) malloc(memsize);
  hipMemcpy(outputCPU, outputGPU, memsize, hipMemcpyDeviceToHost);

  // check output
  for (int i=0; i<NUM_INPUTS; i++) {
    if ((2 <= i) && (i <= 5)) { // inputs are nan, output should be true
      if (outputCPU[i] == 0) {
	      failed(
          "__hisnan() returned false for %f (input idx = %d)\n",
          static_cast<float>(inputCPU[i]),
          i);
      }
    }
    else { // inputs are NOT nan, output should be false
      if (outputCPU[i] != 0) {
	      failed(
          "__hisnan() returned true for %f (input idx = %d)\n",
          static_cast<float>(inputCPU[i]),
          i);
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
  hipLaunchKernelGGL(
    kernel_hisinf, dim3(1), dim3(NUM_INPUTS), 0, 0, inputGPU, outputGPU);

  // copy output from device
  int* outputCPU = (int*) malloc(memsize);
  hipMemcpy(outputCPU, outputGPU, memsize, hipMemcpyDeviceToHost);

  // check output
  for (int i=0; i<NUM_INPUTS; i++) {
    if ((0 <= i) && (i <= 1)) { // inputs are inf, output should be true
      if (outputCPU[i] == 0) {
	      failed(
          "__hisinf() returned false for %f (input idx = %d)\n",
          static_cast<float>(inputCPU[i]),
          i);
      }
    }
    else { // inputs are NOT inf, output should be false
      if (outputCPU[i] != 0) {
	      failed(
          "__hisinf() returned true for %f (input idx = %d)\n",
          static_cast<float>(inputCPU[i]),
          i);
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
  bool* result{nullptr};
  hipHostMalloc(&result, sizeof(result));

  result[0] = false;
  hipLaunchKernelGGL(
    __halfMath, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result, __half{1});
  hipDeviceSynchronize();

  if (!result[0]) { failed("Failed __half tests."); }

  result[0] = false;
  hipLaunchKernelGGL(
    __half2Math, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result, __half2{1, 1});
  hipDeviceSynchronize();

  if (!result[0]) { failed("Failed __half2 tests."); }

  hipHostFree(result);

  // run some functional checks
  checkFunctional();

  passed();
}
