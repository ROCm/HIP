/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include"hip/hip_runtime.h"
__device__ int deviceGlobal = 1;

extern "C" __global__ void matmulK(int clockrate, int* A, int* B, int* C,
                                   int N) {
  int ROW = blockIdx.y*blockDim.y+threadIdx.y;
  int COL = blockIdx.x*blockDim.x+threadIdx.x;
  int tmpSum = 0;
  if ((ROW < N) && (COL < N)) {
    // each thread computes one element of the block sub-matrix
    for (int i = 0; i < N; i++) {
      tmpSum += A[ROW * N + i] * B[i * N + COL];
    }
    C[ROW * N + COL] = tmpSum;
  }
}

extern "C" __global__ void KernelandExtraParams(int* A, int* B, int* C,
  int *D, int N) {
  int ROW = blockIdx.y*blockDim.y+threadIdx.y;
  int COL = blockIdx.x*blockDim.x+threadIdx.x;
  int tmpSum = 0;
  if (ROW < N && COL < N) {
    // each thread computes one element of the block sub-matrix
    for (int i = 0; i < N; i++) {
      tmpSum += A[ROW * N + i] * B[i * N + COL];
    }
  }
  C[ROW * N + COL] = tmpSum;
  D[ROW * N + COL] = tmpSum;
}

extern "C" __global__ void SixteenSecKernel(int clockrate) {
  uint64_t wait_t = 16000,
  start = clock64()/clockrate, cur;
  do { cur = clock64()/clockrate-start;}while (cur < wait_t);
}

extern "C" __global__ void TwoSecKernel(int clockrate) {
  if (deviceGlobal == 0x2222) {
    deviceGlobal = 0x3333;
  }
  uint64_t wait_t = 2000,
  start = clock64()/clockrate, cur;
  do { cur = clock64()/clockrate-start;}while (cur < wait_t);
  if (deviceGlobal != 0x3333) {
    deviceGlobal = 0x5555;
  }
}

extern "C" __global__ void FourSecKernel(int clockrate) {
  if (deviceGlobal == 1) {
    deviceGlobal = 0x2222;
  }
  uint64_t wait_t = 4000,
  start = clock64()/clockrate, cur;
  do { cur = clock64()/clockrate-start;}while (cur < wait_t);
  if (deviceGlobal == 0x2222) {
    deviceGlobal = 0x4444;
  }
}

