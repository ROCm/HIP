/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#include <hip/hip_runtime.h>

namespace HipTest {
template <typename T> __global__ void vectorADD(const T* A_d, const T* B_d, T* C_d, size_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < NELEM; i += stride) {
    C_d[i] = A_d[i] + B_d[i];
  }
}


template <typename T>
__global__ void vectorADDReverse(const T* A_d, const T* B_d, T* C_d, size_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
    C_d[i] = A_d[i] + B_d[i];
  }
}


template <typename T> __global__ void addCount(const T* A_d, T* C_d, size_t NELEM, int count) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  // Deliberately do this in an inefficient way to increase kernel runtime
  for (int i = 0; i < count; i++) {
    for (size_t i = offset; i < NELEM; i += stride) {
      C_d[i] = A_d[i] + (T)count;
    }
  }
}


template <typename T>
__global__ void addCountReverse(const T* A_d, T* C_d, int64_t NELEM, int count) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  // Deliberately do this in an inefficient way to increase kernel runtime
  for (int i = 0; i < count; i++) {
    for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
      C_d[i] = A_d[i] + (T)count;
    }
  }
}

template<typename T>
__device__ void waitKernel(uint64_t wait_sec, T clockrate) {
  uint64_t start = clock64()/clockrate, cur;
  do { cur = clock64()/clockrate-start;}while (cur < (wait_sec*1000));
}

template<typename T>
__global__ void TwoSecKernel_GlobalVar(int globalvar, int clockrate) {
  if (globalvar == 0x2222) {
    globalvar = 0x3333;
  }
  waitKernel(2, clockrate);
  if (globalvar != 0x3333) {
    globalvar = 0x5555;
  }
}

template<typename T>
__global__ void FourSecKernel_GlobalVar(int globalvar, int clockrate) {
  if (globalvar == 1) {
    globalvar = 0x2222;
  }
  waitKernel(4, clockrate);
  if (globalvar == 0x2222) {
    globalvar = 0x4444;
  }
}


template <typename T> __global__ void memsetReverse(T* C_d, T val, int64_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
    C_d[i] = val;
  }
}

template <typename T> __global__ void vector_square(const T* A_d, T* C_d, size_t N_ELMTS) {
  size_t gputhread = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = gputhread; i < N_ELMTS; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
  }
}

}  // namespace HipTest
