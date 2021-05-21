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


template <typename T> __global__ void memsetReverse(T* C_d, T val, int64_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
    C_d[i] = val;
  }
}
}  // namespace HipTest