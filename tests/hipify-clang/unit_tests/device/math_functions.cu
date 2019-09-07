// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
// Test to warn only on device functions umin and umax as unsupported, but not on user defined ones.
// ToDo: change lit testing in order to parse the output.

#define LEN 1024
#define SIZE LEN * sizeof(float)
// CHECK: #include <hip/hip_runtime.h>
#include <algorithm>

namespace my {
  unsigned int umin(unsigned int arg1, unsigned int arg2) {
    return (arg1 < arg2) ? arg1 : arg2;
  }
  unsigned int umax(unsigned int arg1, unsigned int arg2) {
    return (arg1 > arg2) ? arg1 : arg2;
  }
}

__global__ void uint_arithm(float* A, float* B, float* C, unsigned int u1, unsigned int u2)
{
  unsigned int _umin = umin(u1, u2);
  unsigned int _umax = umax(u1, u2);
  int i = threadIdx.x;
  A[i] = i + _umin;
  B[i] = i + _umax;
  C[i] = A[i] + B[i];
}

int main() {
  unsigned int u1 = 33;
  unsigned int u2 = 34;
  unsigned int _min = my::umin(u1, u2);
  unsigned int _max = my::umax(u1, u2);
  float *A, *B, *C;
  // CHECK: hipMalloc((void**)&A, SIZE);
  cudaMalloc((void**)&A, SIZE);
  // CHECK: hipMalloc((void**)&B, SIZE);
  cudaMalloc((void**)&B, SIZE);
  // CHECK: hipMalloc((void**)&C, SIZE);
  cudaMalloc((void**)&C, SIZE);
  dim3 dimGrid(LEN / 512, 1, 1);
  dim3 dimBlock(512, 1, 1);
  // CHECK: hipLaunchKernelGGL(uint_arithm, dim3(dimGrid), dim3(dimBlock), 0, 0, A, B, C, u1, u2);
  uint_arithm<<<dimGrid, dimBlock>>>(A, B, C, u1, u2);
  return _min < _max;
}
