// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
// CHECK: #include <hip/hip_runtime.h>

#include <cuda.h>

__global__ void axpy_kernel(float a, float* x, float* y) {
  y[threadIdx.x] = a * x[threadIdx.x];
}

void axpy(float a, float* x, float* y) {

#ifdef SOME_MACRO
  // CHECK: hipLaunchKernelGGL(axpy_kernel, dim3(1), dim3(1), 0, 0, a, y, x);
  axpy_kernel <<<1, 1>>> (a, y, x);
#endif

#ifndef SOME_MACRO
  // CHECK: hipLaunchKernelGGL(axpy_kernel, dim3(1), dim3(2), 0, 0, a, y, x);
  axpy_kernel <<<1, 2>>> (a, y, x);
#endif

#ifdef SOME_MACRO
  // CHECK: hipLaunchKernelGGL(axpy_kernel, dim3(1), dim3(3), 0, 0, a, y, x);
  axpy_kernel <<<1, 3>>> (a, y, x);
#else
  // CHECK: hipLaunchKernelGGL(axpy_kernel, dim3(1), dim3(4), 0, 0, a, x, y);
  axpy_kernel <<<1, 4>>> (a, x, y);
#endif

}