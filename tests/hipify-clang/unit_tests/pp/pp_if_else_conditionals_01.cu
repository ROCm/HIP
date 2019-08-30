// RUN: %run_test hipify "%s" "%t" %hipify_args "--skip-excluded-preprocessor-conditional-blocks" %clang_args
// CHECK: #include <hip/hip_runtime.h>

__global__ void axpy_kernel(float a, float* x, float* y) {
  y[threadIdx.x] = a * x[threadIdx.x];
}

void axpy(float a, float* x, float* y) {
float* y_new = nullptr;
#ifdef SOME_MACRO
  y_new = x;
  // CHECK: axpy_kernel <<<1, 1>>> (a, y_new, x);
  axpy_kernel <<<1, 1>>> (a, y_new, x);
#endif

#ifndef SOME_MACRO
  y_new = y;
  // CHECK: hipLaunchKernelGGL(axpy_kernel, dim3(1), dim3(2), 0, 0, a, y_new, x);
  axpy_kernel <<<1, 2>>> (a, y_new, x);
#endif

#ifdef SOME_MACRO
  // CHECK: axpy_kernel <<<1, 3>>> (a, y, x);
  axpy_kernel <<<1, 3>>> (a, y, x);
#else
  // CHECK: hipLaunchKernelGGL(axpy_kernel, dim3(1), dim3(4), 0, 0, a, x, y);
  axpy_kernel <<<1, 4>>> (a, x, y);
#endif

#ifdef SOME_MACRO
  // CHECK: axpy_kernel <<<1, 5>>> (a, y, x);
  axpy_kernel <<<1, 5>>> (a, y, x);
#elif defined SOME_MACRO_1
  // CHECK: axpy_kernel <<<1, 6>>> (a, x, y);
  axpy_kernel <<<1, 6>>> (a, x, y);
#else
  // CHECK: hipLaunchKernelGGL(axpy_kernel, dim3(1), dim3(7), 0, 0, a, x, y);
  axpy_kernel <<<1, 7>>> (a, x, y);
#endif

#ifndef SOME_MACRO
  // CHECK: hipLaunchKernelGGL(axpy_kernel, dim3(1), dim3(8), 0, 0, a, y, x);
  axpy_kernel <<<1, 8>>> (a, y, x);
#elif !defined(SOME_MACRO_1)
  // CHECK: axpy_kernel <<<1, 9>>> (a, x, y);
  axpy_kernel <<<1, 9>>> (a, x, y);
#else
  // CHECK: axpy_kernel <<<1, 10>>> (a, x, y);
  axpy_kernel <<<1, 10>>> (a, x, y);
#endif

}