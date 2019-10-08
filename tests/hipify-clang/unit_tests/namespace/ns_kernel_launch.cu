// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>

__global__ void test_0() {
  int a = 10;
}

namespace first {
  __global__ void test_1() {
    int b = 20;
  }
  namespace second {
    __global__ void test_2() {
      int c = 30;
    }
  }
}

int main() {
  // CHECK: hipLaunchKernelGGL(::test_0, dim3(1), dim3(1), 0, 0);
  ::test_0<<<1, 1>>>();
  // CHECK: hipLaunchKernelGGL(first::test_1, dim3(1), dim3(1), 0, 0);
  first::test_1<<<1, 1>>>();
  // CHECK: hipLaunchKernelGGL(first::second::test_2, dim3(1), dim3(1), 0, 0);
  first::second::test_2<<<1, 1>>>();
  return 0;
}
