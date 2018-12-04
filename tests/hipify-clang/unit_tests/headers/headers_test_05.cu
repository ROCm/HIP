// RUN: %run_test hipify "%s" "%t" %cuda_args

// CHECK: #pragma once
// CHECK-NEXT: #include <hip/hip_runtime.h>
#pragma once
// CHECK-NOT: #include <hip/hip_runtime.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
  return 0;
}

