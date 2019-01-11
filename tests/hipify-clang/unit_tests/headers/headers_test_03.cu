// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #pragma once
// CHECK-NEXT: #include <hip/hip_runtime.h>
#pragma once
// CHECK-NOT: #include <hip/hip_runtime.h>
int main(int argc, char* argv[]) {
  return 0;
}

