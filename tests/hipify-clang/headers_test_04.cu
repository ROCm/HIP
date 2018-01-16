// RUN: %run_test hipify "%s" "%t" %cuda_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NEXT: #include <stdio.h>
// CHECK-NEXT: #include <iostream>
#include <stdio.h>
#include <iostream>
// CHECK-NOT: #include <hip/hip_runtime.h>
int main(int argc, char* argv[]) {
  return 0;
}

