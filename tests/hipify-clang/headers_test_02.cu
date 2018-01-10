// RUN: %run_test hipify "%s" "%t" %cuda_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK-NOT: #include<cuda.h>
#include <cuda.h>
