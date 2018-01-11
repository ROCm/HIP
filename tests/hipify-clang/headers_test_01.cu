// RUN: %run_test hipify "%s" "%t" %cuda_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
// CHECK-NOT: #include<cuda_runtime.h>
#include <cuda_runtime.h>
