// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include "hipblas.h"
// CHECK-NOT: #include "cublas.h"
// CHECK: #include <stdio.h>
#include "cublas_v2.h"
#include "cublas.h"
#include <stdio.h>
