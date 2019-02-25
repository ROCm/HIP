// RUN: %run_test hipify "%s" "%t" %hipify_args "-roc" %clang_args

// NOTE: Nonworking code just for conversion testing

// CHECK: #include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// CHECK: #include "caffe2/operators/hip/spatial_batch_norm_op_miopen.hip"
#include "caffe2/operators/spatial_batch_norm_op.h"
// CHECK: #include "caffe2/core/hip/common_miopen.h"
#include "caffe2/core/common_cudnn.h"
