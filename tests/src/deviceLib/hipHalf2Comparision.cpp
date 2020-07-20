/*
Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


/* HIT_START
 * BUILD: %t %s ../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include "hip/hip_fp16.h"

#define test_passed(test_name) \
  printf("%s %s  PASSED!%s\n", KGRN, #test_name, KNRM);

enum half2Op {
  HALF2_OP_HEQ2 = 0,
  HALF2_OP_HNE2,
  HALF2_OP_HLE2,
  HALF2_OP_HGE2,
  HALF2_OP_HLT2,
  HALF2_OP_HGT2,
  HALF2_OP_MAX
};

enum half2Test {
  HALF2_TEST_FUNCTION = 0,
  HALF2_TEST_NAN,
  HALF2_TEST_MAX
};

// Kernels for half2 comparision functions

__global__
void __half2Compare(float* result_D, __half2 a, int n, int half2Op,
            int testType) {
  size_t gputhread = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = gputhread; i < n; i += stride) {
    switch (half2Op) {
      case HALF2_OP_HEQ2:
        if (testType == HALF2_TEST_FUNCTION) {
          result_D[i] = __high2float(__heq2(__hadd2(a, __half2{1, 1}),
                                     __half2{2, 2}));
        } else {
          result_D[i] = __high2float(__heq2(__h2div(a, __half2{0, 0}),
                                     __half2{0, 0}));
        }
        break;
      case HALF2_OP_HNE2:
          result_D[i] = __high2float(__hne2(__hadd2(a, __half2{1, 1}),
                                     __half2{2, 2}));
        break;
      case HALF2_OP_HLE2:
        if (testType == HALF2_TEST_FUNCTION) {
          result_D[i] = __high2float(__hle2(__hadd2(a, __half2{1, 1}),
                                     __half2{3, 3}));
        } else {
          result_D[i] = __high2float(__hle2(__h2div(a, __half2{0, 0}),
                                     __half2{0, 0}));
        }
        break;
      case HALF2_OP_HGE2:
        if (testType == HALF2_TEST_FUNCTION) {
          result_D[i] = __high2float(__hge2(__hadd2(a, __half2{1, 1}),
                                     __half2{2, 2}));
        } else {
          result_D[i] = __high2float(__hge2(__h2div(a, __half2{0, 0}),
                                     __half2{0, 0}));
        }
        break;
      case HALF2_OP_HLT2:
        if (testType == HALF2_TEST_FUNCTION) {
          result_D[i] = __high2float(__hlt2(__hadd2(a, __half2{1, 1}),
                                     __half2{3, 3}));
        } else {
          result_D[i] = __high2float(__hlt2(__h2div(a, __half2{0, 0}),
                                     __half2{0, 0}));
        }
        break;
      case HALF2_OP_HGT2:
        if (testType == HALF2_TEST_FUNCTION) {
          result_D[i] = __high2float(__hgt2(__hadd2(a, __half2{1, 1}),
                                     __half2{3, 3}));
        } else {
          result_D[i] = __high2float(__hgt2(__h2div(a, __half2{0, 0}),
                                     __half2{0, 0}));
        }
        break;
    }
  }
}

static bool isFailed(float expectedValue, float *result_H, int size) {
  for (int index = 0; index < size; index++) {
    if (expectedValue != result_H[index]) {
      return true;
    }
  }
  return false;
}

int main() {
  const int n = 64;
  float* result_H = reinterpret_cast<float*>(malloc(n*sizeof(float)));
  float* result_D;
  bool bFunctionalTestFailed = false;
  bool bNanTestFailed = false;
  int index = 0;
  HIPCHECK(hipMalloc(&result_D, n*sizeof(float)));

  // kernel launch and hipmemcpy operation to get return value for heq2
  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{1, 1}, n, HALF2_OP_HEQ2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(1.0, result_H, n)) {
    printf("heq2: failure when arguments are equal\n");
    bFunctionalTestFailed = true;
  }

  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{2, 2}, n, HALF2_OP_HEQ2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(0.0, result_H, n)) {
    printf("heq2: failure when arguments are not equal\n");
    bFunctionalTestFailed = true;
  }

  // kernel launch and hipmemcpy operation to get return value for hne2
  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{2, 2}, n, HALF2_OP_HNE2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(1.0, result_H, n)) {
    printf("hne2: failure when arguments are not equal\n");
    bFunctionalTestFailed = true;
  }

  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{1, 1}, n, HALF2_OP_HNE2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(0.0, result_H, n)) {
    printf("hne2: failure when arguments are equal\n");
    bFunctionalTestFailed = true;
  }

  // kernel launch and hipmemcpy operation to get return value for hle2
  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{1, 1}, n, HALF2_OP_HLE2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(1.0, result_H, n)) {
    printf("hle2: failure when argument is less than equal\n");
    bFunctionalTestFailed = true;
  }

  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{2, 2}, n, HALF2_OP_HLE2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(1.0, result_H, n)) {
    printf("hle2: failure when argument is equal\n");
    bFunctionalTestFailed = true;
  }

  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{3, 3}, n, HALF2_OP_HLE2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(0.0, result_H, n)) {
    printf("hle2: failure when argument is greater\n");
    bFunctionalTestFailed = true;
  }

  // kernel launch and hipmemcpy operation to get return value for hge2
  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{2, 2}, n, HALF2_OP_HGE2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(1.0, result_H, n)) {
    printf("hge2: failure when argument is greater\n");
    bFunctionalTestFailed = true;
  }

  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{1, 1}, n, HALF2_OP_HGE2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(1.0, result_H, n)) {
    printf("hge2: failure when argument is equal\n");
    bFunctionalTestFailed = true;
  }

  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{0, 0}, n, HALF2_OP_HGE2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(0.0, result_H, n)) {
    printf("hge2: failure when argument is less\n");
    bFunctionalTestFailed = true;
  }

  // kernel launch and hipmemcpy operation to get return value for hlt2
  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{1, 1}, n, HALF2_OP_HLT2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(1.0, result_H, n)) {
    printf("hlt2: failure when argument is less\n");
    bFunctionalTestFailed = true;
  }

  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{2, 2}, n, HALF2_OP_HLT2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(0.0, result_H, n)) {
    printf("hlt2: failure when argument is equal\n");
    bFunctionalTestFailed = true;
  }

  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{3, 3}, n, HALF2_OP_HLT2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(0.0, result_H, n)) {
    printf("hlt2: failure when argument is greater\n");
    bFunctionalTestFailed = true;
  }

  // kernel launch and hipmemcpy operation to get return value for hgt2
  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{3, 3}, n, HALF2_OP_HGT2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(1.0, result_H, n)) {
    printf("hgt2: failure when argument is greater\n");
    bFunctionalTestFailed = true;
  }

  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{2, 2}, n, HALF2_OP_HGT2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(0.0, result_H, n)) {
    printf("hgt2: failure when argument is equal\n");
    bFunctionalTestFailed = true;
  }

  hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                     result_D, __half2{1, 1}, n, HALF2_OP_HGT2,
                     HALF2_TEST_FUNCTION);
  hipDeviceSynchronize();
  HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                     hipMemcpyDeviceToHost));
  if (isFailed(0.0, result_H, n)) {
    printf("hgt2: failure when argument is less\n");
    bFunctionalTestFailed = true;
  }

  for (int nanFunctionTest = HALF2_OP_HEQ2; nanFunctionTest < HALF2_OP_MAX;
       nanFunctionTest++) {
    // HNE2 will not have a NaN test
    if (nanFunctionTest != HALF2_OP_HNE2) {
      hipLaunchKernelGGL(__half2Compare, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                         result_D, __half2{0, 0}, n, nanFunctionTest,
                         HALF2_TEST_NAN);
      hipDeviceSynchronize();
      HIPCHECK(hipMemcpy(result_H, result_D, n*sizeof(float),
                         hipMemcpyDeviceToHost));
      if (isFailed(0.0, result_H, n)) {
        printf("NaN test failed for half function: %d\n", nanFunctionTest);
        bNanTestFailed = true;
      }
    }
  }

  hipFree(result_D);
  free(result_H);

  if ((false == bFunctionalTestFailed) && (false == bNanTestFailed)) {
    passed();
  } else {
    failed("Some Half2 tests failed");
  }

  return 0;
}

