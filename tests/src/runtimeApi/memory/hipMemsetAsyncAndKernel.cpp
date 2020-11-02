/*
 * Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
*/

/*
 * Test for checking order of execution of device kernel and
 * hipMemsetAsync apis on all gpus
 */

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#define ITER 10
#define N 1024 * 1024

unsigned blocks = 0;

template <typename T>
__global__ void vector_square(T* B_d, T* C_d, size_t M) {
  for (int i=0 ; i < M ; i++) {
    C_d[i] = B_d[i] * B_d[i];
  }
}

template <typename T>
class MemSetTest {
 public:
  T *A_h, *B_d, *B_h, *C_d;
  T memSetVal;
  size_t Nbytes;
  bool testResult = true;
  int validateCount = 0;
  hipStream_t stream;

  void memAllocate(T memSetValue) {
    memSetVal = memSetValue;
    Nbytes = N * sizeof(T);

    A_h = reinterpret_cast<T*>(malloc(Nbytes));
    HIPASSERT(A_h != NULL);
    HIPCHECK(hipMalloc(&B_d , Nbytes));
    B_h = reinterpret_cast<T*>(malloc(Nbytes));
    HIPASSERT(B_h != NULL);
    HIPCHECK(hipMalloc(&C_d , Nbytes));

    for (int i = 0 ; i < N ; i++) {
      B_h[i] = i;
    }
    HIPCHECK(hipMemcpy(B_d , B_h , Nbytes , hipMemcpyHostToDevice));
    HIPCHECK(hipStreamCreate(&stream));
  }

  void memDeallocate() {
    HIPCHECK(hipFree(B_d)); HIPCHECK(hipFree(C_d));
    free(B_h); free(A_h);
    HIPCHECK(hipStreamDestroy(stream));
  }

  void validateExecutionOrder() {
    for (int p = 0 ; p < N ; p++) {
      if (A_h[p] == memSetVal) {
        validateCount+= 1;
      }
    }
  }

  bool resultAfterAllIterations() {
    testResult = (validateCount == (ITER * N)) ? true : false;
    memDeallocate();
    return testResult;
  }
};

bool testhipMemsetAsyncWithKernel() {
  MemSetTest <char> obj;
  obj.memAllocate(memsetval);
  for (int k = 0 ; k < ITER ; k++) {
    hipLaunchKernelGGL(vector_square, dim3(blocks), dim3(threadsPerBlock), 0,
                       obj.stream, obj.B_d, obj.C_d, N);
    HIPCHECK(hipMemsetAsync(obj.C_d , obj.memSetVal , N , obj.stream));
    HIPCHECK(hipStreamSynchronize(obj.stream));
    HIPCHECK(hipMemcpy(obj.A_h , obj.C_d , obj.Nbytes , hipMemcpyDeviceToHost));

    obj.validateExecutionOrder();
  }
  return obj.resultAfterAllIterations();
}

bool testhipMemsetD32AsyncWithKernel() {
  MemSetTest <int32_t> obj;
  obj.memAllocate(memsetD32val);
  for (int k = 0 ; k < ITER ; k++) {
    hipLaunchKernelGGL(vector_square, dim3(blocks), dim3(threadsPerBlock), 0,
                       obj.stream, obj.B_d, obj.C_d, N);
    HIPCHECK(hipMemsetD32Async(obj.C_d , obj.memSetVal , N , obj.stream));
    HIPCHECK(hipStreamSynchronize(obj.stream));
    HIPCHECK(hipMemcpy(obj.A_h, obj.C_d, obj.Nbytes, hipMemcpyDeviceToHost));

    obj.validateExecutionOrder();
  }
  return obj.resultAfterAllIterations();
}

bool testhipMemsetD16AsyncWithKernel() {
  MemSetTest <int16_t> obj;
  obj.memAllocate(memsetD16val);
  for (int k = 0 ; k < ITER ; k++) {
    hipLaunchKernelGGL(vector_square, dim3(blocks), dim3(threadsPerBlock), 0,
                       obj.stream, obj.B_d, obj.C_d, N);
    HIPCHECK(hipMemsetD16Async(obj.C_d , obj.memSetVal , N , obj.stream));
    HIPCHECK(hipStreamSynchronize(obj.stream));
    HIPCHECK(hipMemcpy(obj.A_h , obj.C_d, obj.Nbytes , hipMemcpyDeviceToHost));

    obj.validateExecutionOrder();
  }
  return obj.resultAfterAllIterations();
}

bool testhipMemsetD8AsyncWithKernel() {
  MemSetTest <char> obj;
  obj.memAllocate(memsetD8val);
  for (int k = 0; k < ITER; k++) {
    hipLaunchKernelGGL(vector_square, dim3(blocks), dim3(threadsPerBlock), 0,
                       obj.stream, obj.B_d, obj.C_d, N);
    HIPCHECK(hipMemsetD8Async(obj.C_d, obj.memSetVal, N, obj.stream));
    HIPCHECK(hipStreamSynchronize(obj.stream));
    HIPCHECK(hipMemcpy(obj.A_h, obj.C_d, obj.Nbytes, hipMemcpyDeviceToHost));

    obj.validateExecutionOrder();
  }
  return obj.resultAfterAllIterations();
}

int main() {
  bool testResult = true;
  int numDevices = 0;
  blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  printf("blocks: %u\n", blocks);

  HIPCHECK(hipGetDeviceCount(&numDevices));
  printf("total number of gpus in the system: %d\n", numDevices);

  for (int i = 0; i < numDevices; i++) {
    HIPCHECK(hipSetDevice(i));
    printf("test running on gpu %d\n", i);

    testResult &= testhipMemsetAsyncWithKernel();
    if (!(testResult)) {
      printf("Mismatch in order of execution of hipMemsetAsync and kernel\n");
    }

    testResult &= testhipMemsetD32AsyncWithKernel();
    if (!(testResult)) {
      printf("Mismatch in order of execution of hipMemsetD32Async and kernel\n");
    }

    testResult &= testhipMemsetD16AsyncWithKernel();
    if (!(testResult)) {
      printf("Mismatch in order of execution of hipMemsetD16Async and kernel\n");
    }

    testResult &= testhipMemsetD8AsyncWithKernel();
    if (!(testResult)) {
      printf("Mismatch in order of execution of hipMemsetD8Async and kernel\n");
    }
  }

  if (testResult) {
    printf("Execution order of Kernel and hipMemsetAsync apis on "
           "all gpus is correct!\n");
    passed();
  } else {
      failed("One or more hipMemsetAsync tests failed\n");
  }
}
