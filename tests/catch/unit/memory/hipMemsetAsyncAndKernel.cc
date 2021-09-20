/*
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#define ITER 6
#define N 1024 * 1024

constexpr auto blocksPerCU = 6;  // to hide latency
constexpr auto threadsPerBlock = 256;
static unsigned blocks = 0;


template <typename T>
class MemSetKernelTest {
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
    HIP_ASSERT(A_h != nullptr);
    HIP_CHECK(hipMalloc(&B_d , Nbytes));
    B_h = reinterpret_cast<T*>(malloc(Nbytes));
    HIP_ASSERT(B_h != nullptr);
    HIP_CHECK(hipMalloc(&C_d , Nbytes));

    for (int i = 0 ; i < N ; i++) {
      B_h[i] = i;
    }
    HIP_CHECK(hipMemcpy(B_d , B_h , Nbytes , hipMemcpyHostToDevice));
    HIP_CHECK(hipStreamCreate(&stream));
  }

  void memDeallocate() {
    HIP_CHECK(hipFree(B_d)); HIP_CHECK(hipFree(C_d));
    free(B_h); free(A_h);
    HIP_CHECK(hipStreamDestroy(stream));
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

static bool testhipMemsetAsyncWithKernel() {
  MemSetKernelTest<char> obj;
  constexpr char memsetval = 0x42;

  obj.memAllocate(memsetval);
  for (int k = 0 ; k < ITER ; k++) {
    hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                    dim3(threadsPerBlock), 0, obj.stream, obj.B_d, obj.C_d, N);
    HIP_CHECK(hipMemsetAsync(obj.C_d , obj.memSetVal , N , obj.stream));
    HIP_CHECK(hipStreamSynchronize(obj.stream));
    HIP_CHECK(hipMemcpy(obj.A_h, obj.C_d, obj.Nbytes, hipMemcpyDeviceToHost));

    obj.validateExecutionOrder();
  }
  return obj.resultAfterAllIterations();
}

static bool testhipMemsetD32AsyncWithKernel() {
  MemSetKernelTest <int32_t> obj;
  constexpr int memsetD32val = 0xDEADBEEF;

  obj.memAllocate(memsetD32val);
  for (int k = 0 ; k < ITER ; k++) {
    hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                    dim3(threadsPerBlock), 0, obj.stream, obj.B_d, obj.C_d, N);
    HIP_CHECK(hipMemsetD32Async((hipDeviceptr_t)obj.C_d , obj.memSetVal,
                                                              N, obj.stream));
    HIP_CHECK(hipStreamSynchronize(obj.stream));
    HIP_CHECK(hipMemcpy(obj.A_h, obj.C_d, obj.Nbytes, hipMemcpyDeviceToHost));

    obj.validateExecutionOrder();
  }
  return obj.resultAfterAllIterations();
}

static bool testhipMemsetD16AsyncWithKernel() {
  MemSetKernelTest <int16_t> obj;
  constexpr int16_t memsetD16val = 0xDEAD;

  obj.memAllocate(memsetD16val);
  for (int k = 0 ; k < ITER ; k++) {
    hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                    dim3(threadsPerBlock), 0, obj.stream, obj.B_d, obj.C_d, N);
    HIP_CHECK(hipMemsetD16Async((hipDeviceptr_t)obj.C_d , obj.memSetVal,
                                                              N, obj.stream));
    HIP_CHECK(hipStreamSynchronize(obj.stream));
    HIP_CHECK(hipMemcpy(obj.A_h, obj.C_d, obj.Nbytes, hipMemcpyDeviceToHost));

    obj.validateExecutionOrder();
  }
  return obj.resultAfterAllIterations();
}

static bool testhipMemsetD8AsyncWithKernel() {
  MemSetKernelTest <char> obj;
  constexpr char memsetD8val = 0xDE;

  obj.memAllocate(memsetD8val);
  for (int k = 0; k < ITER; k++) {
    hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                    dim3(threadsPerBlock), 0, obj.stream, obj.B_d, obj.C_d, N);
    HIP_CHECK(hipMemsetD8Async((hipDeviceptr_t)obj.C_d, obj.memSetVal,
                                                              N, obj.stream));
    HIP_CHECK(hipStreamSynchronize(obj.stream));
    HIP_CHECK(hipMemcpy(obj.A_h, obj.C_d, obj.Nbytes, hipMemcpyDeviceToHost));

    obj.validateExecutionOrder();
  }
  return obj.resultAfterAllIterations();
}


/*
 * Test for checking order of execution of device kernel and
 * hipMemsetAsync apis on all gpus
 */
TEST_CASE("Unit_hipMemsetAsync_VerifyExecutionWithKernel") {
  int numDevices = 0;
  bool ret;

  blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGetDeviceCount(&numDevices));
  REQUIRE(numDevices > 0);

  auto devNum = GENERATE_COPY(range(0, numDevices));
  HIP_CHECK(hipSetDevice(devNum));

  SECTION("hipMemsetAsync With Kernel") {
    ret = testhipMemsetAsyncWithKernel();
    REQUIRE(ret == true);
  }

  SECTION("hipMemsetD32Async With Kernel") {
    ret = testhipMemsetD32AsyncWithKernel();
    REQUIRE(ret == true);
  }

  SECTION("hipMemsetD16Async With Kernel") {
    ret = testhipMemsetD16AsyncWithKernel();
    REQUIRE(ret == true);
  }

  SECTION("hipMemsetD8Async With Kernel") {
    ret = testhipMemsetD8AsyncWithKernel();
    REQUIRE(ret == true);
  }
}
