/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

static void hipGraphLaunchWithMode(hipStream_t stream,
                                   hipStreamCaptureMode mode) {
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  constexpr size_t N = 1024;
  size_t Nbytes = N * sizeof(float);

  int *A_d, *C_d;
  int *A_h, *C_h;
  HipTest::initArrays<int>(&A_d, nullptr, &C_d, &A_h, nullptr, &C_h, N, false);

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&mode));

  HIP_CHECK(hipStreamBeginCapture(stream, mode));

  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                              dim3(threadsPerBlock), 0, stream, A_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  // Validate end capture is successful
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      UNSCOPED_INFO("A and C not matching at " << i);
    }
  }
  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
}

void threadFuncCaptureMode(hipStream_t stream, hipStreamCaptureMode mode) {
  hipGraphLaunchWithMode(stream, mode);
}

/**
 * Functional Test for APIs - hipThreadExchangeStreamCaptureMode
 1) Call ModeGlobal from main thread & ModeGlobal from other thread
 2) Call ModeGlobal from main thread & ModeThreadLocal from other thread
 3) Call ModeGlobal from main thread & ModeRelaxed from other thread
 4) Call ModeThreadLocal from main thread & ModeGlobal from other thread
 5) Call ModeThreadLocal from main thread & ModeThreadLocal from other thread
 6) Call ModeThreadLocal from main thread & ModeRelaxed from other thread
 7) Call ModeRelaxed from main thread & ModeGlobal from other thread
 8) Call ModeRelaxed from main thread & ModeThreadLocal from other thread
 9) Call ModeRelaxed from main thread & ModeRelaxed from other thread
 */

TEST_CASE("Unit_hipThreadExchangeStreamCaptureMode_Functional") {
  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));

  SECTION("Call ModeGlobal from main thread & ModeGlobal from other thread") {
    hipGraphLaunchWithMode(stream, hipStreamCaptureModeGlobal);
    std::thread t1(threadFuncCaptureMode, stream, hipStreamCaptureModeGlobal);
    t1.join();
  }
  SECTION("Call ModeGlobal from main thread & ModeThreadLocal from other") {
    hipGraphLaunchWithMode(stream, hipStreamCaptureModeGlobal);
    std::thread t2(threadFuncCaptureMode, stream,
                   hipStreamCaptureModeThreadLocal);
    t2.join();
  }
  SECTION("Call ModeGlobal from main thread & ModeRelaxed from other thread") {
    hipGraphLaunchWithMode(stream, hipStreamCaptureModeGlobal);
    std::thread t3(threadFuncCaptureMode, stream, hipStreamCaptureModeRelaxed);
    t3.join();
  }
  SECTION("Call ModeThreadLocal from main thread & ModeGlobal from other") {
    hipGraphLaunchWithMode(stream, hipStreamCaptureModeThreadLocal);
    std::thread t4(threadFuncCaptureMode, stream, hipStreamCaptureModeGlobal);
    t4.join();
  }
  SECTION("Call ModeThreadLocal from main thread & ThreadLocal from other") {
    hipGraphLaunchWithMode(stream, hipStreamCaptureModeThreadLocal);
    std::thread t5(threadFuncCaptureMode, stream,
                   hipStreamCaptureModeThreadLocal);
    t5.join();
  }
  SECTION("Call ModeThreadLocal from main thread & ModeRelaxed from other") {
    hipGraphLaunchWithMode(stream, hipStreamCaptureModeThreadLocal);
    std::thread t6(threadFuncCaptureMode, stream, hipStreamCaptureModeRelaxed);
    t6.join();
  }
  SECTION("Call ModeRelaxed from main thread & ModeGlobal from other thread") {
    hipGraphLaunchWithMode(stream, hipStreamCaptureModeRelaxed);
    std::thread t7(threadFuncCaptureMode, stream, hipStreamCaptureModeGlobal);
    t7.join();
  }
  SECTION("Call ModeRelaxed from main thread & ModeThreadLocal from other") {
    hipGraphLaunchWithMode(stream, hipStreamCaptureModeRelaxed);
    std::thread t8(threadFuncCaptureMode, stream,
                   hipStreamCaptureModeThreadLocal);
    t8.join();
  }
  SECTION("Call ModeRelaxed from main thread & ModeRelaxed from other") {
    hipGraphLaunchWithMode(stream, hipStreamCaptureModeRelaxed);
    std::thread t9(threadFuncCaptureMode, stream, hipStreamCaptureModeRelaxed);
    t9.join();
  }

  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Negative Test for APIs - hipThreadExchangeStreamCaptureMode
 1) Pass Mode as nullptr
 2) Pass Mode as -1
 3) Pass Mode as INT_MAX
 4) Pass Mode other than existing 3 mode (hipStreamCaptureModeRelaxed + 1)
 */

#if HT_AMD  // getting error in Cuda Setup
TEST_CASE("Unit_hipThreadExchangeStreamCaptureMode_Negative") {
  hipError_t ret;
  hipStreamCaptureMode mode;

  SECTION("Pass Mode as nullptr") {
    ret = hipThreadExchangeStreamCaptureMode(nullptr);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Pass Mode as -1") {
    mode = hipStreamCaptureMode(-1);
    ret = hipThreadExchangeStreamCaptureMode(&mode);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Pass Mode as INT_MAX") {
    mode = hipStreamCaptureMode(INT_MAX);
    ret = hipThreadExchangeStreamCaptureMode(&mode);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Pass Mode as hipStreamCaptureModeRelaxed + 1") {
    mode = hipStreamCaptureMode(hipStreamCaptureModeRelaxed + 1);
    ret = hipThreadExchangeStreamCaptureMode(&mode);
    REQUIRE(ret == hipErrorInvalidValue);
  }
}
#endif

