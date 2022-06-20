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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Negative Testcase Scenarios :
1) Pass stream as nullptr and verify there is no crash, api returns error code.
2) Pass graph as nullptr and verify there is no crash, api returns error code.
3) Pass graph as nullptr and  and stream as hipStreamPerThread verify there
   is no crash, api returns error code.
4) End capture on stream where capture has not yet started and verify
   error code is returned.
5) Destroy stream and try to end capture.
6) Destroy Graph and try to end capture.
7) Begin capture on a thread with mode other than hipStreamCaptureModeRelaxed
   and try to end capture from different thread. Expect to return
   hipErrorStreamCaptureWrongThread.
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

TEST_CASE("Unit_hipStreamEndCapture_Negative") {
  hipError_t ret;
  SECTION("Pass stream as nullptr") {
    hipGraph_t graph;
    ret = hipStreamEndCapture(nullptr, &graph);
    REQUIRE(hipErrorIllegalState == ret);
  }
#if HT_NVIDIA
  SECTION("Pass graph as nullptr") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    ret = hipStreamEndCapture(stream, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
    HIP_CHECK(hipStreamDestroy(stream));
  }
  SECTION("Pass graph as nullptr and stream as hipStreamPerThread") {
    ret = hipStreamEndCapture(hipStreamPerThread, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#endif
  SECTION("End capture on stream where capture has not yet started") {
    hipStream_t stream;
    hipGraph_t graph;
    HIP_CHECK(hipStreamCreate(&stream));
    ret = hipStreamEndCapture(stream, &graph);
    REQUIRE(hipErrorIllegalState == ret);
    HIP_CHECK(hipStreamDestroy(stream));
  }
  SECTION("Destroy stream and try to end capture") {
    hipStream_t stream;
    hipGraph_t graph;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    HIP_CHECK(hipStreamDestroy(stream));
    ret = hipStreamEndCapture(stream, &graph);
    REQUIRE(hipErrorContextIsDestroyed == ret);
  }
  SECTION("Destroy graph and try to end capture in between") {
    hipStream_t stream{nullptr};
    hipGraph_t graph{nullptr};
    constexpr unsigned blocks = 512;
    constexpr unsigned threadsPerBlock = 256;
    constexpr size_t N = 100000;
    size_t Nbytes = N * sizeof(float);
    float *A_d, *C_d;
    float *A_h, *C_h;

    A_h = reinterpret_cast<float*>(malloc(Nbytes));
    C_h = reinterpret_cast<float*>(malloc(Nbytes));
    REQUIRE(A_h != nullptr);
    REQUIRE(C_h != nullptr);

    // Fill with Phi + i
    for (size_t i = 0; i < N; i++) {
      A_h[i] = 1.618f + i;
    }

    HIP_CHECK(hipMalloc(&A_d, Nbytes));
    HIP_CHECK(hipMalloc(&C_d, Nbytes));
    REQUIRE(A_d != nullptr);
    REQUIRE(C_d != nullptr);

    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

    HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream));
    hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                                dim3(threadsPerBlock), 0, stream, A_d, C_d, N);
    HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

    HIP_CHECK(hipGraphDestroy(graph));
    ret = hipStreamEndCapture(stream, &graph);
    REQUIRE(hipSuccess == ret);

    free(A_h);
    free(C_h);
    HIP_CHECK(hipFree(A_d));
    HIP_CHECK(hipFree(C_d));
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

static void thread_func(hipStream_t stream, hipGraph_t graph) {
  HIP_ASSERT(hipErrorStreamCaptureWrongThread ==
             hipStreamEndCapture(stream, &graph));
}

TEST_CASE("Unit_hipStreamEndCapture_Thread_Negative") {
  hipStream_t stream{nullptr};
  hipGraph_t graph{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  constexpr size_t N = 100000;
  size_t Nbytes = N * sizeof(float);
  float *A_d, *C_d;
  float *A_h, *C_h;

  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 1.618f + i;
  }

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                              dim3(threadsPerBlock), 0, stream, A_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

  std::thread t(thread_func, stream, graph);
  t.join();

#if HT_AMD
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
#endif

  free(A_h);
  free(C_h);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphDestroy(graph));
}

