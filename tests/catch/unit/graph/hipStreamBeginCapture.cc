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
Testcase Scenarios :
 1) Initiate stream capture with different modes on custom stream.
 Capture stream sequence and replay the sequence in multiple iterations.
 2) End capture and validate that API returns captured graph for
 all possible modes on custom stream.
 3) Initiate stream capture with different modes on hipStreamPerThread.
 Capture stream sequence and replay the sequence in multiple iterations.
 4) End capture and validate that API returns captured graph for
 all possible modes on hipStreamPerThread.
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>


constexpr size_t N = 1000000;
constexpr int LAUNCH_ITERS = 50;


bool CaptureStreamAndLaunchGraph(float *A_d, float *C_d, float *A_h,
                 float *C_h, hipStreamCaptureMode mode, hipStream_t stream) {
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  size_t Nbytes = N * sizeof(float);

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

  // Replay the recorded sequence multiple times
  for (int i = 0; i < LAUNCH_ITERS; i++) {
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
  }

  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      UNSCOPED_INFO("A and C not matching at " << i);
      return false;
    }
  }
  return true;
}

/**
 * Basic Functional Test for API capturing custom stream and replaying sequence.
 * Test exercises the API on available/possible modes.
 * Stream capture with different modes behave the same when supported/
 * safe apis are used in sequence.
 */
TEST_CASE("Unit_hipStreamBeginCapture_BasicFunctional") {
  float *A_d, *C_d;
  float *A_h, *C_h;
  size_t Nbytes = N * sizeof(float);
  hipStream_t stream;
  bool ret;

  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
      A_h[i] = 1.618f + i;
  }

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);

  SECTION("Capture stream and launch graph when mode is global") {
    ret = CaptureStreamAndLaunchGraph(A_d, C_d, A_h, C_h,
                                          hipStreamCaptureModeGlobal, stream);
    REQUIRE(ret == true);
  }

  SECTION("Capture stream and launch graph when mode is local") {
    ret = CaptureStreamAndLaunchGraph(A_d, C_d, A_h, C_h,
                                     hipStreamCaptureModeThreadLocal, stream);
    REQUIRE(ret == true);
  }

  SECTION("Capture stream and launch graph when mode is relaxed") {
    ret = CaptureStreamAndLaunchGraph(A_d, C_d, A_h, C_h,
                                         hipStreamCaptureModeRelaxed, stream);
    REQUIRE(ret == true);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  free(A_h);
  free(C_h);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
}

/**
 * Perform capture on hipStreamPerThread, launch the graph and verify results.
 */
TEST_CASE("Unit_hipStreamBeginCapture_hipStreamPerThread") {
  float *A_d, *C_d;
  float *A_h, *C_h;
  size_t Nbytes = N * sizeof(float);
  hipStream_t stream{hipStreamPerThread};
  bool ret;

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

  SECTION("Capture hipStreamPerThread and launch graph when mode is global") {
    ret = CaptureStreamAndLaunchGraph(A_d, C_d, A_h, C_h,
                                           hipStreamCaptureModeGlobal, stream);
    REQUIRE(ret == true);
  }

  SECTION("Capture hipStreamPerThread and launch graph when mode is local") {
    ret = CaptureStreamAndLaunchGraph(A_d, C_d, A_h, C_h,
                                      hipStreamCaptureModeThreadLocal, stream);
    REQUIRE(ret == true);
  }

  SECTION("Capture hipStreamPerThread and launch graph when mode is relaxed") {
    ret = CaptureStreamAndLaunchGraph(A_d, C_d, A_h, C_h,
                                          hipStreamCaptureModeRelaxed, stream);
    REQUIRE(ret == true);
  }

  free(A_h);
  free(C_h);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
}
