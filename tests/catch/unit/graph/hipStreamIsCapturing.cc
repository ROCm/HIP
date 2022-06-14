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
#include <hip_test_kernels.hh>

constexpr unsigned blocks = 512;
constexpr unsigned threadsPerBlock = 256;
constexpr size_t N = 100000;
constexpr size_t Nbytes = N * sizeof(float);

/**
API - hipStreamIsCapturing
Negative Testcase Scenarios : Negative
  1) Check capture status with null pCaptureStatus.
  2) Check capture status with hipStreamPerThread and null pCaptureStatus.
Functional Testcase Scenarios :
  1) Check capture status with null stream.
  2) Check capture status with hipStreamPerThread.
  3) Functional : Create a stream, call api and check
     capture status is hipStreamCaptureStatusNone.
  4) Functional : Start capturing a stream and check
     capture status returned as hipStreamCaptureStatusActive.
  5) Functional : Stop capturing a stream and check
     status is returned as hipStreamCaptureStatusNone.
  6) Functional : Use hipStreamPerThread, call api and check
     capture status is hipStreamCaptureStatusNone.
  7) Functional : Start capturing using hipStreamPerThread and check
     capture status returned as hipStreamCaptureStatusActive.
  8) Functional : Stop capturing using hipStreamPerThread and check
     status is returned as hipStreamCaptureStatusNone.
*/


TEST_CASE("Unit_hipStreamIsCapturing_Negative") {
  hipError_t ret;
  hipStream_t stream{};

  SECTION("Check capture status with null pCaptureStatus.") {
    ret = hipStreamIsCapturing(stream, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Check capture status with hipStreamPerThread and"
                 " nullptr as pCaptureStatus.") {
    ret = hipStreamIsCapturing(hipStreamPerThread, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
}

TEST_CASE("Unit_hipStreamIsCapturing_Functional_Basic") {
  hipStreamCaptureStatus cStatus;

  SECTION("Check capture status with null stream.") {
    HIP_CHECK(hipStreamIsCapturing(nullptr, &cStatus));
    REQUIRE(hipStreamCaptureStatusNone == cStatus);
  }
  SECTION("Check capture status with hipStreamPerThread.") {
    HIP_CHECK(hipStreamIsCapturing(hipStreamPerThread, &cStatus));
    REQUIRE(hipStreamCaptureStatusNone == cStatus);
  }
}

/**
Testcase Scenarios :
  1) Functional : Create a stream, call api and check
     capture status is hipStreamCaptureStatusNone.
  2) Functional : Start capturing a stream and check
     capture status returned as hipStreamCaptureStatusActive.
  3) Functional : Stop capturing a stream and check
     status is returned as hipStreamCaptureStatusNone.
*/

TEST_CASE("Unit_hipStreamIsCapturing_Functional") {
  float *A_d, *C_d;
  float *A_h, *C_h;
  hipStream_t stream{nullptr};
  hipGraph_t graph{nullptr};
  hipStreamCaptureStatus cStatus;

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

  SECTION("Check the stream capture status before start capturing.") {
    HIP_CHECK(hipStreamIsCapturing(stream, &cStatus));
    REQUIRE(hipStreamCaptureStatusNone == cStatus);
  }

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  SECTION("Start capturing a stream and check the status.") {
    HIP_CHECK(hipStreamIsCapturing(stream, &cStatus));
    REQUIRE(hipStreamCaptureStatusActive == cStatus);
  }

  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                              dim3(threadsPerBlock), 0, stream, A_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  SECTION("Stop capturing a stream and check the status.") {
    HIP_CHECK(hipStreamIsCapturing(stream, &cStatus));
    REQUIRE(hipStreamCaptureStatusNone == cStatus);
  }

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));

  free(A_h);
  free(C_h);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
}

/**
Testcase Scenarios :
  1) Functional : Use hipStreamPerThread, call api and check
     capture status is hipStreamCaptureStatusNone.
  2) Functional : Start capturing using hipStreamPerThread and check
     capture status returned as hipStreamCaptureStatusActive.
  3) Functional : Stop capturing using hipStreamPerThread and check
     status is returned as hipStreamCaptureStatusNone.
*/

TEST_CASE("Unit_hipStreamIsCapturing_hipStreamPerThread") {
  float *A_d, *C_d;
  float *A_h, *C_h;
  hipGraph_t graph{nullptr};
  hipStreamCaptureStatus cStatus;

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

  SECTION("Check the stream capture status before start capturing.") {
    HIP_CHECK(hipStreamIsCapturing(hipStreamPerThread, &cStatus));
    REQUIRE(hipStreamCaptureStatusNone == cStatus);
  }

  HIP_CHECK(hipStreamBeginCapture(hipStreamPerThread,
                                  hipStreamCaptureModeGlobal));

  SECTION("Start capturing a stream and check the status.") {
    HIP_CHECK(hipStreamIsCapturing(hipStreamPerThread, &cStatus));
    REQUIRE(hipStreamCaptureStatusActive == cStatus);
  }

  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice,
                                             hipStreamPerThread));

  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, hipStreamPerThread));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
           dim3(threadsPerBlock), 0, hipStreamPerThread, A_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost,
                                             hipStreamPerThread));

  HIP_CHECK(hipStreamEndCapture(hipStreamPerThread, &graph));

  SECTION("Stop capturing a stream and check the status.") {
    HIP_CHECK(hipStreamIsCapturing(hipStreamPerThread, &cStatus));
    REQUIRE(hipStreamCaptureStatusNone == cStatus);
  }

  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
  HIP_CHECK(hipGraphDestroy(graph));

  free(A_h);
  free(C_h);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
}
