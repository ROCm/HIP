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

Testcase Scenarios
------------------
Functional:
1) Start stream capture and get capture info v2. Verify api is success, capture status is hipStreamCaptureStatusActive,
 identifier returned is valid/non-zero, graph object is returned.
2) When stream capture is in progress, create dependent nodes by creating multistream dependencies and verify the api returns
 valid dependent nodes.
3) End stream capture and get capture info. Verify api is success, capture status is hipStreamCaptureStatusNone and
 identifier/graph/nodes are not returned by api.
4) When optional parameters are not passed, make sure api still returns capture status of stream.
5) Begin capture on hipStreamPerThread, get capture info v2 and validate results.
6) Perform multiple captures and verify the identifier returned is unique.

Parameter Validation/Negative:
1) Capture Status location as nullptr and verify api returns error code.
2) Stream as nullptr and verify api returns error code.

*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

constexpr size_t N = 1000000;
constexpr int LAUNCH_ITERS = 1;

/**
 * Validates stream capture infov2, launches graph and verifies results
 */
void validateStreamCaptureInfoV2(hipStream_t mstream) {
  hipStream_t stream1{nullptr}, stream2{nullptr}, streamForLaunch{nullptr};
  hipEvent_t memcpyEvent1, memsetEvent2, forkStreamEvent;
  hipGraph_t graph{nullptr}, capInfoGraph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  const hipGraphNode_t *nodelist{};
  size_t Nbytes = N * sizeof(float), numDependencies;
  float *A_d, *C_d;
  float *A_h, *C_h;
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);
  HIP_CHECK(hipStreamCreate(&streamForLaunch));

  // Initialize input buffer
  for (size_t i = 0; i < N; ++i) {
      A_h[i] = 3.146f + i;  // Pi
  }

  // Create cross stream dependencies.
  // memset/memcpy operations are done on stream1 and stream2
  // and they are joined back to mainstream
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventCreate(&memcpyEvent1));
  HIP_CHECK(hipEventCreate(&memsetEvent2));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));

  HIP_CHECK(hipStreamBeginCapture(mstream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(forkStreamEvent, mstream));
  HIP_CHECK(hipStreamWaitEvent(stream1, forkStreamEvent, 0));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
  HIP_CHECK(hipMemsetAsync(A_d, 0, Nbytes, stream1));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipEventRecord(memcpyEvent1, stream1));
  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream2));
  HIP_CHECK(hipEventRecord(memsetEvent2, stream2));
  HIP_CHECK(hipStreamWaitEvent(mstream, memcpyEvent1, 0));
  HIP_CHECK(hipStreamWaitEvent(mstream, memsetEvent2, 0));

  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  unsigned long long capSequenceID = 0;  // NOLINT
  constexpr int numDepsCreated = 2;      // Num of dependencies created

  HIP_CHECK(hipStreamGetCaptureInfo_v2(mstream, &captureStatus,
                &capSequenceID, &capInfoGraph, &nodelist, &numDependencies));

  // verify capture status is active, sequence id is valid, graph is returned,
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(capSequenceID > 0);
  REQUIRE(capInfoGraph != nullptr);
  REQUIRE(numDependencies == numDepsCreated);

  // verify dependency nodes list returned is the one we created.
  hipGraphNodeType type(hipGraphNodeTypeEmpty);

  HIP_CHECK(hipGraphNodeGetType(nodelist[0], &type));
  if ((type != hipGraphNodeTypeMemset) && (type != hipGraphNodeTypeMemcpy)) {
    INFO("Type0 returned as " << type);
    REQUIRE(false);
  }

  HIP_CHECK(hipGraphNodeGetType(nodelist[1], &type));
  if ((type != hipGraphNodeTypeMemset) && (type != hipGraphNodeTypeMemcpy)) {
    INFO("Type1 returned as " << type);
    REQUIRE(false);
  }

  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                              dim3(threadsPerBlock), 0, mstream, A_d, C_d, N);

  // End capture and verify graph is returned
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, mstream));
  HIP_CHECK(hipStreamEndCapture(mstream, &graph));
  REQUIRE(graph != nullptr);

  // verify capture status is inactive and other params are not updated
  capSequenceID = 0;
  capInfoGraph = nullptr;
  numDependencies = 0;
  nodelist = nullptr;
  HIP_CHECK(hipStreamGetCaptureInfo_v2(mstream, &captureStatus,
                &capSequenceID, &capInfoGraph, &nodelist, &numDependencies));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);
  REQUIRE(capSequenceID == 0);
  REQUIRE(capInfoGraph == nullptr);
  REQUIRE(nodelist == nullptr);
  REQUIRE(numDependencies == 0);

  // Verify api still returns capture status when optional args are not passed
  HIP_CHECK(hipStreamGetCaptureInfo_v2(mstream, &captureStatus));
  REQUIRE(captureStatus == hipStreamCaptureStatusNone);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Replay the recorded sequence multiple times
  for (int i = 0; i < LAUNCH_ITERS; i++) {
    HIP_CHECK(hipGraphLaunch(graphExec, streamForLaunch));
  }

  HIP_CHECK(hipStreamSynchronize(streamForLaunch));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForLaunch));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(forkStreamEvent));
  HIP_CHECK(hipEventDestroy(memcpyEvent1));
  HIP_CHECK(hipEventDestroy(memsetEvent2));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      INFO("A and C not matching at " << i << " C_h[i] " << C_h[i]
                                           << " A_h[i] " << A_h[i]);
      REQUIRE(false);
    }
  }
  free(A_h);
  free(C_h);
}

/**
 * Basic Functional Test for stream capture and getting capture info V2.
 * Regular/custom stream is used for stream capture.
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_v2_BasicFunctional") {
  hipStream_t streamForCapture;

  HIP_CHECK(hipStreamCreate(&streamForCapture));
  validateStreamCaptureInfoV2(streamForCapture);
  HIP_CHECK(hipStreamDestroy(streamForCapture));
}

/**
 * Test performs stream capture on hipStreamPerThread and validates
 * capture info V2.
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_v2_hipStreamPerThread") {
  validateStreamCaptureInfoV2(hipStreamPerThread);
}

/**
 * Test starts stream capture on multiple streams and verifies uniqueness of
 * identifiers returned from capture Info V2.
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_v2_UniqueID") {
  constexpr int numStreams = 100;
  hipStream_t streams[numStreams]{};
  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  std::vector<int> idlist;
  unsigned long long capSequenceID{};  // NOLINT
  hipGraph_t graph{nullptr};

  for (int i = 0; i < numStreams; i++) {
    HIP_CHECK(hipStreamCreate(&streams[i]));
    HIP_CHECK(hipStreamBeginCapture(streams[i], hipStreamCaptureModeGlobal));
    HIP_CHECK(hipStreamGetCaptureInfo_v2(streams[i], &captureStatus,
                                 &capSequenceID, nullptr, nullptr, nullptr));
    REQUIRE(captureStatus == hipStreamCaptureStatusActive);
    REQUIRE(capSequenceID > 0);
    idlist.push_back(capSequenceID);
  }

  for (int i = 0; i < numStreams; i++) {
    for (int j = i+1; j < numStreams; j++) {
      if (idlist[i] == idlist[j]) {
        INFO("Same identifier returned for stream "
                                          << i << " and stream " << j);
        REQUIRE(false);
      }
    }
  }

  for (int i = 0; i < numStreams; i++) {
    HIP_CHECK(hipStreamEndCapture(streams[i], &graph));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}

/**
 * Parameter validation/Negative tests for api
 */
TEST_CASE("Unit_hipStreamGetCaptureInfo_v2_ParamValidation") {
  hipError_t ret;
  hipStream_t stream;
  float *A_d;
  hipGraph_t graph{}, capInfoGraph{};
  hipStreamCaptureStatus captureStatus;
  unsigned long long capSequenceID;  // NOLINT
  size_t numDependencies;
  const hipGraphNode_t *nodelist{};
  constexpr int numBytes{100};

  HIP_CHECK(hipMalloc(&A_d, numBytes));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemsetAsync(A_d, 0, numBytes, stream));

  SECTION("Capture Status location as nullptr") {
    ret = hipStreamGetCaptureInfo_v2(stream, nullptr,
                &capSequenceID, &capInfoGraph, &nodelist, &numDependencies);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("Stream as nullptr") {
    ret = hipStreamGetCaptureInfo_v2(nullptr, &captureStatus,
                &capSequenceID, &capInfoGraph, &nodelist, &numDependencies);
    if ((ret != hipErrorUnknown) && (ret != hipErrorStreamCaptureImplicit)) {
      INFO("Ret : " << ret);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(A_d));
}
