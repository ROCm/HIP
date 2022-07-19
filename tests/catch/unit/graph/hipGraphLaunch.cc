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

/* Test verifies hipGraphLaunch API
Negative scenarios -
1) Pass graphExec as nullptr and verify api returns error code.
2) Pass pGraphExec as nullptr and stream as hipStreamPerThread and verify  api returns error code.
3) Pass pGraphExec as empty object and verify  api returns error code.
4) Destroy executable graph and try to launch it. Make sure api should not crash and it should returns error code.
5) Destroy stream and try to launch respective executable graph. Make sure api should not crash and it should returns error code.
6) Destroy actual graph created and try to launch respective executable graph.
   Check api should execute properly without crash or error code.
Functional Scenario -
1) Check basic functionality with stream as hipStreamPerThread
2) Test hipGraphLaunch call on multiple devices.
*/

TEST_CASE("Unit_hipGraphLaunch_Negative") {
  hipError_t ret;
  SECTION("Pass pGraphExec as nullptr") {
    hipStream_t stream{};
    ret = hipGraphLaunch(nullptr, stream);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pGraphExec as nullptr and stream as hipStreamPerThread") {
    ret = hipGraphLaunch(nullptr, hipStreamPerThread);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pGraphExec as empty object") {
    hipGraphExec_t graphExec{};
    hipStream_t stream{};
    ret = hipGraphLaunch(graphExec, stream);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Destroy executable graph and try to launch it") {
    constexpr size_t Nbytes = 1024;
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipStream_t stream;
    hipGraphNode_t memsetNode;

    char *devData;
    HIP_CHECK(hipMalloc(&devData, Nbytes));

    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipStreamCreate(&stream));

    hipMemsetParams memsetParams{};
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(devData);
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = Nbytes;
    memsetParams.height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                    &memsetParams));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    // Launch again after destroy graph exec object.
    ret = hipGraphLaunch(graphExec, stream);
    REQUIRE(hipErrorInvalidValue == ret);

    HIP_CHECK(hipFree(devData));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream));
  }
/* In this case in CUDA setup this api call is giving - unknown error (999)
   So enabling this test for both AMD and CUDA by checking with hipSuccess */
  SECTION("Destroy stream and try to launch respective executable graph") {
    constexpr size_t Nbytes = 1024;
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipStream_t stream;
    hipGraphNode_t memsetNode;

    char *devData;
    HIP_CHECK(hipMalloc(&devData, Nbytes));

    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipStreamCreate(&stream));

    hipMemsetParams memsetParams{};
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(devData);
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = Nbytes;
    memsetParams.height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                    &memsetParams));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipStreamDestroy(stream));
    // Launch again after destroy stream
    ret = hipGraphLaunch(graphExec, stream);
    REQUIRE(hipSuccess != ret);

    HIP_CHECK(hipFree(devData));
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
  }
  SECTION("Destroy graph and try to launch respective executable graph") {
    constexpr size_t Nbytes = 1024;
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipStream_t stream;
    hipGraphNode_t memsetNode;

    char *devData;
    HIP_CHECK(hipMalloc(&devData, Nbytes));

    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipStreamCreate(&stream));

    hipMemsetParams memsetParams{};
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(devData);
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = Nbytes;
    memsetParams.height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                    &memsetParams));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipGraphDestroy(graph));
    // Launch again after destroy graph
    ret = hipGraphLaunch(graphExec, stream);
    REQUIRE(hipSuccess == ret);

    HIP_CHECK(hipFree(devData));
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

TEST_CASE("Unit_hipGraphLaunch_Functional_hipStreamPerThread") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(char);
  constexpr size_t val = 0;
  constexpr size_t updateVal = 2;
  char *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  char *A_h{nullptr}, *B_h{nullptr};

  HipTest::initArrays<char>(&A_d, &B_d, &C_d,
                            &A_h, &B_h, nullptr, N, false);

  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memsetNode;

  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(C_d);
  memsetParams.value = val;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                  &memsetParams));

  std::vector<hipGraphNode_t> dependencies;
  dependencies.push_back(memsetNode);

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = updateVal;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, dependencies.data(),
                                  dependencies.size(), &memsetParams));
  HIP_CHECK(hipGraphMemsetNodeSetParams(memsetNode, &memsetParams));
  dependencies.push_back(memsetNode);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));

  // Validating the result
  for (size_t i = 0; i < Nbytes; i++) {
    if (A_h[i] != updateVal) {
      WARN("Validation failed at- " << i << " A_h[i] " << A_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<char>(A_d, B_d, C_d,
                            A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

static void hipGraphLaunch_test() {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(char);
  constexpr size_t val = 0;
  constexpr size_t updateVal = 1;
  char *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  char *A_h{nullptr}, *B_h{nullptr};

  HipTest::initArrays<char>(&A_d, &B_d, &C_d,
                            &A_h, &B_h, nullptr, N, false);

  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t streamForGraph;
  hipGraphNode_t memsetNode;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(C_d);
  memsetParams.value = val;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                  &memsetParams));

  std::vector<hipGraphNode_t> dependencies;
  dependencies.push_back(memsetNode);

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = updateVal;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, dependencies.data(),
                                  dependencies.size(), &memsetParams));
  HIP_CHECK(hipGraphMemsetNodeSetParams(memsetNode, &memsetParams));
  dependencies.push_back(memsetNode);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));

  // Validating the result
  for (size_t i = 0; i < Nbytes; i++) {
    if (A_h[i] != updateVal) {
      WARN("Validation failed at- " << i << " A_h[i] " << A_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<char>(A_d, B_d, C_d,
                            A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

TEST_CASE("Unit_hipGraphLaunch_Functional_multidevice_test") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices > 0) {
    for (int i = 0; i < numDevices; i++) {
      HIP_CHECK(hipSetDevice(i));
      hipGraphLaunch_test();
    }
  } else {
    SUCCEED("Skipped the testcase as there is no device to test.");
  }
}
