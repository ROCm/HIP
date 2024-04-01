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
Functional-
1) Instantiate a graph with memset node, obtain executable graph and update the
   hipMemsetParams node params with set. Make sure they are taking effect.
Negative-
1) Pass hGraphExec as nullptr and verify api returns error code.
2) Pass graph node as nullptr and verify api returns error code.
3) Pass different hipGraphNode_t which was not used in graphExec and verify api returns error code.
4) Pass Pass different Graph which was not used in graphExec and verify api returns error code.
5) Pass pNodeParams as nullptr and verify api returns error code.
6) Pass pNodeParams as empty structure object and verify api returns error code.
7) Pass hipMemsetParams::dst as nullptr, api should return error code.
8) Pass hipMemsetParams::element size other than 1, 2, or 4 and check api should return error code.
9) Pass hipMemsetParams::height as zero and check api should return error code.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

/* Test verifies hipGraphExecMemsetNodeSetParams API Negative scenarios.
 */
TEST_CASE("Unit_hipGraphExecMemsetNodeSetParams_Negative") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(char);
  constexpr size_t val = 0;
  char *devData, *hOutputData;

  HIP_CHECK(hipMalloc(&devData, Nbytes));
  hOutputData = reinterpret_cast<char *>(malloc(Nbytes));
  REQUIRE(hOutputData != nullptr);
  memset(hOutputData, 0,  Nbytes);

  hipGraph_t graph;
  hipError_t ret;
  hipGraphExec_t graphExec;
  hipStream_t streamForGraph;
  hipGraphNode_t memsetNode;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  hipMemsetParams mParams{};
  memset(&mParams, 0, sizeof(mParams));
  mParams.dst = reinterpret_cast<void*>(devData);
  mParams.value = val;
  mParams.pitch = 0;
  mParams.elementSize = sizeof(char);
  mParams.width = Nbytes;
  mParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &mParams));

  std::vector<hipGraphNode_t> dependencies;
  dependencies.push_back(memsetNode);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Pass hGraphExec as nullptr") {
    ret = hipGraphExecMemsetNodeSetParams(nullptr, memsetNode, &mParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hGraphNode as nullptr") {
    ret = hipGraphExecMemsetNodeSetParams(graphExec, nullptr, &mParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass different hGraphNode which was not used in graphExec") {
    hipGraphNode_t memsetNode1{};
    ret = hipGraphExecMemsetNodeSetParams(graphExec, memsetNode1, &mParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass different Graph which was not used in graphExec") {
    hipGraph_t graph1;
    HIP_CHECK(hipGraphCreate(&graph1, 0));
    HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph1, nullptr, 0, &mParams));
    ret = hipGraphExecMemsetNodeSetParams(graphExec, memsetNode, &mParams);
    REQUIRE(hipErrorInvalidValue == ret);
    HIP_CHECK(hipGraphDestroy(graph1));
  }
  SECTION("Pass pNodeParams as nullptr") {
    ret = hipGraphExecMemsetNodeSetParams(graphExec, memsetNode, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#if HT_NVIDIA
  SECTION("Pass pNodeParams as empty structure object") {
    hipMemsetParams mParmTemp{};
    ret = hipGraphExecMemsetNodeSetParams(graphExec, memsetNode, &mParmTemp);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#endif
  SECTION("Pass hipMemsetParams::dst as nullptr") {
    mParams.dst = nullptr;
    ret = hipGraphExecMemsetNodeSetParams(graphExec, memsetNode, &mParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#if HT_NVIDIA
  SECTION("Pass hipMemsetParams::element size other than 1, 2, or 4") {
    mParams.dst = reinterpret_cast<void*>(devData);
    mParams.elementSize = 9;
    ret = hipGraphExecMemsetNodeSetParams(graphExec, memsetNode, &mParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hipMemsetParams::height as zero") {
    mParams.elementSize = sizeof(char);
    mParams.height = 0;
    ret = hipGraphExecMemsetNodeSetParams(graphExec, memsetNode, &mParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#endif

  free(hOutputData);
  HIP_CHECK(hipFree(devData));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/* Test verifies hipGraphExecMemsetNodeSetParams API Functional scenarios.
 */
TEST_CASE("Unit_hipGraphExecMemsetNodeSetParams_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(char);
  constexpr size_t val = 0;
  constexpr size_t updateVal = 2;
  char *devData, *devData1, *hOutputData, *hOutputData1;

  HIP_CHECK(hipMalloc(&devData, Nbytes));
  HIP_CHECK(hipMalloc(&devData1, Nbytes));
  hOutputData = reinterpret_cast<char *>(malloc(Nbytes));
  REQUIRE(hOutputData != nullptr);
  memset(hOutputData, updateVal,  Nbytes);
  hOutputData1 = reinterpret_cast<char *>(malloc(Nbytes));
  REQUIRE(hOutputData1 != nullptr);
  memset(hOutputData1, 0,  Nbytes);

  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t streamForGraph;
  hipGraphNode_t memsetNode;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(devData);
  memsetParams.value = val;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                  &memsetParams));

  std::vector<hipGraphNode_t> dependencies;
  dependencies.push_back(memsetNode);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(devData1);
  memsetParams.value = updateVal;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;

  REQUIRE(hipSuccess == hipGraphExecMemsetNodeSetParams(graphExec, memsetNode,
                                                        &memsetParams));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  HIP_CHECK(hipMemcpy(hOutputData1, devData1, Nbytes, hipMemcpyDeviceToHost));
  HipTest::checkArray(hOutputData, hOutputData1, Nbytes, 1);

  free(hOutputData);
  free(hOutputData1);
  HIP_CHECK(hipFree(devData));
  HIP_CHECK(hipFree(devData1));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
