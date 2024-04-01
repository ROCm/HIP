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

/**
Testcase Scenarios :
Negative -
1) Pass node as nullptr and verify api returns error code.
2) Pass pNodeParams as nullptr and verify api returns error code.
Functional -
1) Create a graph, add Memset node to graph with desired node params.
   Verify api fetches the node params mentioned while adding Memset node.
2) Set Memset node params with hipGraphMemsetNodeSetParams,
   now get the params and verify both are same.
*/

#include <hip_test_common.hh>

/* Test verifies hipGraphMemsetNodeGetParams API Negative scenarios.
 */
TEST_CASE("Unit_hipGraphMemsetNodeGetParams_Negative") {
  hipError_t ret;
  hipGraph_t graph;
  hipGraphNode_t memsetNode;

  HIP_CHECK(hipGraphCreate(&graph, 0));

  char *devData;
  HIP_CHECK(hipMalloc(&devData, 1024));
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(devData);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = 1024;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                  &memsetParams));
  SECTION("Pass node as nullptr") {
    ret = hipGraphMemsetNodeGetParams(nullptr, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass GetNodeParams as nullptr") {
    ret = hipGraphMemsetNodeGetParams(memsetNode, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HIP_CHECK(hipFree(devData));
  HIP_CHECK(hipGraphDestroy(graph));
}

/* Test verifies hipGraphMemsetNodeGetParams API Functional scenarios.
 */

bool memsetNodeCompare(hipMemsetParams *mNode1, hipMemsetParams *mNode2) {
  if (mNode1->dst != mNode2->dst)
    return false;
  if (mNode1->elementSize != mNode2->elementSize)
    return false;
  if (mNode1->height != mNode2->height)
    return false;
  if (mNode1->pitch != mNode2->pitch)
    return false;
  if (mNode1->value != mNode2->value)
    return false;
  if (mNode1->width != mNode2->width)
    return false;
  return true;
}

TEST_CASE("Unit_hipGraphMemsetNodeGetParams_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(char);
  constexpr size_t val = 0;

  char *devData;
  HIP_CHECK(hipMalloc(&devData, Nbytes));

  hipGraph_t graph;
  hipGraphNode_t memsetNode;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipStream_t streamForGraph;
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
  SECTION("Get Memset Param and verify.") {
    hipMemsetParams memsetGetParams;
    REQUIRE(hipSuccess == hipGraphMemsetNodeGetParams(memsetNode,
                                                      &memsetGetParams));
    // Validating the result
    REQUIRE(true == memsetNodeCompare(&memsetParams, &memsetGetParams));
  }
  SECTION("Set memset node params then Get and verify.") {
    constexpr size_t updateVal = 2;
    char *devData1;
    HIP_CHECK(hipMalloc(&devData1, Nbytes));
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(devData1);
    memsetParams.value = updateVal;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = Nbytes;
    memsetParams.height = 1;

    hipMemsetParams memsetGetParams;
    REQUIRE(hipSuccess == hipGraphMemsetNodeSetParams(memsetNode,
                                                      &memsetParams));
    REQUIRE(hipSuccess == hipGraphMemsetNodeGetParams(memsetNode,
                                                      &memsetGetParams));
    // Validating the result
    REQUIRE(true == memsetNodeCompare(&memsetParams, &memsetGetParams));
    HIP_CHECK(hipFree(devData1));
  }
  HIP_CHECK(hipFree(devData));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
