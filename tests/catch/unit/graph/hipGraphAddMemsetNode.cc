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
Negative Testcase Scenarios for api hipGraphAddMemsetNode :
1) Pass pGraphNode as nullptr and check if api returns error.
2) Pass pGraphNode as un-initialize object and check.
3) Pass Graph as nullptr and check if api returns error.
4) Pass Graph as empty object(skipping graph creation), api should return error code.
5) Pass pDependencies as nullptr, api should return success.
6) Pass numDependencies is max(size_t) and pDependencies is not valid ptr, api expected to return error code.
7) Pass pDependencies is nullptr, but numDependencies is non-zero, api expected to return error.
8) Pass pMemsetParams as nullptr and check if api returns error code.
9) Pass pMemsetParams as un-initialize object and check if api returns error code.
10) Pass hipMemsetParams::dst as nullptr should return error code.
11) Pass hipMemsetParams::element size other than 1, 2, or 4 and check api should return error code.
12) Pass hipMemsetParams::height as zero and check api should return error code.
*/

#include <hip_test_common.hh>

/**
 * Negative Test for API hipGraphAddMemsetNode
 */

TEST_CASE("Unit_hipGraphAddMemsetNode_Negative") {
  hipError_t ret;
  hipGraph_t graph;
  hipGraphNode_t memsetNode;
  char *devData;

  HIP_CHECK(hipMalloc(&devData, 1024));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(devData);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = 1024;
  memsetParams.height = 1;

  SECTION("Pass pGraphNode as nullptr") {
    ret = hipGraphAddMemsetNode(nullptr, graph, nullptr, 0, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pGraphNode as un-initialize object") {
    hipGraphNode_t memsetNode_1;
    ret = hipGraphAddMemsetNode(&memsetNode_1, graph,
                                nullptr, 0, &memsetParams);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass graph as nullptr") {
    ret = hipGraphAddMemsetNode(&memsetNode, nullptr,
                                nullptr, 0, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass Graph as empty object") {
    hipGraph_t graph_1{};
    ret = hipGraphAddMemsetNode(&memsetNode, graph_1,
                                nullptr, 0, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pDependencies as nullptr") {
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass numDependencies is max and pDependencies is not valid ptr") {
    ret = hipGraphAddMemsetNode(&memsetNode, graph,
                                nullptr, INT_MAX, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pDependencies as nullptr, but numDependencies is non-zero") {
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 9, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pMemsetParams as nullptr") {
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pMemsetParams as un-initialize object") {
    hipMemsetParams memsetParams1;
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                &memsetParams1);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hipMemsetParams::dst as nullptr") {
    memsetParams.dst = nullptr;
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hipMemsetParams::element size other than 1, 2, or 4") {
    memsetParams.dst = reinterpret_cast<void*>(devData);
    memsetParams.elementSize = 9;
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hipMemsetParams::height as zero") {
    memsetParams.elementSize = sizeof(char);
    memsetParams.height = 0;
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HIP_CHECK(hipFree(devData));
  HIP_CHECK(hipGraphDestroy(graph));
}
