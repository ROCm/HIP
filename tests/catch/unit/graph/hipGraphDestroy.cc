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

/**
Negative Testcase Scenarios :
1) Pass graph as nullptr and verify.
2) Destroy already destroyed graph and check api returns error code.
3) Destroy graph when is in use and make sure api handles it gracefully.
*/

TEST_CASE("Unit_hipGraphDestroy_Negative") {
  hipError_t ret;
  SECTION("Deleting HipGraph with nullptr") {
    ret = hipGraphDestroy(nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Destroy already destroyed graph and check api returns error code") {
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphDestroy(graph));

    ret = hipGraphDestroy(graph);
    REQUIRE(hipErrorIllegalState == ret);
  }
  SECTION("Destroy graph when is in use and make sure api handles"
          " it gracefully.") {
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipStream_t streamForGraph;
    hipGraphNode_t memsetNode;

    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipStreamCreate(&streamForGraph));

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

    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    REQUIRE(graphExec != nullptr);

    HIP_CHECK(hipGraphDestroy(graph));

    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));

    HIP_CHECK(hipFree(devData));
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipStreamDestroy(streamForGraph));
  }
}

