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
3) Passing hipMemsetParams::dst as nullptr should return error code.
4) Passing hipMemsetParams::element size other than 1, 2, or 4 and check.
5) Passing hipMemsetParams::height as zero and check api should return error code.
Functional -
1) Add Memset node to graph, update the node params with set and
   launch the graph and check the set params are executing properly.
2) Add Memset node to graph, launch graph, then update the Memset node params
   with set and launch the graph and check updated params are taking effect.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

/* Test verifies hipGraphMemsetNodeSetParams API invalid params scenarios.
 */
TEST_CASE("Unit_hipGraphMemsetNodeSetParams_InvalidParams") {
  hipError_t ret;
  hipGraph_t graph;
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
  SECTION("Pass node as nullptr") {
    ret = hipGraphMemsetNodeSetParams(nullptr, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass GetNodeParams as nullptr") {
    ret = hipGraphMemsetNodeSetParams(memsetNode, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass dest as nullptr") {
    memsetParams.dst = nullptr;
    ret = hipGraphMemsetNodeSetParams(memsetNode, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#if HT_NVIDIA
  SECTION("Pass element size other than 1, 2, or 4") {
    memsetParams.dst = reinterpret_cast<void*>(devData);
    memsetParams.elementSize = 9;
    ret = hipGraphMemsetNodeSetParams(memsetNode, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass height as zero or negative") {
    memsetParams.elementSize = 2;
    memsetParams.height = 0;
    ret = hipGraphMemsetNodeSetParams(memsetNode, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#endif
  HIP_CHECK(hipFree(devData));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

static void validate_result(char *hData, size_t size, char val) {
  // Validating the result
  for (size_t i = 0; i < size; i++) {
    if (hData[i] != val) {
      WARN("Validation failed at- " << i << " hData[i] " << hData[i]);
      REQUIRE(false);
    }
  }
}

/* Test verifies hipGraphMemsetNodeSetParams API Functional scenarios.
 */
TEST_CASE("Unit_hipGraphMemsetNodeSetParams_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(char);
  constexpr size_t val = 0;
  constexpr size_t updateVal = 1;
  constexpr size_t updateVal2 = 2;
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

  SECTION("Update the memsetNode and check") {
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
    validate_result(A_h, Nbytes, updateVal);
  }
  SECTION("Update the memsetNode again and check") {
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(B_d);
    memsetParams.value = updateVal2;
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

    HIP_CHECK(hipMemcpy(B_h, B_d, Nbytes, hipMemcpyDeviceToHost));
    validate_result(B_h, Nbytes, updateVal2);
  }

  HipTest::freeArrays<char>(A_d, B_d, C_d,
                            A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
