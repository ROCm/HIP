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
Testcase Scenarios of hipGraphHostNodeSetParams API:

Functional:
1. Creates graph, Adds HostNode, update hostNode params using hipGraphHostNodeSetParams API
   and validates the result
2. Create graph, Add Graphnodes and clones the graph. Add Hostnode to the cloned graph, update
   hostNode params using hipGraphHostNodeSetParams API and validate the result

Negative:

1) Pass pGraphNode as nullptr and verify api doesn’t crash, returns error code.
2) Pass pNodeParams as nullptr and verify api doesn’t crash, returns error code.
3) Pass hipHostNodeParams::hipHostFn_t as nullptr and verify api doesn't crash, returns error code.
4) Pass unintialized host params
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#define SIZE 1024

static void callbackfunc(void *A_h) {
  int *A = reinterpret_cast<int *>(A_h);
  for (int i = 0; i < SIZE; i++) {
     A[i] = i;
  }
}

static void callbackfunc_setparams(void *B_h) {
  int *B = reinterpret_cast<int *>(B_h);
  for (int i = 0; i < SIZE; i++) {
     B[i] = i * i;
  }
}

/*
This testcase verifies the negative scenarios of
hipGraphHostNodeSetParams API
*/
TEST_CASE("Unit_hipGraphHostNodeSetParams_Negative") {
  constexpr size_t N = 1024;
  hipGraph_t graph;
  int *A_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, &C_d,
                           &A_h, nullptr, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams;
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph,
                                 nullptr,
                                 0, &hostParams));
#if HT_NVIDIA
  SECTION("Passing nullptr to graph node") {
    REQUIRE(hipGraphHostNodeSetParams(nullptr, &hostParams)
                                      == hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to hostParams") {
    REQUIRE(hipGraphHostNodeSetParams(hostNode, nullptr)
                                      == hipErrorInvalidValue);
  }
#endif

  SECTION("Passing nullptr to host func") {
    hostParams.fn = nullptr;
    REQUIRE(hipGraphHostNodeSetParams(hostNode, &hostParams)
                                      == hipErrorInvalidValue);
  }


  SECTION("Passing unintialized hostParams") {
    hipHostNodeParams unintParams = {0, 0};
    REQUIRE(hipGraphHostNodeSetParams(hostNode, &unintParams)
                                      == hipErrorInvalidValue);
  }
  HIP_CHECK(hipGraphDestroy(graph));
}
/*
This testcase verifies hipGraphHostNodeSetParams API in cloned graph
Creates graph, Add graph nodes and clone the graph
Add HostNode to the cloned graph,update the host params using
hipGraphHostNodeSetParams API and validates the result
*/
TEST_CASE("Unit_hipGraphHostNodeSetParams_ClonedGraphwithHostNode") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, &C_d,
                           &A_h, nullptr, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_C,
                 memcpyD2H_AC;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr,
                                    0, C_d, C_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_AC, graph, nullptr,
                                    0, A_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_C,
                                    &memcpyD2H_AC, 1));

  hipGraph_t clonedgraph;
  HIP_CHECK(hipGraphClone(&clonedgraph, graph));

  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams;
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, clonedgraph,
                                 nullptr,
                                 0, &hostParams));


  hipHostNodeParams sethostParams;
  sethostParams.fn = callbackfunc_setparams;
  sethostParams.userData = C_h;
  HIP_CHECK(hipGraphHostNodeSetParams(hostNode, &sethostParams));


  // Instantiate and launch the cloned graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, clonedgraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != static_cast<int>(i * i)) {
      INFO("Validation failed i " <<  i  << "C_h[i] "<< C_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
This testcase verifies the following scenario
Create graph, Adds host node to the graph,
updates the host params using hipGraphHostNodeSetParams API
and validates the result
*/
TEST_CASE("Unit_hipGraphHostNodeSetParams_BasicFunc") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, &C_d,
                           &A_h, nullptr, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyD2H_AC, memcpyH2D_C;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr,
                                    0, C_d, C_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_AC, graph, nullptr,
                                    0, A_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams;
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph,
                                 nullptr,
                                 0, &hostParams));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_C,
                                    &memcpyD2H_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H_AC,
                                    &hostNode, 1));

  hipHostNodeParams sethostParams;
  sethostParams.fn = callbackfunc_setparams;
  sethostParams.userData = C_h;
  HIP_CHECK(hipGraphHostNodeSetParams(hostNode, &sethostParams));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != static_cast<int>(i * i)) {
      INFO("Validation failed i " <<  i  << "C_h[i] "<< C_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
