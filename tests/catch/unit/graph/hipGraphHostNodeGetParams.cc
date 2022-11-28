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
Testcase Scenarios of hipGraphHostNodeGetParams API:

Functional Scenarios:
1) Create a graph, add Host node to graph with desired  node params. Verify api fetches the node params
   which were mentioned while adding the host node.
2) Set host node params with hipGraphHostNodeSetParams, now get the params and verify both are same.
3) Create graph, Add Graphnodes and clones the graph. Add Hostnode to the cloned graph, update
   hostNode params using hipGraphHostNodeSetParams API  now get the params and verify both are same

Negative Scenarios:

1) Pass pGraphNode as nullptr and verify api doesn’t crash, returns error code.
2) Pass pNodeParams as nullptr and verify api doesn’t crash, returns error code.
3) Pass unintialized graph node
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
hipGraphHostNodeGetParams API
*/
TEST_CASE("Unit_hipGraphHostNodeGetParams_Negative") {
  constexpr size_t N = 1024;
  hipGraph_t graph;
  int *A_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, &C_d,
                           &A_h, nullptr, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph,
                                 nullptr,
                                 0, &hostParams));
  hipHostNodeParams GethostParams;

  SECTION("Passing nullptr to graph node") {
    REQUIRE(hipGraphHostNodeGetParams(nullptr, &GethostParams)
                                      == hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to hostParams") {
    REQUIRE(hipGraphHostNodeGetParams(hostNode, nullptr)
                                      == hipErrorInvalidValue);
  }

  SECTION("Passing unintialized graphNode") {
    hipGraphNode_t unint_graphnode{nullptr};
    REQUIRE(hipGraphHostNodeGetParams(unint_graphnode, &GethostParams)
                                      == hipErrorInvalidValue);
  }
  HIP_CHECK(hipGraphDestroy(graph));
}
/*
This testcase verifies hipGraphHostNodeGetParams API in cloned graph
Creates graph, Add graph nodes and clone the graph
Add HostNode to the cloned graph, update hostNode using hipGraphHostNodeSetParams,
then get the host node params using hipGraphHostNodeGetParams API  and
compare it.
*/
TEST_CASE("Unit_hipGraphHostNodeGetParams_ClonedGraphwithHostNode") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, &C_d,
                           &A_h, nullptr, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_C,
                 memcpyD2H_AC;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr,
                                    0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr,
                                    0, C_d, C_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_AC, graph, nullptr,
                                    0, A_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A,
                                    &memcpyD2H_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_C,
                                    &memcpyD2H_AC, 1));

  hipGraph_t clonedgraph;
  HIP_CHECK(hipGraphClone(&clonedgraph, graph));

  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, clonedgraph,
                                 nullptr,
                                 0, &hostParams));
  hipHostNodeParams sethostParams = {0, 0};
  sethostParams.fn = callbackfunc_setparams;
  sethostParams.userData = C_h;
  HIP_CHECK(hipGraphHostNodeSetParams(hostNode, &sethostParams));
  hipHostNodeParams gethostParams;
  HIP_CHECK(hipGraphHostNodeGetParams(hostNode, &gethostParams));
  REQUIRE(memcmp(&sethostParams, &gethostParams, sizeof(hipHostNodeParams))
                      == 0);
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
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedgraph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
}

/*
This testcase verifies the following scenarios
Create graph, Adds host node to the graph, updates it
with hipGraphHostNodeSetParams and gets the host node
params using hipGraphHostNodeGetParams API and validates
it
*/
void hipGraphHostNodeGetParams_func(bool setparams) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, &C_d,
                           &A_h, nullptr, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyD2H_AC, memcpyH2D_C;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr,
                                    0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr,
                                    0, C_d, C_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_AC, graph, nullptr,
                                    0, A_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  hipGraphNode_t hostNode;
  hipHostNodeParams hostParams = {0, 0};
  hostParams.fn = callbackfunc;
  hostParams.userData = A_h;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph,
                                 nullptr,
                                 0, &hostParams));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A,
                                    &memcpyD2H_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_C,
                                    &memcpyD2H_AC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H_AC,
                                    &hostNode, 1));

  if (setparams) {
    hipHostNodeParams sethostParams = {0, 0};
    sethostParams.fn = callbackfunc_setparams;
    sethostParams.userData = C_h;
    HIP_CHECK(hipGraphHostNodeSetParams(hostNode, &sethostParams));

    hipHostNodeParams gethostParams;
    HIP_CHECK(hipGraphHostNodeGetParams(hostNode, &gethostParams));
    REQUIRE(memcmp(&sethostParams, &gethostParams, sizeof(hipHostNodeParams))
        == 0);
  } else {
    hipHostNodeParams gethostParams;
    HIP_CHECK(hipGraphHostNodeGetParams(hostNode, &gethostParams));
    REQUIRE(memcmp(&hostParams, &gethostParams, sizeof(hipHostNodeParams))
        == 0);
  }
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result

  if (setparams) {
    for (size_t i = 0; i < N; i++) {
      if (C_h[i] != static_cast<int>(i * i)) {
        INFO("Validation failed i " <<  i  << "C_h[i] "<< C_h[i]);
        REQUIRE(false);
      }
    }
  } else {
    for (size_t i = 0; i < N; i++) {
      if (A_h[i] != static_cast<int>(i)) {
        INFO("Validation failed i " <<  i  << "C_h[i] "<< C_h[i]);
        REQUIRE(false);
      }
    }
  }

  HipTest::freeArrays<int>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
/*
This testcase verifies hipGraphHostNodeGetParams API by
adding host node to graph and gets the host params and
validates it
*/
TEST_CASE("Unit_hipGraphHostNodeGetParams_BasicFunc") {
  hipGraphHostNodeGetParams_func(false);
}

/*
This testcase verifies hipGraphHostNodeGetParams API by
adding host node to graph, updates host node params
using hipGraphHostNodeSetParams  and gets the host params
validates it
*/
TEST_CASE("Unit_hipGraphHostNodeGetParams_SetParams") {
  hipGraphHostNodeGetParams_func(true);
}
