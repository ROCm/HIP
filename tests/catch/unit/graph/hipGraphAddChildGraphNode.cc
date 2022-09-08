/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
Testcase Scenarios of hipGraphAddChildGraphNode API:

Functional:
1. Create child graph as root node and execute the main graph.
2. Create multiple child graph nodes and check the behaviour
3. Clone the child graph node, Add new nodes and execute the cloned graph
4. Create child graph, add it to main graph and execute child graph
5. Pass original graph as child graph and execute the org graph

Negative:
1. Pass nullptr to graph node
2. Pass nullptr to graph
3. Pass invalid number of numDepdencies
4. Pass nullptr to child graph
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/*
This testcase verifies the negative scenarios of
hipGraphAddChildGraphNode API
*/
TEST_CASE("Unit_hipGraphAddChildGraphNode_Negative") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, childgraph1;
  int *A_d{nullptr}, *B_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, nullptr,
      &A_h, &B_h, nullptr,
      N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, childGraphNode1;
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph1, nullptr,
        0, A_h, B_d,
        Nbytes, hipMemcpyDeviceToHost));

  SECTION("Pass nullptr to graph noe") {
    REQUIRE(hipGraphAddChildGraphNode(nullptr, graph,
          nullptr, 0, childgraph1)
        == hipErrorInvalidValue);
  }

  SECTION("Pass nullptr to graph") {
    REQUIRE(hipGraphAddChildGraphNode(&childGraphNode1, nullptr,
          nullptr, 0, childgraph1)
        == hipErrorInvalidValue);
  }

  SECTION("Pass nullptr to child graph") {
    REQUIRE(hipGraphAddChildGraphNode(&childGraphNode1, graph,
          nullptr, 0, nullptr)
        == hipErrorInvalidValue);
  }

  SECTION("Pass invalid depdencies") {
    REQUIRE(hipGraphAddChildGraphNode(&childGraphNode1, graph,
          nullptr, 10, childgraph1)
        == hipErrorInvalidValue);
  }
}

/*
This testcase verifies the following scenario
Creates the graph, add the graph as a child node
and verify the number of the nodes in the original graph
*/
TEST_CASE("Unit_hipGraphAddChildGraphNode_OrgGraphAsChildGraph") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *B_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, nullptr, &A_h, &B_h, nullptr, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, childGraphNode1;
  size_t numNodes;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_h, B_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph,
                                      nullptr, 0, graph));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &memcpyH2D_A, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify number of nodes
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  REQUIRE(numNodes == 3);
  HipTest::freeArrays<int>(A_d, B_d, nullptr, A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
This testcase verifies the following scenario
Create graph, Add child nodes to the graph and execute only the
child graph node and verify the behaviour
*/
TEST_CASE("Unit_hipGraphAddChildGraphNode_ExecuteChildGraph") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, childgraph1;
  hipGraphExec_t graphExec;
  int *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(nullptr, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, childGraphNode1, memcpyH2D_C;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph1, nullptr,
                                    0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph1, nullptr,
                                    0, A_h, B_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr,
                                    0, C_d, C_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr,
                                    0, A_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph,
                                      nullptr, 0, childgraph1));

  HIP_CHECK(hipGraphAddDependencies(childgraph1, &memcpyH2D_B,
                                    &memcpyH2D_A, 1));

  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, childgraph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify childgraph execution result
  for (size_t i = 0; i < N; i++) {
    if (B_h[i] != A_h[i]) {
      INFO("Validation failed B_h[i] " <<  B_h[i]  << "A_h[i] "<< A_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(nullptr, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
This testcase verifies the following scenario
creates graph, Add child nodes to graph, clone the graph and execute
the cloned graph
*/
TEST_CASE("Unit_hipGraphAddChildGraphNode_CloneChildGraph") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, childgraph1, clonedgraph;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *B_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, nullptr, &A_h, &B_h, nullptr, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&clonedgraph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, childGraphNode1;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph1, nullptr,
                                    0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph,
                                      nullptr, 0, childgraph1));

  // Added new memcpy node to the cloned graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_h, A_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode1, &memcpyH2D_B, 1));

  // Cloned the graph
  HIP_CHECK(hipGraphClone(&clonedgraph, graph));

  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, clonedgraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify childgraph execution result
  for (size_t i = 0; i < N; i++) {
    if (B_h[i] != A_h[i]) {
      INFO("Validation failed B_h[i] " <<  B_h[i]  << "A_h[i] "<< A_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, B_d, nullptr, A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
This testcase verifies the following scenario
Create graph, add multiple child nodes and validates the
behaviour
*/
TEST_CASE("Unit_hipGraphAddChildGraphNode_MultipleChildNodes") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  size_t NElem{N};
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph, childgraph1, childgraph2;
  hipGraphExec_t graphExec;
  hipKernelNodeParams kernelNodeParams{};
  hipGraphNode_t kernel_vecAdd;
  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, childGraphNode1,
                 childGraphNode2, memcpyD2H_C;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphCreate(&childgraph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph1, nullptr,
                                    0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph2, nullptr,
                                    0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph,
                                      nullptr, 0, childgraph1));
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode2, graph,
                                      nullptr, 0, childgraph2));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr,
                                    0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0,
                                                        &kernelNodeParams));

  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode1,
                                    &childGraphNode2, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode2,
                                    &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2H_C, 1));

  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify childgraph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph2));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
/**
 This testcase verifies hipGraphAddChildGraphNode functionality
 where root node is the child node.
 */
TEST_CASE("Unit_hipGraphAddChildGraphNode_SingleChildNode") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memset_A, memset_B, memsetKer_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  hipMemsetParams memsetParams{};
  size_t NElem{N};
  int memsetVal{};
  hipGraph_t childgraph;
  hipGraphNode_t ChildGraphNode;

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&childgraph, 0));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, childgraph, nullptr, 0,
                                                              &memsetParams));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(B_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_B, childgraph, nullptr, 0,
                                                              &memsetParams));

  void* kernelArgs1[] = {&C_d, &memsetVal, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func =
                       reinterpret_cast<void *>(HipTest::memsetReverse<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&memsetKer_C, childgraph, nullptr, 0,
                                                        &kernelNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph, nullptr,
                                    0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph, nullptr,
                                    0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, childgraph, nullptr,
                                    0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, childgraph, nullptr, 0,
                                                        &kernelNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memset_A, &memcpyH2D_A, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memset_B, &memcpyH2D_B, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memcpyH2D_A,
                                    &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memcpyH2D_B,
                                    &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &memsetKer_C,
                                    &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(childgraph, &kernel_vecAdd,
                                    &memcpyD2H_C, 1));

  HIP_CHECK(hipGraphAddChildGraphNode(&ChildGraphNode, graph,
                                      nullptr, 0, childgraph));
  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify childgraph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
