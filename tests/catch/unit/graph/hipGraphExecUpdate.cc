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
Functional-
1) Make a clone of the created graph and update the executable-graph from a clone or same graph again.
2) Update the executable-graph from a graph and make sure they are taking effect.
Negative-
1) When Pass hGraphExec as nullptr and verify api returns error code.
2) When Pass hGraph as nullptr and verify api returns error code.
3) When Pass hErrorNode_out as nullptr and verify api returns error code.
4) When Pass updateResult_out as nullptr and verify api returns error code.
5) When the a graphExec was updated with with different type of node and verify api returns error code.
6) When a node is deleted in hGraph but not its pair from hGraphExec and verify api returns error code.
7) When a node is deleted in hGraphExec but not its pair from hGraph and verify api returns error code.
8) When grpah dependencies differ but graph have same node and verify api returns error code.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/* Test verifies hipGraphExecUpdate API Negative nullptr check scenarios.
 */
TEST_CASE("Unit_hipGraphExecUpdate_Negative_Basic") {
  hipError_t ret;
  hipGraph_t graph{};
  hipGraphExec_t graphExec{};
  hipGraphNode_t hErrorNode_out{};
  hipGraphExecUpdateResult updateResult_out{};

  SECTION("Pass hGraphExec as nullptr") {
    ret = hipGraphExecUpdate(nullptr, graph, &hErrorNode_out,
                             &updateResult_out);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hGraph as nullptr") {
    ret = hipGraphExecUpdate(graphExec, nullptr, &hErrorNode_out,
                             &updateResult_out);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hErrorNode_out as nullptr") {
    ret = hipGraphExecUpdate(graphExec, graph, nullptr, &updateResult_out);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass updateResult_out as nullptr") {
    ret = hipGraphExecUpdate(graphExec, graph, &hErrorNode_out, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
}

/* Test verifies hipGraphExecUpdate API Negative scenarios.
   When the a graphExec was updated with with different type of node
 */
TEST_CASE("Unit_hipGraphExecUpdate_Negative_TypeChange") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(char);
  constexpr size_t val = 0;
  char *devData;
  int *A_d, *A_h;

  HipTest::initArrays<int>(&A_d, nullptr, nullptr,
                           &A_h, nullptr, nullptr, N, false);

  HIP_CHECK(hipMalloc(&devData, Nbytes));

  hipGraph_t graph, graph2;
  hipGraphExec_t graphExec;
  hipStream_t streamForGraph;
  hipGraphNode_t memsetNode, memcpy_A, hErrorNode_out;
  hipError_t ret;
  hipGraphExecUpdateResult updateResult_out;

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

  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph2, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));

  // graphExec was created before memcpyTemp was added to graph.
  ret = hipGraphExecUpdate(graphExec, graph2, &hErrorNode_out,
                           &updateResult_out);

  REQUIRE(hipGraphExecUpdateErrorNodeTypeChanged == updateResult_out);
  REQUIRE(hipErrorGraphExecUpdateFailure == ret);


  HIP_CHECK(hipFree(devData));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/* Test verifies hipGraphExecUpdate API Negative scenarios.
   When the count of nodes differ in hGraphExec and hGraph
 */
TEST_CASE("Unit_hipGraphExecUpdate_Negative_CountDiffer") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};

  int *hData = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hData != nullptr);
  memset(hData, 0, Nbytes);

  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C, memcpyTemp;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipError_t ret;
  hipGraph_t graph1, graph2, graph3;
  hipGraphExec_t graphExec1, graphExec2;
  hipStream_t streamForGraph;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph1, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph1, nullptr, 0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph1, nullptr, 0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph1, nullptr, 0,
                                                        &kernelNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &kernel_vecAdd, &memcpy_C, 1));

  // Create a cloned graph and added extra node to it
  HIP_CHECK(hipGraphClone(&graph2, graph1));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyTemp, graph2, nullptr, 0,
                                    C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));

  SECTION("When a node deleted from Graph but not from its pair GraphExec") {
    ret = hipGraphExecUpdate(graphExec2, graph1, &hErrorNode_out,
                             &updateResult_out);
    REQUIRE(hipErrorGraphExecUpdateFailure == ret);
  }
  SECTION("When a node deleted from GraphExec but not from its pair Graph") {
    ret = hipGraphExecUpdate(graphExec1, graph2, &hErrorNode_out,
                                 &updateResult_out);
    REQUIRE(hipErrorGraphExecUpdateFailure == ret);
  }
  SECTION("When the dependent nodes of a pair differ") {
    HIP_CHECK(hipGraphCreate(&graph3, 0));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph3, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph3, nullptr, 0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph3, nullptr, 0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph3, nullptr, 0,
                                                        &kernelNodeParams));
    // Create dependencies
    HIP_CHECK(hipGraphAddDependencies(graph3, &memcpy_A, &kernel_vecAdd, 1));
    HIP_CHECK(hipGraphAddDependencies(graph3, &memcpy_B, &kernel_vecAdd, 1));
    HIP_CHECK(hipGraphAddDependencies(graph3, &memcpy_C, &kernel_vecAdd, 1));

    ret = hipGraphExecUpdate(graphExec1, graph3, &hErrorNode_out,
                             &updateResult_out);
    REQUIRE(hipErrorGraphExecUpdateFailure == ret);
    HIP_CHECK(hipGraphDestroy(graph3));
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
  free(hData);
}

/* Functional Scenario -
1) Make a clone of the created graph and update the executable-graph from a clone graph.
2) Update the executable-graph from a graph and make sure they are taking effect.
*/

TEST_CASE("Unit_hipGraphExecUpdate_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};

  int *hData = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hData != nullptr);
  memset(hData, 0, Nbytes);

  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C;
  hipGraphNode_t kernel_vecAdd, kernel_vecSquare;
  hipKernelNodeParams kernelNodeParams{};
  hipGraph_t graph, graph2, clonedgraph{};
  hipGraphExec_t graphExec;
  hipStream_t streamForGraph;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func =
                   reinterpret_cast<void *>(HipTest::vector_square<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecSquare, graph, nullptr, 0,
                                                        &kernelNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &kernel_vecSquare, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_B, &kernel_vecSquare, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecSquare, &memcpy_C, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Update graphExec with clone graph") {
    HIP_CHECK(hipGraphClone(&clonedgraph, graph));
    HIP_CHECK(hipGraphExecUpdate(graphExec, clonedgraph, &hErrorNode_out,
                                 &updateResult_out));
  }

  // Code for new graph creation with samilar node setup
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph2, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph2, nullptr, 0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph2, nullptr, 0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphMemcpyNodeSetParams1D(memcpy_C, hData, C_d, Nbytes,
                                          hipMemcpyDeviceToHost));

  memset(&kernelNodeParams, 0, sizeof(hipKernelNodeParams));
  void* kernelArgs2[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph2, nullptr, 0,
                                                        &kernelNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &kernel_vecAdd, &memcpy_C, 1));

  // Update the graphExec graph from graph -> graph2
  HIP_CHECK(hipGraphExecUpdate(graphExec, graph2, &hErrorNode_out,
                               &updateResult_out));
  REQUIRE(updateResult_out == hipGraphExecUpdateSuccess);

  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, hData, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(clonedgraph));
  free(hData);
}
