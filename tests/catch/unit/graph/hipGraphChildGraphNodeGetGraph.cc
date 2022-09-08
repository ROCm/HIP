/*Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
Testcase Scenarios of hipGraphChildGraphNodeGetGraph API:

Functional Scenarios:
1. Get the child graph node from the original graph and execute it

Negative Scenarios:
1. Pass nullptr to graph
2. Pass nullptr to graphnode
3. Pass uninitialized graph node
4. Pass orginial graph node instead of child graph node
**/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/*
This testcase verifies the following scenario
Create graph, add multiple child nodes and gets the
graph of one of the child nodes using hipGraphChildGraphNodeGetGraph API
executes it and validates the results
*/
TEST_CASE("Unit_hipGraphChildGraphNodeGetGraph_Functional") {
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
  hipGraph_t Getgraph;
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(childGraphNode1, &Getgraph));
  // Instantiate and launch the child graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, Getgraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));


  // Verify child graph execution result
  HIP_CHECK(hipMemcpy(C_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    if (A_h[i] != C_h[i]) {
      INFO("Validation failed " << A_h[i] << C_h[i]);
      REQUIRE(false);
    }
  }
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph2));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
This testcase verifies the negative scenarios
of hipGraphChildGraphNodeGetGraph API
*/
TEST_CASE("Unit_hipGraphChildGraphNodeGetGraph_Negative") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, childgraph1;
  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, childGraphNode1;
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph1, nullptr,
                                    0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph,
                                      nullptr, 0, childgraph1));

  hipGraph_t Getgraph;
  SECTION("nullptr to child node") {
    REQUIRE((hipGraphChildGraphNodeGetGraph(nullptr, &Getgraph))
                                            == hipErrorInvalidValue);
  }
#if HT_NVIDIA
  SECTION("nullptr to graph") {
    REQUIRE((hipGraphChildGraphNodeGetGraph(childGraphNode1, nullptr))
                                            == hipErrorInvalidValue);
  }

  SECTION("Passing parent instead of child graph node") {
    REQUIRE((hipGraphChildGraphNodeGetGraph(memcpyH2D_A, &Getgraph))
                                            == hipErrorInvalidValue);
  }

  SECTION("Passing unintialized node") {
    hipGraphNode_t unint_node;
    REQUIRE((hipGraphChildGraphNodeGetGraph(unint_node, &Getgraph))
                                            == hipErrorInvalidValue);
  }
#endif
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(childgraph1));
}
