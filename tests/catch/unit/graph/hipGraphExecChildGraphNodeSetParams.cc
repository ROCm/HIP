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
  Testcase Scenarios of hipGraphExecChildGraphNodeSetParams API:

  Functional Scenarios:
  1. Create child graph, Instantiate the graph and update the child graph
     using hipGraphExecChildGraphNodeSetParams API
  2. Create child graph with topology, Instantiate the graph
     and update the child graph
     using hipGraphExecChildGraphNodeSetParams API

  Negative Scenarios:
  1. Pass nullptr to child graph
  2. Pass nullptr to graphnode
  3. Pass nullptr to graphExec
  4. Pass uninitialized graph node
  5. Pass orginial graph node instead of child graph node
  6. Change topology of child graph node
 **/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>


TEST_CASE("Unit_hipGraphExecChildGraphNodeSetParams_Negative") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, childgraph1, childgraph2;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  std::vector<hipGraphNode_t> childdependencies;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphCreate(&childgraph2, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_A,
                 memcpyH2D_B_child, childGraphNode1;
  HIP_CHECK(hipMemcpy(C_d, C_h, Nbytes, hipMemcpyHostToDevice));

  // Adding MemcpyNode to graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr,
        0, A_d, A_h,
        Nbytes, hipMemcpyHostToDevice));

  // Adding memcpyNode to childgraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph1, nullptr,
        0, B_d, A_d,
        Nbytes, hipMemcpyDeviceToDevice));

  // Adding childnode to graph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph,
        nullptr, 0, childgraph1));

  // Adding memcpynode to graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, graph, nullptr,
        0, B_h, B_d,
        Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &childGraphNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode1, &memcpyD2H_A, 1));

  // Adding memcpynode to new childgraph which is used to update the
  // childgraph node
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B_child, childgraph2, nullptr,
        0, B_d, C_d,
        Nbytes, hipMemcpyDeviceToDevice));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Pass nullptr to graphExec") {
    REQUIRE(hipGraphExecChildGraphNodeSetParams(nullptr, childGraphNode1,
                                                childgraph2)
                                                == hipErrorInvalidValue);
  }
  SECTION("Pass nullptr to child graph node") {
    REQUIRE(hipGraphExecChildGraphNodeSetParams(graphExec, nullptr,
                                                childgraph2)
                                                == hipErrorInvalidValue);
  }
  SECTION("Pass nullptr to child graph") {
    REQUIRE(hipGraphExecChildGraphNodeSetParams(graphExec,
                                                childGraphNode1,
                                                nullptr)
                                                == hipErrorInvalidValue);
  }
  SECTION("Passing parent graph instead of child graph") {
    REQUIRE(hipGraphExecChildGraphNodeSetParams(graphExec,
                                                childGraphNode1, graph)
                                                != hipSuccess);
  }
  SECTION("Updating the child graph topology") {
    hipGraphNode_t newnode;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&newnode, childgraph2, nullptr,
                                      0, B_d, C_d,
                                      Nbytes, hipMemcpyDeviceToDevice));
    HIP_CHECK(hipGraphAddDependencies(childgraph2, &memcpyH2D_B_child,
                                      &newnode, 1));

    REQUIRE(hipGraphExecChildGraphNodeSetParams(graphExec, childGraphNode1,
                                                childgraph2)
                                                != hipSuccess);
  }

  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(childgraph2));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
   This testcase verifies the following scenario
   Create graph, add child node to graph, Instantiate the graph
   and update the child graph node with a new graph
   using hipGraphExecChildGraphNodeSetParams API
   and execute it
   */
TEST_CASE("Unit_hipGraphExecChildGraphNodeSetParams_BasicFunc") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, childgraph1, childgraph2;
  hipGraphExec_t graphExec;
  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  std::vector<hipGraphNode_t> childdependencies;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphCreate(&childgraph2, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_A,
                 memcpyH2D_B_child, childGraphNode1;
  HIP_CHECK(hipMemcpy(C_d, C_h, Nbytes, hipMemcpyHostToDevice));

  // Adding MemcpyNode to graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr,
                                    0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));

  // Adding memcpyNode to childgraph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph1, nullptr,
                                    0, B_d, A_d,
                                    Nbytes, hipMemcpyDeviceToDevice));

  // Adding childnode to graph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph,
                                      nullptr, 0, childgraph1));

  // Adding memcpynode to graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, graph, nullptr,
                                    0, B_h, B_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &childGraphNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode1, &memcpyD2H_A, 1));

  // Adding memcpynode to new childgraph which is used to update the
  // childgraph node
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B_child, childgraph2, nullptr,
                                    0, B_d, C_d,
                                    Nbytes, hipMemcpyDeviceToDevice));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Update the childgraph node
  HIP_CHECK(hipGraphExecChildGraphNodeSetParams(graphExec, childGraphNode1,
                                                childgraph2));

  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  for (unsigned int i = 0; i < N; i++) {
    if (B_h[i] != C_h[i]) {
      WARN("Validation failed " << B_h[i] << "\t" << C_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(childgraph2));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
   This testcase verifies the following scenario
   Create graph, Create child graph with a topology and
   add child node to graph, Instantiate the graph
   and update the child graph node with a new graph
   using hipGraphExecChildGraphNodeSetParams API
   and execute it
   */
TEST_CASE("Unit_hipGraphExecChildGraphNodeSetParams_ChildTopology") {
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
  std::vector<hipGraphNode_t> childdependencies, childdependencies1;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyH2D_C, childGraphNode1,
                 memcpyD2H_A, memcpyD2D_AB;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph1, 0));
  HIP_CHECK(hipGraphCreate(&childgraph2, 0));

  // Adding memcpy node to graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr,
                                    0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));

  // Adding memcpy node to child graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2D_AB, childgraph1, nullptr,
                                    0, B_d, A_d,
                                    Nbytes, hipMemcpyDeviceToDevice));
  childdependencies.push_back(memcpyD2D_AB);

  // Adding memcpy node to child graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph1,
                                    childdependencies.data(),
                                    childdependencies.size(), B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));

  // Adding memcpy node to child graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, childgraph1,
                                    childdependencies.data(),
                                    childdependencies.size(), C_d, C_h,
                                    Nbytes, hipMemcpyHostToDevice));

  childdependencies.clear();
  childdependencies.push_back(memcpyH2D_B);
  childdependencies.push_back(memcpyH2D_C);

  void* kernelArgs2[] = {&B_d, &C_d, &A_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;

  // Adding kernel node to child graph
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, childgraph1,
                                  childdependencies.data(),
                                  childdependencies.size(),
                                  &kernelNodeParams));

  // Adding child node to graph
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode1, graph,
                                      nullptr, 0, childgraph1));

  // Adding memcpy node to graph
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, graph, nullptr,
                                    0, A_h, A_d,
                                    Nbytes, hipMemcpyDeviceToHost));


  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &childGraphNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &childGraphNode1, &memcpyD2H_A, 1));

  // Creating another child graph for updating parameters with the same topology
  // and passing the new child graph to hipGraphExecChildGraphNodeSetParams API
  hipGraphNode_t memcpyD2D_AB1, memcpyH2D_B1, memcpyH2D_C1, kernel_vecAdd1;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2D_AB1, childgraph2, nullptr,
                                    0, B_d, A_d,
                                    Nbytes, hipMemcpyDeviceToDevice));
  childdependencies.clear();
  childdependencies.push_back(memcpyD2D_AB1);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B1, childgraph2,
                                    childdependencies.data(),
                                    childdependencies.size(), B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C1, childgraph2,
                                    childdependencies.data(),
                                    childdependencies.size(), C_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));

  childdependencies.clear();
  childdependencies.push_back(memcpyH2D_B1);
  childdependencies.push_back(memcpyH2D_C1);

  void* kernelArgs21[] = {&B_d, &C_d, &A_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs21);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd1, childgraph2,
                                  childdependencies.data(),
                                  childdependencies.size(),
                                  &kernelNodeParams));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphExecChildGraphNodeSetParams(graphExec,
                                                childGraphNode1, childgraph2));

  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify child graph execution result
  HipTest::checkVectorADD(B_h, B_h, A_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph1));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
