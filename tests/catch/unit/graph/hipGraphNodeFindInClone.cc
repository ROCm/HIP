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

/*
Testcase Scenarios of hipGraphNodeFindInClone API:

Negative:

1) Pass nullptr to graph node
2) pass nullptr to original graph node
3) pass nullptr to clonedGraph
4) Pass original graph in place of the cloned graph
5) Pass invalid originalNode
6) Destroy the graph node in the original graph
   and try to get the deleted graph node
   from the cloned graph
7) Clone the graph,Add node to Original graph
   and try to find the original node in the cloned graph


Functional:

1) Get the graph node from the cloned graph corresponding to the original node
2) Create and clone the graph, modify the original graph and clone the graph again,
   then try to find the newly added graph node  from the cloned graph

*/

#include<hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>


/* This test covers the negative scenarios of
   hipGraphNodeFindInClone API */

TEST_CASE("Unit_hipGraphNodeFindInClone_Negative") {
  hipGraph_t graph;
  hipGraph_t clonedgraph;
  hipGraphNode_t graphnode, newnode;
  hipGraphNode_t clonedgraphnode;
  HIP_CHECK(hipGraphCreate(&graph, 0));


  int *A_d, *A_h, *B_d, *B_h;
  HipTest::initArrays<int>(&A_d, &B_d, nullptr, &A_h,
                           &B_h, nullptr, 1024, false);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&graphnode, graph, nullptr, 0, A_d, A_h,
                                    1024, hipMemcpyHostToDevice));
  // Cloned the graph
  HIP_CHECK(hipGraphClone(&clonedgraph, graph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&newnode, graph, nullptr, 0, B_d, B_h,
                                    1024, hipMemcpyHostToDevice));

  SECTION("Passing nullptr to Cloned graph") {
    REQUIRE(hipGraphNodeFindInClone(&clonedgraphnode, graphnode, nullptr)
                                    == hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to original graph") {
    REQUIRE(hipGraphNodeFindInClone(nullptr, graphnode, clonedgraph)
                                    == hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to graph node") {
    REQUIRE(hipGraphNodeFindInClone(&clonedgraphnode, nullptr, clonedgraph)
                                    == hipErrorInvalidValue);
  }
#if HT_NVIDIA
  SECTION("Pass uncloned graph") {
    REQUIRE(hipGraphNodeFindInClone(&clonedgraphnode, graphnode, graph)
                                    == hipErrorInvalidValue);
  }

  SECTION("Destroy the graph node and find in cloned graph") {
    HIP_CHECK(hipGraphDestroyNode(graphnode));
    REQUIRE(hipGraphNodeFindInClone(&clonedgraphnode, graphnode,
                                    clonedgraph)
        == hipErrorInvalidValue);
  }
#endif

  SECTION("Pass invalid original graphnode") {
    hipGraphNode_t unintialized_graphnode{nullptr};
    REQUIRE(hipGraphNodeFindInClone(&clonedgraphnode, unintialized_graphnode,
                                    graph)
                                    == hipErrorInvalidValue);
  }

  SECTION("Find node in cloned graph which is only present in original graph") {
    REQUIRE(hipGraphNodeFindInClone(&clonedgraphnode, newnode,
                                    clonedgraph) == hipErrorInvalidValue);
  }


  HipTest::freeArrays<int>(A_d, B_d, nullptr,
                           A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedgraph));
}


void hipGraphNodeFindInClone_Func(bool ModifyOrigGraph = false) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph, clonedgraph;
  hipGraphNode_t memset_A, memset_B, memsetKer_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C, memcpyD2D_C,
                 memcpyD2H_C_new;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipMemsetParams memsetParams{};
  int memsetVal{};
  size_t NElem{N};

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                                                              &memsetParams));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(B_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_B, graph, nullptr, 0,
                                                              &memsetParams));

  void* kernelArgs1[] = {&C_d, &memsetVal, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func =
                       reinterpret_cast<void *>(HipTest::memsetReverse<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&memsetKer_C, graph, nullptr, 0,
                                                        &kernelNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                   Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h,
                                   Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d,
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

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_B, &memcpyH2D_B, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memsetKer_C, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2H_C, 1));


  if (ModifyOrigGraph) {
    // Cloned the graph
    HIP_CHECK(hipGraphClone(&clonedgraph, graph));
    // Modify Original graph by adding new dependency
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2D_C, graph, nullptr, 0,
                                      C_d, B_d,
                                      Nbytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C_new, graph, nullptr, 0,
                                      C_h, C_d,
                                      Nbytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2D_C, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2D_C,
                                      &memcpyD2H_C_new, 1));
  }
  // Cloned the graph
  HIP_CHECK(hipGraphClone(&clonedgraph, graph));
  hipGraphNode_t clonedgraphnode;
  if (ModifyOrigGraph) {
    REQUIRE(hipGraphNodeFindInClone(&clonedgraphnode,
                                    memcpyD2H_C_new, clonedgraph)
                                    == hipSuccess);
  } else {
    REQUIRE(hipGraphNodeFindInClone(&clonedgraphnode,
                                    memcpyH2D_A, clonedgraph)
                                    == hipSuccess);
  }
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedgraph));
}

TEST_CASE("Unit_hipGraphNodeFindInClone_Functional") {
  SECTION("hipGraphNodeFindInClone Basic Functionality") {
    hipGraphNodeFindInClone_Func();
  }
  SECTION("hipGraphNodeFindInClone Modify Original graph") {
    hipGraphNodeFindInClone_Func(true);
  }
}
