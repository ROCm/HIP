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
Testcase Scenarios
------------------
Functional ::
1) Add nodes to graph and get nodes. Verify the added nodes are present in returned list.
2) Pass nodes as nullptr and verify numNodes returns actual number of nodes added to graph.
3) If numNodes passed is greater than the actual number of nodes, the remaining entries in nodes
will be set to NULL, and the number of nodes actually obtained will be returned in numNodes.

Argument Validation ::
1) Pass graph as nullptr and verify api returns error code.
2) Pass numNodes as nullptr and other params as valid values. Expect api to return error code.
3) When there are no nodes in graph, expect numNodes to be set to zero.
4) Pass numNodes less than actual number of nodes. Expect api to populate requested number of node entries
and does update numNodes.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/**
 * Functional Test for hipGraphGetNodes API fetching node list
 */
TEST_CASE("Unit_hipGraphGetNodes_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  constexpr auto addlEntries = 4;
  hipGraph_t graph;
  hipGraphNode_t memcpyNode, kernelNode;
  hipKernelNodeParams kernelNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  std::vector<hipGraphNode_t> dependencies, nodelist;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, A_d, A_h,
                                   Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  nodelist.push_back(memcpyNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, B_d, B_h,
                                   Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  nodelist.push_back(memcpyNode);

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, dependencies.data(),
                                  dependencies.size(), &kernelNodeParams));
  dependencies.clear();
  dependencies.push_back(kernelNode);
  nodelist.push_back(kernelNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, dependencies.data(),
                                    dependencies.size(), C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  nodelist.push_back(memcpyNode);

  // Get numNodes by passing nodes as nullptr.
  // verify : numNodes is set to actual number of nodes added
  size_t numNodes{};
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  INFO("Num of nodes returned by GetNodes : " << numNodes);
  REQUIRE(numNodes == nodelist.size());

  // Request for extra/additional nodes.
  // verify : totNodes is reset to actual number of nodes
  // verify : additional entries in nodes are set to nullptr
  size_t totNodes = numNodes + addlEntries;
  int numBytes = sizeof(hipGraphNode_t) * totNodes;
  hipGraphNode_t* nodes = reinterpret_cast<hipGraphNode_t *>(malloc(numBytes));
  REQUIRE(nodes != nullptr);
  HIP_CHECK(hipGraphGetNodes(graph, nodes, &totNodes));
  REQUIRE(totNodes == nodelist.size());
  for (auto i = numNodes; i < numNodes + addlEntries; i++) {
    REQUIRE(nodes[i] == nullptr);
  }

  // Verify added nodes are present in the node entries returned
  for (auto Node : nodelist) {
    bool found = false;
    for (size_t i = 0; i < numNodes; i++) {
      if (Node == nodes[i]) {
        found = true;
        break;
      }
    }

    if (!found) {
      INFO("Added node " << Node << " not present in returned list");
      REQUIRE(false);
    }
  }

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  free(nodes);
}

/**
 * Test performs api parameter validation by passing various values
 * as input and output parameters and validates the behavior.
 * Test will include both negative and positive scenarios.
 */
TEST_CASE("Unit_hipGraphGetNodes_ParamValidation") {
  hipStream_t stream{nullptr};
  hipGraph_t graph{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  constexpr size_t N = 1000000;
  size_t Nbytes = N * sizeof(float), numNodes{};
  float *A_d, *C_d;
  float *A_h, *C_h;
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                              dim3(threadsPerBlock), 0, stream, A_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  INFO("Num of nodes returned by GetNodes : " << numNodes);

  int numBytes = sizeof(hipGraphNode_t) * numNodes;
  hipGraphNode_t* nodes = reinterpret_cast<hipGraphNode_t *>(malloc(numBytes));
  REQUIRE(nodes != nullptr);

  SECTION("graph as nullptr") {
    hipError_t ret = hipGraphGetNodes(nullptr, nodes, &numNodes);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("numNodes as nullptr") {
    hipError_t ret = hipGraphGetNodes(graph, nodes, nullptr);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("no nodes in graph") {
    hipGraph_t emptyGraph{};
    HIP_CHECK(hipGraphCreate(&emptyGraph, 0));
    HIP_CHECK(hipGraphGetNodes(emptyGraph, nullptr, &numNodes));
    REQUIRE(numNodes == 0);
  }

  SECTION("numNodes less than actual number of nodes") {
    size_t numPartNodes = numNodes - 1;
    hipGraphNodeType nodeType;
    HIP_CHECK(hipGraphGetNodes(graph, nodes, &numPartNodes));

    // verify numPartNodes is unchanged
    REQUIRE(numPartNodes == numNodes - 1);
    // verify partial node list returned has valid nodes
    for (size_t i = 0; i < numPartNodes; i++) {
      HIP_CHECK(hipGraphNodeGetType(nodes[i], &nodeType));
      REQUIRE(nodeType >= 0);
      REQUIRE(nodeType < hipGraphNodeTypeCount);
    }
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  free(A_h);
  free(C_h);
  free(nodes);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
}
