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
Testcase Scenarios :
Functional ::
 1) Add nodes to graph with and without dependencies, verify the api returns list of
 root nodes (i.e., nodes without dependencies).
 2) Pass nodes as nullptr and verify api returns actual number of root nodes added to graph.
 3) If NumRootNodes passed is greater than the actual number of root nodes, the remaining entries in
 nodes list will be set to NULL, and the number of nodes actually obtained will be returned in NumRootNodes.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/**
 * Functional Test for API fetching root node list
 */
TEST_CASE("Unit_hipGraphGetRootNodes_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  constexpr auto addlEntries = 5;
  hipGraph_t graph;
  hipGraphNode_t memcpyNode, kernelNode;
  hipKernelNodeParams kernelNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  std::vector<hipGraphNode_t> dependencies, rootnodelist;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, A_d, A_h,
                                   Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  rootnodelist.push_back(memcpyNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, B_d, B_h,
                                   Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  rootnodelist.push_back(memcpyNode);

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
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, dependencies.data(),
                                    dependencies.size(), C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  // Get numRootNodes by passing rootnodes list as nullptr.
  // verify : numRootNodes is set to actual number of root nodes added
  size_t numRootNodes{};
  HIP_CHECK(hipGraphGetRootNodes(graph, nullptr, &numRootNodes));
  INFO("Num of nodes returned by GetRootNodes : " << numRootNodes);
  REQUIRE(numRootNodes == rootnodelist.size());

  // Request for extra/additional nodes.
  // verify : totNodes is reset to actual number of root nodes present
  // verify : additional entries in rootnodes list are set to nullptr
  size_t totNodes = numRootNodes + addlEntries;
  int numBytes = sizeof(hipGraphNode_t) * totNodes;
  hipGraphNode_t* rootnodes =
                 reinterpret_cast<hipGraphNode_t *>(malloc(numBytes));
  REQUIRE(rootnodes != nullptr);
  HIP_CHECK(hipGraphGetRootNodes(graph, rootnodes, &totNodes));
  REQUIRE(totNodes == rootnodelist.size());
  for (auto i = numRootNodes; i < numRootNodes + addlEntries; i++) {
    REQUIRE(rootnodes[i] == nullptr);
  }

  // Verify added nodes(without dependencies) are present
  // in the root nodes fetched.
  for (auto Node : rootnodelist) {
    bool found = false;
    for (size_t i = 0; i < numRootNodes; i++) {
      if (Node == rootnodes[i]) {
        found = true;
        break;
      }
    }

    if (!found) {
      INFO("Returned root node " << Node << " not present in added list");
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
  free(rootnodes);
}
