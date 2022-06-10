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
Testcase Scenarios :
 1) Create a graph and add nodes with dependencies manually. Perform
    selective removal of dependencies and make sure they are taking
    effect using hipGraphGetEdges() API.
 2) Generate graph by capturing stream. Perform selective removal of
    dependencies and make sure they are taking effect using
    hipGraphGetEdges() API.
 3) Pass numDependencies as 0 and verify api returns success but doesn't
    remove the depedencies.
 4) Create a graph and add nodes with dependencies manually. Perform
    selective removal of dependency and add new dependency. Verify the
    change by executing the updated graph.
 5) Negative Test Cases
    - Pass graph parameter as nullptr.
    - Pass from node parameter as nullptr.
    - Pass to node parameter as nullptr.
    - Pass uninitialized graph.
    - Node passed in "to" parameter does not exist in graph.
    - Remove non existing dependency.
    - Remove the same dependency twice.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#define TOTAL_NUM_OF_EDGES 6

/**
 * Kernel Functions to perform square and return in the same
 * input memory location.
 */
static __global__ void vector_square(int* A_d, size_t N_ELMTS) {
  size_t gputhread = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  int temp = 0;
  for (size_t i = gputhread; i < N_ELMTS; i += stride) {
    temp = A_d[i] * A_d[i];
    A_d[i] = temp;
  }
}

/**
 * Scenario 1 and Scenario 3: Validate hipGraphRemoveDependencies
 * for manually created graph.
 */
TEST_CASE("Unit_hipGraphRemoveDependencies_Func_Manual") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memset_A, memset_B, memsetKer_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
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

  SECTION("scenario 1") {
    // Remove some dependencies
    constexpr size_t numEdgesRemoved = 3;
    HIP_CHECK(hipGraphRemoveDependencies(graph, &memcpyH2D_A,
                                        &kernel_vecAdd, 1));
    HIP_CHECK(hipGraphRemoveDependencies(graph, &memcpyH2D_B,
                                        &kernel_vecAdd, 1));
    HIP_CHECK(hipGraphRemoveDependencies(graph, &memsetKer_C,
                                        &kernel_vecAdd, 1));
    // Validate manually with hipGraphGetEdges() API
    hipGraphNode_t fromnode[TOTAL_NUM_OF_EDGES]{};
    hipGraphNode_t tonode[TOTAL_NUM_OF_EDGES]{};
    size_t numEdges = TOTAL_NUM_OF_EDGES;
    HIP_CHECK(hipGraphGetEdges(graph, fromnode, tonode, &numEdges));

    hipGraphNode_t expected_from_nodes[numEdgesRemoved] = {memcpyH2D_A,
    memcpyH2D_B, memsetKer_C};
    hipGraphNode_t expected_to_nodes[numEdgesRemoved] = {kernel_vecAdd,
    kernel_vecAdd, kernel_vecAdd};
    bool nodeFound;
    int found_count = 0;
    for (size_t idx_from = 0; idx_from < numEdgesRemoved; idx_from++) {
      nodeFound = false;
      int idx = 0;
      for (; idx < TOTAL_NUM_OF_EDGES; idx++) {
        if (expected_from_nodes[idx_from] == fromnode[idx]) {
          nodeFound = true;
          break;
        }
      }
      if (nodeFound && (tonode[idx] == expected_to_nodes[idx_from])) {
        found_count++;
      }
    }
    // Ensure none of the nodes are discovered
    REQUIRE(0 == found_count);
    // Validate with returned number of edges from hipGraphGetEdges() API
    numEdges = 0;
    HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
    size_t numEdgesExpected = TOTAL_NUM_OF_EDGES - numEdgesRemoved;
    REQUIRE(numEdgesExpected == numEdges);
  }

  SECTION("scenario 3") {
    HIP_CHECK(hipGraphRemoveDependencies(graph, &memcpyH2D_A,
                                        &kernel_vecAdd, 0));
    size_t numEdges = 0;
    HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
    size_t numEdgesExpected = TOTAL_NUM_OF_EDGES;
    REQUIRE(numEdgesExpected == numEdges);
  }
  // Destroy
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Scenario 2: Validate hipGraphRemoveDependencies for stream captured graph.
 */
TEST_CASE("Unit_hipGraphRemoveDependencies_Func_StrmCapture") {
  hipStream_t stream1, stream2, stream3;
  hipEvent_t forkStreamEvent, memsetEvent1, memsetEvent2;
  hipGraph_t graph;
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};
  int memsetVal{};

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  // Create streams and events
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));
  HIP_CHECK(hipEventCreate(&memsetEvent1));
  HIP_CHECK(hipEventCreate(&memsetEvent2));
  // Begin stream capture
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(forkStreamEvent, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
  HIP_CHECK(hipStreamWaitEvent(stream3, forkStreamEvent, 0));
  // Add operations to stream3
  hipLaunchKernelGGL(HipTest::memsetReverse<int>,
                    dim3(blocks), dim3(threadsPerBlock), 0, stream3,
                    C_d, memsetVal, NElem);
  HIP_CHECK(hipEventRecord(memsetEvent1, stream3));
  // Add operations to stream2
  HIP_CHECK(hipMemsetAsync(B_d, 0, Nbytes, stream2));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream2));
  HIP_CHECK(hipEventRecord(memsetEvent2, stream2));
  // Add operations to stream1
  HIP_CHECK(hipMemsetAsync(A_d, 0, Nbytes, stream1));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream1, memsetEvent2, 0));
  HIP_CHECK(hipStreamWaitEvent(stream1, memsetEvent1, 0));
  hipLaunchKernelGGL(HipTest::vectorADD<int>,
                    dim3(blocks), dim3(threadsPerBlock), 0, stream1,
                    A_d, B_d, C_d, NElem);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost,
                           stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));
  hipGraphNode_t* nodes{nullptr};
  size_t numNodes = 0, numEdges = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nodes, &numNodes));
  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  REQUIRE(7 == numNodes);
  REQUIRE(TOTAL_NUM_OF_EDGES == numEdges);
  // Get the edges and remove one edge. Verify edge is removed.
  hipGraphNode_t fromnode[TOTAL_NUM_OF_EDGES]{};
  hipGraphNode_t tonode[TOTAL_NUM_OF_EDGES]{};
  HIP_CHECK(hipGraphGetEdges(graph, fromnode, tonode, &numEdges));
  HIP_CHECK(hipGraphRemoveDependencies(graph, &fromnode[0],
                                       &tonode[0], 1));
  // Verify
  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  size_t expected_num_edges = TOTAL_NUM_OF_EDGES - 1;
  REQUIRE(expected_num_edges == numEdges);
  // Destroy
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream3));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
 * Scenario 4: Dynamically modify dependencies in a graph using
 * hipGraphRemoveDependencies and verify the computation.
 */
TEST_CASE("Unit_hipGraphRemoveDependencies_ChangeComputeFunc") {
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd, kernel_square;
  hipKernelNodeParams kernelNodeParams{};
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
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
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2H_C, 1));
  // Instantiate and execute Graph
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  // Validate
  bool bMismatch = false;
  for (size_t idx = 0; idx < NElem; idx++) {
    if (C_h[idx] != (A_h[idx] + B_h[idx])) {
      bMismatch = true;
      break;
    }
  }
  REQUIRE(false == bMismatch);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  // Remove dependency memcpyH2D_B -> kernel_vecAdd and
  // add new dependencies memcpyH2D_B -> kernel_square -> kernel_vecAdd
  // Square kernel
  void* kernelArgs1[] = {&B_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func =
                       reinterpret_cast<void *>(vector_square);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_square, graph, nullptr, 0,
                                                        &kernelNodeParams));
  hipGraphRemoveDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1);
  // Add new dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_square, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_square,
                                    &kernel_vecAdd, 1));
  size_t numEdges = 0, numNodes = 0;
  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  REQUIRE(4 == numEdges);
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  REQUIRE(5 == numNodes);
  // Instantiate and execute graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  // Validate
  bMismatch = false;
  for (size_t idx = 0; idx < NElem; idx++) {
    if (C_h[idx] != (A_h[idx] + B_h[idx]*B_h[idx])) {
      bMismatch = true;
      break;
    }
  }
  REQUIRE(false == bMismatch);
  // Destroy
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Scenario 5: Negative Tests
 */
TEST_CASE("Unit_hipGraphRemoveDependencies_Negative") {
  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event_start, event_end;
  HIP_CHECK(hipEventCreateWithFlags(&event_start, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&event_end, hipEventDisableTiming));
  // memset node
  constexpr size_t Nbytes = 1024;
  char *A_d;
  hipGraphNode_t memset_A;
  hipMemsetParams memsetParams{};
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                                   &memsetParams));
  // create event record node
  hipGraphNode_t event_node_start, event_node_end;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_node_start, graph, nullptr, 0,
                                                            event_start));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_node_end, graph, nullptr, 0,
                                                            event_end));
  // create empty node
  hipGraphNode_t emptyNode{};
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0));
  // Add dependencies between nodes
  HIP_CHECK(hipGraphAddDependencies(graph, &event_node_start, &memset_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &event_node_end, 1));

  SECTION("graph is nullptr") {
    REQUIRE(hipErrorInvalidValue ==
        hipGraphRemoveDependencies(nullptr, &event_node_start, &memset_A, 1));
  }

  SECTION("from is nullptr") {
    REQUIRE(hipErrorInvalidValue ==
        hipGraphRemoveDependencies(graph, nullptr, &memset_A, 1));
  }

  SECTION("to is nullptr") {
    REQUIRE(hipErrorInvalidValue ==
        hipGraphRemoveDependencies(graph, &event_node_start, nullptr, 1));
  }

  SECTION("graph is uninitialized") {
    hipGraph_t graph_uninit{};
    REQUIRE(hipErrorInvalidValue ==
        hipGraphRemoveDependencies(graph_uninit, &event_node_start,
                                   nullptr, 1));
  }

  SECTION("non existing node") {
    REQUIRE(hipErrorInvalidValue ==
        hipGraphRemoveDependencies(graph, &event_node_start,
                                   &emptyNode, 1));
  }

  SECTION("remove non existing dependency") {
    REQUIRE(hipErrorInvalidValue ==
        hipGraphRemoveDependencies(graph, &event_node_start,
                                   &event_node_end, 1));
  }

  SECTION("remove same dependency twice") {
    HIP_CHECK(hipGraphRemoveDependencies(graph, &event_node_start,
                                   &memset_A, 1));
    REQUIRE(hipErrorInvalidValue ==
        hipGraphRemoveDependencies(graph, &event_node_start,
                                   &memset_A, 1));
  }

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event_end));
  HIP_CHECK(hipEventDestroy(event_start));
}
