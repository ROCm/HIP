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
 1) Simple Scenario: Create an event node and add it to graph.
Instantiate and Launch the Graph. Wait for the event to complete.
The operation must succeed without any failures.
 2) Add different kinds of nodes to graph and add dependencies to nodes.
Create an event record node at the end. Instantiate and Launch the Graph.
Wait for the event to complete. Verify the results. Event is created using
hipEventCreate.
 3) Add different kinds of nodes to graph and add dependencies to nodes.
Create event record nodes at the beginning and end. Instantiate and Launch
the Graph. Wait for the event to complete. Verify the results. Also verify
the elapsed time. Events are created using hipEventCreate.
 4) Add different kinds of nodes to graph and add dependencies to nodes.
Create an event record node at the end.  Instantiate and Launch graph.
Wait for the event to complete. Verify the results. Event is created
using hipEventCreateWithFlags (for different flag values).
 5) Create event record node at the beginning with
flag = hipEventDisableTiming, a memset node and event record nodes at the
end. Instantiate and Launch the Graph. Wait for the event to complete.
Verify that hipEventElapsedTime() returns error.
 6) Validate scenario 2 by running the graph multiple times in a loop
(100 times) after instantiation.
 7) Negative Scenarios
    - Output node is a nullptr.
    - Input graph is a nullptr.
    - Input dependencies is a nullptr.
    - Input event is a nullptr.
    - Input graph is uninitialized.
    - Input event is uninitialized.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/**
 * Scenario 1: Create s simple graph with just one event record
 * node and instantiate and launch the graph.
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Functional_Simple") {
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  hipGraphNode_t eventrec;
  HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, nullptr, 0,
                                                            event));
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  // Wait for event
  HIP_CHECK(hipEventSynchronize(event));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Local test function
 */
static void validateAddEventRecordNode(bool measureTime, bool withFlags,
                            int nstep, unsigned flag = 0) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memset_A, memset_B, memsetKer_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t ker_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  hipMemsetParams memsetParams{};
  int memsetVal{};
  size_t NElem{N};
  HIP_CHECK(hipStreamCreate(&streamForGraph));
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

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d,
                                  A_h, Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d,
                                  B_h, Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h,
                                  C_d, Nbytes, hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&ker_vecAdd, graph, nullptr, 0,
                                                        &kernelNodeParams));
  hipEvent_t eventstart, eventend;
  if (withFlags) {
    HIP_CHECK(hipEventCreateWithFlags(&eventstart, flag));
    HIP_CHECK(hipEventCreateWithFlags(&eventend, flag));
  } else {
    HIP_CHECK(hipEventCreate(&eventstart));
    HIP_CHECK(hipEventCreate(&eventend));
  }
  hipGraphNode_t event_start, event_final;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_start, graph, nullptr, 0,
                                                            eventstart));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_final, graph, nullptr, 0,
                                                            eventend));
  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &event_start, &memset_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &event_start, &memset_B, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_B, &memcpyH2D_B, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &ker_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &ker_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memsetKer_C, &ker_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &ker_vecAdd, &memcpyD2H_C, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H_C, &event_final, 1));
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  for (int istep = 0; istep < nstep; istep++) {
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    // Wait for eventend
    HIP_CHECK(hipEventSynchronize(eventend));
    // Verify graph execution result
    HipTest::checkVectorADD(A_h, B_h, C_h, N);
    if (measureTime) {
      // Verify event record time difference_type
      float t = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&t, eventstart, eventend));
      REQUIRE(t > 0.0f);
    }
  }
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(eventstart));
  HIP_CHECK(hipEventDestroy(eventend));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Scenario 2: Validate event record nodes created without flags.
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Functional_WithoutFlags") {
  // Create events without flags using hipEventCreate and
  // elapsed time is not validated
  validateAddEventRecordNode(false, false, 1);
}

/**
 * Scenario 3: Validate elapsed time between 2 recorded events.
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Functional_ElapsedTime") {
  // Create events without flags using hipEventCreate and
  // elapsed time is validated
  validateAddEventRecordNode(true, false, 1);
}

/**
 * Scenario 4: Validate event record nodes created with different
 * event flags.
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Functional_WithFlags") {
  // Create events with different flags using hipEventCreate and
  // elapsed time is not validated
  SECTION("Flag = hipEventDefault") {
    validateAddEventRecordNode(false, true, 1, hipEventDefault);
  }

  SECTION("Flag = hipEventBlockingSync") {
    validateAddEventRecordNode(false, true, 1, hipEventBlockingSync);
  }

  SECTION("Flag = hipEventDisableTiming") {
    validateAddEventRecordNode(false, true, 1, hipEventDisableTiming);
  }
}

/**
 * Scenario 5: Validate hipGraphAddEventRecordNode by executing graph
 * 100 times in a loop.
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_MultipleRun") {
  validateAddEventRecordNode(false, false, 100);
}

/**
 * Scenario 6: Validate hipGraphAddEventRecordNode with time disabled events.
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Functional_TimingDisabled") {
  constexpr size_t Nbytes = 1024;
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipEvent_t event_start, event_end;
  HIP_CHECK(hipEventCreateWithFlags(&event_start, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&event_end, hipEventDisableTiming));
  // memset node
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

  hipGraphNode_t event_node_start, event_node_end;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_node_start, graph, nullptr, 0,
                                                            event_start));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_node_end, graph, nullptr, 0,
                                                            event_end));
  // Add dependencies between nodes
  HIP_CHECK(hipGraphAddDependencies(graph, &event_node_start, &memset_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &event_node_end, 1));
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  // Wait for event
  HIP_CHECK(hipEventSynchronize(event_end));
  // Validate hipEventElapsedTime returns error code because timing is
  // disabled for start and end event nodes.
  float t;
  REQUIRE(hipSuccess != hipEventElapsedTime(&t, event_start, event_end));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event_end));
  HIP_CHECK(hipEventDestroy(event_start));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Scenario 7: All negative tests
 */
TEST_CASE("Unit_hipGraphAddEventRecordNode_Negative") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  hipGraphNode_t eventwait;
  SECTION("pGraphNode = nullptr") {
    REQUIRE(hipErrorInvalidValue == hipGraphAddEventRecordNode(nullptr,
                                    graph, nullptr, 0, event));
  }

  SECTION("graph = nullptr") {
    REQUIRE(hipErrorInvalidValue == hipGraphAddEventRecordNode(&eventwait,
                                    nullptr, nullptr, 0, event));
  }

  SECTION("pDependencies = nullptr and numDependencies != 0") {
    REQUIRE(hipErrorInvalidValue == hipGraphAddEventRecordNode(&eventwait,
                                    graph, nullptr, 1, event));
  }

  SECTION("event = nullptr") {
    REQUIRE(hipErrorInvalidValue == hipGraphAddEventRecordNode(&eventwait,
                                    graph, nullptr, 0, nullptr));
  }

  SECTION("graph is uninitialized") {
    hipGraph_t graph_uninit{};
    REQUIRE(hipErrorInvalidValue == hipGraphAddEventRecordNode(&eventwait,
                                    graph_uninit, nullptr, 0, nullptr));
  }

  SECTION("event is uninitialized") {
    hipEvent_t event_uninit{};
    REQUIRE(hipErrorInvalidValue == hipGraphAddEventRecordNode(&eventwait,
                                    graph, nullptr, 0, event_uninit));
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
}
