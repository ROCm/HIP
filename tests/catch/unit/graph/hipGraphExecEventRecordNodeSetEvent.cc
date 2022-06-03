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
 1) Create a graph with event record nodes as follows:
    event_record_start_node(event1) --> MemcpyH2DNode --> kernel -->
    MemcpyD2HNode --> event_record_stop_node(event2).Instantiate the graph.
    Set a different event 'event3' in event_record_stop_node using
    hipGraphExecEventRecordNodeSetEvent. Launch the graph. Verify the
    hipGraphExecEventRecordNodeSetEvent functionality by measuring the
    time difference between event2 & event1 and between event3 and
    event1.
 2) Scenario to verify that hipGraphExecEventRecordNodeSetEvent does not
    impact the graph and changes only the executable graph.
    Create an event record node with event1 and add it to graph. Instantiate
    the graph to create an executable graph. Change the event in the
    executable graph to event2. Verify that the event record node still
    contains event1.
 3) Negative Scenarios
    - Input executable graph is a nullptr.
    - Input node is a nullptr.
    - Input event to set is a nullptr.
    - Input executable graph is uninitialized.
    - Input node is uninitialized.
    - Input event is uninitialized.
    - Event record node does not exist in graph.
    - Input node is a memset node.
    - Input node is a event wait node.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#define GRID_DIM 512
#define BLK_DIM 512
#define LEN (GRID_DIM * BLK_DIM)

/**
 * Kernel Functions to copy.
 */
static __global__ void copy_ker_func(int* a, int* b) {
  int tx = hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;
  if (tx < LEN) b[tx] = a[tx];
}

/**
 * Scenario 1: Functional scenario (See description Above)
 */
TEST_CASE("Unit_hipGraphExecEventRecordNodeSetEvent_Functional") {
  size_t memsize = LEN*sizeof(int);
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Create events
  hipEvent_t event_start, event1_end, event2_end;
  HIP_CHECK(hipEventCreate(&event_start));
  HIP_CHECK(hipEventCreate(&event1_end));
  HIP_CHECK(hipEventCreate(&event2_end));
  // Create nodes with event_start and event1_end
  hipGraphNode_t event_start_rec, event_end_rec;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_start_rec, graph, nullptr, 0,
                                                          event_start));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_end_rec, graph, nullptr, 0,
                                                          event1_end));
  int *inp_h, *inp_d, *out_h, *out_d;
  // Allocate host buffers
  inp_h = reinterpret_cast<int*>(malloc(memsize));
  REQUIRE(inp_h != nullptr);
  out_h = reinterpret_cast<int*>(malloc(memsize));
  REQUIRE(out_h != nullptr);
  // Allocate device buffers
  HIP_CHECK(hipMalloc(&inp_d, memsize));
  HIP_CHECK(hipMalloc(&out_d, memsize));
  // Initialize host buffer
  for (uint32_t i = 0; i < LEN; i++) {
    inp_h[i] = i;
    out_h[i] = 0;
  }
  // graph creation ...........
  // Create memcpy and kernel nodes for graph
  hipGraphNode_t memcpyH2D, memcpyD2H, kernelnode;
  hipKernelNodeParams kernelNodeParams{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D, graph, nullptr, 0, inp_d,
                inp_h, memsize, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H, graph, nullptr, 0,
                out_h, out_d, memsize, hipMemcpyDeviceToHost));
  void* kernelArgs1[] = {&inp_d, &out_d};
  kernelNodeParams.func = reinterpret_cast<void *>(copy_ker_func);
  kernelNodeParams.gridDim = dim3(GRID_DIM);
  kernelNodeParams.blockDim = dim3(BLK_DIM);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelnode, graph, nullptr, 0,
                                  &kernelNodeParams));

  // Create dependencies for graph
  HIP_CHECK(hipGraphAddDependencies(graph, &event_start_rec,
                                    &memcpyH2D, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D,
                                    &kernelnode, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernelnode,
                                    &memcpyD2H, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H,
                                    &event_end_rec, 1));
  // Instantiate and launch the graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  // Change the event at event_end_rec node to event2_end
  HIP_CHECK(hipGraphExecEventRecordNodeSetEvent(graphExec,
                                            event_end_rec, event2_end));

  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  // Wait for graph to complete
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  // Validate output
  bool btestPassed = true;
  for (uint32_t i = 0; i < LEN; i++) {
    if (out_h[i] != inp_h[i]) {
      btestPassed = false;
      break;
    }
  }
  REQUIRE(btestPassed == true);
  // Validate the changed events
  float t = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&t, event_start, event2_end));
  REQUIRE(t > 0.0f);
  // Since event1_end is never recorded, hipEventElapsedTime
  // should return error code.
  REQUIRE(hipErrorInvalidResourceHandle ==
          hipEventElapsedTime(&t, event_start, event1_end));
  // Free resources
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(inp_d));
  HIP_CHECK(hipFree(out_d));
  free(inp_h);
  free(out_h);
  HIP_CHECK(hipEventDestroy(event_start));
  HIP_CHECK(hipEventDestroy(event1_end));
  HIP_CHECK(hipEventDestroy(event2_end));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Scenario 2: This test verifies that changes to executable graph does
 * not impact the original graph.
 */
TEST_CASE("Unit_hipGraphExecEventRecordNodeSetEvent_VerifyEventNotChanged") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event1, event2, event_out;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  hipGraphNode_t eventrec;
  HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, nullptr, 0,
                                                          event1));
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphExecEventRecordNodeSetEvent(graphExec,
                                                eventrec, event2));
  HIP_CHECK(hipGraphEventRecordNodeGetEvent(eventrec, &event_out));
  // validate set event and get event are same
  REQUIRE(event1 == event_out);
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event2));
  HIP_CHECK(hipEventDestroy(event1));
}

/**
 * Scenario 3: Negative Tests
 */
TEST_CASE("Unit_hipGraphExecEventRecordNodeSetEvent_Negative") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  hipGraphNode_t eventrec;
  HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, nullptr, 0,
                                                            event1));
  // Create memset
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

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("hGraphExec = nullptr") {
    REQUIRE(hipErrorInvalidValue ==
        hipGraphExecEventRecordNodeSetEvent(nullptr, eventrec, event2));
  }

  SECTION("hNode = nullptr") {
    REQUIRE(hipErrorInvalidValue ==
        hipGraphExecEventRecordNodeSetEvent(graphExec, nullptr, event2));
  }

  SECTION("event = nullptr") {
    REQUIRE(hipErrorInvalidValue ==
        hipGraphExecEventRecordNodeSetEvent(graphExec, eventrec, nullptr));
  }

  SECTION("hGraphExec is uninitialized") {
    hipGraphExec_t graphExec1{};
    REQUIRE(hipErrorInvalidValue ==
        hipGraphExecEventRecordNodeSetEvent(graphExec1, eventrec, event2));
  }

  SECTION("hNode is uninitialized") {
    hipGraphNode_t dummy{};
    REQUIRE(hipErrorInvalidValue ==
        hipGraphExecEventRecordNodeSetEvent(graphExec, dummy, event2));
  }

  SECTION("event is uninitialized") {
    hipEvent_t event_dummy{};
    REQUIRE(hipErrorInvalidValue ==
        hipGraphExecEventRecordNodeSetEvent(graphExec, eventrec,
                                            event_dummy));
  }

  SECTION("event record node does not exist") {
    hipGraph_t graph1;
    HIP_CHECK(hipGraphCreate(&graph1, 0));
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph1, nullptr, 0,
                                    &memsetParams));
    hipGraphExec_t graphExec1;
    HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
    REQUIRE(hipErrorInvalidValue ==
        hipGraphExecEventRecordNodeSetEvent(graphExec1, eventrec, event2));
    HIP_CHECK(hipGraphExecDestroy(graphExec1));
    HIP_CHECK(hipGraphDestroy(graph1));
  }

  SECTION("pass memset node as hNode") {
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                                    &memsetParams));
    REQUIRE(hipErrorInvalidValue ==
        hipGraphExecEventRecordNodeSetEvent(graphExec, memset_A, event2));
  }

  SECTION("pass event wait node as hNode") {
    hipGraphNode_t event_wait_node;
    HIP_CHECK(hipGraphAddEventWaitNode(&event_wait_node, graph, nullptr, 0,
                                                            event1));
    REQUIRE(hipErrorInvalidValue ==
        hipGraphExecEventRecordNodeSetEvent(graphExec, event_wait_node,
                                            event2));
  }

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
}
