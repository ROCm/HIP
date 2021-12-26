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
 1) Set a different type of event using hipGraphEventWaitNodeSetEvent and
    validate using hipGraphEventWaitNodeGetEvent.
 2) Create a graph1 with memset (Value 1) node, event record node (event A)
    and memset (Value 2) node and event record node (event B). Create a
    graph2 with Event Wait (event A ) node and memcpyd2h. Instantiate
    graph1 on stream1 and graph2 on stream2. Set the Event Wait node event
    to B using hipGraphEventWaitNodeSetEvent. Launch graphs. Wait for the
    event to complete. Verify the results.
 3) Negative Scenarios
    - Input node parameter is passed as nullptr.
    - Input event parameter is passed as nullptr.
    - Input node is an empty node.
    - Input node is a memset node.
    - Input node is a event record node.
    - Input node is an uninitialized node.
    - Input event is an uninitialized node.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#define LEN 512

/**
 * Local Function
 */
static void validateEventWaitNodeSetEvent(unsigned flag) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Create events
  hipEvent_t event1, event2, event_out;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreateWithFlags(&event2, flag));
  hipGraphNode_t eventwait;
  HIP_CHECK(hipGraphAddEventWaitNode(&eventwait, graph, nullptr, 0,
                                                            event1));
  // Set a different event
  HIP_CHECK(hipGraphEventWaitNodeSetEvent(eventwait, event2));
  HIP_CHECK(hipGraphEventWaitNodeGetEvent(eventwait, &event_out));
  // validate set event and get event are same
  REQUIRE(event2 == event_out);
  // Free resources
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
}

/**
 * Local Function
 */
static void setEventRecordNode() {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Create events
  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  hipGraphNode_t eventrec;
  HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, nullptr, 0,
                                                            event1));
  // Set a different event eventrec using hipGraphEventWaitNodeSetEvent
  REQUIRE(hipErrorInvalidValue ==
          hipGraphEventWaitNodeSetEvent(eventrec, event2));
  // Free resources
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
}

/**
 * Scenario 2
 */
TEST_CASE("Unit_hipGraphEventWaitNodeSetEvent_SetProp") {
  size_t memsize = LEN * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, LEN);
  size_t NElem{LEN};
  hipGraph_t graph1, graph2;
  hipStream_t streamForGraph1, streamForGraph2;
  hipGraphExec_t graphExec1, graphExec2;
  HIP_CHECK(hipStreamCreate(&streamForGraph1));
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph2));

  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreateWithFlags(&event1, hipEventDefault));
  HIP_CHECK(hipEventCreateWithFlags(&event2, hipEventBlockingSync));
  hipGraphNode_t event_rec_node, event_wait_node;
  int *inp_h, *inp_d, *out_h_g1, *out_d_g1, *out_h_g2, *out_d_g2;
  // Allocate host buffers
  inp_h = reinterpret_cast<int*>(malloc(memsize));
  REQUIRE(inp_h != nullptr);
  out_h_g1 = reinterpret_cast<int*>(malloc(memsize));
  REQUIRE(out_h_g1 != nullptr);
  out_h_g2 = reinterpret_cast<int*>(malloc(memsize));
  REQUIRE(out_h_g2 != nullptr);
  // Allocate device buffers
  HIP_CHECK(hipMalloc(&inp_d, memsize));
  HIP_CHECK(hipMalloc(&out_d_g1, memsize));
  HIP_CHECK(hipMalloc(&out_d_g2, memsize));
  // Initialize host buffer
  for (uint32_t i = 0; i < LEN; i++) {
    inp_h[i] = i;
    out_h_g1[i] = 0;
    out_h_g2[i] = 0;
  }
  // Graph1 creation ...........
  // Create event1 record node in graph1
  HIP_CHECK(hipGraphAddEventRecordNode(&event_rec_node, graph1, nullptr, 0,
                                                            event1));

  // Create memcpy and kernel nodes for graph1
  hipGraphNode_t memcpyH2D, memcpyD2H_1, kernelnode_1;
  hipKernelNodeParams kernelNodeParams1{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D, graph1, nullptr, 0, inp_d,
                inp_h, memsize, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_1, graph1, nullptr, 0,
                out_h_g1, out_d_g1, memsize, hipMemcpyDeviceToHost));

  void* kernelArgs1[] = {&inp_d, &out_d_g1, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams1.func =
  reinterpret_cast<void *>(HipTest::vector_square<int>);
  kernelNodeParams1.gridDim = dim3(blocks);
  kernelNodeParams1.blockDim = dim3(threadsPerBlock);
  kernelNodeParams1.sharedMemBytes = 0;
  kernelNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams1.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelnode_1, graph1, nullptr, 0,
                                  &kernelNodeParams1));
  // Create dependencies for graph1
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpyH2D,
                                    &event_rec_node, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &event_rec_node,
                                    &kernelnode_1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &kernelnode_1,
                                    &memcpyD2H_1, 1));

  // Graph2 creation ...........
  // Create event1 record node in graph2
  HIP_CHECK(hipGraphAddEventWaitNode(&event_wait_node, graph2, nullptr, 0,
                                                            event1));
  // Create memcpy and kernel nodes for graph2
  hipGraphNode_t memcpyD2H_2, kernelnode_2;
  hipKernelNodeParams kernelNodeParams2{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_2, graph2, nullptr, 0,
                out_h_g2, out_d_g2, memsize, hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&inp_d, &out_d_g2, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams2.func =
  reinterpret_cast<void *>(HipTest::vector_cubic<int>);
  kernelNodeParams2.gridDim = dim3(blocks);
  kernelNodeParams2.blockDim = dim3(threadsPerBlock);
  kernelNodeParams2.sharedMemBytes = 0;
  kernelNodeParams2.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams2.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelnode_2, graph2, nullptr, 0,
                                  &kernelNodeParams2));
  // Create dependencies for graph2
  HIP_CHECK(hipGraphAddDependencies(graph2, &event_wait_node,
                                    &kernelnode_2, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &kernelnode_2,
                                    &memcpyD2H_2, 1));

  // Instantiate and launch the graphs
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
  // Set event
  HIP_CHECK(hipGraphEventRecordNodeSetEvent(event_rec_node, event2));
  HIP_CHECK(hipGraphEventWaitNodeSetEvent(event_wait_node, event2));

  HIP_CHECK(hipGraphLaunch(graphExec1, streamForGraph1));
  HIP_CHECK(hipGraphLaunch(graphExec2, streamForGraph2));
  HIP_CHECK(hipStreamSynchronize(streamForGraph1));
  HIP_CHECK(hipStreamSynchronize(streamForGraph2));
  // Validate output
  bool btestPassed1 = true;
  for (uint32_t i = 0; i < LEN; i++) {
    if (out_h_g1[i] != (inp_h[i]*inp_h[i])) {
      btestPassed1 = false;
      break;
    }
  }
  REQUIRE(btestPassed1 == true);
  bool btestPassed2 = true;
  for (uint32_t i = 0; i < LEN; i++) {
    if (out_h_g2[i] != (inp_h[i]*inp_h[i]*inp_h[i])) {
      btestPassed2 = false;
      break;
    }
  }
  REQUIRE(btestPassed2 == true);
  // Destroy all resources
  HIP_CHECK(hipFree(inp_d));
  HIP_CHECK(hipFree(out_d_g1));
  HIP_CHECK(hipFree(out_d_g2));
  free(inp_h);
  free(out_h_g1);
  free(out_h_g2);
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
  HIP_CHECK(hipStreamDestroy(streamForGraph1));
  HIP_CHECK(hipStreamDestroy(streamForGraph2));
}
/**
 * Scenario 1
 */
TEST_CASE("Unit_hipGraphEventWaitNodeSetEvent_SetGet") {
  SECTION("Flag = hipEventDefault") {
    validateEventWaitNodeSetEvent(hipEventDefault);
  }

  SECTION("Flag = hipEventBlockingSync") {
    validateEventWaitNodeSetEvent(hipEventBlockingSync);
  }

  SECTION("Flag = hipEventDisableTiming") {
    validateEventWaitNodeSetEvent(hipEventDisableTiming);
  }
}

/**
 * Scenario 3
 */
TEST_CASE("Unit_hipGraphEventWaitNodeSetEvent_Negative") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  hipGraphNode_t eventwait;
  HIP_CHECK(hipGraphAddEventWaitNode(&eventwait, graph, nullptr, 0,
                                                        event1));
  SECTION("node = nullptr") {
    REQUIRE(hipErrorInvalidValue == hipGraphEventWaitNodeSetEvent(
                                    nullptr, event2));
  }

  SECTION("event = nullptr") {
    REQUIRE(hipErrorInvalidValue == hipGraphEventWaitNodeSetEvent(
                                    eventwait, nullptr));
  }

  SECTION("input node is empty node") {
    hipGraphNode_t EmptyGraphNode;
    HIP_CHECK(hipGraphAddEmptyNode(&EmptyGraphNode, graph, nullptr, 0));
    REQUIRE(hipErrorInvalidValue ==
            hipGraphEventWaitNodeSetEvent(EmptyGraphNode, event2));
  }

  SECTION("input node is memset node") {
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
    REQUIRE(hipErrorInvalidValue ==
            hipGraphEventWaitNodeSetEvent(memset_A, event2));
    HIP_CHECK(hipFree(A_d));
  }

  SECTION("input node is event record node") {
    setEventRecordNode();
  }

  SECTION("input node is uninitialized node") {
    hipGraphNode_t node_uninit{};
    REQUIRE(hipErrorInvalidValue ==
            hipGraphEventWaitNodeSetEvent(node_uninit, event2));
  }

  SECTION("input event is uninitialized") {
    hipEvent_t event_uninit{};
    REQUIRE(hipErrorInvalidValue == hipGraphEventWaitNodeSetEvent(
                                    eventwait, event_uninit));
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
}
