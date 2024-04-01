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
 1) Set a different type of event using hipGraphEventRecordNodeSetEvent and
    validate using hipGraphEventRecordNodeGetEvent.
 2) Add different kinds of nodes to graph and add dependencies to nodes.
    Create an event record node at the end with Default flag. Set a different
    type of event using hipGraphEventRecordNodeSetEvent.  Instantiate and
    Launch graph. Wait for the event to complete. Verify the results.
 3) Negative Scenarios
    - Input node parameter is nullptr.
    - Input event parameter is nullptr.
    - Empty node is passed as input node.
    - Memset node is passed as input node.
    - Input node is an uninitialized node.
    - Input event is an uninitialized event.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/**
 * Local Function: Set Get test
 */
static void validateEventRecordNodeSetEvent(unsigned flag) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Create events
  hipEvent_t event1, event2, event_out;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreateWithFlags(&event2, flag));
  hipGraphNode_t eventrec;
  HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, nullptr, 0,
                                                            event1));
  // Set a different event
  HIP_CHECK(hipGraphEventRecordNodeSetEvent(eventrec, event2));
  HIP_CHECK(hipGraphEventRecordNodeGetEvent(eventrec, &event_out));
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
static void setEventWaitNode() {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Create events
  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  hipGraphNode_t eventwait;
  HIP_CHECK(hipGraphAddEventWaitNode(&eventwait, graph, nullptr, 0,
                                                           event1));
  // Set a different event eventwait using hipGraphEventRecordNodeSetEvent
  REQUIRE(hipErrorInvalidValue ==
          hipGraphEventRecordNodeSetEvent(eventwait, event2));
  // Free resources
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
}

/**
 * Scenario 2: Validate Change of event property in event record node.
 */
TEST_CASE("Unit_hipGraphEventRecordNodeSetEvent_SetEventProperty") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Create events
  hipEvent_t event1_start, event2_start, event1_end, event2_end;
  HIP_CHECK(hipEventCreate(&event1_start));
  HIP_CHECK(hipEventCreate(&event1_end));
  HIP_CHECK(hipEventCreateWithFlags(&event2_start, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&event2_end, hipEventDisableTiming));
  // Create nodes
  hipGraphNode_t event_start_rec, event_end_rec;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_start_rec, graph, nullptr, 0,
                                                          event1_start));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_end_rec, graph, nullptr, 0,
                                                          event1_end));
  // Create memset node
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
  // Create dependencies
  // event_start_rec --> memset_A --> event_end_rec
  HIP_CHECK(hipGraphAddDependencies(graph, &event_start_rec, &memset_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &event_end_rec, 1));

  // Instantiate and launch graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  // Validate by measuring time difference between event_end_rec &
  // event_start_rec
  float t = 0.0f;
  REQUIRE(hipSuccess == hipEventElapsedTime(&t, event1_start,
                                            event1_end));
  REQUIRE(t > 0.0f);
  // Change the event property after instantiation
  HIP_CHECK(hipGraphEventRecordNodeSetEvent(event_start_rec, event2_start));
  HIP_CHECK(hipGraphEventRecordNodeSetEvent(event_end_rec, event2_end));
  // Launch the graph again with the new settings
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  // Validate hipEventElapsedTime() must return
  // hipErrorInvalidHandle when events are created using
  // hipEventDisableTiming flag.
  t = 0.0f;
  REQUIRE(hipErrorInvalidHandle ==
          hipEventElapsedTime(&t, event2_start, event2_end));
  // Free resources
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipEventDestroy(event1_start));
  HIP_CHECK(hipEventDestroy(event2_start));
  HIP_CHECK(hipEventDestroy(event1_end));
  HIP_CHECK(hipEventDestroy(event2_end));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Scenario 1: Validate Set Get test for all Event flags
 */
TEST_CASE("Unit_hipGraphEventRecordNodeSetEvent_SetGet") {
  SECTION("Flag = hipEventDefault") {
    validateEventRecordNodeSetEvent(hipEventDefault);
  }

  SECTION("Flag = hipEventBlockingSync") {
    validateEventRecordNodeSetEvent(hipEventBlockingSync);
  }

  SECTION("Flag = hipEventDisableTiming") {
    validateEventRecordNodeSetEvent(hipEventDisableTiming);
  }
}

/**
 * Scenario 3: Negative Tests
 */
TEST_CASE("Unit_hipGraphEventRecordNodeSetEvent_Negative") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  hipGraphNode_t eventrec;
  HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, nullptr, 0,
                                                            event1));
  SECTION("node = nullptr") {
    REQUIRE(hipErrorInvalidValue == hipGraphEventRecordNodeSetEvent(nullptr,
                                    event2));
  }

  SECTION("event_out = nullptr") {
    REQUIRE(hipErrorInvalidValue == hipGraphEventRecordNodeSetEvent(eventrec,
                                    nullptr));
  }

  SECTION("input node is empty node") {
    hipGraphNode_t EmptyGraphNode;
    HIP_CHECK(hipGraphAddEmptyNode(&EmptyGraphNode, graph, nullptr, 0));
    REQUIRE(hipErrorInvalidValue ==
            hipGraphEventRecordNodeSetEvent(EmptyGraphNode, event2));
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
            hipGraphEventRecordNodeSetEvent(memset_A, event2));
    HIP_CHECK(hipFree(A_d));
  }

  SECTION("input node is event wait node") {
    setEventWaitNode();
  }

  SECTION("input node is uninitialized node") {
    hipGraphNode_t node_uninit{};
    REQUIRE(hipErrorInvalidValue ==
            hipGraphEventRecordNodeSetEvent(node_uninit, event2));
  }

  SECTION("input event is uninitialized") {
    hipEvent_t event_uninit{};
    REQUIRE(hipErrorInvalidValue ==
            hipGraphEventRecordNodeSetEvent(eventrec, event_uninit));
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
}
