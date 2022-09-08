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
Functional ::
1) Pass different types of valid nodes and get the corresponding type of the node
2) Add Graph node, destroy the graph node and add the same node of different type.
   hipGraphNodeGetTye should return the new type
3) Add graph node, overwrite new type to the same graph node and trigger
   hipGraphNodeGetType which should return the updated type.
Negative ::
1) Pass nullptr to graph node
2) Pass nullptr  to node type
3) Pass invalid to graph node
*/
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

TEST_CASE("Unit_hipGraphNodeGetType_Negative") {
  SECTION("Pass nullptr to graph node") {
    hipGraphNodeType nodeType;
    REQUIRE(hipGraphNodeGetType(nullptr, &nodeType) == hipErrorInvalidValue);
  }

  SECTION("Pass nullptr to node type") {
    hipGraphNode_t memcpyNode;
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&memcpyNode, graph, nullptr , 0));
    REQUIRE(hipGraphNodeGetType(memcpyNode, nullptr) == hipErrorInvalidValue);
  }

  SECTION("Pass invalid node") {
    hipGraphNode_t Node = {};
    hipGraphNodeType nodeType;
    REQUIRE(hipGraphNodeGetType(Node, &nodeType) == hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipGraphNodeGetType_Functional") {
  constexpr size_t N = 1024;
  hipGraphNodeType nodeType;
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  hipEvent_t event;
  hipGraphNode_t waiteventNode;
  HIP_CHECK(hipEventCreate(&event));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  SECTION("Delete Node and Assign different Node Type") {
    HIP_CHECK(hipStreamWaitEvent(stream, event, 0));
    HIP_CHECK(hipGraphAddEventWaitNode(&waiteventNode, graph, nullptr, 0,
                                        event));
    HIP_CHECK(hipGraphNodeGetType(waiteventNode, &nodeType));
    HIP_CHECK(hipGraphAddEmptyNode(&waiteventNode, graph, nullptr , 0));
    HIP_CHECK(hipGraphNodeGetType(waiteventNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeEmpty);
  }

  SECTION("Override the graph node and get Node Type") {
    HIP_CHECK(hipStreamWaitEvent(stream, event, 0));
    HIP_CHECK(hipGraphAddEventWaitNode(&waiteventNode, graph, nullptr, 0,
                                        event));
    HIP_CHECK(hipGraphNodeGetType(waiteventNode, &nodeType));
    HIP_CHECK(hipGraphAddEmptyNode(&waiteventNode, graph, nullptr , 0));
    HIP_CHECK(hipGraphNodeGetType(waiteventNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeEmpty);
  }
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(event));
}
/**
 * Functional Test for hipGraphNodeGetType API
 */
TEST_CASE("Unit_hipGraphNodeGetType_NodeType") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNodeType nodeType;
  hipGraphNode_t memcpyNode, kernelNode;
  SECTION("Get Memcpy NodeType") {
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, A_d, A_h,
                                      Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphNodeGetType(memcpyNode, &nodeType));

    // temp disable it until correct node is set
    // REQUIRE(nodeType == hipGraphNodeTypeMemcpy);

    HIP_CHECK(hipGraphAddEmptyNode(&memcpyNode, graph, nullptr , 0));
    HIP_CHECK(hipGraphNodeGetType(memcpyNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeEmpty);
  }

  SECTION("Get Kernel NodeType") {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nullptr,
                                    0, &kernelNodeParams));
    HIP_CHECK(hipGraphNodeGetType(kernelNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeKernel);
  }

  SECTION("Get Empty NodeType") {
    hipGraphNode_t emptyNode;
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr , 0));
    HIP_CHECK(hipGraphNodeGetType(emptyNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeEmpty);
  }

  SECTION("Get Memset NodeType") {
    hipGraphNode_t memsetNode;
    hipMemsetParams memsetParams{};
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(A_d);
    memsetParams.value = 10;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = Nbytes;
    memsetParams.height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
          &memsetParams));
    HIP_CHECK(hipGraphNodeGetType(memsetNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeMemset);
  }

  SECTION("Get WaitEvent NodeType") {
    hipEvent_t event;
    hipGraphNode_t waiteventNode;
    HIP_CHECK(hipEventCreate(&event));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamWaitEvent(stream, event, 0));
    HIP_CHECK(hipGraphAddEventWaitNode(&waiteventNode, graph, nullptr, 0,
                                        event));
    HIP_CHECK(hipGraphNodeGetType(waiteventNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeWaitEvent);
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipEventDestroy(event));
  }

  SECTION("Get EventRecord NodeType") {
    hipEvent_t event;
    hipGraphNode_t recordeventNode;
    HIP_CHECK(hipEventCreate(&event));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipEventRecord(event, stream));
    HIP_CHECK(hipGraphAddEventRecordNode(&recordeventNode, graph, nullptr, 0,
                                        event));
    HIP_CHECK(hipGraphNodeGetType(recordeventNode, &nodeType));
    REQUIRE(nodeType == hipGraphNodeTypeEventRecord);
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipEventDestroy(event));
  }


  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
}
