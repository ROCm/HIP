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
Testcase Scenarios of hipGraphExecMemcpyNodeSetParamsFromSymbol API:
Functional
1) Allocate global symbol memory, Instantiate a graph with memcpy node,
  obtain executable graph and update the node params with set exec api call.
  Make sure they are taking effect.
2) Allocate const symbol memory, Instantiate a graph with memcpy node,
  obtain executable graph and update the node params with set exec api call.
  Make sure they are taking effect.
Negative
1) Pass hGraphExec as nullptr and check if api returns error.
2) Pass GraphNode as nullptr and check if api returns error.
3) Pass destination ptr as nullptr, api expected to return error code.
4) Pass symbol ptr as nullptr, api expected to return error code.
5) Pass count as zero, api expected to return error code.
6) Pass offset+count greater than allocated size, api expected to return error code.
7) Pass same symbol pointer as source ptr and destination ptr, api expected to return error code.
8) Pass Pass both dstn ptr and source ptr as 2 different symbol ptr, api expected to return error code.
9) Copy from device ptr to host ptr but pass kind as different, api expected to return error code.
10) Check with other graph node but pass same graphExec, api expected to return error code.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <limits>
#define SIZE 256

__device__ int globalIn[SIZE];
__device__ int globalOut[SIZE];
__device__ __constant__ int globalConst[SIZE];


/* Test verifies hipGraphExecMemcpyNodeSetParamsFromSymbol API Negative scenarios.
 */
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative") {
  constexpr size_t Nbytes = SIZE * sizeof(int);
  int *A_d{nullptr}, *B_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, nullptr,
                           &A_h, &B_h, nullptr, SIZE, false);

  hipError_t ret;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyToSymbolNode, memcpyFromSymbolNode, memcpyH2D_A;
  std::vector<hipGraphNode_t> dependencies;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyH2D_A);

  // Adding MemcpyNodeToSymbol
  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
                                          dependencies.data(),
                                          dependencies.size(),
                                          HIP_SYMBOL(globalIn),
                                          A_d, Nbytes, 0,
                                          hipMemcpyDeviceToDevice));
  dependencies.clear();
  dependencies.push_back(memcpyToSymbolNode);

  HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbolNode, graph,
                                            dependencies.data(),
                                            dependencies.size(),
                                            B_h,
                                            HIP_SYMBOL(globalConst),
                                            Nbytes, 0,
                                            hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphMemcpyNodeSetParamsFromSymbol(memcpyFromSymbolNode,
                                            B_d,
                                            HIP_SYMBOL(globalIn),
                                            Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
  // Instantiate the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Pass hGraphExec as nullptr") {
    ret = hipGraphExecMemcpyNodeSetParamsFromSymbol(nullptr,
                                                memcpyFromSymbolNode, B_d,
                                                HIP_SYMBOL(globalConst),
                                                Nbytes, 0,
                                                hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass GraphNode as nullptr") {
    ret = hipGraphExecMemcpyNodeSetParamsFromSymbol(graphExec,
                                                nullptr, B_d,
                                                HIP_SYMBOL(globalConst),
                                                Nbytes, 0,
                                                hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass destination ptr as nullptr") {
    ret = hipGraphExecMemcpyNodeSetParamsFromSymbol(graphExec,
                                                memcpyFromSymbolNode, nullptr,
                                                HIP_SYMBOL(globalConst),
                                                Nbytes, 0,
                                                hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass symbol ptr as nullptr") {
    ret = hipGraphExecMemcpyNodeSetParamsFromSymbol(graphExec,
                                                memcpyFromSymbolNode, B_d,
                                                nullptr,
                                                Nbytes, 0,
                                                hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidSymbol == ret);
  }
  SECTION("Pass count as zero") {
    ret = hipGraphExecMemcpyNodeSetParamsFromSymbol(graphExec,
                                                memcpyFromSymbolNode, B_d,
                                                HIP_SYMBOL(globalConst),
                                                0, 0,
                                                hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass offset+count greater than allocated size") {
    ret = hipGraphExecMemcpyNodeSetParamsFromSymbol(graphExec,
                                                memcpyFromSymbolNode, B_d,
                                                HIP_SYMBOL(globalConst),
                                                Nbytes, 10,
                                                hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass same symbol pointer as source and destination ptr") {
    ret = hipGraphExecMemcpyNodeSetParamsFromSymbol(graphExec,
                                                memcpyFromSymbolNode,
                                                HIP_SYMBOL(globalIn),
                                                HIP_SYMBOL(globalIn),
                                                Nbytes, 0,
                                                hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass both dstn ptr and source ptr as 2 different symbol ptr") {
    ret = hipGraphExecMemcpyNodeSetParamsFromSymbol(graphExec,
                                                memcpyFromSymbolNode,
                                                HIP_SYMBOL(globalIn),
                                                HIP_SYMBOL(globalOut),
                                                Nbytes, 0,
                                                hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Copy from device ptr to host ptr but pass kind as different") {
    ret = hipGraphExecMemcpyNodeSetParamsFromSymbol(graphExec,
                                                memcpyFromSymbolNode,
                                                B_h,
                                                HIP_SYMBOL(globalOut),
                                                Nbytes, 0,
                                                hipMemcpyDeviceToDevice);
    REQUIRE(hipSuccess != ret);
  }
  SECTION("Check with other graph node") {
    hipGraphNode_t memcpyFromSymbolNode1{};
    ret = hipGraphExecMemcpyNodeSetParamsFromSymbol(graphExec,
                                                memcpyFromSymbolNode1,
                                                B_d,
                                                HIP_SYMBOL(globalOut),
                                                Nbytes, 0,
                                                hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HipTest::freeArrays<int>(A_d, B_d, nullptr,
                           A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphDestroy(graph));
}

static
void hipGraphExecMemcpyNodeSetParamsFromSymbol_GlobalMem(bool useConstVar) {
  constexpr size_t Nbytes = SIZE * sizeof(int);
  hipGraphNode_t memcpyD2H_B;
  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, &C_d,
                           &A_h, &B_h, nullptr, SIZE, false);

  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyToSymbolNode, memcpyFromSymbolNode, memcpyH2D_A;
  std::vector<hipGraphNode_t> dependencies;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyH2D_A);

  if (useConstVar) {
    HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
          dependencies.data(),
          dependencies.size(),
          HIP_SYMBOL(globalConst),
          A_d, Nbytes, 0,
          hipMemcpyDeviceToDevice));
  } else {
    HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
          dependencies.data(),
          dependencies.size(),
          HIP_SYMBOL(globalIn),
          A_d, Nbytes, 0,
          hipMemcpyDeviceToDevice));
  }
  dependencies.clear();
  dependencies.push_back(memcpyToSymbolNode);

  if (useConstVar) {
    HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbolNode, graph,
                                            dependencies.data(),
                                            dependencies.size(),
                                            C_d,
                                            HIP_SYMBOL(globalConst),
                                            Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
  } else {
    HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbolNode, graph,
                                            dependencies.data(),
                                            dependencies.size(),
                                            C_d,
                                            HIP_SYMBOL(globalIn),
                                            Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
  }
  dependencies.clear();
  dependencies.push_back(memcpyFromSymbolNode);

  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_B, graph, dependencies.data(),
                                    dependencies.size(), B_h, B_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Update the node with B_d destination pointer from C_d
  if (useConstVar) {
    HIP_CHECK(hipGraphExecMemcpyNodeSetParamsFromSymbol(graphExec,
                                            memcpyFromSymbolNode,
                                            B_d,
                                            HIP_SYMBOL(globalConst),
                                            Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
  } else {
    HIP_CHECK(hipGraphExecMemcpyNodeSetParamsFromSymbol(graphExec,
                                            memcpyFromSymbolNode,
                                            B_d,
                                            HIP_SYMBOL(globalIn),
                                            Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
  }

  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  // Validating the result
  for (int i = 0; i < SIZE; i++) {
    if (B_h[i] != A_h[i]) {
       WARN("Validation failed B_h[i] " << B_h[i] << "A_h[i] " << A_h[i]);
       REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, B_d, C_d,
                           A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/* Test verifies hipGraphExecMemcpyNodeSetParamsFromSymbol Functional scenario.
1) Allocate global symbol memory, Instantiate a graph with memcpy node,
  obtain executable graph and update the node params with set exec api call.
  Make sure they are taking effect.
2) Allocate const symbol memory, Instantiate a graph with memcpy node,
  obtain executable graph and update the node params with set exec api call.
  Make sure they are taking effect.
 */
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Functional") {
  SECTION("Check and update with Global Device Symbol Memory") {
    hipGraphExecMemcpyNodeSetParamsFromSymbol_GlobalMem(false);
  }
  SECTION("Check and update with Constant Global Device Symbol Memory") {
    hipGraphExecMemcpyNodeSetParamsFromSymbol_GlobalMem(true);
  }
}
