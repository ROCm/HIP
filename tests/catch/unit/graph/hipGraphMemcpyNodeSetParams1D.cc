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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios :
Functional-
1) Create a graph, add Memcpy node to graph, update the Memcpy node params with set and make sure they are taking effect.
Negative-
1) Pass pGraphNode as nullptr and check if api returns error.
2) Pass destination ptr is nullptr, api expected to return error code.
3) Pass source ptr is nullptr, api expected to return error code.
4) Pass count as zero, api expected to return error code.
5) Pass same pointer as source ptr and destination ptr, api expected to return error code.
6) Pass overlap memory as source ptr and destination ptr where source ptr is ahead of destination ptr, api expected to return error code.
7) Pass overlap memory as source ptr and destination ptr where destination ptr is ahead of source ptr, api expected to return error code.
8) If count is more than allocated size for source and destination ptr, api should return error code.
9) If count is less than allocated size for source and destination ptr, api should return error code.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/* Test verifies hipGraphMemcpyNodeSetParams1D API Negative scenarios.
 */
TEST_CASE("Unit_hipGraphMemcpyNodeSetParams1D_Negative") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  int *A_d, *A_h;
  hipGraphNode_t memcpyNode{};
  hipError_t ret;

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&A_h, Nbytes));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));

  SECTION("Pass pGraphNode as nullptr") {
    ret = hipGraphMemcpyNodeSetParams1D(nullptr, A_d, A_h, Nbytes,
                                        hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass destination ptr is nullptr") {
    ret = hipGraphMemcpyNodeSetParams1D(memcpyNode, nullptr, A_h, Nbytes,
                                        hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass source ptr is nullptr") {
    ret = hipGraphMemcpyNodeSetParams1D(memcpyNode, A_d, nullptr, Nbytes,
                                        hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass count as zero") {
    ret = hipGraphMemcpyNodeSetParams1D(memcpyNode, A_d, A_h, 0,
                                        hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#if HT_AMD
  SECTION("Pass same pointer as source ptr and destination ptr") {
    ret = hipGraphMemcpyNodeSetParams1D(memcpyNode, A_d, A_d, Nbytes,
                                        hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass overlap memory where destination ptr is ahead of source ptr") {
    ret = hipGraphMemcpyNodeSetParams1D(memcpyNode, A_d, A_d-5, Nbytes,
                                        hipMemcpyHostToDevice);
    REQUIRE(hipSuccess == ret);
  }
#endif
  SECTION("Pass overlap memory where source ptr is ahead of destination ptr") {
    ret = hipGraphMemcpyNodeSetParams1D(memcpyNode, A_d+5, A_d, Nbytes-5,
                                        hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Copy more than allocated memory") {
    ret = hipGraphMemcpyNodeSetParams1D(memcpyNode, A_d, A_h, Nbytes+8,
                                        hipMemcpyHostToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Copy less than allocated memory") {
    ret = hipGraphMemcpyNodeSetParams1D(memcpyNode, A_d, A_h, Nbytes-8,
                                        hipMemcpyHostToDevice);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Change the kind from H2D to D2H") {
    ret = hipGraphMemcpyNodeSetParams1D(memcpyNode, A_d, A_h, Nbytes,
                                        hipMemcpyDeviceToHost);
    REQUIRE(hipSuccess == ret);
  }

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(A_h));
  HIP_CHECK(hipGraphDestroy(graph));
}

/* Test verifies hipGraphMemcpyNodeSetParams1D API Functional scenarios.
 */
TEST_CASE("Unit_hipGraphMemcpyNodeSetParams1D_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};

  int *hData = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hData != nullptr);
  memset(hData, 0, Nbytes);

  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t streamForGraph;

  HIP_CHECK(hipStreamCreate(&streamForGraph));

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphMemcpyNodeSetParams1D(memcpyD2H_C, hData, C_d, Nbytes,
                                          hipMemcpyDeviceToHost));

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

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, hData, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphDestroy(graph));
  free(hData);
}

