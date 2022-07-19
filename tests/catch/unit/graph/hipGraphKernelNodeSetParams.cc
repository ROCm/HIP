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
Negative -
1) Pass node as nullptr and verify api returns error code.
2) Pass pNodeParams as nullptr and verify api returns error code.
Functional -
1) Add kernel node to graph with certain kernel params, now update the kernel
   node params with set and check taking effect after launching graph.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/* Test verifies hipGraphKernelNodeSetParams API Negative scenarios.
 */

TEST_CASE("Unit_hipGraphKernelNodeSetParams_Negative") {
  constexpr int N = 1024;
  size_t NElem{N};
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  int *A_d, *B_d, *C_d;
  hipError_t ret;
  hipGraph_t graph;
  hipGraphNode_t kNode;
  hipKernelNodeParams kNodeParams{};
  HIP_CHECK(hipMalloc(&A_d, sizeof(int) * N));
  HIP_CHECK(hipMalloc(&B_d, sizeof(int) * N));
  HIP_CHECK(hipMalloc(&C_d, sizeof(int) * N));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};

  kNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void **>(kernelArgs);
  kNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams));

  SECTION("Pass node as nullptr") {
    ret = hipGraphKernelNodeSetParams(nullptr, &kNodeParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  SECTION("Pass kNodeParams as nullptr") {
    ret = hipGraphKernelNodeSetParams(kNode, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Functional Test for API Set Kernel Params
 */

TEST_CASE("Unit_hipGraphKernelNodeSetParams_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memcpyNode, kNode;
  hipKernelNodeParams kNodeParams{}, kNodeParams1{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  std::vector<hipGraphNode_t> dependencies;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, A_d, A_h,
                                   Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, B_d, B_h,
                                   Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, dependencies.data(),
                                  dependencies.size(), &kNodeParams));

  kNodeParams1.func = reinterpret_cast<void *>(HipTest::vectorSUB<int>);
  kNodeParams1.gridDim = dim3(blocks);
  kNodeParams1.blockDim = dim3(threadsPerBlock);
  kNodeParams1.sharedMemBytes = 0;
  kNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams1.extra = nullptr;
  HIP_CHECK(hipGraphKernelNodeSetParams(kNode, &kNodeParams1));

  dependencies.clear();
  dependencies.push_back(kNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, dependencies.data(),
                                    dependencies.size(), C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  HipTest::checkVectorSUB<int>(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
