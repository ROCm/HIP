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

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

/* Test verifies hipGraphAddKernelNode API Negative scenarios.
 */

TEST_CASE("Unit_hipGraphAddKernelNode_Negative") {
  constexpr int N = 1024;
  size_t NElem{N};
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  int *A_d, *B_d, *C_d;
  hipGraph_t graph;
  hipError_t ret;
  hipGraphNode_t kNode;
  hipKernelNodeParams kNodeParams{};
  std::vector<hipGraphNode_t> dependencies;

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

  SECTION("Pass pGraphNode as nullptr") {
    ret = hipGraphAddKernelNode(nullptr, graph, nullptr, 0, &kNodeParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass Graph as nullptr") {
    ret = hipGraphAddKernelNode(&kNode, nullptr, nullptr, 0, &kNodeParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass invalid numDependencies") {
    ret = hipGraphAddKernelNode(&kNode, graph, nullptr, 11, &kNodeParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass invalid numDependencies and valid list for dependencies") {
    HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams));
    dependencies.push_back(kNode);
    ret = hipGraphAddKernelNode(&kNode, graph,
              dependencies.data(), dependencies.size()+1, &kNodeParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass NodeParams as nullptr") {
    ret = hipGraphAddKernelNode(&kNode, graph,
              dependencies.data(), dependencies.size(), nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass NodeParams func datamember as nullptr") {
    kNodeParams.func = nullptr;
    ret = hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams);
    REQUIRE(hipSuccess != ret);
  }
  SECTION("Pass kernelParams datamember as nullptr") {
    kNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
    kNodeParams.kernelParams = nullptr;
    ret = hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#if HT_AMD
// On Cuda setup this test case getting failed
  SECTION("Try adding kernel node after destroy the already created graph") {
    kNodeParams.kernelParams = reinterpret_cast<void **>(kernelArgs);
    HIP_CHECK(hipGraphDestroy(graph));
    ret = hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#endif

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
}

