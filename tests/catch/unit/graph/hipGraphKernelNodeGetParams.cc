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
Negative -
1) Pass node as nullptr and verify api returns error code.
2) Pass pNodeParams as nullptr and verify api returns error code.
Functional -
1) Create a graph, add kernel node to graph with desired kernel node params.
   Verify api fetches the node params mentioned while adding kernel node.
2) Set kernel node params with hipGraphKernelNodeSetParams,
   now get the params and verify both are same.
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#define THREADS_PER_BLOCK 512

/* Test verifies hipGraphKernelNodeGetParams API Negative scenarios.
 */

TEST_CASE("Unit_hipGraphKernelNodeGetParams_Negative") {
  constexpr int N = 1024;
  size_t NElem{N};
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
  kNodeParams.gridDim = dim3(N / THREADS_PER_BLOCK, 1, 1);
  kNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void **>(kernelArgs);
  kNodeParams.extra = nullptr;

  HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams));

  SECTION("Pass node as nullptr") {
    ret = hipGraphKernelNodeGetParams(nullptr, &kNodeParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  SECTION("Pass kNodeParams as nullptr") {
    ret = hipGraphKernelNodeGetParams(kNode, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipGraphDestroy(graph));
}

static bool dim3_compare(dim3 node1, dim3 node2) {
  if ((node1.x == node2.x) && (node1.y == node2.y) && (node1.z == node2.z))
    return true;
  else
    return false;
}

static bool kernelParam_compare(void **p1, void ** p2) {
  for (int i = 0; i < 4; i++) {
    if (*reinterpret_cast<int *>(p1[i]) != *reinterpret_cast<int *>(p2[i]))
      return false;
  }
  return true;
}

static bool node_compare(hipKernelNodeParams *kNode1,
                         hipKernelNodeParams *kNode2) {
  if (!dim3_compare(kNode1->blockDim, kNode2->blockDim))
    return false;
  if (kNode1->extra != kNode2->extra)
    return false;
  if (kNode1->func != kNode2->func)
    return false;
  if (!dim3_compare(kNode1->gridDim, kNode2->gridDim))
    return false;
  if (!kernelParam_compare(kNode1->kernelParams, kNode2->kernelParams))
    return false;
  if (kNode1->sharedMemBytes != kNode2->sharedMemBytes)
    return false;
  return true;
}

/* Test verifies hipGraphKernelNodeGetParams API Functional scenarios.
 */
TEST_CASE("Unit_hipGraphKernelNodeGetParams_Functional") {
  constexpr int N = 1024;
  size_t NElem{N};
  int *A_d, *B_d, *C_d;
  hipGraph_t graph;
  hipGraphNode_t kNode;
  hipKernelNodeParams kNodeParams{};
  HIP_CHECK(hipMalloc(&A_d, sizeof(int) * N));
  HIP_CHECK(hipMalloc(&B_d, sizeof(int) * N));
  HIP_CHECK(hipMalloc(&C_d, sizeof(int) * N));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};

  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(N / THREADS_PER_BLOCK, 1, 1);
  kNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void **>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kNodeParams));

  SECTION("Get Kernel Param and verify.") {
    hipKernelNodeParams kNodeGetParams;
    HIP_CHECK(hipGraphKernelNodeGetParams(kNode, &kNodeGetParams));
    REQUIRE(true == node_compare(&kNodeParams, &kNodeGetParams));
  }

  SECTION("Set kernel node params then Get Kernel Param and verify.") {
    hipKernelNodeParams kNodeParams1;
    kNodeParams1.func =
          reinterpret_cast<void *>(HipTest::vectorADDReverse<int>);
    kNodeParams1.gridDim = dim3(N / THREADS_PER_BLOCK, 1, 1);
    kNodeParams1.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
    kNodeParams1.sharedMemBytes = 0;
    kNodeParams1.kernelParams = reinterpret_cast<void **>(kernelArgs);
    kNodeParams1.extra = nullptr;

    hipKernelNodeParams kNodeGetParams1;
    HIP_CHECK(hipGraphKernelNodeSetParams(kNode, &kNodeParams1));
    HIP_CHECK(hipGraphKernelNodeGetParams(kNode, &kNodeGetParams1));
    REQUIRE(true == node_compare(&kNodeParams1, &kNodeGetParams1));
  }
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipGraphDestroy(graph));
}
