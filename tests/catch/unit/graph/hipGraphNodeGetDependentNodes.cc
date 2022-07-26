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

Testcase Scenarios
------------------
Functional:
1) Create a graph and add nodes with dependencies. Query for dependent nodes of the node passed and verify the result with dependencies defined.
2) When pDependentNodes is passed as nullptr, verify pNumDependentNodes returns the number of dependent nodes.
3) When pNumDependentNodes is higher than the actual number of dependent nodes, the remaining entries in pDependentNodes will be set to NULL,
 and the number of nodes actually obtained will be returned in pNumDependentNodes.
4) When pNumDependentNodes is lesser than the actual number of dependent nodes, api should return the requested number of nodes in pDependentNodes.

Argument Validation:
1) Add a single node in graph and pass the node to api. Verify the api returns dependent nodes as 0.
2) Pass node as nullptr and verify api doesn’t crash, returns error code.
3) Pass pNumDependentNodes as nullptr and verify api doesn’t crash, returns error code.
4) Pass node as un-initialized/invalid parameter and verify api returns error code.

*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

static __global__ void updateResult(int* C_d, int* Res_d, int val,
                                                  int64_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
    Res_d[i] = C_d[i] + val;
  }
}

static __global__ void vectorSum(const int* A_d, const int* B_d,
                                 const int* C_d, int* Res_d, size_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < NELEM; i += stride) {
    Res_d[i] = A_d[i] + B_d[i] + C_d[i];
  }
}

/**
 * Verify api when GetDependent nodes is requested
 * for actual number of nodes.
 */
static void queryActualNumOfDepNodes(const std::vector<hipGraphNode_t> &Nlist,
                             hipGraphNode_t kernel_vecSqr, size_t numDeps) {
  hipGraphNode_t* depnodes;
  int numBytes = sizeof(hipGraphNode_t) * numDeps;
  depnodes = reinterpret_cast<hipGraphNode_t *>(malloc(numBytes));
  REQUIRE(depnodes != nullptr);
  HIP_CHECK(hipGraphNodeGetDependentNodes(kernel_vecSqr, depnodes, &numDeps));
  REQUIRE(numDeps == Nlist.size());

  // Verify all dependent nodes are present in the node entries returned
  for (auto Node : Nlist) {
    bool found = false;
    for (size_t i = 0; i < numDeps; i++) {
      if (Node == depnodes[i]) {
        found = true;
        break;
      }
    }

    if (!found) {
      INFO("Dependent node " << Node << " not present in returned list");
      REQUIRE(false);
    }
  }
  free(depnodes);
}

/**
 * Verify api when GetDependent nodes queried
 * for greater number than actual number of nodes.
 */
static void queryGreaterNumOfDepNodes(const std::vector<hipGraphNode_t> &Nlist,
                             hipGraphNode_t kernel_vecSqr, size_t numDeps) {
  constexpr auto addlEntries = 4;
  hipGraphNode_t* depnodes;
  size_t totDeps = numDeps + addlEntries;
  int numBytes = sizeof(hipGraphNode_t) * totDeps;
  depnodes = reinterpret_cast<hipGraphNode_t *>(malloc(numBytes));
  REQUIRE(depnodes != nullptr);
  HIP_CHECK(hipGraphNodeGetDependentNodes(kernel_vecSqr, depnodes, &totDeps));
  REQUIRE(totDeps == Nlist.size());

  for (auto i = numDeps; i < numDeps + addlEntries; i++) {
    REQUIRE(depnodes[i] == nullptr);
  }

  // Verify all dependent nodes are present in the node entries returned
  for (auto Node : Nlist) {
    bool found = false;
    for (size_t i = 0; i < numDeps; i++) {
      if (Node == depnodes[i]) {
        found = true;
        break;
      }
    }

    if (!found) {
      INFO("Dependent node " << Node << " not present in returned list");
      REQUIRE(false);
    }
  }
  free(depnodes);
}

/**
 * Verify api when GetDependent nodes queried
 * for lesser number than actual number of nodes.
 */
static void queryLesserNumOfDepNodes(const std::vector<hipGraphNode_t> &Nlist,
                             hipGraphNode_t kernel_vecSqr, size_t numDeps) {
  size_t totDeps = numDeps - 1;
  hipGraphNode_t* depnodes;
  int numBytes = sizeof(hipGraphNode_t) * totDeps;
  size_t count{};
  depnodes = reinterpret_cast<hipGraphNode_t *>(malloc(numBytes));
  REQUIRE(depnodes != nullptr);
  HIP_CHECK(hipGraphNodeGetDependentNodes(kernel_vecSqr, depnodes, &totDeps));
  REQUIRE(totDeps == Nlist.size() - 1);

  // Verify all dependent nodes are present in the node entries returned
  for (auto Node : Nlist) {
    for (size_t i = 0; i < totDeps; i++) {
      if (Node == depnodes[i]) {
        count++;
        break;
      }
    }
  }
  REQUIRE(count == totDeps);
  free(depnodes);
}

/**
 * Functional Test for getting dependent nodes in graph and verifying execution
 */
TEST_CASE("Unit_hipGraphNodeGetDependentNodes_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraphNode_t kernel_vecSqr{}, kernel_vecAdd{};
  hipGraphNode_t kernelmod1{}, kernelmod2{}, kernelmod3{};
  hipGraphNode_t memcpyD2H{}, memcpyH2D_A{};
  hipKernelNodeParams kernelNodeParams{};
  hipGraph_t graph{};
  size_t numDeps{};
  hipStream_t streamForGraph;
  int *A_d, *C_d;
  int *A_h, *C_h;
  int *Res1_d, *Res2_d, *Res3_d;
  int *Sum_d, *Sum_h;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HipTest::initArrays<int>(&A_d, &C_d, &Sum_d, &A_h, &C_h, &Sum_h, N);
  HipTest::initArrays<int>(&Res1_d, &Res2_d, &Res3_d,
                           nullptr, nullptr, nullptr, N);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  // Initialize input buffer and vecsqr result
  for (size_t i = 0; i < N; ++i) {
      A_h[i] = i + 1;
      C_h[i] = A_h[i] * A_h[i];
  }

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                   Nbytes, hipMemcpyHostToDevice));

  void* kernelArgsVS[] = {&A_d, &C_d, reinterpret_cast<void *>(&NElem)};
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func =
                       reinterpret_cast<void *>(HipTest::vector_square<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgsVS);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecSqr, graph, &memcpyH2D_A, 1,
                                                        &kernelNodeParams));

  // Create multiple nodes dependent on vecSqr node.
  // Dependent nodes takes vecSqr input and computes output independently.
  std::vector<hipGraphNode_t> nodelist;
  int incValue1{1};
  void* kernelArgs1[] = {&C_d, &Res1_d, &incValue1,
                                        reinterpret_cast<void *>(&NElem)};
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func =
                       reinterpret_cast<void *>(updateResult);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelmod1, graph, &kernel_vecSqr, 1,
                                                      &kernelNodeParams));
  nodelist.push_back(kernelmod1);

  int incValue2{2};
  void* kernelArgs2[] = {&C_d, &Res2_d, &incValue2,
                                        reinterpret_cast<void *>(&NElem)};
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func =
                       reinterpret_cast<void *>(updateResult);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelmod2, graph, &kernel_vecSqr, 1,
                                                      &kernelNodeParams));
  nodelist.push_back(kernelmod2);

  int incValue3{3};
  void* kernelArgs3[] = {&C_d, &Res3_d, &incValue3,
                                        reinterpret_cast<void *>(&NElem)};
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func =
                       reinterpret_cast<void *>(updateResult);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs3);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelmod3, graph, &kernel_vecSqr, 1,
                                                        &kernelNodeParams));
  nodelist.push_back(kernelmod3);

  HIP_CHECK(hipGraphNodeGetDependentNodes(kernel_vecSqr, nullptr, &numDeps));
  REQUIRE(numDeps == nodelist.size());

  // Verify api When Dependent nodes are requested for actual number of nodes.
  queryActualNumOfDepNodes(nodelist, kernel_vecSqr, numDeps);

  // Verify api When Dependent nodes are requested for more than
  // actual number of nodes.
  queryGreaterNumOfDepNodes(nodelist, kernel_vecSqr, numDeps);

  // Verify api When Dependent nodes are requested for less than
  // actual number of nodes.
  queryLesserNumOfDepNodes(nodelist, kernel_vecSqr, numDeps);

  // Compute sum from all dependent nodes
  void* kernelArgsAdd[] = {&Res1_d, &Res2_d, &Res3_d, &Sum_d,
                                             reinterpret_cast<void *>(&NElem)};
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func =
                       reinterpret_cast<void *>(vectorSum);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgsAdd);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph,
                                   nodelist.data(), nodelist.size(),
                                   &kernelNodeParams));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H, graph, &kernel_vecAdd, 1,
                                    Sum_h, Sum_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if ( Sum_h[i] != ( (C_h[i] + incValue1)
                     + (C_h[i] + incValue2)
                     + (C_h[i] + incValue3) ) ) {
      INFO("Sum not matching at " << i << " Sum_h[i] " << Sum_h[i]
                                       << " C_h[i] " << C_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, C_d, Sum_d, A_h, C_h, Sum_h, false);
  HipTest::freeArrays<int>(Res1_d, Res2_d, Res3_d,
                             nullptr, nullptr, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test performs api parameter validation by passing various values
 * as input and output parameters and validates the behavior.
 * Test will include both negative and positive scenarios.
 */
TEST_CASE("Unit_hipGraphNodeGetDependentNodes_ParamValidation") {
  hipGraph_t graph{};
  const int numBytes = 100;
  size_t numDeps{1};
  hipGraphNode_t memsetNode{}, depnodes{};
  hipError_t ret{};
  char *A_d;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&A_d, numBytes));
  hipMemsetParams memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 1;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numBytes * sizeof(char);
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr,
                                                    0, &memsetParams));

  SECTION("single node in graph") {
    ret = hipGraphNodeGetDependentNodes(memsetNode, &depnodes, &numDeps);

    // Api expected to return success and no dependent nodes.
    REQUIRE(ret == hipSuccess);
    REQUIRE(numDeps == 0);
  }

  SECTION("node as nullptr") {
    ret = hipGraphNodeGetDependentNodes(nullptr, &depnodes, &numDeps);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("NumDependentNodes as nullptr") {
    ret = hipGraphNodeGetDependentNodes(memsetNode, &depnodes, nullptr);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("node as un-initialized/invalid parameter") {
    hipGraphNode_t uninit_node{};
    ret = hipGraphNodeGetDependentNodes(uninit_node, &depnodes, &numDeps);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
}
