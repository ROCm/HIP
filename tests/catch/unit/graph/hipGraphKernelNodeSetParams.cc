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
2) Add kernel node to graph with certain kernel params, now get kernel node parameters
   with hipGraphKernelNodeGetParams, then update the kernel node params with
   hipGraphKernelNodeSetParams, finally check taking effect after launching graph.
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

static __global__ void ker_vec_add(int *A, int *B) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  A[i] = A[i] + B[i];
}

static __global__ void ker_vec_sub(int *A, int *B) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  A[i] = A[i] - B[i];
}

/**
 Internal class for creating nested graphs.
 */
class GraphKernelNodeGetSetParam {
  const int N = 1024;
  size_t Nbytes;
  const int threadsPerBlock = 256;
  const int blocks = (N / threadsPerBlock);
  hipGraphNode_t memcpyH2D_A1, memcpyH2D_A2, memcpyD2H_A3, vec_maths;
  hipGraph_t graph;
  hipKernelNodeParams kerNodeParams { };
  int *A1_d, *A2_d, *A1_h, *A2_h, *A3_h;

 public:
  // Create a nested Graph
  GraphKernelNodeGetSetParam() {
    Nbytes = N * sizeof(int);
    // Allocate device buffers
    HIP_CHECK(hipMalloc(&A1_d, Nbytes));
    HIP_CHECK(hipMalloc(&A2_d, Nbytes));
    // Allocate host buffers
    A1_h = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(A1_h != NULL);
    A2_h = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(A2_h != NULL);
    A3_h = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(A3_h != NULL);
    // Create all the 3 level graphs
    HIP_CHECK(hipGraphCreate(&graph, 0));
    void *kernelArgs[] = { &A1_d, &A2_d };
    kerNodeParams.func = reinterpret_cast<void*>(ker_vec_add);
    kerNodeParams.gridDim = dim3(blocks);
    kerNodeParams.blockDim = dim3(threadsPerBlock);
    kerNodeParams.sharedMemBytes = 0;
    kerNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kerNodeParams.extra = nullptr;
    HIP_CHECK(
        hipGraphAddKernelNode(&vec_maths, graph, nullptr, 0, &kerNodeParams));
    // Add nodes to graph
    HIP_CHECK(
        hipGraphAddMemcpyNode1D(&memcpyH2D_A1, graph, nullptr, 0, A1_d, A1_h,
                                Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipGraphAddMemcpyNode1D(&memcpyH2D_A2, graph, nullptr, 0, A2_d, A2_h,
                                Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipGraphAddMemcpyNode1D(&memcpyD2H_A3, graph, nullptr, 0, A3_h, A1_d,
                                Nbytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A1, &vec_maths, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A2, &vec_maths, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &vec_maths, &memcpyD2H_A3, 1));
  }

  // Fill Random Input Data
  void fillRandInpData() {
    for (int i = 0; i < N; i++) {
      A1_h[i] = (rand() % 256);  //NOLINT
      A2_h[i] = (rand() % 256);  //NOLINT
    }
  }

  hipGraph_t* getRootGraph() {
    return &graph;
  }

  void updateNode() {
    size_t numNodes = 0;
    HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
    hipGraphNode_t *nodes = reinterpret_cast<hipGraphNode_t*>(malloc(
        numNodes * sizeof(hipGraphNode_t)));
    HIP_CHECK(hipGraphGetNodes(graph, nodes, &numNodes));
    // Get the Graph node from the embedded graph
    size_t nodeIdx = 0;
    for (size_t idx = 0; idx < numNodes; idx++) {
      hipGraphNodeType nodeType;
      HIP_CHECK(hipGraphNodeGetType(nodes[idx], &nodeType));
      if (nodeType == hipGraphNodeTypeKernel) {
        nodeIdx = idx;
        break;
      }
    }
    hipKernelNodeParams nodeParam;
    HIP_CHECK(hipGraphKernelNodeGetParams(nodes[nodeIdx], &nodeParam));
    nodeParam.func = reinterpret_cast<void*>(ker_vec_sub);
    HIP_CHECK(hipGraphKernelNodeSetParams(nodes[nodeIdx], &nodeParam));
    free(nodes);
  }

  // Function to validate result
  void validateOutData() {
    HipTest::checkVectorSUB<int>(A1_h, A2_h, A3_h, N);
  }

  // Destroy resources
  ~GraphKernelNodeGetSetParam() {
    // Free all allocated buffers
    HIP_CHECK(hipFree(A2_d));
    HIP_CHECK(hipFree(A1_d));
    free(A3_h);
    free(A2_h);
    free(A1_h);
    HIP_CHECK(hipGraphDestroy(graph));
  }
};

TEST_CASE("Unit_hipGraphKernelNodeGetSetParams_Functional") {
  hipGraph_t *graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  GraphKernelNodeGetSetParam GraphKernelNodeGetSetParamObj;
  graph = GraphKernelNodeGetSetParamObj.getRootGraph();
  GraphKernelNodeGetSetParamObj.updateNode();
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, (*graph), nullptr,
                                nullptr, 0));
  GraphKernelNodeGetSetParamObj.fillRandInpData();
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  GraphKernelNodeGetSetParamObj.validateOutData();
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
}
