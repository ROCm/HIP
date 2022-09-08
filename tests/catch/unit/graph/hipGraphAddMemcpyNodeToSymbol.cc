/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
Testcase Scenarios of hipGraphAddMemcpyNodeToSymbol API:

Functional :

1. Allocate global symbol memory, add the MemcpyNodeToSymbol
   node to the graph and verify  for different memory kinds
2. Allocate const memory  add the MemcpyNodeToSymbol node to
   the graph and verify  for different memory kinds
3. Allocate global symbol memory and  device memory in GPU-0
   and perform MemcpyToSymbol from peer GPU by adding it to the graph node.
4. Allocate const symbol memory and  device memory in GPU-0
   and perform MemcpyToSymbol from peer GPU by adding it to the graph node.
5. Allocate global memory, Add MemcpyToSymbolNode,KernelNode and memcpynode and validating
   the behaviour

Negative :

1) Pass nullptr to graph node
2) Pass nullptr to graph
3) Pass nullptr to dependencies
4) Pass invalid numDependencies
5) Pass nullptr to dst
6) Pass nullptr to symbol
7) Pass invalid count 
8) Pass offset+count greater than allocated size
9) Pass unintialized graph
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <limits>
#define SIZE 256

__device__ int globalIn[SIZE];
__device__ __constant__ int globalConst[SIZE];

__global__ void MemcpyToSymbolKernel(int* B_d) {
  for (int i = 0 ; i < SIZE; i++) {
      B_d[i] = globalIn[i];
  }
}

/* This testcase verifies negative scenarios of
   hipGraphAddMemcpyNodeToSymbol API */
TEST_CASE("Unit_hipGraphAddMemcpyNodeToSymbol_Negative") {
  constexpr size_t Nbytes = SIZE * sizeof(int);
  int *A_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, nullptr,
                           &A_h, &B_h, nullptr, SIZE, false);

  hipGraph_t graph;
  hipGraphNode_t memcpyH2D_A;
  std::vector<hipGraphNode_t> dependencies;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyH2D_A);
#if HT_NVIDIA
  hipGraphNode_t memcpyToSymbolNode;
  SECTION("Passing nullptr to graph") {
    REQUIRE(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, nullptr,
                                          dependencies.data(),
                                          dependencies.size(),
                                          HIP_SYMBOL(globalIn),
                                          A_h, Nbytes, 0,
                                          hipMemcpyDeviceToDevice)
                                          == hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to graph node") {
    REQUIRE(hipGraphAddMemcpyNodeToSymbol(nullptr, graph,
                                          dependencies.data(),
                                          dependencies.size(),
                                          HIP_SYMBOL(globalIn),
                                          A_d, Nbytes, 0,
                                          hipMemcpyDeviceToDevice)
                                          == hipErrorInvalidValue);
  }

  SECTION("Passing size > 1 and dependencies as nullptr") {
    REQUIRE(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
                                          nullptr,
                                          1,
                                          HIP_SYMBOL(globalIn),
                                          A_d, Nbytes, 0,
                                          hipMemcpyDeviceToDevice)
                                          == hipErrorInvalidValue);
  }

  SECTION("Passing invalid dependencies size") {
    REQUIRE(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
                                          dependencies.data(),
                                          10,
                                          HIP_SYMBOL(globalIn),
                                          A_d, Nbytes, 0,
                                          hipMemcpyDeviceToDevice)
                                          == hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to dst") {
    REQUIRE(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
                                          dependencies.data(),
                                          dependencies.size(),
                                          nullptr,
                                          A_d, Nbytes, 0,
                                          hipMemcpyDeviceToDevice)
                                          == hipErrorInvalidSymbol);
  }

  SECTION("Passing nullptr to source") {
    REQUIRE(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
                                          dependencies.data(),
                                          dependencies.size(),
                                          HIP_SYMBOL(globalIn),
                                          nullptr, Nbytes, 0,
                                          hipMemcpyDeviceToDevice)
                                          == hipErrorInvalidValue);
  }

  SECTION("Passing offset+size > max size") {
    REQUIRE(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
                                          dependencies.data(),
                                          dependencies.size(),
                                          HIP_SYMBOL(globalIn),
                                          A_d, Nbytes, 10,
                                          hipMemcpyDeviceToDevice)
                                          == hipErrorInvalidValue);
  }

  SECTION("Passing Max count") {
  REQUIRE(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
                                        dependencies.data(),
                                        dependencies.size(),
                                        HIP_SYMBOL(globalIn),
                                        A_d,
                                        std::numeric_limits<int>::max(), 0,
                                        hipMemcpyDeviceToDevice)
                                        == hipErrorInvalidValue);
  }

  SECTION("Pass Unintialized graph") {
  hipGraph_t unint_graph;
  REQUIRE(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, unint_graph,
                                        dependencies.data(),
                                        dependencies.size(),
                                        HIP_SYMBOL(globalIn),
                                        A_d,
                                        Nbytes, 0,
                                        hipMemcpyDeviceToDevice)
                                        == hipErrorInvalidValue);
  }
#endif

  HipTest::freeArrays<int>(A_d, nullptr, nullptr,
                      A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphDestroy(graph));
}
/*
This function is used to verify the following scenarios
1. Create global variable, allocate Memory in GPU-0 and create dependency graph of
   hipGraphAddMemcpyNodeToSymbol API in GPU-1 and validate the result
2. Allocate global memory, Create dependency graph and validate the result on GPU-0
3. Allocate global const memory, Create dependency graph and validate the result on GPU-0
4. Create global const variable, allocate Memory in GPU-0 and create dependency graph of
   hipGraphAddMemcpyNodeToSymbol API in GPU-1 and validate the result
*/
void hipGraphAddMemcpyNodeToSymbol_GlobalMemory(bool device_ctxchg = false,
                                                bool const_device_var = false) {
  constexpr size_t Nbytes = SIZE * sizeof(int);
  int *A_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, nullptr,
                           &A_h, &B_h, nullptr, SIZE, false);

  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyToSymbolNode, memcpyFromSymbolNode, memcpyH2D_A;
  std::vector<hipGraphNode_t> dependencies;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  if (device_ctxchg) {
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));
  }
  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyH2D_A);

  // Adding MemcpyNodeToSymbol

  if (const_device_var) {
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

  // Adding MemcpyNodeFromSymbol
  if (const_device_var) {
    HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbolNode, graph,
          dependencies.data(),
          dependencies.size(),
          B_h,
          HIP_SYMBOL(globalConst),
          Nbytes, 0, hipMemcpyDeviceToHost));
  } else {
    HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbolNode, graph,
          dependencies.data(),
          dependencies.size(),
          B_h,
          HIP_SYMBOL(globalIn),
          Nbytes, 0, hipMemcpyDeviceToHost));
  }

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  // Validating the result
  for (int i = 0; i < SIZE; i++) {
    if (B_h[i] != A_h[i]) {
       WARN("Validation failed B_h[i] " << B_h[i] << "A_h[i] " << A_h[i]);
       REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, nullptr, nullptr,
                      A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}
/*
This testcase verifies allocating global symbol memory,
add the MemcpyNodeToSymbol node to the graph and
erifying the result
*/
TEST_CASE("Unit_hipGraphAddMemcpyNodeToSymbol_GlobalMemory") {
  hipGraphAddMemcpyNodeToSymbol_GlobalMemory(false, false);
}

/*
This testcase verifies allocating global const symbol memory,
add the MemcpyNodeToSymbol node to the graph and
verifying the result
*/
TEST_CASE("Unit_hipGraphAddMemcpyNodeToSymbol_GlobalConstMemory") {
  hipGraphAddMemcpyNodeToSymbol_GlobalMemory(false, true);
}

#if HT_NVIDIA
/*
This testcase verifies allocating global symbol memory and device variables
in GPU-0 and add the MemcpyNodeToSymbol node to the graph and
verifying the result in GPU-1
*/
TEST_CASE("Unit_hipGraphAddMemcpyNodeToSymbol_GlobalMemoryPeerDevice") {
  int numDevices = 0;
  int canAccessPeer = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    hipDeviceCanAccessPeer(&canAccessPeer, 0, 1);
    if (canAccessPeer) {
      hipGraphAddMemcpyNodeToSymbol_GlobalMemory(true, false);
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}
/*
This testcase verifies allocating global const symbol memory and device variables
in GPU-0 and add the MemcpyNodeToSymbol node to the graph and
verifying the result in GPU-1
*/
TEST_CASE("Unit_hipGraphAddMemcpyNodeToSymbol_GlobalConstMemoryPeerDevice") {
  int numDevices = 0;
  int canAccessPeer = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    hipDeviceCanAccessPeer(&canAccessPeer, 0, 1);
    if (canAccessPeer) {
      hipGraphAddMemcpyNodeToSymbol_GlobalMemory(true, true);
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}
#endif
/*
This testcaser verifies allocating global memory,
Add MemcpyToSymbolNode,KernelNode and memcpynode and validating
the behaviour
*/
TEST_CASE("Unit_hipGraphAddMemcpyNodeToSymbol_MemcpyToSymbolNodeWithKernel") {
  constexpr size_t Nbytes = SIZE * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, SIZE);
  hipGraphNode_t memcpytosymbolkernel, memcpyD2H_B;
  hipKernelNodeParams kernelNodeParams{};
  int *A_d{nullptr}, *B_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, nullptr,
                           &A_h, &B_h, nullptr, SIZE, false);

  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyToSymbolNode, memcpyH2D_A;
  std::vector<hipGraphNode_t> dependencies;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  dependencies.push_back(memcpyH2D_A);

  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph,
                                            dependencies.data(),
                                            dependencies.size(),
                                            HIP_SYMBOL(globalIn),
                                            A_d, Nbytes, 0,
                                            hipMemcpyDeviceToDevice));
  dependencies.clear();
  dependencies.push_back(memcpyToSymbolNode);

  // Adding Kernel node
  void* kernelArgs1[] = {&B_d};
  kernelNodeParams.func =
                       reinterpret_cast<void *>(MemcpyToSymbolKernel);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&memcpytosymbolkernel, graph,
                                  dependencies.data(), dependencies.size(),
                                  &kernelNodeParams));
  dependencies.clear();
  dependencies.push_back(memcpytosymbolkernel);

  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_B, graph, dependencies.data(),
                                    dependencies.size(), B_h, B_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  // Validating the result
  for (int i = 0; i < SIZE; i++) {
    if (B_h[i] != A_h[i]) {
       WARN("Validation failed B_h[i] " << B_h[i] << "A_h[i] " << A_h[i]);
       REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, B_d, nullptr,
                           A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}
