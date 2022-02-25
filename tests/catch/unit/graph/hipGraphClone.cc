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

/*
Testcase Scenarios of hipGraphClone API:

Negative:

1. Pass nullptr to cloned graph
2. pass nullptr to original graph

Functional:

1. Clone the graph,Instantiate and execute the cloned graph
2. Clone the graph and modify the original graph and ensure that the
   cloned graph is not modified
3. Create graph on one GPU device and clone it from peer GPU device
4. Create graph in one thread and clone it from multiple threads.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#define NUM_THREADS 10

/* This test covers the negative scenarios of
   hipGraphClone API */

TEST_CASE("Unit_hipGraphClone_Negative") {
  SECTION("Passing nullptr to Cloned graph") {
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    REQUIRE(hipGraphClone(nullptr, graph) == hipErrorInvalidValue);
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Passing nullptr to original graph") {
    hipGraph_t clonedGraph;
    REQUIRE(hipGraphClone(&clonedGraph, nullptr) == hipErrorInvalidValue);
  }
}
/*
This function creates the graph with dependencies
then performs device context change and clones the cloned graph
Executes the cloned graph and validates the result
*/
void hipGraphClone_DeviceContextChange() {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph, clonedgraph;
  hipGraphExec_t graphExec;
  hipStream_t streamForGraph;
  hipGraphNode_t memcpyH2D_A, memcpyD2H_A;
  int *A_d{nullptr}, *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, nullptr,
                      &A_h, &B_h, nullptr, N, false);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, graph, nullptr, 0, B_h, A_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &memcpyD2H_A, 1));
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipGraphClone(&clonedgraph, graph));
  // Instantiate and launch the original graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, clonedgraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  for (size_t i = 0; i < N; i++) {
    if (A_h[i] != B_h[i]) {
      INFO("Validation failed A_h[i] " << A_h[i] << " B_h[i] " << B_h[i]);
      REQUIRE(false);
    }
  }
  HipTest::freeArrays<int>(A_d, nullptr, nullptr, A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedgraph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
/*
This function does the following
1. Creates the graph with multiple dependencies
   clones the graph and validates the result.
2. Creates the graph, clones the graph and modifies
   the existing graph and execute the cloned graph
   to ensure that cloned graph is not modified
*/
void hipGraphClone_Func(bool ModifyOrigGraph = false) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph, clonedgraph;
  hipGraphNode_t memset_A, memset_B, memsetKer_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C, memcpyD2D_C,
                 memcpyD2H_C_new;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  hipMemsetParams memsetParams{};
  int memsetVal{};
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                                                              &memsetParams));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(B_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_B, graph, nullptr, 0,
                                                              &memsetParams));

  void* kernelArgs1[] = {&C_d, &memsetVal, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func =
                       reinterpret_cast<void *>(HipTest::memsetReverse<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&memsetKer_C, graph, nullptr, 0,
                                                        &kernelNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                   Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h,
                                   Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d,
                                   Nbytes, hipMemcpyDeviceToHost));

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
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_B, &memcpyH2D_B, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memsetKer_C, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2H_C, 1));

  HIP_CHECK(hipGraphClone(&clonedgraph, graph));

  if (ModifyOrigGraph) {
    // Modify Original graph by adding new dependency
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2D_C, graph, nullptr, 0,
                                      C_d, B_d,
                                      Nbytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C_new, graph, nullptr, 0,
                                      C_h, C_d,
                                      Nbytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2D_C, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2D_C,
                                      &memcpyD2H_C_new, 1));

    // Instantiate and launch the original graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));

    for (size_t i= 0; i < NElem; i++) {
      if (C_h[i] != B_h[i]) {
         INFO("Validation failed C_h is " << C_h[i] <<
               "B_h is " << B_h[i]);
         REQUIRE(false);
      }
    }
  }

  // Instantiate and launch the cloned graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, clonedgraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedgraph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
This testcase verifies following scenarios
1. Clones the graph and verify the result
2. Clones the graph, Modify the original graph and
   validate the result of the cloned graph
3. Device context change for cloned graph
*/
TEST_CASE("Unit_hipGraphClone_Functional") {
  SECTION("hipGraphClone Basic Functionality") {
    hipGraphClone_Func();
  }
  SECTION("hipGraphClone Modify Original graph") {
    hipGraphClone_Func(true);
  }

  SECTION("hipGraphClone Device context change") {
    int numDevices = 0;
    int canAccessPeer = 0;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    if (numDevices > 1) {
      HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
      if (canAccessPeer) {
        hipGraphClone_DeviceContextChange();
      } else {
        SUCCEED("Machine does not seem to have P2P");
      }
    } else {
      SUCCEED("skipped the testcase as no of devices is less than 2");
    }
  }
}

/*
This testcase creates the graph with dependencies
then creates multiple threads and clones the graph
in each thread and executes the cloned graph
hipGraphClone is failing in CUDA in multi threaded
scenario so excluded for nvidia
*/
#if HT_AMD
TEST_CASE("Unit_hipGraphClone_MultiThreaded") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphNode_t memcpyH2D_A, memcpyD2H_A;
  int *A_d{nullptr}, *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, nullptr,
                      &A_h, &B_h, nullptr, N, false);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, graph, nullptr, 0, B_h, A_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &memcpyD2H_A, 1));
  std::vector<std::thread> threads;
  auto lambdaFunc = [&](){
    hipGraph_t clonedgraph;
    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphClone(&clonedgraph, graph));
    // Instantiate and launch the cloned graph
    HIP_CHECK(hipGraphInstantiate(&graphExec, clonedgraph, nullptr,
          nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));

    for (size_t i = 0; i < N; i++) {
      if (A_h[i] != B_h[i]) {
        INFO("Validation failed A_h[i] " << A_h[i] << " B_h[i] " << B_h[i]);
        REQUIRE(false);
      }
    }

    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(clonedgraph));
  };
  for (int i = 0; i < NUM_THREADS; i++) {
    std::thread t(lambdaFunc);
    threads.push_back(std::move(t));
  }
  for (auto &t : threads) {
    t.join();
  }
  HipTest::freeArrays<int>(A_d, nullptr, nullptr, A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphDestroy(graph));
}
#endif
