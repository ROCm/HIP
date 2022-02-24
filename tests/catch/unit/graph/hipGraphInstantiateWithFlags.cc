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

/*
hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph, unsigned long long flags);
Testcase Scenarios of hipGraphInstantiateWithFlags API:

Negative:
1) Pass nullptr to pGraphExec
2) Pass nullptr to graph
4) Pass invalid flag

Functional:

1) Create dependencies graph and instantiate the graph
2) Create graph in one GPU device and instantiate, launch in peer GPU device
3) Create stream capture graph and instantite the graph
4) Create stream capture graph in one GPU device  and instantite the graph launch
   in peer GPU device

Mapping is missing for NVIDIA platform hence skipping the testcases
*/


#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

constexpr size_t N = 1000000;
#if HT_AMD
/* This test covers the negative scenarios of
   hipGraphInstantiateWithFlags API */
TEST_CASE("Unit_hipGraphInstantiateWithFlags_Negative") {
#if HT_NVIDIA
  SECTION("Passing nullptr pGraphExec") {
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    REQUIRE(hipGraphInstantiateWithFlags(nullptr,
                                         graph, 0) == hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to graph") {
    hipGraphExec_t graphExec;
    REQUIRE(hipGraphInstantiateWithFlags(&graphExec,
                                         nullptr, 0) == hipErrorInvalidValue);
  }

  SECTION("Passing Invalid flag") {
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphExec_t graphExec;
    REQUIRE(hipGraphInstantiateWithFlags(&graphExec, graph, 10) != hipSuccess);
  }
#endif
}
/*
This function verifies the following scenarios
1. Creates dependency graph, Instantiates the graph with flags and verifies it
2. Creates graph on one GPU-1 device and instantiates the graph on peer GPU device
*/
void GraphInstantiateWithFlags_DependencyGraph(bool ctxt_change = false) {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memset_A, memset_B, memsetKer_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  hipMemsetParams memsetParams{};
  int memsetVal{};
  size_t NElem{N};

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

  if (ctxt_change) {
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));
  }
  // Instantiate and launch the cloned graph
  HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/*
This function verifies the following scenarios
1. Creates stream capture graph, Instantiates the graph with flags and verifies it
2. Creates graph on one GPU-1 device and instantiates the graph on peer GPU device
*/
void GraphInstantiateWithFlags_StreamCapture(bool deviceContextChg = false) {
  float *A_d, *C_d;
  float *A_h, *C_h;
  size_t Nbytes = N * sizeof(float);
  hipStream_t stream;
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};

  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
      A_h[i] = 1.618f + i;
  }
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);
  HIP_CHECK(hipGraphCreate(&graph, 0));


  HIP_CHECK(hipStreamCreate(&stream));
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                              dim3(threadsPerBlock), 0, stream, A_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  if (deviceContextChg) {
    HIP_CHECK(hipSetDevice(1));
    HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));
  }

  // Validate end capture is successful
  REQUIRE(graph != nullptr);
  HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));
  REQUIRE(graphExec != nullptr);

  HIP_CHECK(hipGraphLaunch(graphExec, stream));

  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      UNSCOPED_INFO("A and C not matching at " << i);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  free(A_h);
  free(C_h);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
}
/*
This testcase verifies hipGraphInstantiateWithFlags API
by creating dependency graph and instantiate, launching and verifying
the result
*/
TEST_CASE("Unit_hipGraphInstantiateWithFlags_DependencyGraph") {
  GraphInstantiateWithFlags_DependencyGraph();
}
/*
This testcase verifies hipGraphInstantiateWithFlags API
by creating dependency graph on GPU-0 and instantiate, launching and verifying
the result on GPU-1
*/
#if HT_NVIDIA
TEST_CASE("Unit_hipGraphInstantiateWithFlags_DependencyGraphDeviceCtxtChg") {
  int numDevices = 0;
  int canAccessPeer = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      GraphInstantiateWithFlags_DependencyGraph(true);
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}
#endif
/*
This testcase verifies hipGraphInstantiateWithFlags API
by creating capture graph and instantiate, launching and verifying
the result
*/
TEST_CASE("Unit_hipGraphInstantiateWithFlags_StreamCapture") {
  GraphInstantiateWithFlags_StreamCapture();
}

/*
This testcase verifies hipGraphInstantiateWithFlags API
by creating capture graph on GPU-0 and instantiate, launching and verifying
the result on GPU-1
*/
TEST_CASE("Unit_hipGraphInstantiateWithFlags_StreamCaptureDeviceContextChg") {
  int numDevices = 0;
  int canAccessPeer = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      GraphInstantiateWithFlags_StreamCapture(true);
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}
#endif
