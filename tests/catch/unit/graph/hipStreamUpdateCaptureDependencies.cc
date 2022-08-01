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
1) Begin capture on stream and launch work on the stream with dependencies.
Add additional depencies in the flow by calling the api with flag hipStreamAddCaptureDependencies. Verify added dependencies are taking effect.
2) Begin capture on stream and launch work on the stream with dependencies.
Replace the dependency set with new nodes by calling the api with flag hipStreamSetCaptureDependencies. Verify updated dependency list is taking effect.
3) Begin capture on hipStreamPerThread and launch work on the stream with dependencies.
Add additional depencies in the flow by calling the api with flag hipStreamAddCaptureDependencies. Verify added dependencies are taking effect.
4) Begin capture on hipStreamPerThread and launch work on the stream with dependencies.
Replace the dependency set with new nodes by calling the api with flag hipStreamSetCaptureDependencies. Verify updated dependency list is taking effect.

Argument Validation:
1) Pass Dependencies as nullptr and numDeps as 0. Verify api returns success.
2) Pass Dependencies as nullptr and numDeps as nonzero. Verify api fails with error code.
3) When numDeps exceeds actual number of nodes, verify api fails with error code.
4) When Invalid flag is passed, verify api fails with error code.
5) When dependency node is a un-initialized/invalid parameter, verify api fails with error code.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

static constexpr size_t N = 1000000;
static constexpr int LAUNCH_ITERS = 5;

static __global__ void updateResult(double* C_d, int val, int64_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
    C_d[i] = C_d[i] + val;
  }
}

static __global__ void vectorSum(const double* A_d, const double* B_d,
                            const double* C_d, double* Res_d, size_t NELEM) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < NELEM; i += stride) {
    Res_d[i] = A_d[i] + B_d[i] + C_d[i];
  }
}


static void UpdateStreamCaptureDependenciesSet(hipStream_t mstream) {
  hipStream_t stream1{nullptr}, stream2{nullptr}, streamForLaunch{nullptr};
  hipEvent_t memcpyEvent1, memsetEvent2, forkStreamEvent;
  hipGraph_t graph{nullptr}, capInfoGraph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  const hipGraphNode_t *nodelist{};
  size_t Nbytes = N * sizeof(double), numDependencies;
  double *A_d, *C_d;
  double *A_h, *C_h;

  HipTest::initArrays<double>(&A_d, nullptr, &C_d, &A_h, nullptr, &C_h, N);
  HIP_CHECK(hipStreamCreate(&streamForLaunch));

  // Initialize input buffer
  for (size_t i = 0; i < N; ++i) {
      A_h[i] = 3.146f + i;  // Pi
  }

  // Create cross stream dependencies.
  // memset/memcpy operations are done on stream1 and stream2
  // and they are joined back to mainstream
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventCreate(&memcpyEvent1));
  HIP_CHECK(hipEventCreate(&memsetEvent2));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));

  HIP_CHECK(hipStreamBeginCapture(mstream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(forkStreamEvent, mstream));
  HIP_CHECK(hipStreamWaitEvent(stream1, forkStreamEvent, 0));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
  HIP_CHECK(hipMemsetAsync(A_d, 0, Nbytes, stream1));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipEventRecord(memcpyEvent1, stream1));
  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream2));
  HIP_CHECK(hipEventRecord(memsetEvent2, stream2));
  HIP_CHECK(hipStreamWaitEvent(mstream, memcpyEvent1, 0));
  HIP_CHECK(hipStreamWaitEvent(mstream, memsetEvent2, 0));

  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};

  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                              dim3(threadsPerBlock), 0, mstream, A_d, C_d, N);

  // Get dependencies now
  HIP_CHECK(hipStreamGetCaptureInfo_v2(mstream, &captureStatus,
                nullptr, &capInfoGraph, &nodelist, &numDependencies));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(numDependencies == 1);

  // Add node to modify vector sqr result and plug-in the node
  int incValue = 1;
  size_t NElem{N};
  hipGraphNode_t updateNode{};
  hipKernelNodeParams kernelNodeParams{};

  void* kernelArgs[] = {&C_d, &incValue, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func =
                       reinterpret_cast<void *>(updateResult);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&updateNode, capInfoGraph, &nodelist[0],
                                        numDependencies, &kernelNodeParams));

  // Replace capture dependency with new kernel node created.
  // Further nodes captured in stream will depend on the new kernel node.
  HIP_CHECK(hipStreamUpdateCaptureDependencies(mstream, &updateNode, 1,
                                             hipStreamSetCaptureDependencies));

  HIP_CHECK(hipStreamGetCaptureInfo_v2(mstream, &captureStatus,
                  nullptr, &capInfoGraph, &nodelist, &numDependencies));

  // Verify updating dependency is taking effect.
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(numDependencies == 1);
  REQUIRE(nodelist[0] == updateNode);

  // End capture and verify graph is returned
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, mstream));
  HIP_CHECK(hipStreamEndCapture(mstream, &graph));
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Replay the recorded sequence multiple times
  for (int i = 0; i < LAUNCH_ITERS; i++) {
    HIP_CHECK(hipGraphLaunch(graphExec, streamForLaunch));
  }

  HIP_CHECK(hipStreamSynchronize(streamForLaunch));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if ((C_h[i] - 1) != A_h[i] * A_h[i]) {
      INFO("A and C not matching at " << i << " C_h[i] " << C_h[i]
                                           << " A_h[i] " << A_h[i]);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForLaunch));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(forkStreamEvent));
  HIP_CHECK(hipEventDestroy(memcpyEvent1));
  HIP_CHECK(hipEventDestroy(memsetEvent2));
  HipTest::freeArrays<double>(A_d, nullptr, C_d, A_h, nullptr, C_h, false);
}


static void UpdateStreamCaptureDependenciesAdd(hipStream_t mstream) {
  hipStream_t stream1{nullptr}, stream2{nullptr}, streamForLaunch{nullptr};
  hipEvent_t memcpyEvent1, memcpyEvent2, forkStreamEvent;
  hipGraph_t graph{nullptr}, capInfoGraph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  const hipGraphNode_t *nodelist{};
  size_t Nbytes = N * sizeof(double), numDependencies;
  double *A_d, *B_d, *C_d, *Res_d;
  double *A_h, *B_h, *C_h, *Res_h;

  HipTest::initArrays<double>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);
  HIP_CHECK(hipMalloc(&Res_d, Nbytes));
  Res_h = reinterpret_cast<double*>(malloc(Nbytes));
  REQUIRE(Res_h != nullptr);

  HIP_CHECK(hipMemset(C_d, 0, Nbytes));
  HIP_CHECK(hipMemset(Res_d, 0, Nbytes));
  HIP_CHECK(hipStreamCreate(&streamForLaunch));

  // Initialize input buffer
  for (size_t i = 0; i < N; ++i) {
    A_h[i] = i;
    B_h[i] = i + 1.1;
    C_h[i] = i + 2.2;
  }

  // Create cross stream dependencies.
  // memset/memcpy operations are done on stream1 and stream2
  // and they are joined back to mainstream
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipEventCreate(&memcpyEvent1));
  HIP_CHECK(hipEventCreate(&memcpyEvent2));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));

  HIP_CHECK(hipStreamBeginCapture(mstream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(forkStreamEvent, mstream));
  HIP_CHECK(hipStreamWaitEvent(stream1, forkStreamEvent, 0));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
  HIP_CHECK(hipMemsetAsync(A_d, 0, Nbytes, stream1));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipEventRecord(memcpyEvent1, stream1));
  HIP_CHECK(hipMemsetAsync(B_d, 0, Nbytes, stream2));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream2));
  HIP_CHECK(hipEventRecord(memcpyEvent2, stream2));
  HIP_CHECK(hipStreamWaitEvent(mstream, memcpyEvent1, 0));
  HIP_CHECK(hipStreamWaitEvent(mstream, memcpyEvent2, 0));

  hipStreamCaptureStatus captureStatus{hipStreamCaptureStatusNone};
  constexpr int numDepsCreated = 2;      // Num of dependencies created

  HIP_CHECK(hipStreamGetCaptureInfo_v2(mstream, &captureStatus,
                nullptr, &capInfoGraph, &nodelist, &numDependencies));
  REQUIRE(captureStatus == hipStreamCaptureStatusActive);
  REQUIRE(capInfoGraph != nullptr);
  REQUIRE(numDependencies == numDepsCreated);

  // Create memcpy node and add it as additional dependency in graph
  hipMemcpy3DParms myparams{};
  hipGraphNode_t memcpyNodeC{};

  memset(&myparams, 0x0, sizeof(hipMemcpy3DParms));
  myparams.srcPos = make_hipPos(0, 0, 0);
  myparams.dstPos = make_hipPos(0, 0, 0);
  myparams.extent = make_hipExtent(Nbytes, 1, 1);
  myparams.srcPtr = make_hipPitchedPtr(C_h, Nbytes,
                                      N, 1);
  myparams.dstPtr = make_hipPitchedPtr(C_d, Nbytes,
                                      N, 1);
  myparams.kind = hipMemcpyHostToDevice;

  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNodeC, capInfoGraph, nullptr,
                                                              0, &myparams));

  // Add/Append additional dependency MemcpyNodeC to the existing set.
  // Further nodes captured in stream will depend on Memcpy nodes A, B and C.
  HIP_CHECK(hipStreamUpdateCaptureDependencies(mstream, &memcpyNodeC,
                                         1, hipStreamAddCaptureDependencies));
  HIP_CHECK(hipStreamGetCaptureInfo_v2(mstream, &captureStatus,
                nullptr, &capInfoGraph, &nodelist, &numDependencies));

  REQUIRE(numDependencies == numDepsCreated + 1);

  hipLaunchKernelGGL(vectorSum, dim3(blocks), dim3(threadsPerBlock), 0,
                                           mstream, A_d, B_d, C_d, Res_d, N);

  HIP_CHECK(hipMemcpyAsync(Res_h, Res_d, Nbytes, hipMemcpyDeviceToHost,
                                                                   mstream));

  HIP_CHECK(hipStreamEndCapture(mstream, &graph));
  REQUIRE(graph != nullptr);
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Replay the recorded sequence multiple times
  for (int i = 0; i < LAUNCH_ITERS; i++) {
    HIP_CHECK(hipGraphLaunch(graphExec, streamForLaunch));
  }

  HIP_CHECK(hipStreamSynchronize(streamForLaunch));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if (Res_h[i] != A_h[i] + B_h[i] + C_h[i]) {
      INFO("Sum not matching at " << i << " Res_h[i] " << Res_h[i]
                                       << " A_h[i] " << A_h[i]
                                       << " B_h[i] " << B_h[i]
                                       << " C_h[i] " << C_h[i]);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForLaunch));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(forkStreamEvent));
  HIP_CHECK(hipEventDestroy(memcpyEvent1));
  HIP_CHECK(hipEventDestroy(memcpyEvent2));
  HipTest::freeArrays<double>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipFree(Res_d));
  free(Res_h);
}

/**
 * Basic Functional Test for stream capture and updating deps on the fly.
 * Regular/custom stream is used for stream capture.
 */
TEST_CASE("Unit_hipStreamUpdateCaptureDependencies_BasicFunctional") {
  hipStream_t streamForCapture;
  HIP_CHECK(hipStreamCreate(&streamForCapture));

  SECTION("hipStreamAddCaptureDependencies flag with custom stream") {
    UpdateStreamCaptureDependenciesAdd(streamForCapture);
  }

  SECTION("hipStreamSetCaptureDependencies flag with custom stream") {
    UpdateStreamCaptureDependenciesSet(streamForCapture);
  }

  HIP_CHECK(hipStreamDestroy(streamForCapture));
}

/**
 * Test performs stream capture on hipStreamPerThread and updates
 * dependencies on the fly, verifies result.
 */
TEST_CASE("Unit_hipStreamUpdateCaptureDependencies_hipStreamPerThread") {
  SECTION("hipStreamAddCaptureDependencies flag with custom stream") {
    UpdateStreamCaptureDependenciesAdd(hipStreamPerThread);
  }

  SECTION("hipStreamSetCaptureDependencies flag with custom stream") {
    UpdateStreamCaptureDependenciesSet(hipStreamPerThread);
  }
}

/**
 * Test performs api parameter validation by passing various values
 * as input and output parameters and validates the behavior.
 * Test will include both negative and positive scenarios.
 */
TEST_CASE("Unit_hipStreamUpdateCaptureDependencies_ParamValidation") {
  hipGraph_t graph{}, capInfoGraph{nullptr};
  hipStreamCaptureStatus captureStatus;
  size_t numDependencies;
  const hipGraphNode_t *nodelist{};
  const int numBytes = 100;
  hipGraphNode_t memsetNode{};
  hipError_t ret;
  hipStream_t stream;
  char *A_d;
  std::vector<hipGraphNode_t> dependencies;

  HIP_CHECK(hipMalloc(&A_d, numBytes));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemsetAsync(A_d, 0, numBytes, stream));

  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &captureStatus,
                nullptr, &capInfoGraph, &nodelist, &numDependencies));

  hipMemsetParams memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 1;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numBytes * sizeof(char);
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, capInfoGraph, nodelist,
                                         numDependencies, &memsetParams));
  dependencies.push_back(memsetNode);

  SECTION("Dependencies as nullptr and numDeps as 0") {
    ret = hipStreamUpdateCaptureDependencies(stream, nullptr,
                         0, hipStreamAddCaptureDependencies);
    REQUIRE(ret == hipSuccess);
  }

  SECTION("Dependencies as nullptr and numDeps as nonzero") {
    ret = hipStreamUpdateCaptureDependencies(stream, nullptr,
                         dependencies.size(), hipStreamAddCaptureDependencies);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("numDeps exceeding actual number of nodes") {
    ret = hipStreamUpdateCaptureDependencies(stream, dependencies.data(),
                     dependencies.size() + 1, hipStreamAddCaptureDependencies);
    REQUIRE(ret == hipErrorInvalidValue);
  }

#if HT_NVIDIA
  // Tests not supported for amd
  SECTION("Invalid flag") {
    constexpr int invalidFlag = 20;
    ret = hipStreamUpdateCaptureDependencies(stream, dependencies.data(),
                         dependencies.size(), invalidFlag);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("depnode as un-initialized/invalid parameter") {
    hipGraphNode_t uninit_node{};
    ret = hipStreamUpdateCaptureDependencies(stream, &uninit_node,
                       1, hipStreamAddCaptureDependencies);
    REQUIRE(ret == hipErrorInvalidValue);
  }
#endif

  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(A_d));
}
