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
Testcase Scenarios :
 1) Execution Without HIPGraphs : Regular procedure of using stream with async api calls.
 2) Manual HIPGraph : Manual procedure of adding nodes to graphs and mapping dependencies.
 3) HIPGraphs Using StreamCapture : Capturing sequence of operations in stream and launching
 graph with the nodes automatically added.
*/

#include <hip_test_common.hh>

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 1000

static __global__ void reduce(float* d_in, double* d_out) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      d_in[myId] += d_in[myId + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    d_out[blockIdx.x] = d_in[myId];
  }
}

static __global__ void reduceFinal(double* d_in, double* d_out) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      d_in[myId] += d_in[myId + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *d_out = d_in[myId];
  }
}

static void init_input(float* a, size_t size) {
  unsigned int seed = time(nullptr);
  for (size_t i = 0; i < size; i++) {
    a[i] = (HipTest::RAND_R(&seed) & 0xFF) / static_cast<float>(RAND_MAX);
  }
}

/**
 * Regular procedure of using stream with async api calls
 */
static void hipWithoutGraphs(float* inputVec_h, float* inputVec_d,
  double* outputVec_d, double* result_d, size_t inputSize, size_t numOfBlocks) {
  hipStream_t stream1, stream2, stream3;
  hipEvent_t forkStreamEvent, memsetEvent1, memsetEvent2;
  double result_h = 0.0;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));
  HIP_CHECK(hipEventCreate(&memsetEvent1));
  HIP_CHECK(hipEventCreate(&memsetEvent2));
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    HIP_CHECK(hipMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize,
                             hipMemcpyDefault, stream1));
    HIP_CHECK(hipMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks,
                                               stream2));
    HIP_CHECK(hipEventRecord(memsetEvent1, stream2));
    HIP_CHECK(hipMemsetAsync(result_d, 0, sizeof(double), stream3));
    HIP_CHECK(hipEventRecord(memsetEvent2, stream3));
    HIP_CHECK(hipStreamWaitEvent(stream1, memsetEvent1, 0));
    hipLaunchKernelGGL(reduce, dim3(inputSize / THREADS_PER_BLOCK, 1, 1),
                       dim3(THREADS_PER_BLOCK, 1, 1), 0, stream1, inputVec_d,
                       outputVec_d);
    HIP_CHECK(hipStreamWaitEvent(stream1, memsetEvent2, 0));
    hipLaunchKernelGGL(reduceFinal, dim3(1, 1, 1),
                                    dim3(THREADS_PER_BLOCK, 1, 1), 0, stream1,
                                    outputVec_d, result_d);
    HIP_CHECK(hipMemcpyAsync(&result_h, result_d, sizeof(double),
                                                  hipMemcpyDefault, stream1));
    HIP_CHECK(hipStreamSynchronize(stream1));
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto result = std::chrono::duration<double, std::milli>(stop - start);
  INFO("Time taken for hipWithoutGraphs : "
      << std::chrono::duration_cast<std::chrono::milliseconds>(result).count()
      << " millisecs ");
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream3));
  double result_h_cpu = 0.0;
  for (size_t i = 0; i < inputSize; i++) {
    result_h_cpu += inputVec_h[i];
  }

  REQUIRE(result_h_cpu == result_h);
}

/**
 * Capturing sequence of operations in stream and launching graph
 * with the nodes automatically added.
 */
static void hipGraphsUsingStreamCapture(float* inputVec_h, float* inputVec_d,
                                 double* outputVec_d, double* result_d,
                                 size_t inputSize, size_t numOfBlocks) {
  hipStream_t stream1, stream2, stream3, streamForGraph;
  hipEvent_t forkStreamEvent, memsetEvent1, memsetEvent2;
  hipGraph_t graph;
  double result_h = 0.0;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));
  HIP_CHECK(hipEventCreate(&memsetEvent1));
  HIP_CHECK(hipEventCreate(&memsetEvent2));
  auto start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(forkStreamEvent, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
  HIP_CHECK(hipStreamWaitEvent(stream3, forkStreamEvent, 0));
  HIP_CHECK(hipMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize,
                                                 hipMemcpyDefault, stream1));
  HIP_CHECK(hipMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks,
                                                                   stream2));
  HIP_CHECK(hipEventRecord(memsetEvent1, stream2));
  HIP_CHECK(hipMemsetAsync(result_d, 0, sizeof(double), stream3));
  HIP_CHECK(hipEventRecord(memsetEvent2, stream3));
  HIP_CHECK(hipStreamWaitEvent(stream1, memsetEvent1, 0));
  hipLaunchKernelGGL(reduce, dim3(inputSize / THREADS_PER_BLOCK, 1, 1),
                     dim3(THREADS_PER_BLOCK, 1, 1), 0, stream1,
                     inputVec_d, outputVec_d);
  HIP_CHECK(hipStreamWaitEvent(stream1, memsetEvent2, 0));
  hipLaunchKernelGGL(reduceFinal, dim3(1, 1, 1), dim3(THREADS_PER_BLOCK, 1, 1),
                     0, stream1, outputVec_d, result_d);
  HIP_CHECK(hipMemcpyAsync(&result_h, result_d, sizeof(double),
                                                   hipMemcpyDefault, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));
  hipGraphNode_t* nodes{nullptr};
  size_t numNodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nodes, &numNodes));
  INFO("Num of nodes in the graph created using stream capture API"
                                                                  << numNodes);
  HIP_CHECK(hipGraphGetRootNodes(graph, nodes, &numNodes));
  INFO("Num of root nodes in the graph created using"
                                            " stream capture API" << numNodes);
  hipGraphExec_t graphExec;

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  auto start1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  }
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  auto stop = std::chrono::high_resolution_clock::now();
  auto withInit =
                      std::chrono::duration<double, std::milli>(stop - start);
  auto withoutInit =
                      std::chrono::duration<double, std::milli>(stop - start1);
  INFO("Time taken for hipGraphsUsingStreamCapture with Init: "
  << std::chrono::duration_cast<std::chrono::milliseconds>(withInit).count()
  << " milliseconds without Init:"
  << std::chrono::duration_cast<std::chrono::milliseconds>(withoutInit).count()
  << " milliseconds ");

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  double result_h_cpu = 0.0;
  for (size_t i = 0; i < inputSize; i++) {
    result_h_cpu += inputVec_h[i];
  }

  REQUIRE(result_h_cpu == result_h);
}

/**
 * Manual procedure of adding nodes to graphs and mapping dependencies.
 */
static void hipGraphsManual(float* inputVec_h, float* inputVec_d,
         double* outputVec_d, double* result_d, size_t inputSize,
                                              size_t numOfBlocks) {
  hipStream_t streamForGraph;
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  hipGraphNode_t memcpyNode, kernelNode, memsetNode;
  double result_h = 0.0;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  auto start = std::chrono::high_resolution_clock::now();
  hipKernelNodeParams kernelNodeParams{};
  hipMemsetParams memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(outputVec_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(float);
  memsetParams.width = numOfBlocks * 2;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, inputVec_d,
                inputVec_h, sizeof(float) * inputSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr,
                                                            0, &memsetParams));
  nodeDependencies.push_back(memsetNode);
  nodeDependencies.push_back(memcpyNode);
  void* kernelArgs[4] = {reinterpret_cast<void*>(&inputVec_d),
                         reinterpret_cast<void*>(&outputVec_d), &inputSize,
                         &numOfBlocks};
  kernelNodeParams.func = reinterpret_cast<void*>(reduce);
  kernelNodeParams.gridDim = dim3(inputSize / THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                                 nodeDependencies.size(), &kernelNodeParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = result_d;
  memsetParams.value = 0;
  memsetParams.elementSize = sizeof(float);
  memsetParams.width = 2;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                                               &memsetParams));
  nodeDependencies.push_back(memsetNode);
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func = reinterpret_cast<void*>(reduceFinal);
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  void* kernelArgs2[3] = {reinterpret_cast<void*>(&outputVec_d),
                          reinterpret_cast<void*>(&result_d), &numOfBlocks};
  kernelNodeParams.kernelParams = kernelArgs2;
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                                 nodeDependencies.size(), &kernelNodeParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph,
                nodeDependencies.data(), nodeDependencies.size(), &result_h,
                           result_d, sizeof(double), hipMemcpyDeviceToHost));
  nodeDependencies.clear();
  nodeDependencies.push_back(memcpyNode);
  hipGraphExec_t graphExec;
  hipGraphNode_t* nodes{nullptr};
  size_t numNodes{};
  HIP_CHECK(hipGraphGetNodes(graph, nodes, &numNodes));
  INFO("Num of nodes in the graph created using hipGraphs Manual"
                                                                  << numNodes);
  HIP_CHECK(hipGraphGetRootNodes(graph, nodes, &numNodes));
  INFO("Num of root nodes in the graph created using"
                                              " hipGraphs Manual" << numNodes);
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  auto start1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  }
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  auto stop = std::chrono::high_resolution_clock::now();
  auto withInit =
    std::chrono::duration<double, std::milli>(stop - start);
  auto withoutInit =
    std::chrono::duration<double, std::milli>(stop - start1);

  INFO("Time taken for hipGraphsManual with Init: "
  << std::chrono::duration_cast<std::chrono::milliseconds>(withInit).count()
  << " milliseconds without Init:"
  << std::chrono::duration_cast<std::chrono::milliseconds>(withoutInit).count()
  << " milliseconds ");

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  double result_h_cpu = 0.0;
  for (size_t i = 0; i < inputSize; i++) {
    result_h_cpu += inputVec_h[i];
  }

  REQUIRE(result_h_cpu == result_h);
}

/**
 * Tests basic functionality of hipGraph APIs by
 * Execution Without HIPGraphs, Manual HIPGraph, HIPGraphs Using StreamCapture.
 */
TEST_CASE("Unit_hipGraph_BasicFunctional") {
  constexpr size_t size = 1 << 12;
  constexpr size_t maxBlocks = 512;
  float *inputVec_d{nullptr}, *inputVec_h{nullptr};
  double *outputVec_d{nullptr}, *result_d{nullptr};

  INFO("Elements : " << size << " ThreadsPerBlock : " << THREADS_PER_BLOCK);
  INFO("Graph Launch iterations = " << GRAPH_LAUNCH_ITERATIONS);

  hipSetDevice(0);
  inputVec_h = reinterpret_cast<float*>(malloc(sizeof(float) * size));
  REQUIRE(inputVec_h != nullptr);
  HIP_CHECK(hipMalloc(&inputVec_d, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&outputVec_d, sizeof(double) * maxBlocks));
  HIP_CHECK(hipMalloc(&result_d, sizeof(double)));
  init_input(inputVec_h, size);

  SECTION("Execution Without HIPGraphs") {
    hipWithoutGraphs(inputVec_h, inputVec_d, outputVec_d,
                                                    result_d, size, maxBlocks);
  }

  SECTION("Manual HIPGraph") {
    hipGraphsManual(inputVec_h, inputVec_d, outputVec_d,
                                                    result_d, size, maxBlocks);
  }

  SECTION("HIPGraphs Using StreamCapture") {
    hipGraphsUsingStreamCapture(inputVec_h, inputVec_d,
                                       outputVec_d, result_d, size, maxBlocks);
  }

  HIP_CHECK(hipFree(inputVec_d));
  HIP_CHECK(hipFree(outputVec_d));
  HIP_CHECK(hipFree(result_d));
  free(inputVec_h);
}
