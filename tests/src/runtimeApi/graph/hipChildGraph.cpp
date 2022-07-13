/* Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc.
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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>
#include <chrono>
#include <test_common.h>
#include <vector>
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */
#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 100
__global__ void reduce(float* d_in, double* d_out, size_t inputSize, size_t outputSize) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      d_in[myId] += d_in[myId + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    d_out[blockIdx.x] = d_in[myId];
  }
}
__global__ void reduceFinal(double* d_in, double* d_out, size_t inputSize) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      d_in[myId] += d_in[myId + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *d_out = d_in[myId];
  }
}
void init_input(float* a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

bool hipGraphsManual(float* inputVec_h, float* inputVec_d, double* outputVec_d, double* result_d,
                     size_t inputSize, size_t numOfBlocks) {
  hipStream_t streamForGraph;
  hipGraph_t graph, childgraph;
  std::vector<hipGraphNode_t> nodeDependencies;
  hipGraphNode_t memcpyNode, kernelNode, memsetNode1, memsetNode2, childGraphNode;
  double result_h = 0.0;
  HIPCHECK(hipStreamCreate(&streamForGraph));
  auto start = std::chrono::high_resolution_clock::now();
  hipKernelNodeParams kernelNodeParams = {0};
  hipMemsetParams memsetParams = {0};
  memsetParams.dst = (void*)outputVec_d;
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(float);
  memsetParams.width = numOfBlocks * 2;
  memsetParams.height = 1;
  HIPCHECK(hipGraphCreate(&graph, 0));
  HIPCHECK(hipGraphCreate(&childgraph, 0));
  HIPCHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, inputVec_d, inputVec_h,
                                   sizeof(float) * inputSize, hipMemcpyHostToDevice));
  HIPCHECK(hipGraphAddMemsetNode(&memsetNode1, graph, NULL, 0, &memsetParams));

  void* kernelArgs[4] = {(void*)&inputVec_d, (void*)&outputVec_d, &inputSize, &numOfBlocks};
  kernelNodeParams.func = (void*)reduce;
  kernelNodeParams.gridDim = dim3(inputSize / THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = (void**)kernelArgs;
  kernelNodeParams.extra = NULL;
  HIPCHECK(hipGraphAddKernelNode(&kernelNode, childgraph, NULL, 0, &kernelNodeParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = result_d;
  memsetParams.value = 0;
  memsetParams.elementSize = sizeof(float);
  memsetParams.width = 2;
  memsetParams.height = 1;
  HIPCHECK(hipGraphAddMemsetNode(&memsetNode2, childgraph, NULL, 0, &memsetParams));
  nodeDependencies.push_back(memsetNode2);
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func = (void*)reduceFinal;
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  void* kernelArgs2[3] = {(void*)&outputVec_d, (void*)&result_d, &numOfBlocks};
  kernelNodeParams.kernelParams = kernelArgs2;
  kernelNodeParams.extra = NULL;
  HIPCHECK(hipGraphAddKernelNode(&kernelNode, childgraph, nodeDependencies.data(),
                                 nodeDependencies.size(), &kernelNodeParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(memcpyNode);
  nodeDependencies.push_back(memsetNode1);
  HIPCHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nodeDependencies.data(),
                                     nodeDependencies.size(), childgraph));
  nodeDependencies.clear();
  nodeDependencies.push_back(childGraphNode);
  HIPCHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nodeDependencies.data(),
                                   nodeDependencies.size(), &result_h, result_d, sizeof(double),
                                   hipMemcpyDeviceToHost));

  hipGraphExec_t graphExec;
  HIPCHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  auto start1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    HIPCHECK(hipGraphLaunch(graphExec, streamForGraph));
  }
  HIPCHECK(hipStreamSynchronize(streamForGraph));
  double result_h_cpu = 0.0;
  for (int i = 0; i < inputSize; i++) {
    result_h_cpu += inputVec_h[i];
  }
  if (result_h_cpu != result_h) {
    printf("Final reduced sum = %lf %lf\n", result_h_cpu, result_h);
    return false;
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto resultWithInit = std::chrono::duration<double, std::milli>(stop - start);
  auto resultWithoutInit = std::chrono::duration<double, std::milli>(stop - start1);
  std::cout << "Time taken for hipGraphsManual with Init: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(resultWithInit).count()
            << " milliseconds without Init:"
            << std::chrono::duration_cast<std::chrono::milliseconds>(resultWithoutInit).count()
            << " milliseconds " << std::endl;

  hipGraph_t clonedGraph;
  hipGraphExec_t clonedGraphExec;
  HIPCHECK(hipGraphClone(&clonedGraph, graph));

  HIPCHECK(hipGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    HIPCHECK(hipGraphLaunch(clonedGraphExec, streamForGraph));
  }
  HIPCHECK(hipStreamSynchronize(streamForGraph));
  if (result_h_cpu != result_h) {
    printf("Cloned graph final reduced sum = %lf %lf\n", result_h_cpu, result_h);
    return false;
  }

  HIPCHECK(hipGraphExecDestroy(graphExec));
  HIPCHECK(hipGraphExecDestroy(clonedGraphExec));
  HIPCHECK(hipGraphDestroy(graph));
  HIPCHECK(hipStreamDestroy(streamForGraph));

  return true;
}

int main(int argc, char** argv) {
  size_t size = 1 << 12;
  size_t maxBlocks = 512;
  hipSetDevice(0);
  printf("%zu elements\n", size);
  printf("threads per block  = %d\n", THREADS_PER_BLOCK);
  printf("Graph Launch iterations = %d\n", GRAPH_LAUNCH_ITERATIONS);
  float *inputVec_d = NULL, *inputVec_h = NULL;
  double *outputVec_d = NULL, *result_d;
  inputVec_h = (float*)malloc(sizeof(float) * size);
  HIPCHECK(hipMalloc(&inputVec_d, sizeof(float) * size));
  HIPCHECK(hipMalloc(&outputVec_d, sizeof(double) * maxBlocks));
  HIPCHECK(hipMalloc(&result_d, sizeof(double)));
  init_input(inputVec_h, size);
  bool status = hipGraphsManual(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);
  HIPCHECK(hipFree(inputVec_d));
  HIPCHECK(hipFree(outputVec_d));
  HIPCHECK(hipFree(result_d));
  if (!status) {
    failed("Failed during hip graph manual\n");
  }
  passed();
}