/* Copyright (c) 2021-present Advanced Micro Devices, Inc.

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
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t EXCLUDE_HIP_PLATFORM all
 * HIT_END

 */
#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3

__global__ void reduce(float* d_in, double* d_out, size_t inputSize, size_t outputSize) {
  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // do reduction in global mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      d_in[myId] += d_in[myId + s];
    }
    __syncthreads();  // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0) {
    int blkx = blockIdx.x;
    d_out[blockIdx.x] = d_in[myId];
  }
}
__global__ void reduceFinal(double* d_in, double* d_out, size_t inputSize) {
  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // do reduction in global mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      d_in[myId] += d_in[myId + s];
    }
    __syncthreads();  // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0) {
    *d_out = d_in[myId];
  }
}

void init_input(float* a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

bool hipGraphsUsingStreamCapture(float* inputVec_h, float* inputVec_d, double* outputVec_d,
                                 double* result_d, size_t inputSize, size_t numOfBlocks) {
  hipStream_t stream1, stream2, stream3, streamForGraph;
  hipEvent_t forkStreamEvent, memsetEvent1, memsetEvent2;
  hipGraph_t graph;
  double result_h = 0.0;

  HIPCHECK(hipStreamCreate(&stream1));
  HIPCHECK(hipStreamCreate(&stream2));
  HIPCHECK(hipStreamCreate(&stream3));
  HIPCHECK(hipStreamCreate(&streamForGraph));

  HIPCHECK(hipEventCreate(&forkStreamEvent));
  HIPCHECK(hipEventCreate(&memsetEvent1));
  HIPCHECK(hipEventCreate(&memsetEvent2));

  HIPCHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));

  HIPCHECK(hipEventRecord(forkStreamEvent, stream1));
  HIPCHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
  HIPCHECK(hipStreamWaitEvent(stream3, forkStreamEvent, 0));

  HIPCHECK(
      hipMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize, hipMemcpyDefault, stream1));

  HIPCHECK(hipMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream2));

  HIPCHECK(hipEventRecord(memsetEvent1, stream2));

  HIPCHECK(hipMemsetAsync(result_d, 0, sizeof(double), stream3));
  HIPCHECK(hipEventRecord(memsetEvent2, stream3));

  HIPCHECK(hipStreamWaitEvent(stream1, memsetEvent1, 0));

  hipLaunchKernelGGL(reduce, dim3(inputSize / THREADS_PER_BLOCK, 1, 1),
                     dim3(THREADS_PER_BLOCK, 1, 1), 0, stream1, inputVec_d, outputVec_d, inputSize,
                     numOfBlocks);
  HIPCHECK(hipStreamWaitEvent(stream1, memsetEvent2, 0));

  hipLaunchKernelGGL(reduceFinal, dim3(1, 1, 1), dim3(THREADS_PER_BLOCK, 1, 1), 0, stream1,
                     outputVec_d, result_d, numOfBlocks);
  HIPCHECK(hipMemcpyAsync(&result_h, result_d, sizeof(double), hipMemcpyDefault, stream1));

  HIPCHECK(hipStreamEndCapture(stream1, &graph));

  hipGraphExec_t graphExec;
  HIPCHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    HIPCHECK(hipGraphLaunch(graphExec, streamForGraph));
  }
  HIPCHECK(hipStreamSynchronize(streamForGraph));
  HIPCHECK(hipGraphExecDestroy(graphExec));
  HIPCHECK(hipGraphDestroy(graph));
  HIPCHECK(hipStreamDestroy(stream1));
  HIPCHECK(hipStreamDestroy(stream2));
  HIPCHECK(hipStreamDestroy(streamForGraph));
  double result_h_cpu = 0.0;

  for (int i = 0; i < inputSize; i++) {
    result_h_cpu += inputVec_h[i];
  }
  if (result_h_cpu != result_h) {
    printf("Final reduced sum = %lf %lf\n", result_h_cpu, result_h);
    return false;
  }
  return true;
}

bool hipGraphsManual(float* inputVec_h, float* inputVec_d, double* outputVec_d, double* result_d,
                     size_t inputSize, size_t numOfBlocks) {
  hipStream_t streamForGraph;
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  hipGraphNode_t memcpyNode, kernelNode, memsetNode;
  double result_h = 0.0;
  HIPCHECK(hipStreamCreate(&streamForGraph));
  hipKernelNodeParams kernelNodeParams = {0};
  hipMemcpy3DParms memcpyParams = {0};
  hipMemsetParams memsetParams = {0};
  memcpyParams.srcArray = NULL;
  memcpyParams.srcPos = make_hipPos(0, 0, 0);
  memcpyParams.srcPtr = make_hipPitchedPtr(inputVec_h, sizeof(float) * inputSize, inputSize, 1);
  memcpyParams.dstArray = NULL;
  memcpyParams.dstPos = make_hipPos(0, 0, 0);
  memcpyParams.dstPtr = make_hipPitchedPtr(inputVec_d, sizeof(float) * inputSize, inputSize, 1);
  memcpyParams.extent = make_hipExtent(sizeof(float) * inputSize, 1, 1);
  memcpyParams.kind = hipMemcpyHostToDevice;
  memsetParams.dst = (void*)outputVec_d;
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(float);  // elementSize can be max 4 bytes
  memsetParams.width = numOfBlocks * 2;
  memsetParams.height = 1;
  HIPCHECK(hipGraphCreate(&graph, 0));
  HIPCHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
  HIPCHECK(hipGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));
  nodeDependencies.push_back(memsetNode);
  nodeDependencies.push_back(memcpyNode);
  void* kernelArgs[4] = {(void*)&inputVec_d, (void*)&outputVec_d, &inputSize, &numOfBlocks};
  kernelNodeParams.func = (void*)reduce;
  kernelNodeParams.gridDim = dim3(inputSize / THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = (void**)kernelArgs;
  kernelNodeParams.extra = NULL;
  HIPCHECK(hipGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                                 nodeDependencies.size(), &kernelNodeParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = result_d;
  memsetParams.value = 0;
  memsetParams.elementSize = sizeof(float);
  memsetParams.width = 2;
  memsetParams.height = 1;
  HIPCHECK(hipGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));
  nodeDependencies.push_back(memsetNode);
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func = (void*)reduceFinal;
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  void* kernelArgs2[3] = {(void*)&outputVec_d, (void*)&result_d, &numOfBlocks};
  kernelNodeParams.kernelParams = kernelArgs2;
  kernelNodeParams.extra = NULL;
  HIPCHECK(hipGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                                 nodeDependencies.size(), &kernelNodeParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);
  memset(&memcpyParams, 0, sizeof(memcpyParams));
  memcpyParams.srcArray = NULL;
  memcpyParams.srcPos = make_hipPos(0, 0, 0);
  memcpyParams.srcPtr = make_hipPitchedPtr(result_d, sizeof(double), 1, 1);
  memcpyParams.dstArray = NULL;
  memcpyParams.dstPos = make_hipPos(0, 0, 0);
  memcpyParams.dstPtr = make_hipPitchedPtr(&result_h, sizeof(double), 1, 1);
  memcpyParams.extent = make_hipExtent(sizeof(double), 1, 1);
  memcpyParams.kind = hipMemcpyDeviceToHost;
  HIPCHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                 nodeDependencies.size(), &memcpyParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(memcpyNode);
  hipGraphNode_t hostNode;

  hipGraphExec_t graphExec;
  HIPCHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    HIPCHECK(hipGraphLaunch(graphExec, streamForGraph));
  }
  HIPCHECK(hipStreamSynchronize(streamForGraph));
  
  HIPCHECK(hipGraphExecDestroy(graphExec));
  HIPCHECK(hipGraphDestroy(graph));
  HIPCHECK(hipStreamDestroy(streamForGraph));

  double result_h_cpu = 0.0;

  for (int i = 0; i < inputSize; i++) {
    result_h_cpu += inputVec_h[i];
  }
  if (result_h_cpu != result_h) {
    printf("Final reduced sum = %lf %lf\n", result_h_cpu, result_h);
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  size_t size = 1 << 12;  // number of elements to reduce
  size_t maxBlocks = 512;
  // This will pick the best possible CUDA capable device
  int devID = 1;  // TODO: implement: findCudaDevice(argc, (const char**)argv); based of max GFLOPS
                  // incase of multiple devic
  hipSetDevice(0);  //
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
  bool status1 = hipGraphsManual(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);
  bool status2 = hipGraphsUsingStreamCapture(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);
  HIPCHECK(hipFree(inputVec_d));
  HIPCHECK(hipFree(outputVec_d));
  HIPCHECK(hipFree(result_d));
  if(!status1) {
    failed("Failed during hip Graph Manual\n");
  }
  if(!status2) {
    failed("Failed during hip Graphs during stream capture\n");
  }
  passed();
}
