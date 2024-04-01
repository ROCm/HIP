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
#define HIP_TEMPLATE_KERNEL_LAUNCH
#include <hip_test_common.hh>
#include <stdio.h>
#include <ratio>
#include <chrono>

int64_t timeNanos() {
  using namespace std::chrono;
  static auto t0 = steady_clock::now();
  auto timeSpan = duration_cast<std::chrono::nanoseconds>(steady_clock::now() - t0);
  return timeSpan.count();
}

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

__global__ void matrixTranspose(float *out, float *in, const int width) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  out[y * width + x] = in[x * width + y];
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float *output, float *input,
                                 const unsigned int width) {
  for (unsigned int j = 0; j < width; j++) {
    for (unsigned int i = 0; i < width; i++) {
      output[i * width + j] = input[j * width + i];
    }
  }
}

void thread_run(const int iThread) {
  int i = 0;
  int errors = 0;
  float eventMs = 1.0f;
  float *matrix = nullptr;
  float *transposeMatrix = nullptr;
  float *cpuTransposeMatrix = nullptr;
  float *gpuMatrix = nullptr;
  float *gpuTransposeMatrix = nullptr;
  hipDeviceProp_t devProp;
  memset(&devProp, 0, sizeof(devProp));
  HIP_CHECK(hipGetDeviceProperties(&devProp, iThread));
  fprintf(stderr, "[%d] device name = %s\n", iThread, devProp.name);

  HIP_CHECK(hipSetDevice(iThread));
  hipEvent_t start, stop;

  auto time = timeNanos();
  HIP_CHECK(hipEventCreate(&start));
  fprintf(stderr, "[%d] hipEventCreate(&start) cost cpu time %6.3fms\n", iThread,
          (timeNanos() - time)/1000000.0);

  HIP_CHECK(hipEventCreate(&stop));

  matrix = (float*) malloc(NUM * sizeof(float));
  transposeMatrix = (float*) malloc(NUM * sizeof(float));
  cpuTransposeMatrix = (float*) malloc(NUM * sizeof(float));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    matrix[i] = (float) i * 10.0f;
  }

  // allocate the memory on the device side
  HIP_CHECK(hipMalloc((void**) &gpuMatrix, NUM * sizeof(float)));
  HIP_CHECK(hipMalloc((void**) &gpuTransposeMatrix, NUM * sizeof(float)));

  time = timeNanos();
  // Record the start event
  // The first call of hipEventRecord will trigger VirtualDevice creation that will trigger building
  // of BlitLinearSourceCode, which will cost 200+ ms.
  HIP_CHECK(hipEventRecord(start));
  fprintf(stderr, "[%d] hipEventRecord(&start) cost cpu time %6.3fms\n", iThread,
          (timeNanos() - time)/1000000.0);

  time = timeNanos();
  // Memory transfer from host to device
  HIP_CHECK(hipMemcpy(gpuMatrix, matrix, NUM * sizeof(float), hipMemcpyHostToDevice));

  // Record the stop event
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  HIP_CHECK(hipEventElapsedTime(&eventMs, start, stop));

  fprintf(stderr, "[%d] hipMemcpyHostToDevice cost gpu time %6.3fms, cpu time %6.3fms\n",
          iThread, eventMs, (timeNanos() - time)/1000000.0);

  // Record the start event
  HIP_CHECK(hipEventRecord(start));

  time = timeNanos();
  // Lauching kernel from host
  hipLaunchKernelGGL(
      matrixTranspose,
      dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
      dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
      gpuMatrix, WIDTH);
  // Record the stop event
  HIP_CHECK(hipEventRecord(stop));

  fprintf(stderr, "[%d] hipLaunchKernelGGL() cost cpu time %6.3fms\n", iThread,
          (timeNanos() - time)/1000000.0);

  HIP_CHECK(hipEventSynchronize(stop));
  HIP_CHECK(hipEventElapsedTime(&eventMs, start, stop));

  fprintf(stderr, "[%d] kernel Execution cost gpu time %6.3fms, cpu time = %6.3fms\n",
          iThread, eventMs, (timeNanos() - time)/1000000.0);

  // Record the start event
  HIP_CHECK(hipEventRecord(start));

  // Memory transfer from device to host
  HIP_CHECK(hipMemcpy(transposeMatrix, gpuTransposeMatrix, NUM * sizeof(float),
            hipMemcpyDeviceToHost));

  // Record the stop event
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  HIP_CHECK(hipEventElapsedTime(&eventMs, start, stop));

  fprintf(stderr, "[%d] hipMemcpyDeviceToHost cost gpu time %6.3fms\n", iThread, eventMs);

  // CPU MatrixTranspose computation
  matrixTransposeCPUReference(cpuTransposeMatrix, matrix, WIDTH);

  // verify the results
  double eps = 1.0E-6;
  for (i = 0; i < NUM; i++) {
    if (std::abs(transposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
      errors++;
    }
  }
  if (errors != 0) {
    fprintf(stderr, "[%d] FAILED: %d errors\n", iThread, errors);
  } else {
    fprintf(stderr, "[%d] PASSED\n", iThread);
  }
  // free the resources on device side
  HIP_CHECK(hipFree(gpuMatrix));
  HIP_CHECK(hipFree(gpuTransposeMatrix));

  // free the resources on host side
  free(matrix);
  free(transposeMatrix);
  free(cpuTransposeMatrix);
  REQUIRE(errors == 0);
}

void testEventMGpuMThreads(int nThreads = 1) {
  int iThread = 0;
  std::thread *threads = new std::thread[nThreads];
  for (iThread = 0; iThread < nThreads; iThread++) {
    threads[iThread] = std::thread(thread_run, iThread);
  }
  for (iThread = 0; iThread < nThreads; iThread++) {
    threads[iThread].join();
  }
  delete []threads;
}

TEST_CASE("Unit_hipEventMGpuMThreads_1") {
  testEventMGpuMThreads(1);
}

TEST_CASE("Unit_hipEventMGpuMThreads_2") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    testEventMGpuMThreads(numDevices);
  } else {
    SUCCEED("skipped the testcase as number of devices is less than 2");
  }
}

TEST_CASE("Unit_hipEventMGpuMThreads_3") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    fprintf(stderr, "First round\n");
    testEventMGpuMThreads(numDevices);
    fprintf(stderr, "Second round\n");
    testEventMGpuMThreads(numDevices);
  } else {
    SUCCEED("skipped the testcase as number of devices is less than 2");
  }
}
