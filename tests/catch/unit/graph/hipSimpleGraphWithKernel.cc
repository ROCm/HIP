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
 Simple test to demonstrate usage of graph.
 Compares implementation with and without using graphs.
*/

#include <hip_test_common.hh>

#define N 1024 * 1024
#define NSTEP 1000
#define NKERNEL 25
#define CONSTANT 5.34

static __global__ void simpleKernel(float* out_d, float* in_d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) out_d[idx] = CONSTANT * in_d[idx];
}

static void hipTestWithGraph() {
  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float *in_h, *out_h;
  in_h = new float[N];
  out_h = new float[N];
  for (int i = 0; i < N; i++) {
    in_h[i] = i;
  }

  float *in_d, *out_d;
  HIP_CHECK(hipMalloc(&in_d, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&out_d, N * sizeof(float)));
  HIP_CHECK(hipMemcpy(in_d, in_h, N * sizeof(float), hipMemcpyHostToDevice));

  auto start = std::chrono::high_resolution_clock::now();
  // start CPU wallclock timer
  hipGraph_t graph;
  hipGraphExec_t instance;

  hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
  for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
    simpleKernel<<<dim3(N / 512, 1, 1), dim3(512, 1, 1),
                                             0, stream>>>(out_d, in_d);
  }
  hipStreamEndCapture(stream, &graph);
  hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

  auto start1 = std::chrono::high_resolution_clock::now();
  for (int istep = 0; istep < NSTEP; istep++) {
    hipGraphLaunch(instance, stream);
    hipStreamSynchronize(stream);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto withInit = std::chrono::duration<double, std::milli>(stop - start);
  auto withoutInit = std::chrono::duration<double, std::milli>(stop - start1);

  INFO("Time taken for graph with Init: "
  << std::chrono::duration_cast<std::chrono::milliseconds>(withInit).count()
  << " milliseconds without Init:"
  << std::chrono::duration_cast<std::chrono::milliseconds>(withoutInit).count()
  << " milliseconds ");

  HIP_CHECK(hipMemcpy(out_h, out_d, N * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    if (static_cast<float>(in_h[i] * CONSTANT) != out_h[i]) {
      INFO("Mismatch at indx:" << i << " " << in_h[i] << " " << out_h[i]);
      REQUIRE(false);
    }
  }
  delete[] in_h;
  delete[] out_h;
  HIP_CHECK(hipFree(in_d));
  HIP_CHECK(hipFree(out_d));
}

static void hipTestWithoutGraph() {
  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  INFO("Info: running on device " << deviceId << props.name);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float *in_h, *out_h;
  in_h = new float[N];
  out_h = new float[N];
  for (int i = 0; i < N; i++) {
    in_h[i] = i;
  }

  float *in_d, *out_d;
  HIP_CHECK(hipMalloc(&in_d, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&out_d, N * sizeof(float)));
  HIP_CHECK(hipMemcpy(in_d, in_h, N * sizeof(float), hipMemcpyHostToDevice));

  // start CPU wallclock timer
  auto start = std::chrono::high_resolution_clock::now();
  for (int istep = 0; istep < NSTEP; istep++) {
    for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
      simpleKernel<<<dim3(N / 512, 1, 1), dim3(512, 1, 1),
                                                   0, stream>>>(out_d, in_d);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto result = std::chrono::duration<double, std::milli>(stop - start);
  INFO("Time taken for test without graph: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(result).count()
       << " millisecs ");
  HIP_CHECK(hipMemcpy(out_h, out_d, N * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    if (static_cast<float>(in_h[i] * CONSTANT) != out_h[i]) {
      INFO("Mismatch at indx:" << i << " " << in_h[i] << " " << out_h[i]);
      REQUIRE(false);
    }
  }
  delete[] in_h;
  delete[] out_h;
  HIP_CHECK(hipFree(in_d));
  HIP_CHECK(hipFree(out_d));
}

/**
 * Simple test to demonstrate usage of graph.
 */
TEST_CASE("Unit_hipGraph_SimpleGraphWithKernel") {
  // Sections run test with and without graph.
  SECTION("Run Test Without Graph") {
    hipTestWithoutGraph();
  }

  SECTION("Run Test With Graph") {
    hipTestWithGraph();
  }
}
