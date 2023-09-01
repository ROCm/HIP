#include <stdio.h>
#include <test_common.h>
#include <unistd.h>

#include <chrono>
#include <iostream>

#include "hip/hip_runtime.h"
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */
#define N 1024 * 1024
#define NSTEP 1000
#define NKERNEL 25
#define CONSTANT 5.34

__global__ void simpleKernel(float* out_d, float* in_d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) out_d[idx] = CONSTANT * in_d[idx];
}

bool hipTestWithGraph(int nstep, int nkernel) {
  int deviceId;
  HIPCHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, deviceId));

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  float *in_h, *out_h;
  in_h = new float[N];
  out_h = new float[N];
  for (int i = 0; i < N; i++) {
    in_h[i] = i;
  }

  float *in_d, *out_d;
  HIPCHECK(hipMalloc(&in_d, N * sizeof(float)));
  HIPCHECK(hipMalloc(&out_d, N * sizeof(float)));
  HIPCHECK(hipMemcpy(in_d, in_h, N * sizeof(float), hipMemcpyHostToDevice));

  auto start = std::chrono::high_resolution_clock::now();
  // start CPU wallclock timer
  bool graphCreated = false;
  hipGraph_t graph;
  hipGraphExec_t instance;

  hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
  for (int ikrnl = 0; ikrnl < nkernel; ikrnl++) {
    simpleKernel<<<dim3(N / 512, 1, 1), dim3(512, 1, 1), 0, stream>>>(out_d, in_d);
  }
  hipStreamEndCapture(stream, &graph);
  hipGraphInstantiate(&instance, graph, NULL, NULL, 0);

  auto start1 = std::chrono::high_resolution_clock::now();
  for (int istep = 0; istep < nstep; istep++) {
    hipGraphLaunch(instance, stream);
  }
  hipStreamSynchronize(stream);

  auto stop = std::chrono::high_resolution_clock::now();
  auto resultWithInit = std::chrono::duration<double, std::milli>(stop - start);
  auto resultWithoutInit = std::chrono::duration<double, std::milli>(stop - start1);
  std::cout << "Time taken for graph with Init: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(resultWithInit).count()
            << " milliseconds without Init:"
            << std::chrono::duration_cast<std::chrono::milliseconds>(resultWithoutInit).count()
            << " milliseconds " << std::endl;

  HIPCHECK(hipMemcpy(out_h, out_d, N * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    if (float(in_h[i] * CONSTANT) != out_h[i]) {
      return false;
    }
  }
  delete[] in_h;
  delete[] out_h;
  HIPCHECK(hipFree(in_d));
  HIPCHECK(hipFree(out_d));
  return true;
}

bool hipTestWithoutGraph(int nstep, int nkernel) {
  int deviceId;
  HIPCHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, deviceId));
  printf("info: running on device #%d %s with graph size & launches:%d %d \n", deviceId, props.name,
         nkernel, nstep);

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  float *in_h, *out_h;
  in_h = new float[N];
  out_h = new float[N];
  for (int i = 0; i < N; i++) {
    in_h[i] = i;
  }

  float *in_d, *out_d;
  HIPCHECK(hipMalloc(&in_d, N * sizeof(float)));
  HIPCHECK(hipMalloc(&out_d, N * sizeof(float)));
  HIPCHECK(hipMemcpy(in_d, in_h, N * sizeof(float), hipMemcpyHostToDevice));

  // start CPU wallclock timer
  auto start = std::chrono::high_resolution_clock::now();
  for (int istep = 0; istep < nstep; istep++) {
    for (int ikrnl = 0; ikrnl < nkernel; ikrnl++) {
      simpleKernel<<<dim3(N / 512, 1, 1), dim3(512, 1, 1), 0, stream>>>(out_d, in_d);
    }
  }
  HIPCHECK(hipStreamSynchronize(stream));
  auto stop = std::chrono::high_resolution_clock::now();
  auto result = std::chrono::duration<double, std::milli>(stop - start);
  std::cout << "Time taken for test without graph: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(result).count()
            << " millisecs " << std::endl;
  HIPCHECK(hipMemcpy(out_h, out_d, N * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    if (float(in_h[i] * CONSTANT) != out_h[i]) {
      return false;
    }
  }
  delete[] in_h;
  delete[] out_h;
  HIPCHECK(hipFree(in_d));
  HIPCHECK(hipFree(out_d));
  return true;
}

int main(int argc, char* argv[]) {
  bool status1, status2;
  if (argc == 3) {
    status1 = hipTestWithoutGraph(atoi(argv[1]), atoi(argv[2]));
    status2 = hipTestWithGraph(atoi(argv[1]), atoi(argv[2]));
  } else {
    status1 = hipTestWithoutGraph(NSTEP, NKERNEL);
    status2 = hipTestWithGraph(NSTEP, NKERNEL);
  }
  if (!status1) {
    failed("Failed during test without hip graph\n");
  }
  if (!status2) {
    failed("Failed during test with graph\n");
  }
  passed();
}
