// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>

#define K_THREADS 64
#define K_INDEX() ((gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x)
#define RND() ((rand() & 0x7FFF) / float(0x8000))
#define ERRORCHECK() cErrorCheck(__FILE__, __LINE__)

// CHECK: hipEvent_t t##_start, t##_end;            \
// CHECK: hipEventCreate(&t##_start);               \
// CHECK: hipEventCreate(&t##_end);
#define TIMER_CREATE(t)                      \
  cudaEvent_t t##_start, t##_end;            \
  cudaEventCreate(&t##_start);               \
  cudaEventCreate(&t##_end);

// CHECK: hipEventRecord(t##_start);                   \
// CHECK: hipEventSynchronize(t##_start);
#define TIMER_START(t)                          \
  cudaEventRecord(t##_start);                   \
  cudaEventSynchronize(t##_start);              \

// CHECK: hipEventRecord(t##_start);                                 \
// CHECK: hipEventSynchronize(t##_start);                            \
// CHECK: hipEventRecord(t##_end);                                   \
// CHECK: hipEventSynchronize(t##_end);                              \
// CHECK: hipEventElapsedTime(&t, t##_start, t##_end);
#define TIMER_END(t)                                          \
  cudaEventRecord(t##_start);                                 \
  cudaEventSynchronize(t##_start);                            \
  cudaEventRecord(t##_end);                                   \
  cudaEventSynchronize(t##_end);                              \
  cudaEventElapsedTime(&t, t##_start, t##_end);               


inline void cErrorCheck(const char *file, int line) {
// CHECK: hipDeviceSynchronize();
// CHECK: hipError_t err = hipGetLastError();
// CHECK: if (err != hipSuccess) {
// CHECK: printf("Error: %s\n", hipGetErrorString(err));
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    printf(" @ %s: %d\n", file, line);
    exit(-1);
  }
}

inline dim3 K_GRID(int n, int threads = K_THREADS) {
  int blocks = (int)ceilf(sqrtf((float)n/threads));
  dim3 grid(blocks, blocks);
  return grid;
}

typedef struct data  {
  int n;
  float4 *r, *v, *f;
} data;

data cpu, gpu;

#define N 20

__global__ void repulsion(data gpu);
__global__ void integration(data gpu);


int main() {
  printf("Cuda Test 1\n");

  int count = 0;
  // CHECK: hipGetDeviceCount(&count);
  cudaGetDeviceCount(&count);
  printf(" %d CUDA devices found\n", count);
  if(!count) {
    ::exit(EXIT_FAILURE);
  }
  // CHECK: hipFree(0);
  cudaFree(0);

  cpu.n = N;

  cpu.r = (float4*)malloc(N * sizeof(float4));
  cpu.v = (float4*)malloc(N * sizeof(float4));
  cpu.f = (float4*)malloc(N * sizeof(float4));

  for(int i = 0; i < N; ++i) {
    cpu.v[i] = make_float4(0,0,0,0);
    cpu.r[i] = make_float4(RND(), RND(), RND(), 0);
    cpu.f[i] = make_float4(0,0.01,0,0);
  }

  gpu = cpu;
  // CHECK: hipMalloc(&gpu.r, N * sizeof(float4));
  // CHECK: hipMalloc(&gpu.v, N * sizeof(float4));
  // CHECK: hipMalloc(&gpu.f, N * sizeof(float4));
  cudaMalloc(&gpu.r, N * sizeof(float4));
  cudaMalloc(&gpu.v, N * sizeof(float4));
  cudaMalloc(&gpu.f, N * sizeof(float4));
  // CHECK: hipMemcpy(gpu.r, cpu.r, cpu.n * sizeof(float4), hipMemcpyHostToDevice);
  // CHECK: hipMemcpy(gpu.v, cpu.v, cpu.n * sizeof(float4), hipMemcpyHostToDevice);
  // CHECK: hipMemcpy(gpu.f, cpu.f, cpu.n * sizeof(float4), hipMemcpyHostToDevice);
  cudaMemcpy(gpu.r, cpu.r, cpu.n * sizeof(float4), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu.v, cpu.v, cpu.n * sizeof(float4), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu.f, cpu.f, cpu.n * sizeof(float4), cudaMemcpyHostToDevice);

  ERRORCHECK();
  float rep;
  TIMER_CREATE(rep);
  TIMER_START(rep);
  // CHECK: hipLaunchKernelGGL(integration, dim3(K_GRID(cpu.n)), dim3(K_THREADS), 0, 0, gpu);
  integration <<< K_GRID(cpu.n), K_THREADS >>>(gpu);

  TIMER_END(rep);
  printf("Took: %f ms\n", rep);
  ERRORCHECK();
  // CHECK: hipMemcpy(cpu.r, gpu.r, cpu.n * sizeof(float4), hipMemcpyDeviceToHost);
  // CHECK: hipMemcpy(cpu.v, gpu.v, cpu.n * sizeof(float4), hipMemcpyDeviceToHost);
  // CHECK: hipMemcpy(cpu.f, gpu.f, cpu.n * sizeof(float4), hipMemcpyDeviceToHost);
  cudaMemcpy(cpu.r, gpu.r, cpu.n * sizeof(float4), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu.v, gpu.v, cpu.n * sizeof(float4), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu.f, gpu.f, cpu.n * sizeof(float4), cudaMemcpyDeviceToHost);
  // CHECK: hipHostFree(cpu.r);
  // CHECK: hipHostFree(cpu.v);
  // CHECK: hipHostFree(cpu.f);
  cudaFreeHost(cpu.r);
  cudaFreeHost(cpu.v);
  cudaFreeHost(cpu.f);
  // CHECK: hipFree(gpu.r);
  // CHECK: hipFree(gpu.v);
  // CHECK: hipFree(gpu.f);
  cudaFree(gpu.r);
  cudaFree(gpu.v);
  cudaFree(gpu.f);
  // CHECK: hipDeviceReset();
  cudaDeviceReset();

  printf("Results: \n");
  for(int i = 0; i < N; ++i) {
    printf("%f, %f, %f \n", cpu.r[i].x, cpu.r[i].y, cpu.r[i].z);
  }

  printf("Ready...\n");
  return 0;
}

__global__ void repulsion(data gpu) {
  int idx = K_INDEX();
  if(idx < N) {
    gpu.r[idx].x = 1;
    gpu.r[idx].y = 1;
    gpu.r[idx].z = 1;
  }
}

#define MULT4(v, s) v.x *= s; v.y *= s; v.z *= s; v.w *= s;
#define ADD4(v1, v2) v1.x += v2.x; v1.y += v2.y; v1.z += v2.z; v1.w += v2.w;

__global__ void integration(data gpu) {
  int i = K_INDEX();
  if(i < N) {
    MULT4(gpu.f[i], 0.01);
    MULT4(gpu.v[i], 0.01);
    ADD4(gpu.v[i], gpu.f[i]);
    ADD4(gpu.r[i], gpu.v[i]);
    gpu.f[i] = make_float4(0,0,0,0);
  }
}
