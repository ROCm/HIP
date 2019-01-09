// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// Kernel definition
__global__ void  vecAdd(float* A, float* B, float* C)
{
  int i = threadIdx.x;
  A[i] = 0;
  B[i] = i;
  C[i] = A[i] + B[i];
}
// CHECK: #include <hip/hip_runtime.h>
#include <stdio.h>
#define SIZE 10
#define KERNELINVOKES  5000000
int vecadd(int  gpudevice, int rank)
{
  int devcheck(int, int);
  devcheck(gpudevice, rank);
  float A[SIZE], B[SIZE], C[SIZE];
  // Kernel invocation
  float *devPtrA;
  float *devPtrB;
  float *devPtrC;
  int memsize = SIZE * sizeof(float);
  // CHECK: hipMalloc((void**)&devPtrA, memsize);
  // CHECK: hipMalloc((void**)&devPtrB, memsize);
  // CHECK: hipMalloc((void**)&devPtrC, memsize);
  cudaMalloc((void**)&devPtrA, memsize);
  cudaMalloc((void**)&devPtrB, memsize);
  cudaMalloc((void**)&devPtrC, memsize);
  // CHECK: hipMemcpy(devPtrA, A, memsize, hipMemcpyHostToDevice);
  // CHECK: hipMemcpy(devPtrB, B, memsize, hipMemcpyHostToDevice);
  cudaMemcpy(devPtrA, A, memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(devPtrB, B, memsize, cudaMemcpyHostToDevice);
  for (int i = 0; i<KERNELINVOKES; i++)
  {
    // CHECK: hipLaunchKernelGGL(vecAdd, dim3(1), dim3(gpudevice), 0, 0, devPtrA, devPtrB, devPtrC);
    vecAdd <<< 1, gpudevice >>>(devPtrA, devPtrB, devPtrC);
  }
  // CHECK: hipMemcpy(C, devPtrC, memsize, hipMemcpyDeviceToHost);
  cudaMemcpy(C, devPtrC, memsize, cudaMemcpyDeviceToHost);
  // calculate only up to gpudevice to show the unique output
  // of each rank's kernel launch
  for (int i = 0; i<gpudevice; i++)
    printf("rank %d: C[%d]=%f\n", rank, i, C[i]);
  // CHECK: hipFree(devPtrA);
  // CHECK: hipFree(devPtrA);
  // CHECK: hipFree(devPtrA);
  cudaFree(devPtrA);
  cudaFree(devPtrA);
  cudaFree(devPtrA);
}
int devcheck(int  gpudevice, int rank)
{
  int device_count = 0;
  int device;   // used with cudaGetDevice() to verify cudaSetDevice()
  // CHECK: hipGetDeviceCount(&device_count);
  cudaGetDeviceCount(&device_count);
  if (gpudevice >= device_count)
  {
    printf("gpudevice >=  device_count ... exiting\n");
    exit(1);
  }
  // CHECK: hipError_t cudareturn;
  // CHECK: hipDeviceProp_t deviceProp;
  // CHECK: hipGetDeviceProperties(&deviceProp, gpudevice);
  cudaError_t cudareturn;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpudevice);
  // CHECK: if (deviceProp.hipWarpSize <= 1)
  if (deviceProp.warpSize <= 1)
  {
    printf("rank %d: warning, CUDA Device Emulation (CPU) detected, exiting\n", rank);
    exit(1);
  }
  // CHECK: cudareturn = hipSetDevice(gpudevice);
  cudareturn = cudaSetDevice(gpudevice);
  // CHECK: if (cudareturn == hipErrorInvalidDevice)
  if (cudareturn == cudaErrorInvalidDevice)
  {
    // CHECK: perror("hipSetDevice returned hipErrorInvalidDevice");
    perror("cudaSetDevice returned cudaErrorInvalidDevice");
  }
  else
  {
    // CHECK: hipGetDevice(&device);
    cudaGetDevice(&device);
    printf("rank %d: cudaGetDevice()=%d\n", rank, device);
  }
}
