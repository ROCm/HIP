// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
// Synthetic test to warn only on device functions umin and umax as unsupported, but not on user defined ones.
// ToDo: change lit testing in order to parse the output.

#define LEN 1024
#define SIZE LEN * sizeof(float)
#define ITER 1024*1024

// CHECK: #include <hip/hip_runtime.h>
#include <algorithm>

#define CUDA_LAUNCH(cuda_call,dimGrid,dimBlock, ...) \
    cuda_call<<<dimGrid,dimBlock>>>(__VA_ARGS__);

__global__ void Inc1(float *Ad, float *Bd) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tx < 1) {
    for (int i = 0; i < ITER; ++i) {
      Ad[tx] = Ad[tx] + 1.0f;
      for (int j = 0; j < 256; ++j) {
        Bd[tx] = Ad[tx];
      }
    }
  }
}

int main() {
  float *A, *Ad, *Bd;
  A = new float[LEN];
  for (int i = 0; i < LEN; ++i) {
    A[i] = 0.0f;
  }
  // CHECK: hipError_t status;
  cudaError_t status;
  // CHECK: status = hipHostRegister(A, SIZE, hipHostRegisterMapped);
  status = cudaHostRegister(A, SIZE, cudaHostRegisterMapped);
  // CHECK: hipHostGetDevicePointer(&Ad, A, 0);
  cudaHostGetDevicePointer(&Ad, A, 0);
  // CHECK: hipMalloc((void**)&Bd, SIZE);
  cudaMalloc((void**)&Bd, SIZE);
  dim3 dimGrid(LEN / 512, 1, 1);
  dim3 dimBlock(512, 1, 1);

  // CHECK: hipLaunchKernelGGL(Inc1, dim3(dimGrid), dim3(dimBlock), 0, 0, Ad, Bd);
  CUDA_LAUNCH(Inc1, dimGrid, dimBlock, Ad, Bd);
}
