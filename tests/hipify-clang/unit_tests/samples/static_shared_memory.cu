// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// Taken from Jonathan Hui blog https://jhui.github.io/2017/03/06/CUDA

#include <stdio.h>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>

__global__ void staticReverse(int *d, int n)
{
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  // Will not conttinue until all threads completed.
  __syncthreads();
  d[t] = s[tr];
}

int main(void)
{
  const int n = 64;
  int a[n], r[n], d[n];

  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  // CHECK: hipMalloc(&d_d, n * sizeof(int));
  cudaMalloc(&d_d, n * sizeof(int));
  // run version with static shared memory
  // CHECK: hipMemcpy(d_d, a, n*sizeof(int), hipMemcpyHostToDevice);
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  // CHECK: hipLaunchKernelGGL(staticReverse, dim3(1), dim3(n), 0, 0, d_d, n);
  staticReverse<<<1,n>>>(d_d, n);
  // CHECK: hipMemcpy(d, d_d, n*sizeof(int), hipMemcpyDeviceToHost);
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
}
