// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#include <iostream>

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>

#define TOKEN_PASTE(X, Y) X ## Y
#define ARG_LIST_AS_MACRO a, device_x, device_y
#define KERNEL_CALL_AS_MACRO axpy<float><<<1, kDataLen>>>
#define KERNEL_NAME_MACRO axpy<float>

// CHECK: #define COMPLETE_LAUNCH hipLaunchKernelGGL(axpy, dim3(1), dim3(kDataLen), 0, 0, a, device_x, device_y)
#define COMPLETE_LAUNCH axpy<<<1, kDataLen>>>(a, device_x, device_y)


template<typename T>
__global__ void axpy(T a, T *x, T *y) {
  y[threadIdx.x] = a * x[threadIdx.x];
}


int main(int argc, char* argv[]) {
  const int kDataLen = 4;

  float a = 2.0f;
  float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
  float host_y[kDataLen];

  // Copy input data to device.
  float* device_x;
  float* device_y;

  // CHECK: hipMalloc(&device_x, kDataLen * sizeof(float));
  cudaMalloc(&device_x, kDataLen * sizeof(float));

#ifdef HERRING
  // CHECK: hipMalloc(&device_y, kDataLen * sizeof(float));
  cudaMalloc(&device_y, kDataLen * sizeof(float));
#else
  // CHECK: hipMalloc(&device_y, kDataLen * sizeof(double));
  cudaMalloc(&device_y, kDataLen * sizeof(double));
#endif

  // CHECK: hipMemcpy(device_x, host_x, kDataLen * sizeof(float), hipMemcpyHostToDevice);
  cudaMemcpy(device_x, host_x, kDataLen * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel in numerous different strange ways to exercise the prerocessor.
  // CHECK: hipLaunchKernelGGL(axpy, dim3(1), dim3(kDataLen), 0, 0, a, device_x, device_y);
  axpy<<<1, kDataLen>>>(a, device_x, device_y);

  // CHECK: hipLaunchKernelGGL(axpy<float>, dim3(1), dim3(kDataLen), 0, 0, a, device_x, device_y);
  axpy<float><<<1, kDataLen>>>(a, device_x, device_y);

  // CHECK: hipLaunchKernelGGL(axpy<float>, dim3(1), dim3(kDataLen), 0, 0, a, TOKEN_PASTE(device, _x), device_y);
  axpy<float><<<1, kDataLen>>>(a, TOKEN_PASTE(device, _x), device_y);

  // CHECK: hipLaunchKernelGGL(axpy<float>, dim3(1), dim3(kDataLen), 0, 0, ARG_LIST_AS_MACRO);
  axpy<float><<<1, kDataLen>>>(ARG_LIST_AS_MACRO);

  // CHECK: hipLaunchKernelGGL(KERNEL_NAME_MACRO, dim3(1), dim3(kDataLen), 0, 0, ARG_LIST_AS_MACRO);
  KERNEL_NAME_MACRO<<<1, kDataLen>>>(ARG_LIST_AS_MACRO);

  // CHECK: hipLaunchKernelGGL(axpy<float>, dim3(1), dim3(kDataLen), 0, 0, ARG_LIST_AS_MACRO);
  KERNEL_CALL_AS_MACRO(ARG_LIST_AS_MACRO);

  // CHECK: COMPLETE_LAUNCH;
  COMPLETE_LAUNCH;


  // Copy output data to host.
  // CHECK: hipDeviceSynchronize();
  cudaDeviceSynchronize();

  // CHECK: hipMemcpy(host_y, device_y, kDataLen * sizeof(float), hipMemcpyDeviceToHost);
  cudaMemcpy(host_y, device_y, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

  // Print the results.
  for (int i = 0; i < kDataLen; ++i) {
    std::cout << "y[" << i << "] = " << host_y[i] << "\n";
  }

  // CHECK: hipDeviceReset();
  cudaDeviceReset();
  return 0;
}
