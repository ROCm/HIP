// RUN: hipify "%s" 2>&1 | FileCheck %s

#include <helper_cuda.h> // for checkCudaErrors

#include <iostream>

__global__ void axpy(float a, float* x, float* y) {
  // CHECK: hipThreadIdx_x
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
  checkCudaErrors(cudaMalloc(&device_x, kDataLen * sizeof(float)));
  checkCudaErrors(cudaMalloc(&device_y, kDataLen * sizeof(float)));
  checkCudaErrors(cudaMemcpy(device_x, host_x, kDataLen * sizeof(float),
                             cudaMemcpyHostToDevice));

  // Launch the kernel.
  // CHECK: hipLaunchKernel(HIP_KERNEL_NAME
  axpy<<<1, kDataLen>>>(a, device_x, device_y);

  // Copy output data to host.
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(host_y, device_y, kDataLen * sizeof(float),
                             cudaMemcpyDeviceToHost));

  // Print the results.
  for (int i = 0; i < kDataLen; ++i) {
    std::cout << "y[" << i << "] = " << host_y[i] << "\n";
  }

  checkCudaErrors(cudaDeviceReset());
  return 0;
}
