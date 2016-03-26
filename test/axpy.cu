// RUN: hipify "%s" -o=%t --

#include <iostream>

__global__ void axpy(float a, float* x, float* y) {
  // RUN: sh -c "test `grep -c -F 'y[hipThreadIdx_x] = a * x[hipThreadIdx_x];' %t` -eq 2"
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
  // RUN: sh -c "test `grep -c -F 'hipMalloc(&device_x, kDataLen * sizeof(float));' %t` -eq 2"
  cudaMalloc(&device_x, kDataLen * sizeof(float));
  // RUN: sh -c "test `grep -c -F 'hipMalloc(&device_y, kDataLen * sizeof(float));' %t` -eq 2"
  cudaMalloc(&device_y, kDataLen * sizeof(float));
  // RUN: sh -c "test `grep -c -F 'hipMemcpy(device_x, host_x, kDataLen * sizeof(float), hipMemcpyHostToDevice);' %t` -eq 2"
  cudaMemcpy(device_x, host_x, kDataLen * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel.
  // RUN: sh -c "test `grep -c -F 'hipLaunchKernel(HIP_KERNEL_NAME(axpy), dim3(1), dim3(kDataLen), 0, 0, a, device_x, device_y);' %t` -eq 2"
  axpy<<<1, kDataLen>>>(a, device_x, device_y);

  // Copy output data to host.
  // RUN: sh -c "test `grep -c -F 'hipDeviceSynchronize();' %t` -eq 2"
  cudaDeviceSynchronize();
  // RUN: sh -c "test `grep -c -F 'hipMemcpy(host_y, device_y, kDataLen * sizeof(float), hipMemcpyDeviceToHost);' %t` -eq 2"
  cudaMemcpy(host_y, device_y, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

  // Print the results.
  for (int i = 0; i < kDataLen; ++i) {
    std::cout << "y[" << i << "] = " << host_y[i] << "\n";
  }

  // RUN: sh -c "test `grep -c -F 'hipDeviceReset();' %t` -eq 2"
  cudaDeviceReset();
  return 0;
}
