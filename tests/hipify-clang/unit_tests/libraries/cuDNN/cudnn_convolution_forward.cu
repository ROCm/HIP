// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
// CHECK: #include "hipDNN.h"
#include "cudnn.h"

// CHECK: hipError_t err = (f); \
// CHECK: if (err != hipSuccess) { \

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}
// CHECK: hipdnnStatus_t err = (f); \
// CHECK: if (err != HIPDNN_STATUS_SUCCESS) { \

#define CUDNN_CALL(f) { \
    cudnnStatus_t err = (f); \
    if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

__global__ void dev_const(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_iota(float *px) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid;
}

void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  // CHECK: CUDA_CALL(hipMemcpy(
  CUDA_CALL(cudaMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        // CHECK: hipMemcpyDeviceToHost));
        cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(4) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

int main() {
  // CHECK: hipdnnHandle_t cudnn;
  cudnnHandle_t cudnn;
  // CHECK: CUDNN_CALL(hipdnnCreate(&cudnn));
  CUDNN_CALL(cudnnCreate(&cudnn));

  // input
  const int in_n = 1;
  const int in_c = 1;
  const int in_h = 5;
  const int in_w = 5;
  std::cout << "in_n: " << in_n << std::endl;
  std::cout << "in_c: " << in_c << std::endl;
  std::cout << "in_h: " << in_h << std::endl;
  std::cout << "in_w: " << in_w << std::endl;
  std::cout << std::endl;
  // CHECK: hipdnnTensorDescriptor_t in_desc;
  cudnnTensorDescriptor_t in_desc;
  // CHECK: CUDNN_CALL(hipdnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  // CHECK: CUDNN_CALL(hipdnnSetTensor4dDescriptor(
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
  // CHECK: in_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

  float *in_data;
  // CHECK: CUDA_CALL(hipMalloc(
  CUDA_CALL(cudaMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));

  // filter
  const int filt_k = 1;
  const int filt_c = 1;
  const int filt_h = 2;
  const int filt_w = 2;
  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;

  // CHECK: hipdnnFilterDescriptor_t filt_desc;
  cudnnFilterDescriptor_t filt_desc;
  // CHECK: CUDNN_CALL(hipdnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  // CHECK: CUDNN_CALL(hipdnnSetFilter4dDescriptor(
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        // CHECK: filt_desc, HIPDNN_DATA_FLOAT, HIPDNN_TENSOR_NCHW,
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

  float *filt_data;
  //  CUDA_CALL(hipMalloc(
  CUDA_CALL(cudaMalloc(
      &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));

  // convolution
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;

  // CHECK: hipdnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  // CUDNN_CALL(hipdnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  // CHECK: CUDNN_CALL(hipdnnSetConvolution2dDescriptor(
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        // CHECK: HIPDNN_CONVOLUTION, HIPDNN_DATA_FLOAT));
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;

  // CHECK: CUDNN_CALL(hipdnnGetConvolution2dForwardOutputDim(
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std::cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;
  // CHECK: hipdnnTensorDescriptor_t out_desc;
  cudnnTensorDescriptor_t out_desc;
  // CHECK: CUDNN_CALL(hipdnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  // CHECK: CUDNN_CALL(hipdnnSetTensor4dDescriptor(
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        // CHECK: out_desc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_FLOAT,
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

  float *out_data;
  // CHECK: CUDA_CALL(hipMalloc(
  CUDA_CALL(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  // algorithm
  // CHECK: hipdnnConvolutionFwdAlgo_t algo;
  cudnnConvolutionFwdAlgo_t algo;
  // CHECK: CUDNN_CALL(hipdnnGetConvolutionForwardAlgorithm(
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
        cudnn,
        in_desc, filt_desc, conv_desc, out_desc,
        // CHECK: HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

  std::cout << "Convolution algorithm: " << algo << std::endl;
  std::cout << std::endl;

  // workspace
  size_t ws_size;
  // CHECK: CUDNN_CALL(hipdnnGetConvolutionForwardWorkspaceSize(
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  float *ws_data;
  // CHECK: CUDA_CALL(hipMalloc(&ws_data, ws_size));
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;

  // perform
  float alpha = 1.f;
  float beta = 0.f;
  // CHECK: hipLaunchKernelGGL(dev_iota, dim3(in_w * in_h), dim3(in_n * in_c), 0, 0, in_data);
  // CHECK: hipLaunchKernelGGL(dev_const, dim3(filt_w * filt_h), dim3(filt_k * filt_c), 0, 0, filt_data, 1.f);
  dev_iota<<<in_w * in_h, in_n * in_c>>>(in_data);
  dev_const<<<filt_w * filt_h, filt_k * filt_c>>>(filt_data, 1.f);
  // CHECK: CUDNN_CALL(hipdnnConvolutionForward(
  CUDNN_CALL(cudnnConvolutionForward(
      cudnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, out_data));

  // results
  std::cout << "in_data:" << std::endl;
  print(in_data, in_n, in_c, in_h, in_w);
  
  std::cout << "filt_data:" << std::endl;
  print(filt_data, filt_k, filt_c, filt_h, filt_w);
  
  std::cout << "out_data:" << std::endl;
  print(out_data, out_n, out_c, out_h, out_w);

  // finalizing
  // CHECK: CUDA_CALL(hipFree(ws_data));
  CUDA_CALL(cudaFree(ws_data));
  // CHECK: CUDA_CALL(hipFree(out_data));
  CUDA_CALL(cudaFree(out_data));
  // CHECK: CUDNN_CALL(hipdnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  // CHECK: CUDNN_CALL(hipdnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  // CHECK: CUDA_CALL(hipFree(filt_data));
  CUDA_CALL(cudaFree(filt_data));
  // CHECK: CUDNN_CALL(hipdnnDestroyFilterDescriptor(filt_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  // CHECK: CUDA_CALL(hipFree(in_data));
  CUDA_CALL(cudaFree(in_data));
  // CHECK: CUDNN_CALL(hipdnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  // CHECK: CUDNN_CALL(hipdnnDestroy(cudnn));
  CUDNN_CALL(cudnnDestroy(cudnn));
  return 0;
}
