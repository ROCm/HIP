// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <stdio.h>
// CHECK: #include <hipDNN.h>
#include <cudnn.h>

/**
 * 
 * Author: Jon Gauthier <jon@gauthiers.net>
 * February 2015
 * 
. * Adopted for CUDA/CUDNN 9.0
 */

void printMatrix(const double *mat, int m, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            printf("%f\n", mat[j * m + i]);
        }
        printf("\n\n");
    }
}

double *makeDiffData(int m, int c) {
  double *diff = (double *) calloc(m * c, sizeof(double));
  for (int j = 0; j < m; j++) {
    int class_ = rand() % c;
    printf("%d class: %d\n", j, class_);
    for (int i = 0; i < c; i++)
      diff[j * c + i] = class_ == i ? -c / (double) m : 0;
  }

  return diff;
}

int main() {
    int m = 5, c = 4, numChannels = 1;

    double *fcLayer = (double *) malloc(m * c * sizeof(double));
    for (int i = 0; i < m; i++) {
        double def = rand() % 25;
        for (int c_idx = 0; c_idx < c; c_idx++) {
            int offset = i * c + c_idx;
            fcLayer[offset] = def;
        }
    }
    printf("FC LAYER:\n");
    printMatrix(fcLayer, c, m);

    double *d_fcLayer;
    // CHECK: hipMalloc((void**) &d_fcLayer, m * c * sizeof(double));
    cudaMalloc((void**) &d_fcLayer, m * c * sizeof(double));
    // CHECK: hipMemcpy(d_fcLayer, fcLayer, m * c * sizeof(double), hipMemcpyHostToDevice);
    cudaMemcpy(d_fcLayer, fcLayer, m * c * sizeof(double), cudaMemcpyHostToDevice);

    double *d_softmaxData;
    // CHECK: hipMalloc((void**) &d_softmaxData, m * c * sizeof(double));
    cudaMalloc((void**) &d_softmaxData, m * c * sizeof(double));

    // CHECK: hipdnnHandle_t handle;
    cudnnHandle_t handle;
    // CHECK: hipdnnCreate(&handle);
    cudnnCreate(&handle);

    float one = 1;
    float zero = 0;

    // softmaxForward(n, c, h, w, dstData, &srcData);
    // CHECK: hipdnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;
    // CHECK: hipdnnCreateTensorDescriptor(&srcTensorDesc);
    // CHECK: hipdnnCreateTensorDescriptor(&sftTensorDesc);
    cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;
    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnCreateTensorDescriptor(&sftTensorDesc);
    // CHECK: hipdnnSetTensor4dDescriptor(srcTensorDesc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_DOUBLE,
    cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
            m, c, 1, 1);
    // CHECK: hipdnnSetTensor4dDescriptor(sftTensorDesc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_DOUBLE,
    cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
            m, c, 1, 1);
    // CHECK: hipdnnSoftmaxForward(handle, HIPDNN_SOFTMAX_ACCURATE, HIPDNN_SOFTMAX_MODE_CHANNEL, &one,
    cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &one,
            srcTensorDesc, d_fcLayer, &zero, sftTensorDesc, d_softmaxData);
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();

    // Copy back
    double *result = (double *) malloc(m * c * sizeof(double));
    // CHECK: hipMemcpy(result, d_softmaxData, m * c * sizeof(double), hipMemcpyDeviceToHost);
    // CHECK: hipDeviceSynchronize();
    cudaMemcpy(result, d_softmaxData, m * c * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Log
    printf("SOFTMAX:\n");
    printMatrix(result, c, m);

    // Try backward
    // CHECK: hipdnnTensorDescriptor_t diffTensorDesc;
    // CHECK: hipdnnCreateTensorDescriptor(&diffTensorDesc);
    // CHECK: hipdnnSetTensor4dDescriptor(diffTensorDesc, HIPDNN_TENSOR_NCHW, HIPDNN_DATA_DOUBLE,
    cudnnTensorDescriptor_t diffTensorDesc;
    cudnnCreateTensorDescriptor(&diffTensorDesc);
    cudnnSetTensor4dDescriptor(diffTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
                               m, c, 1, 1);

    double *d_gradData;
    // CHECK: hipMalloc((void**) &d_gradData, m * c * sizeof(double));
    cudaMalloc((void**) &d_gradData, m * c * sizeof(double));

    double *diffData = makeDiffData(m, c);
    double *d_diffData;
    // CHECK: hipMalloc((void**) &d_diffData, m * c * sizeof(double));
    // CHECK: hipMemcpy(d_diffData, diffData, m * c * sizeof(double), hipMemcpyHostToDevice);
    // CHECK: hipDeviceSynchronize();
    cudaMalloc((void**) &d_diffData, m * c * sizeof(double));
    cudaMemcpy(d_diffData, diffData, m * c * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // CHECK: hipdnnSoftmaxBackward(handle, HIPDNN_SOFTMAX_ACCURATE, HIPDNN_SOFTMAX_MODE_CHANNEL,
    cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                         &one, srcTensorDesc, d_softmaxData, diffTensorDesc, d_diffData, &zero, sftTensorDesc, d_gradData);
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();

    // Copy back
    double *result_backward = (double *) malloc(m * c * sizeof(double));
    // CHECK: hipMemcpy(result_backward, d_gradData, m * c * sizeof(double), hipMemcpyDeviceToHost);
    // CHECK: hipDeviceSynchronize();
    cudaMemcpy(result_backward, d_gradData, m * c * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Log
    printf("GRADIENT:\n");
    printMatrix(result_backward, c, m);

    // Destruct
    free(result);
    free(diffData);
    free(result_backward);
    free(fcLayer);

    // CHECK: hipdnnDestroyTensorDescriptor(srcTensorDesc);
    // CHECK: hipdnnDestroyTensorDescriptor(sftTensorDesc);
    // CHECK: hipdnnDestroyTensorDescriptor(diffTensorDesc);
    // CHECK: hipFree(d_fcLayer);
    // CHECK: hipFree(d_softmaxData);
    // CHECK: hipFree(d_gradData);
    // CHECK: hipFree(d_diffData);
    // CHECK: hipdnnDestroy(handle);
    cudnnDestroyTensorDescriptor(srcTensorDesc);
    cudnnDestroyTensorDescriptor(sftTensorDesc);
    cudnnDestroyTensorDescriptor(diffTensorDesc);
    cudaFree(d_fcLayer);
    cudaFree(d_softmaxData);
    cudaFree(d_gradData);
    cudaFree(d_diffData);
    cudnnDestroy(handle);
}