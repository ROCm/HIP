// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
// CHECK: #include <hip/hip_runtime.h>
#include <math.h>

__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(int argc, char *argv[])
{
    int numElements = 10;
    bool testResult = true;
    float *A, *B;
    // CHECK: hipMallocManaged(&A, numElements * sizeof(float));
    cudaMallocManaged(&A, numElements * sizeof(float));
    // CHECK: hipMallocManaged(&B, numElements * sizeof(float));
    cudaMallocManaged(&B, numElements * sizeof(float));
    for (int i = 0; i < numElements; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(blockSize, 1, 1);
    // CHECK: hipLaunchKernelGGL(add, dim3(dimGrid), dim3(dimBlock), 0, 0, numElements, A, B);
    add<<<dimGrid, dimBlock>>>(numElements, A, B);
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();
    float maxError = 0.0f;
    for (int i = 0; i < numElements; i++)
        maxError = fmax(maxError, fabs(B[i]-3.0f));
    // CHECK: hipFree(A);
    cudaFree(A);
    // CHECK: hipFree(B);
    cudaFree(B);
    if(maxError == 0.0f)
        return 0;
    return -1;
}
