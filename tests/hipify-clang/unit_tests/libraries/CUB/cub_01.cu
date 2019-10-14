// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
// CHECK: #include <hip/hip_runtime.h>
#include <iostream>
// CHECK: #include <hiprand.h>
#include <curand.h>
// CHECK: #include <hipcub/hipcub.hpp>
#include <cub/cub.cuh>

#include <iostream>

// TODO:
// using namespace cub;

template <typename T>
__global__ void sort(const T* data_in, T* data_out){
     // CHECK: typedef ::hipcub::BlockRadixSort<T, 1024, 4> BlockRadixSortT;
     typedef ::cub::BlockRadixSort<T, 1024, 4> BlockRadixSortT;
    __shared__ typename BlockRadixSortT::TempStorage tmp_sort;
    double items[4];
    int i0 = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    for (int i = 0; i < 4; ++i){
        items[i] = data_in[i0 + i];
    }
    BlockRadixSortT(tmp_sort).Sort(items);
    for (int i = 0; i < 4; ++i){
        data_out[i0 + i] = items[i];
    }
}

int main(){
    double* d_gpu = NULL;
    double* result_gpu = NULL;
    double* data_sorted = new double[4096];
    // Allocate memory on the GPU
    // CHECK: hipMalloc(&d_gpu, 4096 * sizeof(double));
    cudaMalloc(&d_gpu, 4096 * sizeof(double));
    // CHECK: hipMalloc(&result_gpu, 4096 * sizeof(double));
    cudaMalloc(&result_gpu, 4096 * sizeof(double));
    // CHECK: hiprandGenerator_t gen;
    curandGenerator_t gen;
    // Create generator
    // CHECK: hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // Fill array with random numbers
    // CHECK: hiprandGenerateNormalDouble(gen, d_gpu, 4096, 0.0, 1.0);
    curandGenerateNormalDouble(gen, d_gpu, 4096, 0.0, 1.0);
    // Destroy generator
    // CHECK: hiprandDestroyGenerator(gen);
    curandDestroyGenerator(gen);
    // Sort data
    // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(sort), dim3(1), dim3(1024), 0, 0, d_gpu, result_gpu);
    sort<<<1, 1024>>>(d_gpu, result_gpu);
    // CHECK: hipMemcpy(data_sorted, result_gpu, 4096 * sizeof(double), hipMemcpyDeviceToHost);
    cudaMemcpy(data_sorted, result_gpu, 4096 * sizeof(double), cudaMemcpyDeviceToHost);
    // Write the sorted data to standard out
    for (int i = 0; i < 4096; ++i){
        std::cout << data_sorted[i] << ", ";
    }
    std::cout << std::endl;
}
