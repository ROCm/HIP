// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
// CHECK: #include <hip/hip_runtime.h>
#include <iostream>
// CHECK: #include <hiprand.h>
#include <curand.h>
// CHECK: #include <hipcub/hipcub.hpp>
#include <cub/cub.cuh>

#include <iostream>

template <int BLOCK_WIDTH, int ITEMS_PER_THREAD,
          // CHECK: hipcub::BlockLoadAlgorithm BLOCK_LOAD_ALGO,
          cub::BlockLoadAlgorithm BLOCK_LOAD_ALGO,
          // CHECK: hipcub::BlockStoreAlgorithm BLOCK_STORE_ALGO,
          cub::BlockStoreAlgorithm BLOCK_STORE_ALGO,
          typename T>
__global__ void sort(const T* data_in, T* data_out){
    // CHECK: typedef hipcub::BlockLoad<T, BLOCK_WIDTH, ITEMS_PER_THREAD, BLOCK_LOAD_ALGO> BlockLoadT;
    typedef cub::BlockLoad<T, BLOCK_WIDTH, ITEMS_PER_THREAD, BLOCK_LOAD_ALGO> BlockLoadT;
    // CHECK: typedef hipcub::BlockRadixSort<T, BLOCK_WIDTH, ITEMS_PER_THREAD> BlockRadixSortT;
    typedef cub::BlockRadixSort<T, BLOCK_WIDTH, ITEMS_PER_THREAD> BlockRadixSortT;
    // CHECK: typedef hipcub::BlockStore<T, BLOCK_WIDTH, ITEMS_PER_THREAD, BLOCK_STORE_ALGO> BlockStoreT;
    typedef cub::BlockStore<T, BLOCK_WIDTH, ITEMS_PER_THREAD, BLOCK_STORE_ALGO> BlockStoreT;
    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockRadixSortT::TempStorage sort;
        typename BlockStoreT::TempStorage store;
    } tmp_storage;
    T items[ITEMS_PER_THREAD];
    BlockLoadT(tmp_storage.load).Load(data_in + blockIdx.x * BLOCK_WIDTH * ITEMS_PER_THREAD, items);
    __syncthreads();
    BlockRadixSortT(tmp_storage.sort).Sort(items);
    __syncthreads();
    BlockStoreT(tmp_storage.store).Store(data_out + blockIdx.x * BLOCK_WIDTH * ITEMS_PER_THREAD, items);
}

int main() {
    double* d_gpu = NULL;
    double* result_gpu = NULL;
    double* data_sorted = new double[1000*4096];
    // Allocate memory on the GPU
    // CHECK: hipMalloc(&d_gpu, 1000*4096 * sizeof(double));
    cudaMalloc(&d_gpu, 1000*4096 * sizeof(double));
    // CHECK: hipMalloc(&result_gpu, 1000*4096 * sizeof(double));
    cudaMalloc(&result_gpu, 1000*4096 * sizeof(double));
    // CHECK: hiprandGenerator_t gen;
    curandGenerator_t gen;
    // Create generator
    // CHECK: hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // Fill array with random numbers
    // CHECK: hiprandGenerateNormalDouble(gen, d_gpu, 1000*4096, 0.0, 1.0);
    curandGenerateNormalDouble(gen, d_gpu, 1000*4096, 0.0, 1.0);
    // Destroy generator
    // CHECK: hiprandDestroyGenerator(gen);
    curandDestroyGenerator(gen);
    // Sort data
    // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(sort<512, 8, hipcub::BLOCK_LOAD_TRANSPOSE, hipcub::BLOCK_STORE_TRANSPOSE>), dim3(1000), dim3(512), 0, 0, d_gpu, result_gpu);
    sort<512, 8, cub::BLOCK_LOAD_TRANSPOSE, cub::BLOCK_STORE_TRANSPOSE><<<1000, 512>>>(d_gpu, result_gpu);
    // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(sort<256, 16, hipcub::BLOCK_LOAD_DIRECT, hipcub::BLOCK_STORE_DIRECT>), dim3(1000), dim3(256), 0, 0, d_gpu, result_gpu);
    sort<256, 16, cub::BLOCK_LOAD_DIRECT, cub::BLOCK_STORE_DIRECT><<<1000, 256>>>(d_gpu, result_gpu);
    // CHECK: hipMemcpy(data_sorted, result_gpu, 1000*4096*sizeof(double), hipMemcpyDeviceToHost);
    cudaMemcpy(data_sorted, result_gpu, 1000*4096*sizeof(double), cudaMemcpyDeviceToHost);
    // Write the sorted data to standard out
    for (int i = 0; i < 4095; ++i) {
        std::cout << data_sorted[i] << ", ";
    }
    std::cout << data_sorted[4095] << std::endl;
}
