// RUN: %run_test hipify "%s" "%t" %cuda_args

/*
* This software contains source code provided by NVIDIA Corporation.
*/

/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <iostream>
#include <helper_cuda.h>         // helper functions for CUDA error check

const int manualBlockSize = 32;

////////////////////////////////////////////////////////////////////////////////
// Test kernel
//
// This kernel squares each array element. Each thread addresses
// himself with threadIdx and blockIdx, so that it can handle any
// execution configuration, including anything the launch configurator
// API suggests.
////////////////////////////////////////////////////////////////////////////////
__global__ void square(int *array, int arrayCount)
{
    // CHECK: HIP_DYNAMIC_SHARED(int, dynamicSmem);
    extern __shared__ int dynamicSmem[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Potential occupancy calculator
//
// The potential occupancy is calculated according to the kernel and
// execution configuration the user desires. Occupancy is defined in
// terms of active blocks per multiprocessor, and the user can convert
// it to other metrics.
//
// This wrapper routine computes the occupancy of kernel, and reports
// it in terms of active warps / maximum warps per SM.
////////////////////////////////////////////////////////////////////////////////
static double reportPotentialOccupancy(void *kernel, int blockSize, size_t dynamicSMem)
{
    int device;
    // CHECK: hipDeviceProp_t prop;
    cudaDeviceProp prop;

    int numBlocks;
    int activeWarps;
    int maxWarps;

    double occupancy;

    // CHECK: checkCudaErrors(hipGetDevice(&device));
    // CHECK: checkCudaErrors(hipGetDeviceProperties(&prop, device));
    // CHECK: checkCudaErrors(hipOccupancyMaxActiveBlocksPerMultiprocessor(
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &numBlocks,
                        kernel,
                        blockSize,
                        dynamicSMem));

    // CHECK: activeWarps = numBlocks * blockSize / prop.hipWarpSize;
    // CHECK: maxWarps = prop.maxThreadsPerMultiProcessor / prop.hipWarpSize;
    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    occupancy = (double)activeWarps / maxWarps;

    return occupancy;
}

////////////////////////////////////////////////////////////////////////////////
// Occupancy-based launch configurator
//
// The launch configurator, cudaOccupancyMaxPotentialBlockSize and
// cudaOccupancyMaxPotentialBlockSizeVariableSMem, suggests a block
// size that achieves the best theoretical occupancy. It also returns
// the minimum number of blocks needed to achieve the occupancy on the
// whole device.
//
// This launch configurator is purely occupancy-based. It doesn't
// translate directly to performance, but the suggestion should
// nevertheless be a good starting point for further optimizations.
//
// This function configures the launch based on the "automatic"
// argument, records the runtime, and reports occupancy and runtime.
////////////////////////////////////////////////////////////////////////////////
static int launchConfig(int *array, int arrayCount, bool automatic)
{
    int blockSize;
    int minGridSize;
    int gridSize;
    size_t dynamicSMemUsage = 0;
    // CHECK: hipEvent_t start;
    // CHECK: hipEvent_t end;
    cudaEvent_t start;
    cudaEvent_t end;

    float elapsedTime;
    
    double potentialOccupancy;
    // CHECK: checkCudaErrors(hipEventCreate(&start));
    // CHECK: checkCudaErrors(hipEventCreate(&end));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));

    if (automatic) {
        // CHECK: checkCudaErrors(hipOccupancyMaxPotentialBlockSize(
        checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
                            &minGridSize,
                            &blockSize,
                            (void*)square,
                            dynamicSMemUsage,
                            arrayCount));

        std::cout << "Suggested block size: " << blockSize << std::endl
                  << "Minimum grid size for maximum occupancy: " << minGridSize << std::endl;
    } else {
        // This block size is too small. Given limited number of
        // active blocks per multiprocessor, the number of active
        // threads will be limited, and thus unable to achieve maximum
        // occupancy.
        //
        blockSize = manualBlockSize;
    }

    // Round up
    //
    gridSize = (arrayCount + blockSize - 1) / blockSize;

    // Launch and profile
    //
    // CHECK: checkCudaErrors(hipEventRecord(start));
    // CHECK: hipLaunchKernelGGL(square, dim3(gridSize), dim3(blockSize), dynamicSMemUsage, 0, array, arrayCount);
    // CHECK: checkCudaErrors(hipEventRecord(end));
    checkCudaErrors(cudaEventRecord(start));
    square<<<gridSize, blockSize, dynamicSMemUsage>>>(array, arrayCount);
    checkCudaErrors(cudaEventRecord(end));
    // CHECK: checkCudaErrors(hipDeviceSynchronize());
    checkCudaErrors(cudaDeviceSynchronize());

    // Calculate occupancy
    //
    potentialOccupancy = reportPotentialOccupancy((void*)square, blockSize, dynamicSMemUsage);

    std::cout << "Potential occupancy: " << potentialOccupancy * 100 << "%" << std::endl;

    // Report elapsed time
    //
    // CHECK: checkCudaErrors(hipEventElapsedTime(&elapsedTime, start, end));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, end));
    std::cout << "Elapsed time: " << elapsedTime << "ms" << std::endl;
    
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// The test
//
// The test generates an array and squares it with a CUDA kernel, then
// verifies the result.
////////////////////////////////////////////////////////////////////////////////
static int test(bool automaticLaunchConfig, const int count = 1000000)
{
    int *array;
    int *dArray;
    int size = count * sizeof(int);

    array = new int[count];

    for (int i = 0; i < count; i += 1) {
        array[i] = i;
    }
    // CHECK: checkCudaErrors(hipMalloc(&dArray, size));
    // CHECK: checkCudaErrors(hipMemcpy(dArray, array, size, hipMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&dArray, size));
    checkCudaErrors(cudaMemcpy(dArray, array, size, cudaMemcpyHostToDevice));

    for (int i = 0; i < count; i += 1) {
        array[i] = 0;
    }

    launchConfig(dArray, count, automaticLaunchConfig);
    // CHECK: checkCudaErrors(hipMemcpy(array, dArray, size, hipMemcpyDeviceToHost));
    // CHECK: checkCudaErrors(hipFree(dArray));
    checkCudaErrors(cudaMemcpy(array, dArray, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(dArray));

    // Verify the return data
    //
    for (int i = 0; i < count; i += 1) {
        if (array[i] != i * i) {
            std::cout << "element " << i << " expected " << i * i << " actual " << array[i] << std::endl;
            return 1;
        }
    }
    delete[] array;

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Sample Main
//
// The sample runs the test with manually configured launch and
// automatically configured launch, and reports the occupancy and
// performance.
////////////////////////////////////////////////////////////////////////////////
int main()
{
    int status;

    std::cout << "starting Simple Occupancy" << std::endl << std::endl;

    std::cout << "[ Manual configuration with " << manualBlockSize
              << " threads per block ]" << std::endl;

    status = test(false);
    if (status) {
        std::cerr << "Test failed\n" << std::endl;
        return -1;
    }

    std::cout << std::endl;

    std::cout << "[ Automatic, occupancy-based configuration ]" << std::endl;
    status = test(true);
    if (status) {
        std::cerr << "Test failed\n" << std::endl;
        return -1;
    }        

    std::cout << std::endl;
    std::cout << "Test PASSED\n" << std::endl;
    
    return 0;
}
