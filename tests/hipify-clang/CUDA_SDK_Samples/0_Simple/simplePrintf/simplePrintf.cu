// RUN: %run_test hipify "%s" "%t" %cuda_args

/*
* This software contains source code provided by NVIDIA Corporation.
*/

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void testKernel(int val)
{
    printf("[%d, %d]:\t\tValue is:%d\n",\
            blockIdx.y*gridDim.x+blockIdx.x,\
            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
            val);
}

int main(int argc, char **argv)
{
    int devID;
    // CHECK: hipDeviceProp_t props;
    cudaDeviceProp props;

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);

    //Get GPU information
    // CHECK: checkCudaErrors(hipGetDevice(&devID));
    // CHECK: checkCudaErrors(hipGetDeviceProperties(&props, devID));
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);

    printf("printf() is called. Output:\n\n");

    //Kernel configuration, where a two-dimensional grid and
    //three-dimensional blocks are configured.
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    // CHECK: hipLaunchKernelGGL(testKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, 10);
    testKernel<<<dimGrid, dimBlock>>>(10);
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}

