// RUN: %run_test hipify "%s" "%t" %cuda_args
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

//
// This sample demonstrates the use of streams for concurrent execution. It also illustrates how to
// introduce dependencies between CUDA streams with the new cudaStreamWaitEvent function introduced
// in CUDA 3.2.
//
// Devices of compute capability 1.x will run the kernels one after another
// Devices of compute capability 2.0 or higher can overlap the kernels
//
#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h>

// This is a kernel that does no real work but runs at least for a specified number of clocks
__global__ void clock_block(clock_t *d_o, clock_t clock_count)
{
    unsigned int start_clock = (unsigned int) clock();

    clock_t clock_offset = 0;

    while (clock_offset < clock_count)
    {
        unsigned int end_clock = (unsigned int) clock();

        // The code below should work like
        // this (thanks to modular arithmetics):
        //
        // clock_offset = (clock_t) (end_clock > start_clock ?
        //                           end_clock - start_clock :
        //                           end_clock + (0xffffffffu - start_clock));
        //
        // Indeed, let m = 2^32 then
        // end - start = end + m - start (mod m).

        clock_offset = (clock_t)(end_clock - start_clock);
    }

    d_o[0] = clock_offset;
}


// Single warp reduction kernel
__global__ void sum(clock_t *d_clocks, int N)
{
    __shared__ clock_t s_clocks[32];

    clock_t my_sum = 0;

    for (int i = threadIdx.x; i < N; i+= blockDim.x)
    {
        my_sum += d_clocks[i];
    }

    s_clocks[threadIdx.x] = my_sum;
    syncthreads();

    for (int i=16; i>0; i/=2)
    {
        if (threadIdx.x < i)
        {
            s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
        }

        syncthreads();
    }

    d_clocks[0] = s_clocks[0];
}

int main(int argc, char **argv)
{
    int nkernels = 8;               // number of concurrent kernels
    int nstreams = nkernels + 1;    // use one more stream than concurrent kernel
    int nbytes = nkernels * sizeof(clock_t);   // number of data bytes
    float kernel_time = 10; // time the kernel should run in ms
    float elapsed_time;   // timing variables
    int cuda_device = 0;

    printf("[%s] - Starting...\n", argv[0]);

    // get number of kernels if overridden on the command line
    if (checkCmdLineFlag(argc, (const char **)argv, "nkernels"))
    {
        nkernels = getCmdLineArgumentInt(argc, (const char **)argv, "nkernels");
        nstreams = nkernels + 1;
    }

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    cuda_device = findCudaDevice(argc, (const char **)argv);

    // CHECK: hipDeviceProp_t deviceProp;
    cudaDeviceProp deviceProp;
    // CHECK: checkCudaErrors(hipGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDevice(&cuda_device));

    // CHECK: checkCudaErrors(hipGetDeviceProperties(&deviceProp, cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    if ((deviceProp.concurrentKernels == 0))
    {
        printf("> GPU does not support concurrent kernel execution\n");
        printf("  CUDA kernel runs will be serialized\n");
    }

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // allocate host memory
    clock_t *a = 0;                     // pointer to the array data in host memory
    // CHECK: checkCudaErrors(hipHostMalloc((void **)&a, nbytes));
    checkCudaErrors(cudaMallocHost((void **)&a, nbytes));

    // allocate device memory
    clock_t *d_a = 0;             // pointers to data and init value in the device memory
    // CHECK: checkCudaErrors(hipMalloc((void **)&d_a, nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));

    // CHECK: hipStream_t *streams = (hipStream_t *) malloc(nstreams * sizeof(hipStream_t));
    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));

    for (int i = 0; i < nstreams; i++)
    {
        // CHECK: checkCudaErrors(hipStreamCreate(&(streams[i])));
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

    // CHECK: hipEvent_t start_event, stop_event;
    // create CUDA event handles
    cudaEvent_t start_event, stop_event;

    // CHECK: checkCudaErrors(hipEventCreate(&start_event));
    // CHECK: checkCudaErrors(hipEventCreate(&stop_event));
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    // the events are used for synchronization only and hence do not need to record timings
    // this also makes events not introduce global sync points when recorded which is critical to get overlap

    // CHECK: hipEvent_t *kernelEvent;
    // CHECK: kernelEvent = (hipEvent_t *) malloc(nkernels * sizeof(hipEvent_t));
    cudaEvent_t *kernelEvent;
    kernelEvent = (cudaEvent_t *) malloc(nkernels * sizeof(cudaEvent_t));

    for (int i = 0; i < nkernels; i++)
    {
        // CHECK: checkCudaErrors(hipEventCreateWithFlags(&(kernelEvent[i]), hipEventDisableTiming));
        checkCudaErrors(cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming));
    }

    //////////////////////////////////////////////////////////////////////
    // time execution with nkernels streams
    clock_t total_clocks = 0;
#if defined(__arm__) || defined(__aarch64__)
    // the kernel takes more time than the channel reset time on arm archs, so to prevent hangs reduce time_clocks.
    clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 1000));
#else
    clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
#endif

    // CHECK: hipEventRecord(start_event, 0);
    cudaEventRecord(start_event, 0);

    // queue nkernels in separate streams and record when they are done
    for (int i=0; i<nkernels; ++i)
    {
        // CHECK: hipLaunchKernelGGL(clock_block, dim3(1), dim3(1), 0, streams[i], &d_a[i], time_clocks);
        clock_block<<<1,1,0,streams[i]>>>(&d_a[i], time_clocks);
        total_clocks += time_clocks;

        // CHECK: checkCudaErrors(hipEventRecord(kernelEvent[i], streams[i]));
        checkCudaErrors(cudaEventRecord(kernelEvent[i], streams[i]));

        // make the last stream wait for the kernel event to be recorded
        // CHECK: checkCudaErrors(hipStreamWaitEvent(streams[nstreams-1], kernelEvent[i],0));
        checkCudaErrors(cudaStreamWaitEvent(streams[nstreams-1], kernelEvent[i],0));
    }

    // queue a sum kernel and a copy back to host in the last stream.
    // the commands in this stream get dispatched as soon as all the kernel events have been recorded
    // CHECK: hipLaunchKernelGGL(sum, dim3(1), dim3(32), 0, streams[nstreams-1], d_a, nkernels);
    // CHECK: checkCudaErrors(hipMemcpyAsync(a, d_a, sizeof(clock_t), hipMemcpyDeviceToHost, streams[nstreams-1]));
    sum<<<1,32,0,streams[nstreams-1]>>>(d_a, nkernels);
    checkCudaErrors(cudaMemcpyAsync(a, d_a, sizeof(clock_t), cudaMemcpyDeviceToHost, streams[nstreams-1]));

    // at this point the CPU has dispatched all work for the GPU and can continue processing other tasks in parallel

    // in this sample we just wait until the GPU is done
    // CHECK: checkCudaErrors(hipEventRecord(stop_event, 0));
    // CHECK: checkCudaErrors(hipEventSynchronize(stop_event));
    // CHECK: checkCudaErrors(hipEventElapsedTime(&elapsed_time, start_event, stop_event));
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

    printf("Expected time for serial execution of %d kernels = %.3fs\n", nkernels, nkernels * kernel_time/1000.0f);
    printf("Expected time for concurrent execution of %d kernels = %.3fs\n", nkernels, kernel_time/1000.0f);
    printf("Measured time for sample = %.3fs\n", elapsed_time/1000.0f);

    bool bTestResult  = (a[0] > total_clocks);

    // release resources
    for (int i = 0; i < nkernels; i++)
    {
        // CHECK: hipStreamDestroy(streams[i]);
        // CHECK: hipEventDestroy(kernelEvent[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(kernelEvent[i]);
    }

    free(streams);
    free(kernelEvent);

    // CHECK: hipEventDestroy(start_event);
    // CHECK: hipEventDestroy(stop_event);
    // CHECK: hipHostFree(a);
    // CHECK: hipFree(d_a);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFreeHost(a);
    cudaFree(d_a);

    if (!bTestResult)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
