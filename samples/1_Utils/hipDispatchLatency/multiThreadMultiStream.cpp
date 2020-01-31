/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <pthread.h>
#include <stdio.h>
#include "hip/hip_runtime.h"
#ifdef __HIP_PLATFORM_HCC__
#include "hip/hip_ext.h"
#endif
#include <iostream>
#include <chrono>
#include <algorithm>

#define NUM_THREADS 2
#define NUM_GROUPS 1
#define GROUP_SIZE 1
#define WARMUP_RUN_COUNT 10
#define TIMING_RUN_COUNT 100
#define TOTAL_RUN_COUNT WARMUP_RUN_COUNT + TIMING_RUN_COUNT
#define BATCH_SIZE 1000

#define FILE_NAME "test_kernel.code"
#define KERNEL_NAME "test"

__global__ void EmptyKernel() { }

void print_timing(std::string test, const std::array<float, TOTAL_RUN_COUNT> &results, int batch = 1) {
    
    float total_us = 0.0f, mean_us = 0.0f, stddev_us = 0.0f;
    
    // skip warm-up runs
    auto start_iter = std::next(results.begin(), WARMUP_RUN_COUNT);
    auto end_iter = results.end();

    // mean
    std::for_each(start_iter, end_iter, [&](const float &run_ms) {
        total_us += (run_ms * 1000) / batch;
    });   
    mean_us = total_us  / TIMING_RUN_COUNT;

   // stddev
    total_us = 0;
    std::for_each(start_iter, end_iter, [&](const float &run_ms) {
        float dev_us = ((run_ms * 1000) / batch) - mean_us;
        total_us += dev_us * dev_us;
    });
    stddev_us = sqrt(total_us / TIMING_RUN_COUNT);

    // display
    printf("\n %s: %.1f us, std: %.1f us\n", test.c_str(), mean_us, stddev_us);
}

void *testEmptyKernelLaunchInStream(void *tid)
{
    int *threadId = (int *)tid;
    std::array<float, TOTAL_RUN_COUNT> results;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipStream_t stream;
    hipStreamCreate(&stream);
    hipModule_t module;
    hipFunction_t function;
    hipModuleLoad(&module, FILE_NAME);
    hipModuleGetFunction(&function, module, KERNEL_NAME);
    void* kernel_params = nullptr;

    /************************************************************************************/
    /* HIP kernel launch enqueue rate:                                                  */
    /* Measure time taken to enqueue a kernel on the GPU                                */
    /************************************************************************************/
    // Timing hipModuleLaunchKernel
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, &kernel_params, nullptr);
        auto stop = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<double, std::milli>(stop - start).count();
    }
    print_timing("hipModuleLaunchKernel enqueue rate Non-Null Stream Thread-"+ std::to_string(*threadId), results);

    // Timing hipLaunchKernelGGL
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream);
        auto stop = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<double, std::milli>(stop - start).count();
    }
    print_timing("hipLaunchKernelGGL enqueue rate Non-Null Stream Thread-"+ std::to_string(*threadId), results);

    /***********************************************************************************/
    /* Single dispatch execution latency using HIP events:                             */
    /* Measures time to start & finish executing a kernel with GPU-scope visibility    */
    /***********************************************************************************/
    //Timing directly the dispatch
#ifdef __HIP_PLATFORM_HCC__
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        hipEventRecord(start, stream);
        hipExtLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream, start, stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&results[i], start, stop);
    }
    print_timing("Timing directly single dispatch latency Non-Null Stream Thread-" + std::to_string(*threadId), results);
#endif

    //Timing around the dispatch
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        hipEventRecord(start, stream);
        hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream);
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&results[i], start, stop);
    }
    print_timing("Timing around single dispatch latency Non-Null Stream Thread- " + std::to_string(*threadId), results);

    /*********************************************************************************/
    /* Batch dispatch execution latency using HIP events:                            */
    /* Measures average time to start & finish executing each dispatch in a batch    */ 
    /*********************************************************************************/

    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        hipEventRecord(start, stream);
        for (int j = 0; j < BATCH_SIZE; j++) {
            hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream);
        }
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&results[i], start, stop);
    }
    print_timing("Batch dispatch latency Non-Null Stream Thread-"+ std::to_string(*threadId), results, BATCH_SIZE);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipModuleUnload(module);
    hipStreamDestroy(stream);
    return NULL;
}

void *testEmptyKernelLaunchInNullStream(void *tid)
{
    int *threadId = (int *)tid;
    std::array<float, TOTAL_RUN_COUNT> results;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipModule_t module;
    hipFunction_t function;
    hipModuleLoad(&module, FILE_NAME);
    hipModuleGetFunction(&function, module, KERNEL_NAME);
    void* kernel_params = nullptr;

    /************************************************************************************/
    /* HIP kernel launch enqueue rate:                                                  */
    /* Measure time taken to enqueue a kernel on the GPU                                */
    /************************************************************************************/ 
    // Timing hipModuleLaunchKernel
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, &kernel_params, nullptr);
        auto stop = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<double, std::milli>(stop - start).count();
    }
    print_timing("hipModuleLaunchKernel enqueue rate Null Stream Thread-"+ std::to_string(*threadId), results);

    // Timing hipLaunchKernelGGL
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, 0);
        auto stop = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<double, std::milli>(stop - start).count();
    }
    print_timing("hipLaunchKernelGGL enqueue rate Null Stream Thread-"+ std::to_string(*threadId), results);

    /***********************************************************************************/
    /* Single dispatch execution latency using HIP events:                             */   
    /* Measures time to start & finish executing a kernel with GPU-scope visibility    */ 
    /***********************************************************************************/
    //Timing directly the dispatch
#ifdef __HIP_PLATFORM_HCC__
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        hipEventRecord(start, 0);
        hipExtLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, 0, start, stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&results[i], start, stop);
    }
    print_timing("Timing directly single dispatch latency Null Stream Thread-" + std::to_string(*threadId), results);
#endif

    //Timing around the dispatch
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        hipEventRecord(start, 0);
        hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, 0);
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&results[i], start, stop);
    }
    print_timing("Timing around single dispatch latency Null Stream Thread-"+std::to_string(*threadId), results);

    /*********************************************************************************/
    /* Batch dispatch execution latency using HIP events:                            */
    /* Measures average time to start & finish executing each dispatch in a batch    */ 
    /*********************************************************************************/

    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        hipEventRecord(start, 0);
        for (int j = 0; j < BATCH_SIZE; j++) {
            hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, 0);
        }
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&results[i], start, stop);
    }
    print_timing("Batch dispatch latency Null Stream Thread-"+ std::to_string(*threadId), results, BATCH_SIZE);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipModuleUnload(module);
    return NULL;
}

int main()
{
    const int num_threads = NUM_THREADS;

    pthread_t threads[num_threads];
    int index[]= {1,2,3,4,5,6,7,8};
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, testEmptyKernelLaunchInNullStream,(void *)&index[i])) {
            printf("Error creating threads \n");
            return 1;
        }        
    }

    for (int i = 0; i < num_threads; i++) {
        if(pthread_join(threads[i], NULL)) {
            printf("Error joining thread \n");
            return 2;
        }
    }
    hipDeviceReset();
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, testEmptyKernelLaunchInStream,(void *)&index[i])) {
            printf("Error creating threads \n");
            return 1;
        }
    }

    for (int i = 0; i < num_threads; i++) {
        if(pthread_join(threads[i], NULL)) {
            printf("Error joining thread \n");
            return 2;
        }
    }
    return 0;
}
