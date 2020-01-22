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

#include "hip/hip_runtime.h"
#ifdef __HIP_PLATFORM_HCC__
#include "hip/hip_ext.h"
#endif
#include <iostream>
#include <time.h>
#include "ResultDatabase.h"
#include <chrono>

#define PRINT_PROGRESS 0

#define check(cmd)                                                                                 \
    {                                                                                              \
        hipError_t status = cmd;                                                                   \
        if (status != hipSuccess) {                                                                \
            printf("error: '%s'(%d) from %s at %s:%d\n", hipGetErrorString(status), status, #cmd,  \
                   __FILE__, __LINE__);                                                            \
            abort();                                                                               \
        }                                                                                          \
    }

#define LEN 1024 * 1024

#define NUM_GROUPS 1
#define GROUP_SIZE 1
#define TEST_ITERS 100
#define DISPATCHES_PER_TEST 100

#define FILE_NAME "test_kernel.code"
#define KERNEL_NAME "test"

__global__ void EmptyKernel() { }

ResultDatabase resultDB;

void print_timing(const char* msg, const std::array<float, TEST_ITERS> &resultArray, int batchSize) {
    for ( auto it = std::next(resultArray.begin()); it != resultArray.end(); ++it ) {
        resultDB.AddResult(std::string(msg), "", "uS", *it * 1000 / batchSize);
        if (PRINT_PROGRESS & 0x1) {
            std::cout << msg << "\t\t" << *it * 1000 / batchSize << " uS" << std::endl;
        }
        if (PRINT_PROGRESS & 0x2) {
            resultDB.DumpSummary(std::cout);
        }
     }
}

int main() {
    std::array<float, TEST_ITERS> results;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipStream_t stream0 = 0;
    hipDevice_t device;
    hipDeviceGet(&device, 0);

    hipCtx_t context;     
    hipCtxCreate(&context, 0, device); 
    hipModule_t module;
    hipFunction_t function;
    check(hipModuleLoad(&module, FILE_NAME));
    check(hipModuleGetFunction(&function, module, KERNEL_NAME));
    void* kernel_params = nullptr;
    
    /************************************************************************************/
    /* HIP kernel launch enqueue rate:                                                  */
    /* Measure time taken to enqueue a kernel on the GPU                                */
    /************************************************************************************/ 

    // Timing hipModuleLaunchKernel
    for (auto i = 0; i < TEST_ITERS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, &kernel_params, nullptr);
        auto stop = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<double, std::milli>(stop - start).count();
    }
    print_timing("TiminghipModuleLaunchKernel", results, 1);

    // Timing hipLaunchKernelGGL
    for (auto i = 0; i < TEST_ITERS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0);
        auto stop = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<double, std::milli>(stop - start).count();
    }
    print_timing("TiminghipLaunchKernelGGL", results, 1);

    /***********************************************************************************/
    /* Single dispatch execution latency using HIP events:                             */   
    /* Measures time to start & finish executing a kernel with GPU-scope visibility    */ 
    /***********************************************************************************/
    //Timing directly the dispatch
#ifdef __HIP_PLATFORM_HCC__
    for (auto i = 0; i < TEST_ITERS; ++i) {
        hipEventRecord(start);
        hipExtLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0, start, stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&results[i], start, stop);
    }
    print_timing("TimingDirectDispatch", results, 1);
#endif
    //Timing around the dispatch
    for (auto i = 0; i < TEST_ITERS; ++i) {
        hipEventRecord(start, 0);
        hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0);
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&results[i], start, stop);
    }
    print_timing("TimingAroundDispatch", results, 1);

    /*********************************************************************************/
    /* Batch dispatch execution latency using HIP events:                            */
    /* Measures average time to start & finish executing each dispatch in a batch    */ 
    /*********************************************************************************/

    for (auto i = 0; i < TEST_ITERS; ++i) {
         hipEventRecord(start);
         for (int i = 0; i < DISPATCHES_PER_TEST; i++) {
             hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0);
         }
         hipEventRecord(stop, 0);
     hipEventSynchronize(stop);
         hipEventElapsedTime(&results[i], start, stop);
    }
    print_timing("TimingAroundBatch", results, DISPATCHES_PER_TEST);

    check(hipEventDestroy(start));
    check(hipEventDestroy(stop));
    check(hipCtxDestroy(context));
    resultDB.DumpSummary(std::cout);
}
