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
#ifdef __HIP_PLATFORM_AMD__
#include "hip/hip_ext.h"
#endif
#include <iostream>
#include <chrono>
#include <algorithm>

#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#define NUM_GROUPS 1
#define GROUP_SIZE 1
#define WARMUP_RUN_COUNT 100
#define TIMING_RUN_COUNT 1000
#define TOTAL_RUN_COUNT WARMUP_RUN_COUNT + TIMING_RUN_COUNT
#define BATCH_SIZE 1000

#define FILE_NAME "test_kernel.code"
#define KERNEL_NAME "test"

#define HIPCHECK(error)                                                         \
{                                                                               \
  hipError_t localError = error;                                                \
  if (localError != hipSuccess)  {                                              \
    printf("error: '%s'(%d) from %s at %s:%d\n",hipGetErrorString(localError),  \
                   localError, #error, __FILE__, __LINE__);                     \
    fflush(NULL);                                                               \
    abort();                                                                    \
  }                                                                             \
}
__global__ void EmptyKernel() { }

class CSVDump
{
    std::string fName;
    std::string delimeter;
    int linesCount;
public:
    CSVDump(std::string filename, std::string delm = ",") :
            fName(filename), delimeter(delm), linesCount(1)
    {}
    template<typename T>
    void addRow(std::string test, T first, T last);
    void addStats(std::string test, float mean, float std, float min, float max);
};
template<typename T>
void CSVDump::addRow(std::string test, T first, T last)
{
    std::fstream file;
    file.open(fName, std::ios::out | (linesCount ? std::ios::app : std::ios::trunc));
    file << test;
    file <<delimeter;
    for (; first != last; )
    {
        file << *first;
        if (++first != last)
            file << delimeter;
    }
    file << "\n";
    linesCount++;
    file.close();
}

void CSVDump::addStats(std::string test, float mean, float std, float min, float max)
{
    std::fstream file;
    file.open(fName, std::ios::out | (linesCount ? std::ios::app : std::ios::trunc));
    file << test;
    file <<delimeter;
    file.precision(3);
    file << mean <<delimeter <<std<<delimeter<<min<<delimeter<<max;
    file << "\n";
    linesCount++;
    file.close();
}

void print_timing(std::string test, const std::array<float, TOTAL_RUN_COUNT> &results, int batch = 1) {
    CSVDump writer("LaunchLatency.csv");
    float total_us = 0.0f, mean_us = 0.0f, stddev_us = 0.0f, min_us = 0.0f, max_us = 0.0f;

    // skip warm-up runs
    auto start_iter = std::next(results.begin(), WARMUP_RUN_COUNT);
    auto end_iter = results.end();
    //writer.addRow(test, start_iter, end_iter);
    // mean
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    std::for_each(start_iter, end_iter, [&](const float &run_ms) {
        total_us += (run_ms * 1000) / batch;
        min = std::min(run_ms, min);
        max = std::max(run_ms, max);
    });
    mean_us = total_us  / TIMING_RUN_COUNT;
    min_us = (min *1000) / batch;
    max_us = (max *1000) / batch;

    // stddev
    total_us = 0;
    std::for_each(start_iter, end_iter, [&](const float &run_ms) {
        float dev_us = ((run_ms * 1000) / batch) - mean_us;
        total_us += dev_us * dev_us;
    });
    stddev_us = sqrt(total_us / TIMING_RUN_COUNT);

    writer.addStats(test, mean_us, stddev_us, min_us, max_us);
    // display
    printf("\n %s: %.1f us, std: %.1f us max: %.1f us min:%.1f us\n", test.c_str(), mean_us, stddev_us, max_us, min_us);
}

int main() {
    hipStream_t stream0 = 0;
    hipDevice_t device;
    HIPCHECK(hipDeviceGet(&device, 0));
    hipCtx_t context;
    HIPCHECK(hipCtxCreate(&context, 0, device));
    hipModule_t module;
    hipFunction_t function;
    HIPCHECK(hipModuleLoad(&module, FILE_NAME));
    HIPCHECK(hipModuleGetFunction(&function, module, KERNEL_NAME));
    void* params = nullptr;

    std::array<float, TOTAL_RUN_COUNT> results;
    hipEvent_t start, stop;
    HIPCHECK(hipEventCreate(&start));
    HIPCHECK(hipEventCreate(&stop));

    /************************************************************************************/
    /* HIP kernel launch enqueue rate:                                                  */
    /* Measure time taken to enqueue a kernel on the GPU                                */
    /************************************************************************************/

    // Timing hipModuleLaunchKernel
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        HIPCHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, &params, nullptr));
        auto stop = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<float, std::milli>(stop - start).count();
    }
    print_timing("hipModuleLaunchKernel enqueue time", results);

    HIPCHECK(hipDeviceSynchronize());

    // Timing hipLaunchKernelGGL
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0);
        auto stop = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<float, std::milli>(stop - start).count();
    }
    print_timing("hipLaunchKernelGGL enqueue time", results);

    HIPCHECK(hipDeviceSynchronize());

#ifdef __HIP_PLATFORM_AMD__
    //Timing hipExtLaunchKernelGGL
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start_chrono = std::chrono::high_resolution_clock::now();
        hipExtLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0, start, stop, 0);
        auto stop_chrono = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<float, std::milli>(stop_chrono - start_chrono).count();
    }
    print_timing("hipExtLaunchKernelGGL enqueue time", results);

    HIPCHECK(hipDeviceSynchronize());
#endif

    //Timing hipExtLaunchKernelGGL
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start_chrono = std::chrono::high_resolution_clock::now();
        hipExtLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0, nullptr, nullptr, 0);
        auto stop_chrono = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<float, std::milli>(stop_chrono - start_chrono).count();
    }
    print_timing("hipExtLaunchKernelGGL w/o events enqueue time", results);

    HIPCHECK(hipDeviceSynchronize());

    /***********************************************************************************/
    /* Single dispatch execution latency using HIP events:                             */
    /* Measures latency to start & finish executing a kernel with GPU-scope visibility    */
    /***********************************************************************************/

    //Timing directly the dispatch
#ifdef __HIP_PLATFORM_AMD__
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        hipExtLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0, start, stop, 0);
        HIPCHECK(hipEventSynchronize(stop));
        HIPCHECK(hipEventElapsedTime(&results[i], start, stop));
    }
    print_timing("Timing directly single dispatch latency", results);
#endif

    //Timing around the dispatch
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        HIPCHECK(hipEventRecord(start, 0));
        hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0);
        HIPCHECK(hipEventRecord(stop, 0));
        HIPCHECK(hipEventSynchronize(stop));
        HIPCHECK(hipEventElapsedTime(&results[i], start, stop));
    }
    print_timing("Timing around single dispatch latency", results);

    //Timing around the dispatch
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start_chrono = std::chrono::high_resolution_clock::now();
        HIPCHECK(hipEventRecord(start, 0));
        hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0);
        HIPCHECK(hipEventRecord(stop, 0));
        HIPCHECK(hipEventSynchronize(stop));
        auto stop_chrono = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<float, std::milli>(stop_chrono - start_chrono).count();
    }
    print_timing("Wall timing around single dispatch with events", results);

    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start_chrono = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0);
        HIPCHECK(hipStreamSynchronize(stream0));
        auto stop_chrono = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<float, std::milli>(stop_chrono - start_chrono).count();
    }
    print_timing("Wall timing around single dispatch without events", results);

#ifdef __HIP_PLATFORM_AMD__
    //Timing around the dispatch with hipExtLaunchKernelGGL
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start_chrono = std::chrono::high_resolution_clock::now();
        HIPCHECK(hipEventRecord(start, 0));
        hipExtLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0, start, stop, 0);
        HIPCHECK(hipEventSynchronize(stop));
        auto stop_chrono = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<float, std::milli>(stop_chrono - start_chrono).count();
    }
    print_timing("Wall timing around single dispatch ExtLaunch with events", results);

    //Timing around the dispatch with hipExtLaunchKernelGGL without events
    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start_chrono = std::chrono::high_resolution_clock::now();
        hipExtLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0, nullptr, nullptr, 0);
        HIPCHECK(hipStreamSynchronize(stream0));
        auto stop_chrono = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<float, std::milli>(stop_chrono - start_chrono).count();
    }
    print_timing("Wall timing around single dispatch ExtLaunch w/o events", results);
#endif

    /*********************************************************************************/
    /* Batch dispatch execution latency using HIP events:                            */
    /* Measures latency to start & finish executing each dispatch in a batch    */
    /*********************************************************************************/

    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        HIPCHECK(hipEventRecord(start, 0));
        for (int j = 0; j < BATCH_SIZE; j++) {
            hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0);
        }
        HIPCHECK(hipEventRecord(stop, 0));
        HIPCHECK(hipEventSynchronize(stop));
        HIPCHECK(hipEventElapsedTime(&results[i], start, stop));
    }
    print_timing("Batch dispatch latency", results, BATCH_SIZE);

    for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
        auto start_chrono = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < BATCH_SIZE; j++) {
            hipLaunchKernelGGL((EmptyKernel), dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0);
        }
        HIPCHECK(hipStreamSynchronize(stream0));
        auto stop_chrono = std::chrono::high_resolution_clock::now();
        results[i] = std::chrono::duration<float, std::milli>(stop_chrono - start_chrono).count();
    }
    print_timing("Wall timing for batch dispatch latency", results, BATCH_SIZE);

    HIPCHECK(hipEventDestroy(start));
    HIPCHECK(hipEventDestroy(stop));
    HIPCHECK(hipCtxDestroy(context));
}
