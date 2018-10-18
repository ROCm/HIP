/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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
#include <iostream>
#include <time.h>
#include "ResultDatabase.h"

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
#define GROUP_SIZE 64
#define TEST_ITERS 20
#define DISPATCHES_PER_TEST 100

const unsigned p_tests = 0xfffffff;


// HCC optimizes away fully NULL kernel calls, so run one that is nearly null:
__global__ void NearlyNull(float* Ad) {
    if (Ad) {
        Ad[0] = 42;
    }
}


ResultDatabase resultDB;


void stopTest(hipEvent_t start, hipEvent_t stop, const char* msg, int iters) {
    float mS = 0;
    check(hipEventRecord(stop));
    check(hipDeviceSynchronize());
    check(hipEventElapsedTime(&mS, start, stop));
    resultDB.AddResult(std::string(msg), "", "uS", mS * 1000 / iters);
    if (PRINT_PROGRESS & 0x1) {
        std::cout << msg << "\t\t" << mS * 1000 / iters << " uS" << std::endl;
    }
    if (PRINT_PROGRESS & 0x2) {
        resultDB.DumpSummary(std::cout);
    }
}


int main() {
    hipError_t err;
    float* Ad;
    check(hipMalloc(&Ad, 4));


    hipStream_t stream;
    check(hipStreamCreate(&stream));


    hipEvent_t start, sync, stop;
    check(hipEventCreate(&start));
    check(hipEventCreateWithFlags(&sync, hipEventBlockingSync));
    check(hipEventCreate(&stop));


    hipStream_t stream0 = 0;


    if (p_tests & 0x1) {
        hipEventRecord(start);
        hipLaunchKernelGGL(NearlyNull, dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0, Ad);
        stopTest(start, stop, "FirstKernelLaunch", 1);
    }


    if (p_tests & 0x2) {
        hipEventRecord(start);
        hipLaunchKernelGGL(NearlyNull, dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0, Ad);
        stopTest(start, stop, "SecondKernelLaunch", 1);
    }


    if (p_tests & 0x4) {
        for (int t = 0; t < TEST_ITERS; t++) {
            hipEventRecord(start);
            for (int i = 0; i < DISPATCHES_PER_TEST; i++) {
                hipLaunchKernelGGL(NearlyNull, dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0, Ad);
                hipEventRecord(sync);
                hipEventSynchronize(sync);
            }
            stopTest(start, stop, "NullStreamASyncDispatchWait", DISPATCHES_PER_TEST);
        }
    }


    if (p_tests & 0x10) {
        for (int t = 0; t < TEST_ITERS; t++) {
            hipEventRecord(start);
            for (int i = 0; i < DISPATCHES_PER_TEST; i++) {
                hipLaunchKernelGGL(NearlyNull, dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream, Ad);
                hipEventRecord(sync);
                hipEventSynchronize(sync);
            }
            stopTest(start, stop, "StreamASyncDispatchWait", DISPATCHES_PER_TEST);
        }
    }

#if 1

    if (p_tests & 0x40) {
        for (int t = 0; t < TEST_ITERS; t++) {
            hipEventRecord(start);
            for (int i = 0; i < DISPATCHES_PER_TEST; i++) {
                hipLaunchKernelGGL(NearlyNull, dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream0, Ad);
            }
            stopTest(start, stop, "NullStreamASyncDispatchNoWait", DISPATCHES_PER_TEST);
        }
    }

    if (p_tests & 0x80) {
        for (int t = 0; t < TEST_ITERS; t++) {
            hipEventRecord(start);
            for (int i = 0; i < DISPATCHES_PER_TEST; i++) {
                hipLaunchKernelGGL(NearlyNull, dim3(NUM_GROUPS), dim3(GROUP_SIZE), 0, stream, Ad);
            }
            stopTest(start, stop, "StreamASyncDispatchNoWait", DISPATCHES_PER_TEST);
        }
    }
#endif
    resultDB.DumpSummary(std::cout);


    check(hipEventDestroy(start));
    check(hipEventDestroy(sync));
    check(hipEventDestroy(stop));
}
