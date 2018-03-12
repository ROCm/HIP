/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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
// Simple test for memset.
// Also serves as a template for other tests.

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS --std=c++11
 * RUN: %t EXCLUDE_HIP_PLATFORM all
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

#ifdef __HIP_PLATFORM_HCC__
#include <hc_am.hpp>
#endif

#define USE_HCC_MEMTRACKER 0 /* Debug flag to show the memtracker periodically */


int elementSizes[] = {1, 16, 1024, 524288, 16 * 1000 * 1000};
int nSizes = sizeof(elementSizes) / sizeof(int);

int enablePeers(int dev0, int dev1) {
    int canAccessPeer01, canAccessPeer10;
    HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer01, dev0, dev1));
    HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer10, dev1, dev0));
    if (!canAccessPeer01 || !canAccessPeer10) {
        return -1;
    }

    HIPCHECK(hipSetDevice(dev0));
    HIPCHECK(hipDeviceEnablePeerAccess(dev1, 0 /*flags*/));
    HIPCHECK(hipSetDevice(dev1));
    HIPCHECK(hipDeviceEnablePeerAccess(dev0, 0 /*flags*/));

    return 0;
};

// Set value of array to specified 32-bit integer:
__global__ void memsetIntKernel(int* ptr, const int val, size_t numElements) {
    int gid = (blockIdx.x * blockDim.x + threadIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (size_t i = gid; i < numElements; i += stride) {
        ptr[i] = val;
    }
};

__global__ void memcpyIntKernel(const int* src, int* dst, size_t numElements) {
    int gid = (blockIdx.x * blockDim.x + threadIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (size_t i = gid; i < numElements; i += stride) {
        dst[i] = src[i];
    }
};


// CHeck arrays in reverse order, to more easily detect cases where
// the copy is "partially" done.
void checkReverse(const int* ptr, int numElements, int expected) {
    for (int i = numElements - 1; i >= 0; i--) {
        if (ptr[i] != expected) {
            printf("i=%d, ptr[](%d) != expected (%d)\n", i, ptr[i], expected);
            assert(ptr[i] == expected);
        }
    }

    printf("test:   OK\n");
}


void runTestImpl(bool stepAIsCopy, bool hostSync, hipStream_t gpu0Stream, hipStream_t gpu1Stream,
                 int numElements, int* dataGpu0_0, int* dataGpu0_1, int* dataGpu1, int* dataHost,
                 int expected) {
    hipEvent_t e;
    if (!hostSync) {
        HIPCHECK(hipEventCreateWithFlags(&e, 0));
    }
    const size_t sizeElements = numElements * sizeof(int);
    printf("test: runTestImpl with %zu bytes %s with hostSync %s\n", sizeElements,
           stepAIsCopy ? "copy" : "kernel", hostSync ? "enabled" : "disabled");

    hipStream_t stepAStream = gpu0Stream;

    if (stepAIsCopy) {
        HIPCHECK(hipMemcpyAsync(dataGpu1, dataGpu0_0, sizeElements, hipMemcpyDeviceToDevice,
                                stepAStream));
    } else {
        unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);
        hipLaunchKernelGGL(memcpyIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, gpu0Stream,
                           dataGpu0_0, dataGpu1, numElements);
    }

    if (!hostSync) {
        HIPCHECK(hipEventRecord(e, stepAStream));
        HIPCHECK(hipStreamWaitEvent(gpu1Stream, e, 0));
    } else {
        HIPCHECK(hipStreamSynchronize(stepAStream));
    }

    HIPCHECK(
        hipMemcpyAsync(dataGpu0_1, dataGpu1, sizeElements, hipMemcpyDeviceToDevice, gpu1Stream));

    if (!hostSync) {
        HIPCHECK(hipEventRecord(e, gpu1Stream));
    } else {
        HIPCHECK(hipStreamSynchronize(gpu1Stream));
    }

    HIPCHECK(hipMemcpyAsync(dataHost, dataGpu0_1, sizeElements, hipMemcpyDeviceToHost, gpu0Stream));
    HIPCHECK(hipStreamSynchronize(gpu0Stream));

    checkReverse(dataHost, numElements, expected);
    if (!hostSync) {
        HIPCHECK(hipEventDestroy(e));
    }
}

void testMultiGpu(int dev0, int dev1, int numElements, bool hostSync) {
    const size_t sizeElements = numElements * sizeof(int);

    int *dataGpu0_0, *dataGpu0_1, *dataGpu1, *dataHost;
    hipStream_t gpu0Stream, gpu1Stream;
    const int expected = 42;
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

    HIPCHECK(hipSetDevice(dev0));

    HIPCHECK(hipMalloc(&dataGpu0_0, sizeElements));
    HIPCHECK(hipMalloc(&dataGpu0_1, sizeElements));
    HIPCHECK(hipStreamCreate(&gpu0Stream));
    hipLaunchKernelGGL(memsetIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, gpu0Stream,
                       dataGpu0_0, expected, numElements);
    HIPCHECK(hipDeviceSynchronize());


    HIPCHECK(hipSetDevice(dev1));
    HIPCHECK(hipMalloc(&dataGpu1, sizeElements));
    HIPCHECK(hipStreamCreate(&gpu1Stream));
    hipLaunchKernelGGL(memsetIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, gpu0Stream,
                       dataGpu1, 0x34, numElements);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipHostMalloc(&dataHost, sizeElements));
    memset(dataHost, 13, sizeElements);

#if USE_HCC_MEMTRACKER
    hc::am_memtracker_print(0x0);
#endif

    printf("  test: init complete\n");
    runTestImpl(true, hostSync, gpu0Stream, gpu1Stream, numElements, dataGpu0_0, dataGpu0_1,
                dataGpu1, dataHost, expected);

    HIPCHECK(hipFree(dataGpu0_0));
    HIPCHECK(hipFree(dataGpu0_1));
    HIPCHECK(hipFree(dataGpu1));
    HIPCHECK(hipHostFree(dataHost));

    HIPCHECK(hipStreamDestroy(gpu0Stream));
    HIPCHECK(hipStreamDestroy(gpu1Stream));
};

int main(int argc, char* argv[]) {
    HipTest::parseStandardArguments(argc, argv, true);


    int dev0 = 0;
    int dev1 = 1;

    int numDevices;
    HIPCHECK(hipGetDeviceCount(&numDevices));
    if (numDevices == 1) {
        printf("warning : test requires atleast two gpus\n");
        passed();
    }

    if (enablePeers(dev0, dev1) == -1) {
        printf("warning : could not find peer gpus\n");
        return -1;
    };

    for (int index = 0; index < nSizes; index++) {
        testMultiGpu(dev0, dev1, elementSizes[index], false /*GPU Synchronization*/);
        testMultiGpu(dev0, dev1, elementSizes[index], true /*Host Synchronization*/);
    }


    passed();
};
