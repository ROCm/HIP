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
 * RUN: %t 
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

#ifdef __HIP_PLATFORM_HCC__
#include <hc_am.hpp>
#endif

#define USE_HSA_COPY 1

int enablePeers(int dev0, int dev1)
{
    int canAccessPeer01, canAccessPeer10;
    HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer01, dev0, dev1));
    HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer10, dev1, dev0));
    if (!canAccessPeer01 || !canAccessPeer10) {
        return -1;
    }

    HIPCHECK(hipSetDevice(dev0));
    HIPCHECK(hipDeviceEnablePeerAccess(dev1, 0/*flags*/));
    HIPCHECK(hipSetDevice(dev1));
    HIPCHECK(hipDeviceEnablePeerAccess(dev0, 0/*flags*/));

    return 0;
};


__global__ void
memsetIntKernel(int * ptr, int val, size_t numElements)
{
    int gid = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    if (gid < numElements) {
        ptr[gid] = val;
    }
};


void checkReverse(const int *ptr, int numElements, int expected) {
    for (int i=numElements-1; i>=0; i--) {
        if (ptr[i] != expected) {
            printf ("i=%d, ptr[](%d) != expected (%d)\n", i, ptr[i], expected);
            assert (ptr[i] == expected);
        }
    }

    printf ("test:   OK\n");
}


void runTest(bool stepAIsCopy, hipStream_t gpu0Stream, hipStream_t gpu1Stream, int numElements,
             int * dataGpu0, int *dataGpu1, int *dataHost, int expected)
{
    hipEvent_t e;
    HIPCHECK(hipEventCreateWithFlags(&e,0));

    printf ("test: runTest with %s\n", stepAIsCopy ? "copy" : "kernel");
    const size_t sizeElements = numElements * sizeof(int);

    hipStream_t stepAStream = gpu0Stream;

    if (stepAIsCopy) {
#ifdef USE_HSA_COPY
        HIPCHECK(hipMemcpyAsync(dataGpu1, dataGpu0, sizeElements, hipMemcpyDeviceToDevice, stepAStream));
#endif
    } else {
        assert(0); // not yet supported.
    }

    HIPCHECK(hipEventRecord(e, stepAStream));
    HIPCHECK(hipStreamWaitEvent(gpu1Stream, e, 0));

    HIPCHECK(hipMemcpyAsync(dataHost, dataGpu1, sizeElements, hipMemcpyDeviceToHost, gpu1Stream));

    HIPCHECK(hipStreamSynchronize(gpu1Stream));

    checkReverse(dataHost, numElements, expected);
}


void testMultiGpu0(int dev0, int dev1, int numElements)
{
    const size_t sizeElements = numElements * sizeof(int);

    int * dataGpu0, *dataGpu1, *dataHost;
    hipStream_t gpu0Stream, gpu1Stream;
    const int expected = 42;
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

    HIPCHECK(hipSetDevice(dev0));

    HIPCHECK(hipMalloc(&dataGpu0, sizeElements));
    HIPCHECK(hipStreamCreate(&gpu0Stream));
    hipLaunchKernelGGL(memsetIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, gpu0Stream,
                       dataGpu0, expected, numElements);
    HIPCHECK(hipDeviceSynchronize());


    HIPCHECK(hipSetDevice(dev1));
    HIPCHECK(hipMalloc(&dataGpu1, sizeElements));
    HIPCHECK(hipStreamCreate(&gpu1Stream));
    hipLaunchKernelGGL(memsetIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, gpu0Stream,
                       dataGpu1, 0x34, numElements);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipHostMalloc(&dataHost, sizeElements));
    memset(dataHost, 13, sizeElements);

#ifdef __HIP_PLATFORM_HCC__
    hc::am_memtracker_print(0x0);
#endif
    
    printf ("  test: init complete\n");

    runTest(true/*stepAIsCopy*/, gpu0Stream, gpu1Stream, numElements, dataGpu0, dataGpu1, dataHost, expected);

};



int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, true);

    int numElements = N;

    int dev0 = 0;
    int dev1 = 1;

    int numDevices;
    HIPCHECK(hipGetDeviceCount(&numDevices));
    if (numDevices == 1) {
        printf("warning : test requires atleast two gpus\n");
        passed();
    }

    if (enablePeers(dev0,dev1) == -1) {
        printf ("warning : could not find peer gpus\n");
        return -1;
    };

    //testMultiGpu0(dev0, dev1, numElements);



    passed();
};
