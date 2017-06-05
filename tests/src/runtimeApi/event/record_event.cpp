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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * HIT_END
 */


#include "test_common.h"

enum SyncMode {
    syncNone,
    syncNullStream,
    syncOtherStream,
};


const char *syncModeString(int syncMode) {
    switch (syncMode) {
        case syncNone:
            return "syncNone";
        case syncNullStream:
            return "syncNullStream";
        case syncOtherStream:
            return "syncOtherStream";
        default:
            return "unknown";
    };
};


void test(int *C_d, int *C_h, int64_t numElements, SyncMode syncMode)
{
    printf ("\ntest: syncMode=%s\n", syncModeString(syncMode));

    size_t sizeBytes = numElements * sizeof(int);

    int count =100;
    int init0 = 0;
    HIPCHECK(hipMemset(C_d, init0, sizeBytes));
    for (int i=0; i<numElements; i++) {
        C_h[i] = -1; // initialize
    }

    hipStream_t stream = 0;

    unsigned flags=0;
    if (syncMode == syncOtherStream) {
        HIPCHECK(hipStreamCreateWithFlags(&stream, flags));
    }

    hipEvent_t neverCreated=0;
    hipEvent_t start, stop, neverRecorded;
    HIPCHECK(hipEventCreate(&start));
    HIPCHECK(hipEventCreate(&stop));
    HIPCHECK(hipEventCreate(&neverRecorded));

    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

    // sandwhich a kernel:
    HIPCHECK(hipEventRecord(start, stream));
    hipLaunchKernelGGL(HipTest::addCountReverse , dim3(blocks), dim3(threadsPerBlock), 0, stream,    C_d, C_h, numElements, count);
    HIPCHECK(hipEventRecord(stop, stream));

    HIPCHECK(hipStreamSynchronize(stream));  // wait for recording to finish...

    float t;
    HIPCHECK_API(hipEventElapsedTime(&t, neverCreated, stop), hipErrorInvalidResourceHandle);
    HIPCHECK_API(hipEventElapsedTime(&t, start, neverCreated),  hipErrorInvalidResourceHandle);

    HIPCHECK_API(hipEventElapsedTime(&t, neverRecorded, stop), hipErrorInvalidResourceHandle);
    HIPCHECK_API(hipEventElapsedTime(&t, start, neverRecorded),  hipErrorInvalidResourceHandle);

    HIPCHECK(hipEventElapsedTime(&t, start, stop));
    assert (t>0.0f);
    printf ("time=%6.2f\n", t);

    HIPCHECK(hipEventElapsedTime(&t, stop, start));
    assert (t<0.0f);
    printf ("negtime=%6.2f\n", t);

    HIPCHECK(hipEventElapsedTime(&t, start, start));
    assert (t==0.0f);
    HIPCHECK(hipEventElapsedTime(&t, stop, stop));
    assert (t==0.0f);


    if (stream) {
        HIPCHECK(hipStreamDestroy(stream));
    }
    HIPCHECK(hipEventDestroy(start));
    HIPCHECK(hipEventDestroy(stop));

    printf ("test:   OK  \n");
}



void runTests(int64_t numElements)
{
    size_t sizeBytes = numElements * sizeof(int);

    printf ("test: starting sequence with sizeBytes=%zu bytes, %6.2f MB\n", sizeBytes, sizeBytes/1024.0/1024.0);


    int *C_h, *C_d;
    HIPCHECK(hipMalloc(&C_d, sizeBytes));
    HIPCHECK(hipHostMalloc(&C_h, sizeBytes));


    {
        test (C_d, C_h, numElements,  syncNone);
        test (C_d, C_h, numElements,  syncNullStream);
        test (C_d, C_h, numElements,  syncOtherStream);
        //test (C_d, C_h, numElements,  syncDevice);
    }


    HIPCHECK(hipFree(C_d));
    HIPCHECK(hipHostFree(C_h));
}


int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, true /*failOnUndefinedArg*/);

    runTests(4000000);

    passed();
}
