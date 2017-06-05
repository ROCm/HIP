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
    syncStream,
    syncStopEvent,
};


const char *syncModeString(int syncMode) {
    switch (syncMode) {
        case syncNone:
            return "syncNone";
        case syncStream:
            return "syncStream";
        case syncStopEvent:
            return "syncStopEvent";
        default:
            return "unknown";
    };
};


void test(unsigned testMask, int *C_d, int *C_h, int64_t numElements, hipStream_t stream, int waitStart, SyncMode syncMode)
{
    if (!(testMask & p_tests)) {
        return;
    }
    printf ("\ntest 0x%3x: stream=%p waitStart=%d syncMode=%s\n", 
            testMask, stream, waitStart, syncModeString(syncMode));

    size_t sizeBytes = numElements * sizeof(int);

    int count =100;
    int init0 = 0;
    HIPCHECK(hipMemset(C_d, init0, sizeBytes));
    for (int i=0; i<numElements; i++) {
        C_h[i] = -1; // initialize
    }

    hipEvent_t neverCreated=0, neverRecorded, timingDisabled;
    HIPCHECK(hipEventCreate(&neverRecorded));
    HIPCHECK(hipEventCreateWithFlags(&timingDisabled, hipEventDisableTiming));

    hipEvent_t start, stop;
    HIPCHECK(hipEventCreate(&start));
    HIPCHECK(hipEventCreate(&stop));

    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

    HIPCHECK(hipEventRecord(timingDisabled, stream));
    // sandwhich a kernel:
    HIPCHECK(hipEventRecord(start, stream));
    hipLaunchKernelGGL(HipTest::addCountReverse , dim3(blocks), dim3(threadsPerBlock), 0, stream,    C_d, C_h, numElements, count);
    HIPCHECK(hipEventRecord(stop, stream));


    if (waitStart) {
        HIPCHECK(hipEventSynchronize(start));
    }

   
    hipError_t expectedStopError = hipSuccess; 

    // How to wait for the events to finish:
    switch (syncMode) {
        case syncNone:
            expectedStopError = hipErrorNotReady;
            break;
        case syncStream:
            HIPCHECK(hipStreamSynchronize(stream));  // wait for recording to finish...
            break;
        case syncStopEvent:
            HIPCHECK(hipEventSynchronize(stop)); 
            break;
        default:
            assert(0);
    };
            

    float t;

    hipError_t e = hipEventElapsedTime(&t, start, start);
    if ((e != hipSuccess) && (e != hipErrorNotReady))  {
        failed ("start event not in expected state, was %d=%s\n", e, hipGetErrorName(e));
    }

    if (e == hipSuccess) 
        assert (t==0.0f);
        

    // stop usually ready unless we skipped the synchronization (syncNone)
    HIPCHECK_API(hipEventElapsedTime(&t, stop, stop), expectedStopError);
    if (e == hipSuccess) 
        assert (t==0.0f);


    e = hipEventElapsedTime(&t, start, stop);
    HIPCHECK_API(e, expectedStopError);
    if (expectedStopError == hipSuccess) 
        assert (t>0.0f);
    printf ("time=%6.2f error=%s\n", t, hipGetErrorName(e));

    e = hipEventElapsedTime(&t, stop, start);
    HIPCHECK_API(e, expectedStopError);
    if (expectedStopError == hipSuccess) 
        assert (t<0.0f);
    printf ("negtime=%6.2f error=%s\n", t, hipGetErrorName(e));



    {
        // Check some error conditions for incomplete events:
        HIPCHECK_API(hipEventElapsedTime(&t, timingDisabled, stop), hipErrorInvalidResourceHandle);
        HIPCHECK_API(hipEventElapsedTime(&t, start, timingDisabled), hipErrorInvalidResourceHandle);

        HIPCHECK_API(hipEventElapsedTime(&t, neverCreated, stop), hipErrorInvalidResourceHandle);
        HIPCHECK_API(hipEventElapsedTime(&t, start, neverCreated),  hipErrorInvalidResourceHandle);

        HIPCHECK_API(hipEventElapsedTime(&t, neverRecorded, stop), hipErrorInvalidResourceHandle);
        HIPCHECK_API(hipEventElapsedTime(&t, start, neverRecorded),  hipErrorInvalidResourceHandle);
    }

    HIPCHECK(hipEventDestroy(start));
    HIPCHECK(hipEventDestroy(stop));

    // Clear out everything:
    HIPCHECK(hipDeviceSynchronize());

    printf ("test:   OK  \n");
}



void runTests(int64_t numElements)
{
    size_t sizeBytes = numElements * sizeof(int);

    printf ("test: starting sequence with sizeBytes=%zu bytes, %6.2f MB\n", sizeBytes, sizeBytes/1024.0/1024.0);


    int *C_h, *C_d;
    HIPCHECK(hipMalloc(&C_d, sizeBytes));
    HIPCHECK(hipHostMalloc(&C_h, sizeBytes));

    hipStream_t stream;
    HIPCHECK(hipStreamCreateWithFlags(&stream, 0x0));

    //for (int waitStart=0; waitStart<2; waitStart++) {
    for (int waitStart=1; waitStart>=0; waitStart--) {
        unsigned W = waitStart ? 0x1000:0;
        test (W | 0x01, C_d, C_h, numElements,  0     , waitStart, syncNone);
        test (W | 0x02, C_d, C_h, numElements,  stream, waitStart, syncNone);
        test (W | 0x04, C_d, C_h, numElements,  0     , waitStart, syncStream);
        test (W | 0x08, C_d, C_h, numElements,  stream, waitStart, syncStream);
        test (W | 0x10, C_d, C_h, numElements,  0,      waitStart, syncStopEvent);
        test (W | 0x20, C_d, C_h, numElements,  stream, waitStart, syncStopEvent);
    }


    HIPCHECK(hipStreamDestroy(stream));
    HIPCHECK(hipFree(C_d));
    HIPCHECK(hipHostFree(C_h));
}


int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, true /*failOnUndefinedArg*/);

    runTests(80000000);

    passed();
}
