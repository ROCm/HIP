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
    syncMarkerThenOtherStream,
    syncMarkerThenOtherNonBlockingStream,
    syncDevice
};


const char* syncModeString(int syncMode) {
    switch (syncMode) {
        case syncNone:
            return "syncNone";
        case syncNullStream:
            return "syncNullStream";
        case syncOtherStream:
            return "syncOtherStream";
        case syncMarkerThenOtherStream:
            return "syncMarkerThenOtherStream";
        case syncMarkerThenOtherNonBlockingStream:
            return "syncMarkerThenOtherNonBlockingStream";
        case syncDevice:
            return "syncDevice";
        default:
            return "unknown";
    };
};


void test(unsigned testMask, int* C_d, int* C_h, int64_t numElements, SyncMode syncMode,
          bool expectMismatch) {
    // This test sends a long-running kernel to the null stream, then tests to see if the
    // specified synchronization technique is effective.
    //
    // Some syncMode are not expected to correctly sync (for example "syncNone").  in these
    // cases the test sets expectMismatch and the check logic below will attempt to ensure that
    // the undesired synchronization did not occur - ie ensure the kernel is still running and did
    // not yet update the stop event.  This can be tricky since if the kernel runs fast enough it
    // may complete before the check.  To prevent this, the addCountReverse has a count parameter
    // which causes it to loop repeatedly, and the results are checked in reverse order.
    //
    // Tests with expectMismatch=true should ensure the kernel finishes correctly. This results
    // are checked and we test to make sure stop event has completed.

    if (!(testMask & p_tests)) {
        return;
    }
    printf("\ntest 0x%02x: syncMode=%s expectMismatch=%d\n", testMask, syncModeString(syncMode),
           expectMismatch);

    size_t sizeBytes = numElements * sizeof(int);

    int count = 100;
    int init0 = 0;
    HIPCHECK(hipMemset(C_d, init0, sizeBytes));
    for (int i = 0; i < numElements; i++) {
        C_h[i] = -1;  // initialize
    }

    hipStream_t otherStream = 0;
    unsigned flags = (syncMode == syncMarkerThenOtherNonBlockingStream) ? hipStreamNonBlocking
                                                                        : hipStreamDefault;
    HIPCHECK(hipStreamCreateWithFlags(&otherStream, flags));
    hipEvent_t stop, otherStreamEvent;
    HIPCHECK(hipEventCreate(&stop));
    HIPCHECK(hipEventCreate(&otherStreamEvent));


    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);
    // Launch kernel into null stream, should result in C_h == count.
    hipLaunchKernelGGL(HipTest::addCountReverse, dim3(blocks), dim3(threadsPerBlock), 0,
                       0 /*stream*/, static_cast<const int*>(C_d), C_h, numElements, count);
    HIPCHECK(hipEventRecord(stop, 0 /*default*/));

    switch (syncMode) {
        case syncNone:
            break;
        case syncNullStream:
            HIPCHECK(hipStreamSynchronize(0));  // wait on host for null stream:
            break;
        case syncOtherStream:
            // Does this synchronize with the null stream?
            HIPCHECK(hipStreamSynchronize(otherStream));
            break;
        case syncMarkerThenOtherStream:
        case syncMarkerThenOtherNonBlockingStream:

            // this may wait for NULL stream depending hipStreamNonBlocking flag above
            HIPCHECK(hipEventRecord(otherStreamEvent, otherStream));

            HIPCHECK(hipStreamSynchronize(otherStream));
            break;
        case syncDevice:
            HIPCHECK(hipDeviceSynchronize());
            break;
        default:
            assert(0);
    };

    hipError_t done = hipEventQuery(stop);

    if (expectMismatch) {
        assert(done == hipErrorNotReady);
    } else {
        assert(done == hipSuccess);
    }

    int mismatches = 0;
    int expected = init0 + count;
    for (int i = 0; i < numElements; i++) {
        bool compareEqual = (C_h[i] == expected);
        if (!compareEqual) {
            mismatches++;
            if (!expectMismatch) {
                printf("C_h[%d] (%d) != %d\n", i, C_h[i], expected);
                assert(C_h[i] == expected);
            }
        }
    }

    if (expectMismatch) {
        assert(mismatches > 0);
    }


    HIPCHECK(hipStreamDestroy(otherStream));
    HIPCHECK(hipEventDestroy(stop));
    HIPCHECK(hipEventDestroy(otherStreamEvent));

    HIPCHECK(hipDeviceSynchronize());

    printf("test:   OK - %d mismatches (%6.2f%%)\n", mismatches,
           ((double)(mismatches)*100.0) / numElements);
}


void runTests(int64_t numElements) {
    size_t sizeBytes = numElements * sizeof(int);

    printf("\n\ntest: starting sequence with sizeBytes=%zu bytes, %6.2f MB\n", sizeBytes,
           sizeBytes / 1024.0 / 1024.0);


    int *C_h, *C_d;
    HIPCHECK(hipMalloc(&C_d, sizeBytes));
    HIPCHECK(hipHostMalloc(&C_h, sizeBytes));


    {
        test(0x01, C_d, C_h, numElements, syncNone, true /*expectMismatch*/);
        test(0x02, C_d, C_h, numElements, syncNullStream, false /*expectMismatch*/);
        test(0x04, C_d, C_h, numElements, syncOtherStream, true /*expectMismatch*/);
        test(0x08, C_d, C_h, numElements, syncDevice, false /*expectMismatch*/);

        // Sending a marker to to null stream may synchronize the otherStream
        //  - other created with hipStreamNonBlocking=0 : synchronization, should match
        //  - other created with hipStreamNonBlocking=1 : no synchronization, may mismatch
        test(0x10, C_d, C_h, numElements, syncMarkerThenOtherStream, false /*expectMismatch*/);

        // TODO - review why this test seems flaky
        // test (0x20, C_d, C_h, numElements,  syncMarkerThenOtherNonBlockingStream, true
        // /*expectMismatch*/);
    }


    HIPCHECK(hipFree(C_d));
    HIPCHECK(hipHostFree(C_h));
}


int main(int argc, char* argv[]) {
    // Can' destroy the default stream:// TODO - move to another test
    HIPCHECK_API(hipStreamDestroy(0), hipErrorInvalidResourceHandle);

    HipTest::parseStandardArguments(argc, argv, true /*failOnUndefinedArg*/);

    runTests(40000000);

    passed();
}
