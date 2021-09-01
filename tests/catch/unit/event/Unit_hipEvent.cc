/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

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
// Tests hipEventRecord and hipEventElapsedTime with different scenarios
// and confirms if these are working as expected
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#include <hip_test_common.hh>

int tests = -1;
enum SyncMode {
    syncNone,
    syncStream,
    syncStopEvent,
};

const char* syncModeString(int syncMode) {
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

void test(unsigned testMask, int* C_d, int* C_h, int64_t numElements, hipStream_t stream,
          int waitStart, SyncMode syncMode) {
    if (!(testMask & tests)) {
        return;
    }
    std::cout << "\ntest " << std::showbase << std::hex << testMask << ": stream=" << stream
         << " waitStart=" << waitStart << " syncMode=" << syncModeString(syncMode) << std::endl;

    size_t sizeBytes = numElements * sizeof(int);

    int count = 100;
    int init0 = 0;
    HIP_CHECK(hipMemset(C_d, init0, sizeBytes));
    for (int i = 0; i < numElements; i++) {
        C_h[i] = -1;  // initialize
    }

    hipEvent_t neverCreated = 0, neverRecorded, timingDisabled;
    HIP_CHECK(hipEventCreate(&neverRecorded));
    HIP_CHECK(hipEventCreateWithFlags(&timingDisabled, hipEventDisableTiming));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    unsigned blocksPerCU = 6;
    unsigned threadsPerBlock = 256;

    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

    HIP_CHECK(hipEventRecord(timingDisabled, stream));
    // sandwhich a kernel:
    HIP_CHECK(hipEventRecord(start, stream));
    hipLaunchKernelGGL(HipTest::addCountReverse, dim3(blocks), dim3(threadsPerBlock), 0, stream,
                       static_cast<const int*>(C_d), C_h, numElements, count);
    HIP_CHECK(hipEventRecord(stop, stream));

    if (waitStart) {
        HIP_CHECK(hipEventSynchronize(start));
    }

    hipError_t expectedStopError = hipSuccess;

    // How to wait for the events to finish:
    switch (syncMode) {
        case syncNone:
            expectedStopError = hipErrorNotReady;
            break;
        case syncStream:
            HIP_CHECK(hipStreamSynchronize(stream));  // wait for recording to finish...
            break;
        case syncStopEvent:
            HIP_CHECK(hipEventSynchronize(stop));
            break;
        default:
            assert(0);
    };

    float t;

    hipError_t e = hipEventElapsedTime(&t, start, start);
    if ((e != hipSuccess) && (e != hipErrorNotReady || syncMode != syncNone)) {
        printf("start event not in expected state, was %d=%s\n", e, hipGetErrorName(e));
        REQUIRE(false);
    }

    if (e == hipSuccess) assert(t == 0.0f);

    // stop usually ready unless we skipped the synchronization (syncNone)
    HIP_ASSERT(hipEventElapsedTime(&t, stop, stop) == expectedStopError);
    if (e == hipSuccess) assert(t == 0.0f);

    e = hipEventElapsedTime(&t, start, stop);
    HIP_ASSERT(e == expectedStopError);
    if (expectedStopError == hipSuccess) assert(t > 0.0f);
    printf("time=%6.2f error=%s\n", t, hipGetErrorName(e));

    e = hipEventElapsedTime(&t, stop, start);
    HIP_ASSERT(e == expectedStopError);
    if (expectedStopError == hipSuccess) assert(t < 0.0f);
    printf("negtime=%6.2f error=%s\n", t, hipGetErrorName(e));

    {
        // Check some error conditions for incomplete events:
        HIP_ASSERT(hipEventElapsedTime(&t, timingDisabled, stop) == hipErrorInvalidHandle);
        HIP_ASSERT(hipEventElapsedTime(&t, start, timingDisabled) == hipErrorInvalidHandle);

        HIP_ASSERT(hipEventElapsedTime(&t, neverCreated, stop) == hipErrorInvalidHandle);
        HIP_ASSERT(hipEventElapsedTime(&t, start, neverCreated) == hipErrorInvalidHandle);

        HIP_ASSERT(hipEventElapsedTime(&t, neverRecorded, stop) == hipErrorInvalidHandle);
        HIP_ASSERT(hipEventElapsedTime(&t, start, neverRecorded) == hipErrorInvalidHandle);
    }

    HIP_CHECK(hipEventDestroy(neverRecorded));
    HIP_CHECK(hipEventDestroy(timingDisabled));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // Clear out everything:
    HIP_CHECK(hipDeviceSynchronize());

    printf("test:   OK  \n");
}

void runTests(int64_t numElements) {
    size_t sizeBytes = numElements * sizeof(int);

    printf("test: starting sequence with sizeBytes=%zu bytes, %6.2f MB\n", sizeBytes,
           sizeBytes / 1024.0 / 1024.0);


    int *C_h, *C_d;
    HIP_CHECK(hipMalloc(&C_d, sizeBytes));
    HIP_CHECK(hipHostMalloc(&C_h, sizeBytes));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, 0x0));

    for (int waitStart = 1; waitStart >= 0; waitStart--) {
        unsigned W = waitStart ? 0x1000 : 0;
        test(W | 0x01, C_d, C_h, numElements, 0, 0, syncNone);
        test(W | 0x02, C_d, C_h, numElements, stream, 0, syncNone);
        test(W | 0x04, C_d, C_h, numElements, 0, waitStart, syncStream);
        test(W | 0x08, C_d, C_h, numElements, stream, waitStart, syncStream);
        test(W | 0x10, C_d, C_h, numElements, 0, waitStart, syncStopEvent);
        test(W | 0x20, C_d, C_h, numElements, stream, waitStart, syncStopEvent);
    }


    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(C_d));
    HIP_CHECK(hipHostFree(C_h));
}


TEST_CASE("Unit_hipEvent") {
  runTests(10000000);
}
