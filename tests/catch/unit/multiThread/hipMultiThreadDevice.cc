/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST_NAMED: %t hipMultiThreadDevice-serial --tests 0x1
 * TEST_NAMED: %t hipMultiThreadDevice-pyramid --tests 0x4
 * TEST_NAMED: %t hipMultiThreadDevice-nearzero --tests 0x10
 * HIT_END
 */

//#include "hip/hip_runtime_api.h"
#include <hip_test_common.hh>

#ifdef _WIN32
#define MAX_BURST_SIZE   40
#else
#define MAX_BURST_SIZE   100
#endif

// Create a lot of streams and then destroy 'em.
void createThenDestroyStreams(int iterations, int burstSize) {
    hipStream_t* streams = new hipStream_t[burstSize];

    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < burstSize; j++) {
            HIPCHECK(hipStreamCreate(&streams[j]));
        }
        for (int j = 0; j < burstSize; j++) {
            HIPCHECK(hipStreamDestroy(streams[j]));
        }
    }

    delete[] streams;
}


void waitStreams(int iterations) {
    // Repeatedly sync and wait for all streams to complete.
    // TO make this interesting, the test has other threads repeatedly adding and removing streams
    // to the device.
    for (int i = 0; i < iterations; i++) {
        HIPCHECK(hipDeviceSynchronize());
    }
}


// Create 3 streams, all creating and destroying streams on the same device.
// Some create many queue, some not many.
//
void multiThread_pyramid(bool serialize, int iters) {
    std::thread t1(createThenDestroyStreams, iters * 1, MAX_BURST_SIZE);
    if (serialize) {
        t1.join();
    }

    std::thread t2(createThenDestroyStreams, iters * 10, 10);
    if (serialize) {
        t2.join();
    }

    std::thread t3(createThenDestroyStreams, iters * 100, 1);
    if (serialize) {
        t3.join();
    }

    if (!serialize) {
        t1.join();
        t2.join();
        t3.join();
    }
}


// Create 3 streams, all creating and destroying streams on the same device.
// Try to keep number of streams near zero, to cause problems.
void multiThread_nearzero(bool serialize, int iters) {
    std::thread t1(createThenDestroyStreams, iters, 1);
    if (serialize) {
        t1.join();
    }

    std::thread t2(createThenDestroyStreams, iters, 1);
    if (serialize) {
        t2.join();
    }

    std::thread t3(waitStreams, iters * 50);
    if (serialize) {
        t3.join();
    }

    if (!serialize) {
        t1.join();
        t2.join();
        t3.join();
    }
}

TEST_CASE("Unit_hipMultiThreadDevice_Streams") {
    // Serial version, just call once:
    createThenDestroyStreams(10, 10);
}

TEST_CASE("Unit_hipMultiThreadDevice_SerialPyramid") {
    multiThread_pyramid(true, 3);
}

TEST_CASE("Unit_hipMultiThreadDevice_ParallelPyramid") {
    multiThread_pyramid(false, 3);
}

TEST_CASE("Unit_hipMultiThreadDevice_NearZero") {
    multiThread_nearzero(false, 1000);
}
