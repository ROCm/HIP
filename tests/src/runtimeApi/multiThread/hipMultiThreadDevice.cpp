/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * RUN_NAMED: %t hipMultiThreadDevice-serial --tests 0x1
 * RUN_NAMED: %t hipMultiThreadDevice-pyramid --tests 0x4
 * RUN_NAMED: %t hipMultiThreadDevice-nearzero --tests 0x10
 * HIT_END
 */

#include "hip/hip_runtime_api.h"
#include "test_common.h"


// Create a lot of streams and then destroy 'em.
void createThenDestroyStreams(int iterations, int burstSize) {
    hipStream_t* streams = new hipStream_t[burstSize];

    for (int i = 0; i < iterations; i++) {
        if (p_verbose & 0x1) {
            printf("%s iter=%d, create %d then destroy %d\n", __func__, i, burstSize, burstSize);
        }
        for (int j = 0; j < burstSize; j++) {
            if (p_verbose & 0x2) {
                printf("  %d.%d streamCreate\n", i, j);
            }
            HIPCHECK(hipStreamCreate(&streams[j]));
        }
        for (int j = 0; j < burstSize; j++) {
            if (p_verbose & 0x2) {
                printf("  %d.%d streamDestroy\n", i, j);
            }
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
    printf("%s creating %d streams\n", __func__, iters * 100);
    std::thread t1(createThenDestroyStreams, iters * 1, 100);
    if (serialize) {
        t1.join();
        printf("t1 done\n");
    }

    std::thread t2(createThenDestroyStreams, iters * 10, 10);
    if (serialize) {
        t2.join();
        printf("t2 done\n");
    }

    std::thread t3(createThenDestroyStreams, iters * 100, 1);
    if (serialize) {
        t3.join();
        printf("t3 done\n");
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
    printf("%s creating %d streams x 3 threads\n", __func__, iters);
    std::thread t1(createThenDestroyStreams, iters, 1);
    if (serialize) {
        t1.join();
        printf("t1 done\n");
    }

    std::thread t2(createThenDestroyStreams, iters, 1);
    if (serialize) {
        t2.join();
        printf("t2 done\n");
    }

    std::thread t3(waitStreams, iters * 50);
    if (serialize) {
        t3.join();
        printf("t3 done\n");
    }

    if (!serialize) {
        t1.join();
        printf("t1 done\n");
        t2.join();
        printf("t2 done\n");
        t3.join();
        printf("t3 done\n");
    }
}

int main(int argc, char* argv[]) {
    HipTest::parseStandardArguments(argc, argv, true);

    // Serial version, just call once:
    if (p_tests & 0x1) {
        printf("\ntest 0x1 : serial createThenDestroyStreams(10) \n");
        createThenDestroyStreams(10, 10);
    };

    /*disable, this takess a while and if the next one works then no need to run serial*/
    if (1 && (p_tests & 0x2)) {
        printf("\ntest 0x2 : serialized multiThread_pyramid(1) \n");
        multiThread_pyramid(true, 3);
    }

    if (p_tests & 0x4) {
        printf("\ntest 0x4 : parallel multiThread_pyramid(1) \n");
        multiThread_pyramid(false, 3);
    }

    // if (p_tests & 0x8) {
    //    printf ("test 0x8 : multiThread_pyramid(100) \n");
    //    multiThread_pyramid(false, 100);
    // }

    if (p_tests & 0x10) {
        printf("\ntest 0x10 : parallel multiThread_nearzero(1000) \n");
        multiThread_nearzero(false, 1000);
    }

    passed();
}
