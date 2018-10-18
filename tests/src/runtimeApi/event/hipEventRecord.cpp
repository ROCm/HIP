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
// Test hipEventRecord serialization behavior.
// Through manual inspection of the reported timestamps, can determine if recording a NULL event
// forces synchronization : set

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t --iterations 10
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

int main(int argc, char* argv[]) {
    HipTest::parseStandardArguments(argc, argv, true);

    unsigned blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 1024) blocks = 1024;
    if (blocks == 0) blocks = 1;

    printf("N=%zu (A+B+C= %6.1f MB total) blocks=%u threadsPerBlock=%u iterations=%d\n", N,
           ((double)3 * N * sizeof(float)) / 1024 / 1024, blocks, threadsPerBlock, iterations);
    printf("iterations=%d\n", iterations);

    size_t Nbytes = N * sizeof(float);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);

    hipEvent_t start, stop;

    // NULL stream check:
    HIPCHECK(hipEventCreate(&start));
    HIPCHECK(hipEventCreate(&stop));


    HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));


    for (int i = 0; i < iterations; i++) {
        //--- START TIMED REGION
        long long hostStart = HipTest::get_time();
        // Record the start event
        HIPCHECK(hipEventRecord(start, NULL));

        hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                        static_cast<const float*>(A_d), static_cast<const float*>(B_d), C_d, N);


        HIPCHECK(hipEventRecord(stop, NULL));
        HIPCHECK(hipEventSynchronize(stop));
        long long hostStop = HipTest::get_time();
        //--- STOP TIMED REGION


        float eventMs = 1.0f;
        HIPCHECK(hipEventElapsedTime(&eventMs, start, stop));
        float hostMs = HipTest::elapsed_time(hostStart, hostStop);

        printf("host_time (gettimeofday)          =%6.3fms\n", hostMs);
        printf("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);
        printf("\n");

        // Make sure timer is timing something...
        HIPASSERT(eventMs > 0.0f);
    }


    HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));


    printf("check:\n");

    HipTest::checkVectorADD(A_h, B_h, C_h, N, true);


    passed();
}
