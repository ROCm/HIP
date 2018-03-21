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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"
#include "test_common.h"

#define WIDTH 1024
#define HEIGHT 1024

#define NUM (WIDTH * HEIGHT)

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define THREADS_PER_BLOCK_Z 1

int main() {
    int* hostA;
    int* hostB;

    int* deviceA;
    int* deviceB;

    int i;
    int errors;

    hostA = (int*)malloc(NUM * sizeof(int));
    hostB = (int*)malloc(NUM * sizeof(int));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        hostB[i] = i;
    }

    HIPCHECK(hipMalloc((void**)&deviceA, NUM * sizeof(int)));
    HIPCHECK(hipMalloc((void**)&deviceB, NUM * sizeof(int)));

    hipStream_t s;
    HIPCHECK(hipStreamCreate(&s));


    // hostB -> deviceB -> hostA
#define ASYNC 1
#if ASYNC
    HIPCHECK(hipMemcpyAsync(deviceB, hostB, NUM * sizeof(int), hipMemcpyHostToDevice, s));
    HIPCHECK(hipMemcpyAsync(hostA, deviceB, NUM * sizeof(int), hipMemcpyDeviceToHost, s));
#else
    HIPCHECK(hipMemcpy(deviceB, hostB, NUM * sizeof(int), hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(hostA, deviceB, NUM * sizeof(int), hipMemcpyDeviceToHost));
#endif

    HIPCHECK(hipStreamSynchronize(s));
    HIPCHECK(hipDeviceSynchronize());

    // verify the results
    errors = 0;
    for (i = 0; i < NUM; i++) {
        if (hostA[i] != (hostB[i])) {
            errors++;
        }
    }

    HIPCHECK(hipStreamDestroy(s));

    HIPCHECK(hipFree(deviceA));
    HIPCHECK(hipFree(deviceB));

    free(hostA);
    free(hostB);

    // hipResetDefaultAccelerator();

    if (errors != 0) {
        HIPASSERT(1 == 2);
    } else {
        passed();
    }

    return errors;
}
