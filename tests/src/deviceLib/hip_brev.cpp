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
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include <hip/device_functions.h>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))


#define WIDTH 32
#define HEIGHT 32

#define NUM (WIDTH * HEIGHT)

#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define THREADS_PER_BLOCK_Z 1


// CPU implementation of bitreverse
template <typename T>
T bitreverse(T num) {
    T count = sizeof(num) * 8 - 1;
    T reverse_num = num;

    num >>= 1;
    while (num) {
        reverse_num <<= 1;
        reverse_num |= num & 1;
        num >>= 1;
        count--;
    }
    reverse_num <<= count;
    return reverse_num;
}

__global__ void HIP_kernel(unsigned int* a, unsigned int* b,
                           unsigned long long int* c, unsigned long long int* d, int width,
                           int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int i = y * width + x;
    if (i < (width * height)) {
        a[i] = __brev(b[i]);
        c[i] = __brevll(d[i]);
    }
}


using namespace std;

int main() {
    unsigned int* hostA;
    unsigned int* hostB;
    unsigned long long int* hostC;
    unsigned long long int* hostD;

    unsigned int* deviceA;
    unsigned int* deviceB;
    unsigned long long int* deviceC;
    unsigned long long int* deviceD;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    cout << "hip Device prop succeeded " << endl;


    int i;
    int errors;

    hostA = (unsigned int*)malloc(NUM * sizeof(unsigned int));
    hostB = (unsigned int*)malloc(NUM * sizeof(unsigned int));
    hostC = (unsigned long long int*)malloc(NUM * sizeof(unsigned long long int));
    hostD = (unsigned long long int*)malloc(NUM * sizeof(unsigned long long int));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        hostB[i] = i;
        hostD[i] = i;
    }

    HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(unsigned int)));
    HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(unsigned int)));
    HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(unsigned long long int)));
    HIP_ASSERT(hipMalloc((void**)&deviceD, NUM * sizeof(unsigned long long int)));

    HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_ASSERT(
        hipMemcpy(deviceD, hostD, NUM * sizeof(unsigned long long int), hipMemcpyHostToDevice));


    hipLaunchKernelGGL(HIP_kernel, dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA, deviceB, deviceC,
                    deviceD, WIDTH, HEIGHT);


    HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_ASSERT(
        hipMemcpy(hostC, deviceC, NUM * sizeof(unsigned long long int), hipMemcpyDeviceToHost));

    // verify the results
    errors = 0;
    for (i = 0; i < NUM; i++) {
        if (hostA[i] != bitreverse(hostB[i])) {
            errors++;
        }
    }
    if (errors != 0) {
        cout << "__brev() FAILED\n" << endl;
        return -1;
    } else {
        cout << "__brev() checked!" << endl;
    }
    errors = 0;
    for (i = 0; i < NUM; i++) {
        if (hostC[i] != bitreverse(hostD[i])) {
            errors++;
        }
    }
    if (errors != 0) {
        cout << "__brevll() FAILED" << endl;
        return -1;
    } else {
        cout << "__brevll() checked!" << endl;
    }

    cout << "__brev() and __brevll() PASSED!" << endl;

    HIP_ASSERT(hipFree(deviceA));
    HIP_ASSERT(hipFree(deviceB));
    HIP_ASSERT(hipFree(deviceC));
    HIP_ASSERT(hipFree(deviceD));

    free(hostA);
    free(hostB);
    free(hostC);
    free(hostD);

    return errors;
}
