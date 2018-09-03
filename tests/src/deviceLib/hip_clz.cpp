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
#define WIDTH 8
#define HEIGHT 8
#define NUM (WIDTH * HEIGHT)

#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define THREADS_PER_BLOCK_Z 1

unsigned int firstbit_u32(unsigned int a) {
    if (a == 0) {
        return 32;
    }
    unsigned int pos = 0;
    while ((int)a > 0) {
        a <<= 1;
        pos++;
    }
    return pos;
}

unsigned int firstbit_u64(unsigned long long int a) {
    if (a == 0) {
        return 64;
    }
    unsigned int pos = 0;
    while ((long long int)a > 0) {
        a <<= 1;
        pos++;
    }
    return pos;
}

// Check implicit conversion will not cause ambiguity.
__device__ void test_ambiguity() {
  short s;
  unsigned short us;
  float f;
  int i;
  unsigned int ui;
  __clz(f);
  __clz(s);
  __clz(us);
  __clzll(f);
  __clzll(i);
  __clzll(ui);
}

__global__ void HIP_kernel(hipLaunchParm lp, unsigned int* a, unsigned int* b, unsigned int* c,
                           unsigned long long int* d, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int i = y * width + x;
    if (i < (width * height)) {
        a[i] = __clz(b[i]);
        c[i] = __clzll(d[i]);
    }
}

using namespace std;

int main() {
    unsigned int* hostA;
    unsigned int* hostB;
    unsigned int* hostC;
    unsigned long long int* hostD;

    unsigned int* deviceA;
    unsigned int* deviceB;
    unsigned int* deviceC;
    unsigned long long int* deviceD;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    cout << "hip Device prop succeeded " << endl;

    unsigned int i;
    int errors;

    hostA = (unsigned int*)malloc(NUM * sizeof(unsigned int));
    hostB = (unsigned int*)malloc(NUM * sizeof(unsigned int));
    hostC = (unsigned int*)malloc(NUM * sizeof(unsigned int));
    hostD = (unsigned long long int*)malloc(NUM * sizeof(unsigned long long int));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        hostB[i] = 419430 * i;
        hostD[i] = i;
    }

    HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(unsigned int)));
    HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(unsigned int)));
    HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(unsigned int)));
    HIP_ASSERT(hipMalloc((void**)&deviceD, NUM * sizeof(unsigned long long int)));

    HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_ASSERT(
        hipMemcpy(deviceD, hostD, NUM * sizeof(unsigned long long int), hipMemcpyHostToDevice));

    hipLaunchKernel(HIP_kernel, dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA, deviceB, deviceC,
                    deviceD, WIDTH, HEIGHT);


    HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(hostC, deviceC, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));

    // verify the results
    errors = 0;
    for (i = 0; i < NUM; i++) {
        printf("gpu_clz =%d, cpu_clz =%d \n", hostA[i], firstbit_u32(hostB[i]));
        if (hostA[i] != firstbit_u32(hostB[i])) {
            errors++;
        }
    }
    if (errors != 0) {
        cout << "FAILED clz" << endl;
        return -1;
    } else {
        cout << "__clz() checked!" << endl;
    }
    errors = 0;
    for (i = 0; i < NUM; i++) {
        printf("gpu_clzll =%d, cpu_clzll =%d \n", hostC[i], firstbit_u64(hostD[i]));
        if (hostC[i] != firstbit_u64(hostD[i])) {
            errors++;
        }
    }
    if (errors != 0) {
        cout << "FAILED clz" << endl;
        return -1;
    } else {
        cout << "__clzll() checked!" << endl;
    }

    cout << "clz test PASSED!" << endl;

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
