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
 * HIT_END
 */


#include <cstdio>
#include "hip/hip_runtime.h"

__global__ void Kernel(volatile float* hostRes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    hostRes[tid] = tid + 1;
    __threadfence_system();
    // expecting that the data is getting flushed to host here!
    // time waster for-loop (sleep)
    for (int timeWater = 0; timeWater < 100000000; timeWater++)
        ;
}

int main() {
    size_t blocks = 2;
    volatile float* hostRes;
    hipHostMalloc((void**)&hostRes, blocks * sizeof(float), hipHostMallocMapped);
    hostRes[0] = 0;
    hostRes[1] = 0;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Kernel), dim3(1), dim3(blocks), 0, 0, hostRes);
    int eleCounter = 0;
    while (eleCounter < blocks) {
        // blocks until the value changes
        while (hostRes[eleCounter] == 0)
            ;
        printf("%f\n", hostRes[eleCounter]);
        ;
        eleCounter++;
    }
    hipHostFree((void*)hostRes);
    return 0;
}
