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
 * BUILD: %t %s
 * RUN: %t
 * HIT_END
 */

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include "test_common.h"
#include <hip/device_functions.h>

#include <iostream>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

#define LEN 512
#define SIZE LEN << 2

#define TEST_DEBUG (0)

__global__ void kernel_trig(float* In, float* sin_d, float* cos_d, float* tan_d,
                            float* sin_pd, float* cos_pd) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    sin_d[tid] = sinf(In[tid]);
    cos_d[tid] = cosf(In[tid]);
    tan_d[tid] = tanf(In[tid]);
    sincosf(In[tid], &sin_pd[tid], &cos_pd[tid]);
}

int main() {
    float *In, *sin_h, *cos_h, *tan_h, *sin_ph, *cos_ph;
    float *In_d, *sin_d, *cos_d, *tan_d, *sin_pd, *cos_pd;
    int errors = 0;
    In = new float[LEN];
    sin_h = new float[LEN];
    cos_h = new float[LEN];
    tan_h = new float[LEN];
    sin_ph = new float[LEN];
    cos_ph = new float[LEN];
    for (int i = 0; i < LEN; i++) {
        In[i] = 1.0f;
        sin_h[i] = 0.0f;
        cos_h[i] = 0.0f;
        tan_h[i] = 0.0f;
        sin_ph[i] = 0.0f;
        cos_ph[i] = 0.0f;
    }
    HIP_ASSERT(hipMalloc((void**)&In_d, SIZE));
    HIP_ASSERT(hipMalloc((void**)&sin_d, SIZE));
    HIP_ASSERT(hipMalloc((void**)&cos_d, SIZE));
    HIP_ASSERT(hipMalloc((void**)&tan_d, SIZE));
    HIP_ASSERT(hipMalloc((void**)&sin_pd, SIZE));
    HIP_ASSERT(hipMalloc((void**)&cos_pd, SIZE));

    hipMemcpy(In_d, In, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(kernel_trig, dim3(LEN, 1, 1), dim3(1, 1, 1), 0, 0,
                    In_d, sin_d, cos_d, tan_d,
                    sin_pd, cos_pd);
    hipMemcpy(sin_h, sin_d, SIZE, hipMemcpyDeviceToHost);
    hipMemcpy(cos_h, cos_d, SIZE, hipMemcpyDeviceToHost);
    hipMemcpy(tan_h, tan_d, SIZE, hipMemcpyDeviceToHost);
    hipMemcpy(sin_ph, sin_pd, SIZE, hipMemcpyDeviceToHost);
    hipMemcpy(cos_ph, cos_pd, SIZE, hipMemcpyDeviceToHost);
    for (int i = 0; i < LEN; i++) {
        if (sin_h[i] != sin_ph[i] || cos_h[i] != cos_ph[i] || tan_h[i] * cos_h[i] != sin_h[i]) {
            errors++;
#if TEST_DEBUG
            std::cout << "Check Failed!" << std::endl;
            std::cout << " sin_h: " << sin_h[i] << " sin_ph: " << sin_ph[i] << "\n"
                      << " cos_h: " << cos_h[i] << " cos_ph:" << cos_ph[i] << "\n"
                      << " tan_h * cos_h: " << tan_h[i] * cos_h[i] << " sin_h[i]: " << sin_h[i] << "\n";
#endif
        }
    }

    HIP_ASSERT(hipFree(In_d));
    HIP_ASSERT(hipFree(sin_d));
    HIP_ASSERT(hipFree(cos_d));
    HIP_ASSERT(hipFree(tan_d));
    HIP_ASSERT(hipFree(sin_pd));
    HIP_ASSERT(hipFree(cos_pd));

    if (errors != 0) {
        std::cout << "hip_trig FAILED!" << std::endl;
        return -1;
    } else {
        std::cout << "hip_trig PASSED!" << std::endl;
    }
    return errors;
}
