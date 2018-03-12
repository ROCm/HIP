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
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * RUN: %t
 * HIT_END
 */

#include <stdio.h>
#include <thread>
#include <chrono>
#include "hip/hip_runtime.h"
#include "test_common.h"

#ifdef __HIP_PLATFORM_HCC__
#define HIPRT_CB
#endif

__global__ void vector_square(float* C_d, float* A_d, size_t N) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

float *A_h, *C_h;
bool cbDone = false;

static void HIPRT_CB Callback(hipStream_t stream, hipError_t status, void* userData) {
    for (size_t i = 0; i < N; i++) {
        if (C_h[i] != A_h[i] * A_h[i]) {
            warn("Data mismatch %zu", i);
        }
    }
    printf("PASSED!\n");
    cbDone = true;
}

int main(int argc, char* argv[]) {
    float *A_d, *C_d;
    size_t Nbytes = N * sizeof(float);

    A_h = (float*)malloc(Nbytes);
    HIPCHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    C_h = (float*)malloc(Nbytes);
    HIPCHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess);

    // Fill with Phi + i
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.618f + i;
    }

    HIPCHECK(hipMalloc(&A_d, Nbytes));
    HIPCHECK(hipMalloc(&C_d, Nbytes));

    hipStream_t mystream;
    HIPCHECK(hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));

    HIPCHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, mystream));

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;
    hipLaunchKernelGGL((vector_square), dim3(blocks), dim3(threadsPerBlock), 0, mystream, C_d, A_d,
                       N);

    HIPCHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, mystream));
    HIPCHECK(hipStreamAddCallback(mystream, Callback, NULL, 0));

    while (!cbDone) std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
