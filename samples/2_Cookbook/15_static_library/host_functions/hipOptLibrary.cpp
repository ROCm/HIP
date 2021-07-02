/*
 * Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>

#define HIP_ASSERT(status) assert(status == hipSuccess)
#define LEN 512

__global__ void copy(uint32_t* A, uint32_t* B) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    B[tid] = A[tid];
}

void run_test1() {
    uint32_t *A_h, *B_h, *A_d, *B_d;
    size_t valbytes = LEN * sizeof(uint32_t);

    A_h = (uint32_t*)malloc(valbytes);
    B_h = (uint32_t*)malloc(valbytes);
    for (uint32_t i = 0; i < LEN; i++) {
        A_h[i] = i;
        B_h[i] = 0;
    }

    HIP_ASSERT(hipMalloc((void**)&A_d, valbytes));
    HIP_ASSERT(hipMalloc((void**)&B_d, valbytes));

    HIP_ASSERT(hipMemcpy(A_d, A_h, valbytes, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(copy, dim3(LEN/64), dim3(64), 0, 0, A_d, B_d);
    HIP_ASSERT(hipMemcpy(B_h, B_d, valbytes, hipMemcpyDeviceToHost));

    for (uint32_t i = 0; i < LEN; i++) {
        assert(A_h[i] == B_h[i]);
    }

    HIP_ASSERT(hipFree(A_d));
    HIP_ASSERT(hipFree(B_d));
    free(A_h);
    free(B_h);
    std::cout << "Test Passed!\n";
}
