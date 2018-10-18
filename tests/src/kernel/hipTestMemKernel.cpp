/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include "test_common.h"

#define LEN8 8 * 4
#define LEN9 9 * 4
#define LEN10 10 * 4
#define LEN11 11 * 4
#define LEN12 12 * 4

__global__ void MemCpy8(uint8_t* In, uint8_t* Out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    memcpy(Out + tid * 8, In + tid * 8, 8);
}

__global__ void MemCpy9(uint8_t* In, uint8_t* Out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    memcpy(Out + tid * 9, In + tid * 9, 9);
}

__global__ void MemCpy10(uint8_t* In, uint8_t* Out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    memcpy(Out + tid * 10, In + tid * 10, 10);
}

__global__ void MemCpy11(uint8_t* In, uint8_t* Out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    memcpy(Out + tid * 11, In + tid * 11, 11);
}

__global__ void MemCpy12(uint8_t* In, uint8_t* Out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    memcpy(Out + tid * 12, In + tid * 12, 12);
}

__global__ void MemSet8(uint8_t* In) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    memset(In + tid * 8, 1, 8);
}

__global__ void MemSet9(uint8_t* In) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    memset(In + tid * 9, 1, 9);
}

__global__ void MemSet10(uint8_t* In) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    memset(In + tid * 10, 1, 10);
}

__global__ void MemSet11(uint8_t* In) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    memset(In + tid * 11, 1, 11);
}

__global__ void MemSet12(uint8_t* In) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    memset(In + tid * 12, 1, 12);
}

int main() {
    uint8_t *A, *Ad, *B, *Bd, *C, *Cd;
    A = new uint8_t[LEN8];
    B = new uint8_t[LEN8];
    C = new uint8_t[LEN8];
    for (uint32_t i = 0; i < LEN8; i++) {
        A[i] = i;
        B[i] = 0;
        C[i] = 0;
    }
    hipMalloc((void**)&Ad, LEN8);
    hipMalloc((void**)&Bd, LEN8);
    hipMalloc((void**)&Cd, LEN8);
    hipMemcpy(Ad, A, LEN8, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(MemCpy8, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Ad, Bd);
    hipLaunchKernelGGL(MemSet8, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Cd);
    hipMemcpy(B, Bd, LEN8, hipMemcpyDeviceToHost);
    hipMemcpy(C, Cd, LEN8, hipMemcpyDeviceToHost);
    for (uint32_t i = 0; i < LEN8; i++) {
        assert(A[i] == B[i]);
        assert(C[i] == 1);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);

    A = new uint8_t[LEN9];
    B = new uint8_t[LEN9];
    C = new uint8_t[LEN9];
    for (uint32_t i = 0; i < LEN9; i++) {
        A[i] = i;
        B[i] = 0;
        C[i] = 0;
    }
    hipMalloc((void**)&Ad, LEN9);
    hipMalloc((void**)&Bd, LEN9);
    hipMalloc((void**)&Cd, LEN9);
    hipMemcpy(Ad, A, LEN9, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(MemCpy9, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Ad, Bd);
    hipLaunchKernelGGL(MemSet9, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Cd);
    hipMemcpy(B, Bd, LEN9, hipMemcpyDeviceToHost);
    hipMemcpy(C, Cd, LEN9, hipMemcpyDeviceToHost);
    for (uint32_t i = 0; i < LEN9; i++) {
        assert(A[i] == B[i]);
        assert(C[i] == 1);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);

    A = new uint8_t[LEN10];
    B = new uint8_t[LEN10];
    C = new uint8_t[LEN10];
    for (uint32_t i = 0; i < LEN10; i++) {
        A[i] = i;
        B[i] = 0;
        C[i] = 0;
    }
    hipMalloc((void**)&Ad, LEN10);
    hipMalloc((void**)&Bd, LEN10);
    hipMalloc((void**)&Cd, LEN10);
    hipMemcpy(Ad, A, LEN10, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(MemCpy10, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Ad, Bd);
    hipLaunchKernelGGL(MemSet10, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Cd);
    hipMemcpy(B, Bd, LEN10, hipMemcpyDeviceToHost);
    hipMemcpy(C, Cd, LEN10, hipMemcpyDeviceToHost);
    for (uint32_t i = 0; i < LEN10; i++) {
        assert(A[i] == B[i]);
        assert(C[i] == 1);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);

    A = new uint8_t[LEN11];
    B = new uint8_t[LEN11];
    C = new uint8_t[LEN11];
    for (uint32_t i = 0; i < LEN11; i++) {
        A[i] = i;
        B[i] = 0;
        C[i] = 0;
    }
    hipMalloc((void**)&Ad, LEN11);
    hipMalloc((void**)&Bd, LEN11);
    hipMalloc((void**)&Cd, LEN11);
    hipMemcpy(Ad, A, LEN11, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(MemCpy11, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Ad, Bd);
    hipLaunchKernelGGL(MemSet11, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Cd);
    hipMemcpy(B, Bd, LEN11, hipMemcpyDeviceToHost);
    hipMemcpy(C, Cd, LEN11, hipMemcpyDeviceToHost);
    for (uint32_t i = 0; i < LEN11; i++) {
        assert(A[i] == B[i]);
        assert(C[i] == 1);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);

    A = new uint8_t[LEN12];
    B = new uint8_t[LEN12];
    C = new uint8_t[LEN12];
    for (uint32_t i = 0; i < LEN12; i++) {
        A[i] = i;
        B[i] = 0;
        C[i] = 0;
    }
    hipMalloc((void**)&Ad, LEN12);
    hipMalloc((void**)&Bd, LEN12);
    hipMalloc((void**)&Cd, LEN12);
    hipMemcpy(Ad, A, LEN12, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(MemCpy12, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Ad, Bd);
    hipLaunchKernelGGL(MemSet12, dim3(2, 1, 1), dim3(2, 1, 1), 0, 0, Cd);
    hipMemcpy(B, Bd, LEN12, hipMemcpyDeviceToHost);
    hipMemcpy(C, Cd, LEN12, hipMemcpyDeviceToHost);
    for (uint32_t i = 0; i < LEN12; i++) {
        assert(A[i] == B[i]);
        assert(C[i] == 1);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);

    passed();
}
