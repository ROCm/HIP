// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

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

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>

#define NUM 1024
#define SIZE 1024 * 4

__device__ int globalIn[NUM];
__device__ int globalOut[NUM];

__global__ void Assign(int* Out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    Out[tid] = globalIn[tid];
    globalOut[tid] = globalIn[tid];
}

__device__ __constant__ int globalConst[NUM];

__global__ void checkAddress(int* addr, bool* out) {
    *out = (globalConst == addr);
}

int main() {
    int *A, *Am, *B, *Ad, *C, *Cm;
    A = new int[NUM];
    B = new int[NUM];
    C = new int[NUM];
    for (int i = 0; i < NUM; ++i) {
        A[i] = -1 * i;
        B[i] = 0;
        C[i] = 0;
    }
    // CHECK: hipMalloc((void**)&Ad, SIZE);
    cudaMalloc((void**)&Ad, SIZE);
    // CHECK: hipHostMalloc((void**)&Am, SIZE);
    cudaMallocHost((void**)&Am, SIZE);
    // CHECK: hipHostMalloc((void**)&Cm, SIZE);
    cudaMallocHost((void**)&Cm, SIZE);
    for (int i = 0; i < NUM; ++i) {
        Am[i] = -1 * i;
        Cm[i] = 0;
    }
    // CHECK: hipStream_t stream = NULL;
    cudaStream_t stream = NULL;
    // CHECK: hipStreamCreate(&stream);
    cudaStreamCreate(&stream);
    // CHECK: hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), Am, SIZE, 0, hipMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(globalIn, Am, SIZE, 0, cudaMemcpyHostToDevice, stream);
    // CHECK: hipStreamSynchronize(stream);
    cudaStreamSynchronize(stream);
    // CHECK: hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    Assign<<<dim3(1, 1, 1), dim3(NUM, 1, 1)>>>(Ad);
    // CHECK: hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost);
    cudaMemcpy(B, Ad, SIZE, cudaMemcpyDeviceToHost);
    // CHECK: hipMemcpyFromSymbolAsync(Cm, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost, stream);
    cudaMemcpyFromSymbolAsync(Cm, globalOut, SIZE, 0, cudaMemcpyDeviceToHost, stream);
    // CHECK: hipStreamSynchronize(stream);
    cudaStreamSynchronize(stream);
    for (int i = 0; i < NUM; ++i) {
        assert(Am[i] == B[i]);
        assert(Am[i] == Cm[i]);
    }
    for (int i = 0; i < NUM; ++i) {
        A[i] = -2 * i;
        B[i] = 0;
    }
    // CHECK: hipMemcpyToSymbol(HIP_SYMBOL(globalIn), A, SIZE, 0, hipMemcpyHostToDevice);
    cudaMemcpyToSymbol(globalIn, A, SIZE, 0, cudaMemcpyHostToDevice);
    // CHECK: hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    Assign<<<dim3(1, 1, 1), dim3(NUM, 1, 1)>>>(Ad);
    // CHECK: hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost);
    cudaMemcpy(B, Ad, SIZE, cudaMemcpyDeviceToHost);
    // CHECK: hipMemcpyFromSymbol(C, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(C, globalOut, SIZE, 0, cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM; ++i) {
        assert(A[i] == B[i]);
        assert(A[i] == C[i]);
    }
    for (int i = 0; i < NUM; ++i) {
        A[i] = -3 * i;
        B[i] = 0;
    }
    // CHECK: hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), A, SIZE, 0, hipMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(globalIn, A, SIZE, 0, cudaMemcpyHostToDevice, stream);
    // CHECK: hipStreamSynchronize(stream);
    cudaStreamSynchronize(stream);
    // CHECK: hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    Assign<<<dim3(1, 1, 1), dim3(NUM, 1, 1)>>>(Ad);
    // CHECK: hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost);
    cudaMemcpy(B, Ad, SIZE, cudaMemcpyDeviceToHost);
    // CHECK: hipMemcpyFromSymbolAsync(C, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost, stream);
    cudaMemcpyFromSymbolAsync(C, globalOut, SIZE, 0, cudaMemcpyDeviceToHost, stream);
    // CHECK: hipStreamSynchronize(stream);
    cudaStreamSynchronize(stream);
    for (int i = 0; i < NUM; ++i) {
        assert(A[i] == B[i]);
        assert(A[i] == C[i]);
    }
    bool *checkOkD;
    bool checkOk = false;
    size_t symbolSize = 0;
    int *symbolAddress;
    // CHECK: hipGetSymbolSize(&symbolSize, HIP_SYMBOL(globalConst));
    cudaGetSymbolSize(&symbolSize, globalConst);
    // CHECK: hipGetSymbolAddress((void**) &symbolAddress, HIP_SYMBOL(globalConst));
    cudaGetSymbolAddress((void**) &symbolAddress, globalConst);
    // CHECK: hipMalloc((void**)&checkOkD, sizeof(bool));
    cudaMalloc((void**)&checkOkD, sizeof(bool));
    // CHECK: hipLaunchKernelGGL(checkAddress, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, symbolAddress, checkOkD);
    checkAddress<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(symbolAddress, checkOkD);
    // CHECK: hipMemcpy(&checkOk, checkOkD, sizeof(bool), hipMemcpyDeviceToHost);
    cudaMemcpy(&checkOk, checkOkD, sizeof(bool), cudaMemcpyDeviceToHost);
    // CHECK: hipFree(checkOkD);
    cudaFree(checkOkD);
    assert(checkOk);
    assert(symbolSize == SIZE);
    // CHECK: hipHostFree(Am);
    cudaFreeHost(Am);
    // CHECK: hipHostFree(Cm);
    cudaFreeHost(Cm);
    // CHECK: hipFree(Ad);
    cudaFree(Ad);
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
