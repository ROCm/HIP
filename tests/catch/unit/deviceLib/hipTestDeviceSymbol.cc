/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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


/* Test Case Description: Calling hipMemcpyTo/FromSymbolAsync() using user
   declared stream obj and hipStreamPerThread*/

#include <hip_test_common.hh>
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

TEST_CASE("Unit_hipMemcpyToSymbolAsync_ToNFrom") {
    int *A, *Am, *B, *Ad, *C, *Cm;
    A = new int[NUM];
    B = new int[NUM];
    C = new int[NUM];
    for (int i = 0; i < NUM; i++) {
        A[i] = -1 * i;
        B[i] = 0;
        C[i] = 0;
    }

    HIP_CHECK(hipMalloc((void**)&Ad, SIZE));
    HIP_CHECK(hipHostMalloc((void**)&Am, SIZE));
    HIP_CHECK(hipHostMalloc((void**)&Cm, SIZE));
    for (int i = 0; i < NUM; i++) {
        Am[i] = -1 * i;
        Cm[i] = 0;
    }

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), Am, SIZE, 0,
              hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpyFromSymbolAsync(Cm, HIP_SYMBOL(globalOut), SIZE, 0,
              hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (int i = 0; i < NUM; i++) {
        assert(Am[i] == B[i]);
        assert(Am[i] == Cm[i]);
    }

    for (int i = 0; i < NUM; i++) {
        A[i] = -2 * i;
        B[i] = 0;
    }

    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(globalIn), A, SIZE, 0,
                                hipMemcpyHostToDevice));
    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpyFromSymbol(C, HIP_SYMBOL(globalOut), SIZE, 0,
                                  hipMemcpyDeviceToHost));
    for (int i = 0; i < NUM; i++) {
        assert(A[i] == B[i]);
        assert(A[i] == C[i]);
    }

    for (int i = 0; i < NUM; i++) {
        A[i] = -3 * i;
        B[i] = 0;
    }
    SECTION("Calling hipMemcpyTo/FromSymbol using user declared stream obj") {
      HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), A, SIZE, 0,
                                        hipMemcpyHostToDevice, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
      HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpyFromSymbolAsync(C, HIP_SYMBOL(globalOut), SIZE, 0,
                                         hipMemcpyDeviceToHost, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    SECTION("Calling hipMemcpyTo/FromSymbol using hipStreamPerThread") {
      HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), A, SIZE, 0,
                                        hipMemcpyHostToDevice, hipStreamPerThread));
      HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
      hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
      HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpyFromSymbolAsync(C, HIP_SYMBOL(globalOut), SIZE, 0,
                                         hipMemcpyDeviceToHost, hipStreamPerThread));
      HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    }
    for (int i = 0; i < NUM; i++) {
        assert(A[i] == B[i]);
        assert(A[i] == C[i]);
    }

    bool *checkOkD;
    bool checkOk = false;
    size_t symbolSize = 0;
    int *symbolAddress;
    HIP_CHECK(hipGetSymbolSize(&symbolSize, HIP_SYMBOL(globalConst)));
    HIP_CHECK(hipGetSymbolAddress((void**) &symbolAddress, HIP_SYMBOL(globalConst)));
    HIP_CHECK(hipMalloc((void**)&checkOkD, sizeof(bool)));
    hipLaunchKernelGGL(checkAddress, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, symbolAddress, checkOkD);
    HIP_CHECK(hipMemcpy(&checkOk, checkOkD, sizeof(bool), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(checkOkD));
    HIP_ASSERT(checkOk);
    HIP_ASSERT((symbolSize == SIZE));

    HIP_CHECK(hipHostFree(Am));
    HIP_CHECK(hipHostFree(Cm));
    HIP_CHECK(hipFree(Ad));
    delete[] A;
    delete[] B;
    delete[] C;
}
