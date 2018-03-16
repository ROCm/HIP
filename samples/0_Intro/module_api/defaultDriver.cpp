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

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <iostream>
#include <fstream>
#include <vector>

#define LEN 64
#define SIZE LEN << 2

#define fileName "test.co"
#define kernel_name "vadd"

int main() {
    float *A, *B, *C;
    hipDeviceptr_t Ad, Bd, Cd;
    A = new float[LEN];
    B = new float[LEN];
    C = new float[LEN];

    for (uint32_t i = 0; i < LEN; i++) {
        A[i] = i * 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    hipInit(0);
    hipDevice_t device;
    hipCtx_t context;
    hipDeviceGet(&device, 0);
    hipCtxCreate(&context, 0, device);

    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMalloc((void**)&Cd, SIZE);

    hipMemcpyHtoD(Ad, A, SIZE);
    hipMemcpyHtoD(Bd, B, SIZE);
    hipMemcpyHtoD(Cd, C, SIZE);

    hipModule_t Module;
    hipFunction_t Function;
    hipModuleLoad(&Module, fileName);
    hipModuleGetFunction(&Function, Module, kernel_name);

    int n = LEN;
    void* args[4] = {&Ad, &Bd, &Cd, &n};

    hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, args, nullptr);

    hipMemcpyDtoH(C, Cd, SIZE);
    int mismatchCount = 0;
    for (uint32_t i = 0; i < LEN; i++) {
        if (A[i] + B[i] != C[i]) {
            mismatchCount++;
            std::cout << "error: mismatch " << A[i] << " + " << B[i] << " != " << C[i] << std::endl;
        }
    }

    if (mismatchCount == 0) {
        std::cout << "PASSED!\n";
    } else {
        std::cout << "FAILED!\n";
    };

    hipCtxDestroy(context);
    return 0;
}
