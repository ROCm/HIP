/*
Copyright (c) 2017 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip/hip_hcc.h>

#define LEN 64
#define SIZE LEN * sizeof(float)

#define fileName "vcpy_kernel.code"
float myDeviceGlobal;
float myDeviceGlobalArray[16];
#define HIP_CHECK(cmd)                                                                             \
    {                                                                                              \
        hipError_t status = cmd;                                                                   \
        if (status != hipSuccess) {                                                                \
            std::cout << "error: #" << status << " (" << hipGetErrorString(status)                 \
                      << ") at line:" << __LINE__ << ":  " << #cmd << std::endl;                   \
            abort();                                                                               \
        }                                                                                          \
    }

int main() {
    float *A, *B;
    float *Ad, *Bd;
    A = new float[LEN];
    B = new float[LEN];

    for (uint32_t i = 0; i < LEN; i++) {
        A[i] = i * 1.0f;
        B[i] = 0.0f;
    }

    hipInit(0);
    hipDevice_t device;
    hipCtx_t context;
    hipDeviceGet(&device, 0);
    hipCtxCreate(&context, 0, device);

    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);

    hipMemcpyHtoD(hipDeviceptr_t(Ad), A, SIZE);
    hipMemcpyHtoD((hipDeviceptr_t)(Bd), B, SIZE);
    hipModule_t Module;
    HIP_CHECK(hipModuleLoad(&Module, fileName));

    float myDeviceGlobal_h = 42.0;
    myDeviceGlobal = 42.0;

#define ARRAY_SIZE 16

    float myDeviceGlobalArray_h[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        myDeviceGlobalArray_h[i] = i * 1000.0f;
        myDeviceGlobalArray[i] = i * 1000.0f;
    }

    struct {
        void* _Ad;
        void* _Bd;
    } args;

    args._Ad = (void*) Ad;
    args._Bd = (void*) Bd;

    size_t size = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    {
        hipFunction_t Function;
        HIP_CHECK(hipModuleGetFunction(&Function, Module, "hello_world"));
        HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL, (void**)&config));

        hipMemcpyDtoH(B, Bd, SIZE);

        int mismatchCount = 0;
        for (uint32_t i = 0; i < LEN; i++) {
            if (A[i] != B[i]) {
                mismatchCount++;
                std::cout << "error: mismatch " << A[i] << " != " << B[i] << std::endl;
                if (mismatchCount >= 10) {
                    break;
                }
            }
        }

        if (mismatchCount == 0) {
            std::cout << "PASSED!\n";
        } else {
            std::cout << "FAILED!\n";
        };
    }

    {
        hipFunction_t Function;
        HIP_CHECK(hipModuleGetFunction(&Function, Module, "test_globals"));
        HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL, (void**)&config));

        hipMemcpyDtoH(B, Bd, SIZE);

        int mismatchCount = 0;
        for (uint32_t i = 0; i < LEN; i++) {
            float expected = A[i] + myDeviceGlobal_h + myDeviceGlobalArray_h[i % 16];
            if (expected != B[i]) {
                mismatchCount++;
                std::cout << "error: mismatch " << expected << " != " << B[i] << std::endl;
                if (mismatchCount >= 10) {
                    break;
                }
            }
        }

        if (mismatchCount == 0) {
            std::cout << "PASSED!\n";
        } else {
            std::cout << "FAILED!\n";
        };
    }

    hipCtxDestroy(context);
    return 0;
}
