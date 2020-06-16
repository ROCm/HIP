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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD_CMD: vcpy_kernel.code %hc --genco %S/vcpy_kernel.cpp -o vcpy_kernel.code
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>

#include "test_common.h"

#define LEN 64
#define SIZE LEN << 2

#define fileName "vcpy_kernel.code"
#define kernel_name "hello_world"

#define HIP_CHECK(status)                                                                          \
    if (status != hipSuccess) {                                                                    \
        std::cout << "Got Status: " << status << " at Line: " << __LINE__ << std::endl;            \
        exit(0);                                                                                   \
    }

int main() {
    float *A, *B;
    hipDeviceptr_t Ad, Bd;
    A = new float[LEN];
    B = new float[LEN];

    for (uint32_t i = 0; i < LEN; i++) {
        A[i] = i * 1.0f;
        B[i] = 0.0f;
    }

    HIPCHECK(hipInit(0));

    hipDevice_t device;
    hipCtx_t context;
    HIPCHECK(hipDeviceGet(&device, 0));
    HIPCHECK(hipCtxCreate(&context, 0, device));

    HIPCHECK(hipMalloc((void**)&Ad, SIZE));
    HIPCHECK(hipMalloc((void**)&Bd, SIZE));
    HIPCHECK(hipMemcpyHtoD(Ad, A, SIZE));
    HIPCHECK(hipMemcpyHtoD(Bd, B, SIZE));

    hipModule_t Module;
    hipFunction_t Function;
    HIPCHECK(hipModuleLoad(&Module, fileName));
    HIPCHECK(hipModuleGetFunction(&Function, Module, kernel_name));

    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));

    struct {
        void* _Ad;
        void* _Bd;
    } args;
    args._Ad = (void*) Ad;
    args._Bd = (void*) Bd;
    size_t size = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};
    HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, stream, NULL, (void**)&config));

    HIPCHECK(hipStreamDestroy(stream));

    HIPCHECK(hipMemcpyDtoH(B, Bd, SIZE));

    for (uint32_t i = 0; i < LEN; i++) {
        assert(A[i] == B[i]);
    }

    HIPCHECK(hipModuleUnload(Module));
    HIPCHECK(hipCtxDestroy(context));
    passed();
}
