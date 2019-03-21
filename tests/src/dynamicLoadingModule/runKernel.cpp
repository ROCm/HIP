/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <vector>

#include <hip/hip_runtime.h>

#define LEN 64
#define SIZE LEN * sizeof(float)

#define fileName "vcpy_kernel.code"
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

    void* handle = dlopen("libhip_hcc.so", RTLD_LAZY);
    if (!handle) {
        std::cout << dlerror() << "\n";
        return -1;
    }

    void* sym_hipInit = dlsym(handle, "hipInit");
    void* sym_hipDeviceGet = dlsym(handle, "hipDeviceGet");
    void* sym_hipMalloc = dlsym(handle, "hipMalloc");
    void* sym_hipMemcpyHtoD = dlsym(handle, "hipMemcpyHtoD");
    void* sym_hipMemcpyDtoH = dlsym(handle, "hipMemcpyDtoH");
    void* sym_hipModuleLoad = dlsym(handle, "hipModuleLoad");
    void* sym_hipModuleGetGlobal = dlsym(handle, "hipModuleGetGlobal");
    void* sym_hipModuleGetFunction = dlsym(handle, "hipModuleGetFunction");
    void* sym_hipModuleLaunchKernel = dlsym(handle, "hipModuleLaunchKernel");

    hipError_t (*dyn_hipInit)(unsigned int) = reinterpret_cast<hipError_t(*)(unsigned int)>(sym_hipInit);
    hipError_t (*dyn_hipDeviceGet)(hipDevice_t*, int) = reinterpret_cast<hipError_t (*)(hipDevice_t*, int)>(sym_hipDeviceGet);
    hipError_t (*dyn_hipMalloc)(void**, size_t) = reinterpret_cast<hipError_t (*)(void**, size_t)>(sym_hipMalloc);
    hipError_t (*dyn_hipMemcpyHtoD)(hipDeviceptr_t, void*, size_t) = reinterpret_cast<hipError_t (*)(hipDeviceptr_t, void*, size_t)>(sym_hipMemcpyHtoD);
    hipError_t (*dyn_hipMemcpyDtoH)(void*, hipDeviceptr_t, size_t) = reinterpret_cast<hipError_t (*)(void*, hipDeviceptr_t, size_t)>(sym_hipMemcpyDtoH);
    hipError_t (*dyn_hipModuleLoad)(hipModule_t*, const char*) = reinterpret_cast<hipError_t (*)(hipModule_t*, const char*)>(sym_hipModuleLoad);
    hipError_t (*dyn_hipModuleGetGlobal)(hipDeviceptr_t*, size_t*, hipModule_t, const char*) = reinterpret_cast<hipError_t (*)(hipDeviceptr_t*, size_t*, hipModule_t, const char*)>(sym_hipModuleGetGlobal);
    hipError_t (*dyn_hipModuleGetFunction)(hipFunction_t*, hipModule_t, const char*) = reinterpret_cast<hipError_t (*)(hipFunction_t*, hipModule_t, const char*)>(sym_hipModuleGetFunction);
    hipError_t (*dyn_hipModuleLaunchKernel)(hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, hipStream_t, void**, void**) = reinterpret_cast<hipError_t (*)(hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, hipStream_t, void**, void**)>(sym_hipModuleLaunchKernel);

    dyn_hipInit(0);
    hipDevice_t device;
    dyn_hipDeviceGet(&device, 0);

    dyn_hipMalloc((void**)&Ad, SIZE);
    dyn_hipMalloc((void**)&Bd, SIZE);

    dyn_hipMemcpyHtoD(hipDeviceptr_t(Ad), A, SIZE);
    dyn_hipMemcpyHtoD((hipDeviceptr_t)(Bd), B, SIZE);
    hipModule_t Module;
    HIP_CHECK(dyn_hipModuleLoad(&Module, fileName));

    float myDeviceGlobal_h = 42.0;
    float* deviceGlobal;
    size_t deviceGlobalSize;
    HIP_CHECK(dyn_hipModuleGetGlobal((void**)&deviceGlobal, &deviceGlobalSize, Module, "myDeviceGlobal"));
    *deviceGlobal = 42.0;

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
        HIP_CHECK(dyn_hipModuleGetFunction(&Function, Module, "hello_world"));
        HIP_CHECK(dyn_hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL, (void**)&config));

        dyn_hipMemcpyDtoH(B, Bd, SIZE);

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
        HIP_CHECK(dyn_hipModuleGetFunction(&Function, Module, "test_globals"));
        HIP_CHECK(dyn_hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL, (void**)&config));

        dyn_hipMemcpyDtoH(B, Bd, SIZE);

        int mismatchCount = 0;
        for (uint32_t i = 0; i < LEN; i++) {
            float expected = A[i] + myDeviceGlobal_h;
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

    return 0;
}
