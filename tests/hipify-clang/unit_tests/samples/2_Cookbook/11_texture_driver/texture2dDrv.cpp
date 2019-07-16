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
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>

#define fileName "tex2dKernel.code"
// CHECK: texture<float, 2, hipReadModeElementType> tex;
texture<float, 2, cudaReadModeElementType> tex;
bool testResult = false;

// CHECK: hipError_t status = cmd;
// CHECK: if (status != hipSuccess) {
// CHECK: std::cout << "error: #" << status << " (" << hipGetErrorString(status)
#define CUDACHECK(cmd)                                                                             \
    {                                                                                              \
        cudaError_t status = cmd;                                                                  \
        if (status != cudaSuccess) {                                                               \
            std::cout << "error: #" << status << " (" << cudaGetErrorString(status)                \
                      << ") at line:" << __LINE__ << ":  " << #cmd << std::endl;                   \
            abort();                                                                               \
        }                                                                                          \
    }

bool runTest(int argc, char** argv) {
    unsigned int width = 256;
    unsigned int height = 256;
    unsigned int size = width * height * sizeof(float);
    float* hData = (float*)malloc(size);
    memset(hData, 0, size);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            hData[i * width + j] = i * width + j;
        }
    }
    // CHECK: hipModule_t Module;
    CUmodule Module;
    // CHECK: hipModuleLoad(&Module, fileName);
    cuModuleLoad(&Module, fileName);

    // CHECK: hipArray * array;
    CUarray array;
    // CHECK: HIP_ARRAY_DESCRIPTOR desc;
    CUDA_ARRAY_DESCRIPTOR desc;
    // CHECK: desc.Format = HIP_AD_FORMAT_FLOAT;
    desc.Format = CU_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = width;
    desc.Height = height;
    // CHECK: hipArrayCreate(&array, &desc);
    cuArrayCreate(&array, &desc);

    // CHECK: hip_Memcpy2D copyParam;
    CUDA_MEMCPY2D copyParam;
    memset(&copyParam, 0, sizeof(copyParam));
    // CHECK: copyParam.dstMemoryType = hipMemoryTypeArray;
    copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParam.dstArray = array;
    // CHECK: copyParam.srcMemoryType = hipMemoryTypeHost;
    copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
    copyParam.srcHost = hData;
    copyParam.srcPitch = width * sizeof(float);
    copyParam.WidthInBytes = copyParam.srcPitch;
    copyParam.Height = height;
    // CHECK: hipMemcpyParam2D(&copyParam);
    cuMemcpy2D(&copyParam);

    // CHECK: textureReference* texref;
    CUtexref_st* texref;
    // CHECK: hipModuleGetTexRef(&texref, Module, "tex");
    cuModuleGetTexRef(&texref, Module, "tex");
    // CHECK: hipTexRefSetAddressMode(texref, 0, hipAddressModeWrap);
    cuTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_WRAP);
    // CHECK: hipTexRefSetAddressMode(texref, 1, hipAddressModeWrap);
    cuTexRefSetAddressMode(texref, 1, CU_TR_ADDRESS_MODE_WRAP);
    // CHECK: hipTexRefSetFilterMode(texref, hipFilterModePoint);
    cuTexRefSetFilterMode(texref, CU_TR_FILTER_MODE_POINT);
    // CHECK: hipTexRefSetFlags(texref, 0);
    cuTexRefSetFlags(texref, 0);
    // CHECK: hipTexRefSetFormat(texref, HIP_AD_FORMAT_FLOAT, 1);
    cuTexRefSetFormat(texref, CU_AD_FORMAT_FLOAT, 1);
    // CHECK: hipTexRefSetArray(texref, array, HIP_TRSA_OVERRIDE_FORMAT);
    cuTexRefSetArray(texref, array, CU_TRSA_OVERRIDE_FORMAT);

    float* dData = NULL;
    // CHECK: hipMalloc((void**)&dData, size);
    cudaMalloc((void**)&dData, size);

    struct {
        void* _Ad;
        unsigned int _Bd;
        unsigned int _Cd;
    } args;
    args._Ad = (void*) dData;
    args._Bd = width;
    args._Cd = height;

    size_t sizeTemp = sizeof(args);

    // CHECK: void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
    // CHECK: &sizeTemp, HIP_LAUNCH_PARAM_END};
    void* config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &args, CU_LAUNCH_PARAM_BUFFER_SIZE,
                      &sizeTemp, CU_LAUNCH_PARAM_END};

    // CHECK: hipFunction_t Function;
    CUfunction Function;
    // CHECK: hipModuleGetFunction(&Function, Module, "tex2dKernel");
    cuModuleGetFunction(&Function, Module, "tex2dKernel");

    int temp1 = width / 16;
    int temp2 = height / 16;
    // CHECK: hipModuleLaunchKernel(Function, 16, 16, 1, temp1, temp2, 1, 0, 0, NULL, (void**)&config);
    cuLaunchKernel(Function, 16, 16, 1, temp1, temp2, 1, 0, 0, NULL, (void**)&config);
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();

    float* hOutputData = (float*)malloc(size);
    memset(hOutputData, 0, size);
    // CHECK: hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost);
    cudaMemcpy(hOutputData, dData, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (hData[i * width + j] != hOutputData[i * width + j]) {
                printf("Difference [ %d %d ]:%f ----%f\n", i, j, hData[i * width + j],
                       hOutputData[i * width + j]);
                testResult = false;
                break;
            }
        }
    }
    // CHECK: hipFree(dData);
    cudaFree(dData);
    // CHECK: hipFreeArray(hipArray_t(array));
    cudaFreeArray(cudaArray_t(array));
    return true;
}

int main(int argc, char** argv) {
    // CHECK: hipInit(0);
    cuInit(0);
    testResult = runTest(argc, argv);
    printf("%s ...\n", testResult ? "PASSED" : "FAILED");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
    return 0;
}
