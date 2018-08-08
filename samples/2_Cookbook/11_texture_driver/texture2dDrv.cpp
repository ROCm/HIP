/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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
//#include "hip/hip_runtime_api.h"
#include <iostream>
#include <fstream>
#include <vector>
//#include <hip/hip_hcc.h>

#define fileName "tex2dKernel.code"

texture<float, 2, hipReadModeElementType> tex;
bool testResult = false;

#define HIP_CHECK(cmd)                                                                             \
    {                                                                                              \
        hipError_t status = cmd;                                                                   \
        if (status != hipSuccess) {                                                                \
            std::cout << "error: #" << status << " (" << hipGetErrorString(status)                 \
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
    hipModule_t Module;
    HIP_CHECK(hipModuleLoad(&Module, fileName));

    hipArray* array;
    HIP_ARRAY_DESCRIPTOR desc;
    desc.format = HIP_AD_FORMAT_FLOAT;
    desc.numChannels = 1;
    desc.width = width;
    desc.height = height;
    hipArrayCreate(&array, &desc);

    hip_Memcpy2D copyParam;
    memset(&copyParam, 0, sizeof(copyParam));
    copyParam.dstMemoryType = hipMemoryTypeArray;
    copyParam.dstArray = array;
    copyParam.srcMemoryType = hipMemoryTypeHost;
    copyParam.srcHost = hData;
    copyParam.srcPitch = width * sizeof(float);
    copyParam.widthInBytes = copyParam.srcPitch;
    copyParam.height = height;
    hipMemcpyParam2D(&copyParam);

    textureReference* texref;
    hipModuleGetTexRef(&texref, Module, "tex");
    hipTexRefSetAddressMode(texref, 0, hipAddressModeWrap);
    hipTexRefSetAddressMode(texref, 1, hipAddressModeWrap);
    hipTexRefSetFilterMode(texref, hipFilterModePoint);
    hipTexRefSetFlags(texref, 0);
    hipTexRefSetFormat(texref, HIP_AD_FORMAT_FLOAT, 1);
    hipTexRefSetArray(texref, array, HIP_TRSA_OVERRIDE_FORMAT);

    float* dData = NULL;
    hipMalloc((void**)&dData, size);

    struct {
        void* _Ad;
        unsigned int _Bd;
        unsigned int _Cd;
    } args;
    args._Ad = (void*) dData;
    args._Bd = width;
    args._Cd = height;

    size_t sizeTemp = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &sizeTemp, HIP_LAUNCH_PARAM_END};

    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "tex2dKernel"));

    int temp1 = width / 16;
    int temp2 = height / 16;
    HIP_CHECK(
        hipModuleLaunchKernel(Function, 16, 16, 1, temp1, temp2, 1, 0, 0, NULL, (void**)&config));
    hipDeviceSynchronize();

    float* hOutputData = (float*)malloc(size);
    memset(hOutputData, 0, size);
    hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost);

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
    hipFree(dData);
    hipFreeArray(array);
    return true;
}

int main(int argc, char** argv) {
    hipInit(0);
    testResult = runTest(argc, argv);
    printf("%s ...\n", testResult ? "PASSED" : "FAILED");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
    return 0;
}
