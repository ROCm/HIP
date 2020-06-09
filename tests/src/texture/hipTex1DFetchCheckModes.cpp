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

/* HIT_START
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_RUNTIME rocclr
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "../test_common.h"

#define N 16
#define offset 3
__global__ void tex1dKernel(float *val, hipTextureObject_t obj) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N)
        val[k] = tex1Dfetch<float>(obj, k+offset);
}

int runTest(hipTextureAddressMode, hipTextureFilterMode);

int main(int argc, char **argv) {
    int testResult = runTest(hipAddressModeClamp,hipFilterModePoint);
    testResult = runTest(hipAddressModeClamp,hipFilterModeLinear);
    testResult = runTest(hipAddressModeWrap,hipFilterModePoint);
    testResult = runTest(hipAddressModeWrap,hipFilterModeLinear);
    if(testResult) {
        passed();
    } else {
        exit(EXIT_FAILURE);
    }
}

int runTest(hipTextureAddressMode addressMode, hipTextureFilterMode filterMode) {

    int testResult = 1;

    hipCtx_t HipContext;
    hipDevice_t HipDevice;
    int deviceID = 0;
    hipDeviceGet(&HipDevice, deviceID);
    hipCtxCreate(&HipContext, 0, HipDevice);

    // Allocating the required buffer on gpu device
    float *texBuf, *texBufOut;
    float val[N], output[N];
    
    for (int i = 0; i < N; i++) {
        val[i] = i+1;
        output[i] = 0.0;
    }

    HIPCHECK(hipMalloc(&texBuf, N * sizeof(float)));
    HIPCHECK(hipMalloc(&texBufOut, N * sizeof(float)));
    HIPCHECK(hipMemcpy(texBuf, val, N * sizeof(float), hipMemcpyHostToDevice));
    HIPCHECK(hipMemset(texBufOut, 0, N * sizeof(float)));
    hipResourceDesc resDescLinear;

    memset(&resDescLinear, 0, sizeof(resDescLinear));
    resDescLinear.resType = hipResourceTypeLinear;
    resDescLinear.res.linear.devPtr = texBuf;
    resDescLinear.res.linear.desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
    resDescLinear.res.linear.sizeInBytes = N * sizeof(float);

    hipTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = hipReadModeElementType;

    texDesc.addressMode[0] = addressMode;
    texDesc.addressMode[1] = addressMode;
    texDesc.filterMode = filterMode;   
    texDesc.normalizedCoords = false;

    // Creating texture object
    hipTextureObject_t texObj = 0;
    HIPCHECK(hipCreateTextureObject(&texObj, &resDescLinear, &texDesc, NULL));

    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(N , 1, 1);

    hipLaunchKernelGGL(tex1dKernel, dim3(dimGrid), dim3(dimBlock), 0, 0,
                       texBufOut, texObj);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipMemcpy(output, texBufOut, N * sizeof(float), hipMemcpyDeviceToHost));

    for (int i = offset; i < N; i++) {
        if (output[i-offset] != val[i]) {
            testResult = 0;
            break;
        }
    }
    if(testResult){
        for(int i = N-offset; i < N; i++){
           if (output[i] != 0){
               testResult = 0;
               break;
           }
        }
    }
    HIPCHECK(hipDestroyTextureObject(texObj));
    HIPCHECK(hipFree(texBuf));
    HIPCHECK(hipFree(texBufOut));
    return testResult;
}
