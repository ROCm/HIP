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

/*HIT_START
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

#define N 512

__global__ void tex1dKernel(float *val, hipTextureObject_t obj) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N)
        val[k] = tex1Dfetch<float>(obj, k);
}

int runTest(void);

int main(int argc, char **argv) {
    int testResult = runTest();
    if(testResult) {
        passed();
    } else {
        exit(EXIT_FAILURE);
    }
}

int runTest() {
    int testResult = 1;
    // Allocating the required buffer on gpu device
    float *texBuf, *texBufOut;
    float val[N], output[N];
    for (int i = 0; i < N; i++) {
        val[i] = (i + 1) * (i + 1);
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

    // Creating texture object
    hipTextureObject_t texObj = 0;
    HIPCHECK(hipCreateTextureObject(&texObj, &resDescLinear, &texDesc, NULL));

    dim3 dimBlock(64, 1, 1);
    dim3 dimGrid(N / dimBlock.x, 1, 1);

    hipLaunchKernelGGL(tex1dKernel, dim3(dimGrid), dim3(dimBlock), 0, 0,
                       texBufOut, texObj);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipMemcpy(output, texBufOut, N * sizeof(float), hipMemcpyDeviceToHost));

    for(int i = 0; i < N; i++)
        if (output[i] != val[i]) {
            testResult = 0;
            break;
        }

    HIPCHECK(hipDestroyTextureObject(texObj));
    HIPCHECK(hipFree(texBuf));
    HIPCHECK(hipFree(texBufOut));
    return testResult;
}
