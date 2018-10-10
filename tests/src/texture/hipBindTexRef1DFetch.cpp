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


/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

#define N 512

texture<float, 1, hipReadModeElementType> tex;

__global__ void kernel(float *out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x<N){
      out[x] = tex1Dfetch(tex, x);
  }
}

int runTest(void);

int main(int argc, char **argv) {
    int testResult = runTest();
    if (testResult) {
        passed();
    } else {
        exit(EXIT_FAILURE);
    }
}

int runTest() {
    int testResult = 1;
    float *texBuf;
    float val[N], output[N];
    size_t offset = 0;
    float *devBuf;
    for (int i = 0; i < N; i++) {
        val[i] = (float)i;
        output[i] = 0.0;
    }
    hipChannelFormatDesc chanDesc =
        hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);

    HIPCHECK(hipMalloc(&texBuf, N * sizeof(float)));
    HIPCHECK(hipMalloc(&devBuf, N * sizeof(float)));
    HIPCHECK(hipMemcpy(texBuf, val, N * sizeof(float), hipMemcpyHostToDevice));
  
    tex.addressMode[0] = hipAddressModeClamp;
    tex.addressMode[1] = hipAddressModeClamp;
    tex.filterMode = hipFilterModePoint;
    tex.normalized = 0;

    HIPCHECK(hipBindTexture(&offset, tex, (void *)texBuf, chanDesc, N * sizeof(float)));
    HIPCHECK(hipGetTextureAlignmentOffset(&offset,&tex));

    dim3 dimBlock(64, 1, 1);
    dim3 dimGrid(N / dimBlock.x, 1, 1);

    hipLaunchKernelGGL(kernel, dim3(dimGrid), dim3(dimBlock), 0, 0, devBuf);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(output, devBuf, N * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        if (output[i] != val[i]) {
            testResult = 0;
            break;
        }
    }
    HIPCHECK(hipUnbindTexture(&tex));
    HIPCHECK(hipFree(texBuf));
    HIPCHECK(hipFree(devBuf));
    return testResult;
}
