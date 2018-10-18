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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANUMTY OF ANY KIND, EXPRESS OR
IMPLIED, INUMCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNUMESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANUMY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INUM AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INUM CONUMECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <hip/hip_runtime.h>

#define WIDTH 32

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

using namespace std;

__global__ void matrixTranspose_static_shared(float* out, float* in,
                                              const int width) {
    __shared__ float sharedMem[WIDTH * WIDTH];

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    sharedMem[y * width + x] = in[x * width + y];

    __syncthreads();

    out[y * width + x] = sharedMem[y * width + x];
}

__global__ void matrixTranspose_dynamic_shared(float* out, float* in,
                                               const int width) {
    // declare dynamic shared memory
    HIP_DYNAMIC_SHARED(float, sharedMem)

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    sharedMem[y * width + x] = in[x * width + y];

    __syncthreads();

    out[y * width + x] = sharedMem[y * width + x];
}

void MultipleStream(float** data, float* randArray, float** gpuTransposeMatrix,
                    float** TransposeMatrix, int width) {
    const int num_streams = 2;
    hipStream_t streams[num_streams];

    for (int i = 0; i < num_streams; i++) hipStreamCreate(&streams[i]);

    for (int i = 0; i < num_streams; i++) {
        hipMalloc((void**)&data[i], NUM * sizeof(float));
        hipMemcpyAsync(data[i], randArray, NUM * sizeof(float), hipMemcpyHostToDevice, streams[i]);
    }

    hipLaunchKernelGGL(matrixTranspose_static_shared,
                    dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, streams[0],
                    gpuTransposeMatrix[0], data[0], width);

    hipLaunchKernelGGL(matrixTranspose_dynamic_shared,
                    dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), sizeof(float) * WIDTH * WIDTH,
                    streams[1], gpuTransposeMatrix[1], data[1], width);

    for (int i = 0; i < num_streams; i++)
        hipMemcpyAsync(TransposeMatrix[i], gpuTransposeMatrix[i], NUM * sizeof(float),
                       hipMemcpyDeviceToHost, streams[i]);
}

int main() {
    hipSetDevice(0);

    float *data[2], *TransposeMatrix[2], *gpuTransposeMatrix[2], *randArray;

    int width = WIDTH;

    randArray = (float*)malloc(NUM * sizeof(float));

    TransposeMatrix[0] = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix[1] = (float*)malloc(NUM * sizeof(float));

    hipMalloc((void**)&gpuTransposeMatrix[0], NUM * sizeof(float));
    hipMalloc((void**)&gpuTransposeMatrix[1], NUM * sizeof(float));

    for (int i = 0; i < NUM; i++) {
        randArray[i] = (float)i * 1.0f;
    }

    MultipleStream(data, randArray, gpuTransposeMatrix, TransposeMatrix, width);

    hipDeviceSynchronize();

    // verify the results
    int errors = 0;
    double eps = 1.0E-6;
    for (int i = 0; i < NUM; i++) {
        if (std::abs(TransposeMatrix[0][i] - TransposeMatrix[1][i]) > eps) {
            printf("%d stream0: %f stream1  %f\n", i, TransposeMatrix[0][i], TransposeMatrix[1][i]);
            errors++;
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("stream PASSED!\n");
    }

    free(randArray);
    for (int i = 0; i < 2; i++) {
        hipFree(data[i]);
        hipFree(gpuTransposeMatrix[i]);
        free(TransposeMatrix[i]);
    }

    hipDeviceReset();
    return 0;
}
