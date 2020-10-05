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
 * TEST: %t
 * HIT_END
 */

#include <iostream>
#include <hip/hip_runtime.h>
#include "test_common.h"

#define WIDTH 4

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
template <typename T>
__global__ void matrixTranspose(T* out, T* in, const int width) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    T val = in[x];
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) out[i * width + j] = __shfl(val, j * width + i);
    }
}

// CPU implementation of matrix transpose
template <typename T>
void matrixTransposeCPUReference(T* output, T* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

template<typename T>
void runTest() {
    T* Matrix;
    T* TransposeMatrix;
    T* cpuTransposeMatrix;

    T* gpuMatrix;
    T* gpuTransposeMatrix;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    int i;
    int errors;

    Matrix = (T*)malloc(NUM * sizeof(T));
    TransposeMatrix = (T*)malloc(NUM * sizeof(T));
    cpuTransposeMatrix = (T*)malloc(NUM * sizeof(T));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix[i] = (T)i * 10l;
    }

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatrix, NUM * sizeof(T));
    hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(T));

    // Memory transfer from host to device
    hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(T), hipMemcpyHostToDevice);

    // Lauching kernel from host
    hipLaunchKernelGGL(matrixTranspose<T>, dim3(1), dim3(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y), 0, 0,
                    gpuTransposeMatrix, gpuMatrix, WIDTH);

    // Memory transfer from device to host
    hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(T), hipMemcpyDeviceToHost);

    // CPU MatrixTranspose computation
    matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    errors = 0;
    double eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
        if (TransposeMatrix[i] != cpuTransposeMatrix[i]) {
            errors++;
        }
    }

    // free the resources on device side
    hipFree(gpuMatrix);
    hipFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

    if (errors != 0) {
        failed("Mismatch present");
    }
}

int main() {
    runTest<int>();
    runTest<float>();
    runTest<long>();
    runTest<long long>();
    passed();
}
