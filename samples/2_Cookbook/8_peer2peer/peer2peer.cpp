/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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
#include <assert.h>
#define WIDTH 32

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

using namespace std;

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"

#define failed(...)                                                                                \
    printf("%serror: ", KRED);                                                                     \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("error: TEST FAILED\n%s", KNRM);                                                        \
    abort();

#define HIPCHECK(error)                                                                            \
    {                                                                                              \
        hipError_t localError = error;                                                             \
        if ((localError != hipSuccess)&& (localError != hipErrorPeerAccessAlreadyEnabled)&&        \
                     (localError != hipErrorPeerAccessNotEnabled )) {                              \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, hipGetErrorString(localError),  \
                   localError, #error, __FILE__, __LINE__, KNRM);                                  \
            failed("API returned error code.");                                                    \
        }                                                                                          \
    }

void checkPeer2PeerSupport() {
    int gpuCount;
    int canAccessPeer;

    HIPCHECK(hipGetDeviceCount(&gpuCount));

    for (int currentGpu = 0; currentGpu < gpuCount; currentGpu++) {
        HIPCHECK(hipSetDevice(currentGpu));

        for (int peerGpu = 0; peerGpu < currentGpu; peerGpu++) {
            if (currentGpu != peerGpu) {
                HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer, currentGpu, peerGpu));
                printf("currentGpu#%d canAccessPeer: peerGpu#%d=%d\n", currentGpu, peerGpu,
                       canAccessPeer);
            }

            HIPCHECK(hipSetDevice(peerGpu));
            HIPCHECK(hipDeviceReset());
        }
        HIPCHECK(hipSetDevice(currentGpu));
        HIPCHECK(hipDeviceReset());
    }
}

void enablePeer2Peer(int currentGpu, int peerGpu) {
    int canAccessPeer;

    // Must be on a multi-gpu system:
    assert(currentGpu != peerGpu);

    HIPCHECK(hipSetDevice(currentGpu));
    hipDeviceCanAccessPeer(&canAccessPeer, currentGpu, peerGpu);

    if (canAccessPeer == 1) {
        HIPCHECK(hipDeviceEnablePeerAccess(peerGpu, 0));
    } else
        printf("peer2peer transfer not possible between the selected gpu devices");
}

void disablePeer2Peer(int currentGpu, int peerGpu) {
    int canAccessPeer;

    // Must be on a multi-gpu system:
    assert(currentGpu != peerGpu);

    HIPCHECK(hipSetDevice(currentGpu));
    hipDeviceCanAccessPeer(&canAccessPeer, currentGpu, peerGpu);

    if (canAccessPeer == 1) {
        HIPCHECK(hipDeviceDisablePeerAccess(peerGpu));
    } else
        printf("peer2peer disable not required");
}


__global__ void matrixTranspose_static_shared(float* out, float* in,
                                              const int width) {
    __shared__ float sharedMem[WIDTH * WIDTH];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    sharedMem[y * width + x] = in[x * width + y];

    __syncthreads();

    out[y * width + x] = sharedMem[y * width + x];
}

__global__ void matrixTranspose_dynamic_shared(float* out, float* in,
                                               const int width) {
    extern __shared__ float sharedMem[];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    sharedMem[y * width + x] = in[x * width + y];

    __syncthreads();

    out[y * width + x] = sharedMem[y * width + x];
}

int main() {
    checkPeer2PeerSupport();

    int gpuCount;
    int currentGpu, peerGpu;

    HIPCHECK(hipGetDeviceCount(&gpuCount));

    if (gpuCount < 2) {
        printf("Peer2Peer application requires atleast 2 gpu devices");
        return 0;
    }

    currentGpu = 0;
    peerGpu = (currentGpu + 1);

    printf("currentGpu=%d peerGpu=%d (Total no. of gpu = %d)\n", currentGpu, peerGpu, gpuCount);

    float *data[2], *TransposeMatrix[2], *gpuTransposeMatrix[2], *randArray;

    int width = WIDTH;

    randArray = (float*)malloc(NUM * sizeof(float));

    for (int i = 0; i < NUM; i++) {
        randArray[i] = (float)i * 1.0f;
    }

    enablePeer2Peer(currentGpu, peerGpu);

    HIPCHECK(hipSetDevice(currentGpu));
    TransposeMatrix[0] = (float*)malloc(NUM * sizeof(float));
    hipMalloc((void**)&gpuTransposeMatrix[0], NUM * sizeof(float));
    hipMalloc((void**)&data[0], NUM * sizeof(float));
    hipMemcpy(data[0], randArray, NUM * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(matrixTranspose_static_shared,
                    dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix[0],
                    data[0], width);
    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipSetDevice(peerGpu));
    TransposeMatrix[1] = (float*)malloc(NUM * sizeof(float));
    hipMalloc((void**)&gpuTransposeMatrix[1], NUM * sizeof(float));
    hipMalloc((void**)&data[1], NUM * sizeof(float));
    hipMemcpy(data[1], gpuTransposeMatrix[0], NUM * sizeof(float), hipMemcpyDeviceToDevice);

    hipLaunchKernelGGL(matrixTranspose_dynamic_shared,
                    dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), sizeof(float) * WIDTH * WIDTH,
                    0, gpuTransposeMatrix[1], data[1], width);

    hipMemcpy(TransposeMatrix[1], gpuTransposeMatrix[1], NUM * sizeof(float),
              hipMemcpyDeviceToHost);

    hipDeviceSynchronize();

    disablePeer2Peer(currentGpu, peerGpu);

    // verify the results
    int errors = 0;
    double eps = 1.0E-6;
    for (int i = 0; i < NUM; i++) {
        if (std::abs(randArray[i] - TransposeMatrix[1][i]) > eps) {
            printf("%d cpu: %f gpu peered data  %f\n", i, randArray[i], TransposeMatrix[1][i]);
            errors++;
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("Peer2Peer PASSED!\n");
    }

    free(randArray);
    for (int i = 0; i < 2; i++) {
        hipFree(data[i]);
        hipFree(gpuTransposeMatrix[i]);
        free(TransposeMatrix[i]);
    }

    HIPCHECK(hipSetDevice(peerGpu));
    HIPCHECK(hipDeviceReset());

    HIPCHECK(hipSetDevice(currentGpu));
    HIPCHECK(hipDeviceReset());

    return 0;
}
