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
// Simple test for hipExtLaunchMultiKernelMultiDevice API. It can be tested on
// single GPU or multi GPUs.

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"
#include <stdio.h>

#define MAX_GPUS 8
/* 
 * Square each element in the array A and write to array C.
 */
#define NUM_KERNEL_ARGS 3
__global__ void
vector_square(float *C_d, float *A_d, size_t N)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x ;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

int main(int argc, char *argv[])
{
    float *A_d[MAX_GPUS], *C_d[MAX_GPUS];
    float *A_h, *C_h;
    size_t N = 1000000;
    size_t Nbytes = N * sizeof(float);

    int nGpu = 0;
    HIPCHECK(hipGetDeviceCount(&nGpu));
    if (nGpu < 1) {
        printf ("info: didn't find any GPU!\n");
        return 0;
    }
    if (nGpu > MAX_GPUS) {
        nGpu = MAX_GPUS;
    }

    printf ("info: allocate host mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
    A_h = (float*)malloc(Nbytes);
    HIPCHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    C_h = (float*)malloc(Nbytes);
    HIPCHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    // Fill with Phi + i
    for (size_t i = 0; i < N; i++)
    {
        A_h[i] = 1.618f + i; 
    }

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    hipStream_t stream[MAX_GPUS];
    for (int i = 0; i < nGpu; i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipStreamCreateWithFlags(&stream[i], hipStreamNonBlocking));

        hipDeviceProp_t props;
        HIPCHECK(hipGetDeviceProperties(&props, i/*deviceID*/));
        printf ("info: running on bus 0x%2x %s\n", props.pciBusID, props.name);

        printf ("info: allocate device mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
        HIPCHECK(hipMalloc(&A_d[i], Nbytes));
        HIPCHECK(hipMalloc(&C_d[i], Nbytes));


        printf ("info: copy Host2Device\n");
        HIPCHECK ( hipMemcpy(A_d[i], A_h, Nbytes, hipMemcpyHostToDevice));
    }

    hipLaunchParams *launchParamsList = reinterpret_cast<hipLaunchParams *>(
            malloc(sizeof(hipLaunchParams)*nGpu));

    void *args[MAX_GPUS * NUM_KERNEL_ARGS];

    for (int i = 0; i < nGpu; i++) {
        args[i * NUM_KERNEL_ARGS]     = &C_d[i];
        args[i * NUM_KERNEL_ARGS + 1] = &A_d[i];
        args[i * NUM_KERNEL_ARGS + 2] = &N;
        launchParamsList[i].func  =
                reinterpret_cast<void *>(vector_square);
        launchParamsList[i].gridDim   = dim3(blocks);
        launchParamsList[i].blockDim  = dim3(threadsPerBlock);
        launchParamsList[i].sharedMem = 0;
        launchParamsList[i].stream    = stream[i];
        launchParamsList[i].args      = args + i * NUM_KERNEL_ARGS;
    }

    printf ("info: launch vector_square kernel with hipExtLaunchMultiKernelMultiDevice API\n");
    hipExtLaunchMultiKernelMultiDevice(launchParamsList, nGpu, 0);

    for (int j = 0; j < nGpu; j++) {
        hipStreamSynchronize(stream[j]);

        hipDeviceProp_t props;
        HIPCHECK(hipGetDeviceProperties(&props, j/*deviceID*/));
        printf ("info: checking result on bus 0x%2x %s\n", props.pciBusID, props.name);

        printf ("info: copy Device2Host\n");
        HIPCHECK( hipMemcpy(C_h, C_d[j], Nbytes, hipMemcpyDeviceToHost));

        printf ("info: check result\n");
        for (size_t i = 0; i < N; i++)  {
            if (C_h[i] != A_h[i] * A_h[i]) {
                HIPCHECK(hipErrorUnknown);
            }
        }
    }
    
    printf ("PASSED!\n");
}
