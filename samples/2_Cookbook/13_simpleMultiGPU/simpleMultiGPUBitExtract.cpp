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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Simple test showing how to use multi GPU


#include "hip/hip_runtime.h"
// System includes
#include <stdio.h>
#include <assert.h>

// HIP runtime
#include <hip/hip_runtime.h>

#define CHECK(cmd)                                                                             \
{                                                                                              \
    hipError_t error = cmd;                                                                    \
    if (error != hipSuccess) {                                                                 \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                __FILE__, __LINE__);                                                           \
        exit(EXIT_FAILURE);                                                                    \
    }                                                                                          \
}

__global__ void bit_extract_kernel(const uint32_t* d_input, uint32_t* d_output, size_t N)
{
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if( offset < N)
    {
        d_output[offset] = ((d_input[offset] & 0xf00) >> 8);
    }
}

void initialData(uint32_t *ip, int size)
{
    static uint32_t data = 0;
    for (int i = 0; i < size; i++)
    {
        ip[i] = data++;
    }
    return;
}

int main(int argc , char** argv)
{
    int ngpus;
    hipGetDeviceCount(&ngpus);
    printf(" HIP-capable devices: %i\n", ngpus);

    uint32_t **d_input, **d_output;
    uint32_t **h_input, **h_output;
    hipStream_t * stream;

    //input array size
    size_t N = 1000000000;
    size_t NPerGPU = N/ngpus;
    size_t NbytesPerGPU = NPerGPU * sizeof(uint32_t);

    d_input  = (uint32_t **)malloc(sizeof(uint32_t *)*ngpus);
    d_output = (uint32_t **)malloc(sizeof(uint32_t *)*ngpus);

    h_input  = (uint32_t **)malloc(sizeof(uint32_t *)*ngpus);
    h_output = (uint32_t **)malloc(sizeof(uint32_t *)*ngpus);
    stream   = (hipStream_t *) malloc(sizeof(hipStream_t) * ngpus);
    printf("info: Allocatedd total memory %zu, memory per GPU %zu\n",N,NPerGPU);
    for (int i = 0; i < ngpus; i++)
    {
        // set current device
        CHECK(hipSetDevice(i));

        // allocate device memory
        CHECK(hipMalloc((void **) &d_input[i] , NbytesPerGPU));
        CHECK(hipMalloc((void **) &d_output[i], NbytesPerGPU));

        // allocate page locked host memory for asynchronous data transfer
        CHECK(hipHostMalloc((void **) &h_input[i], NbytesPerGPU));

        // allocate page locked host memory for asynchronous data transfer
        CHECK(hipHostMalloc((void **) &h_output[i], NbytesPerGPU));

        // create streams for timing and synchronizing
        CHECK(hipStreamCreate(&stream[i]));

        initialData(h_input[i], NPerGPU);
    }
    printf("info: distributing workload across multiple devices \n");
    // distributing workload across multiple devices
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(hipSetDevice(i));
        CHECK(hipMemcpyAsync(d_input[i], h_input[i], NbytesPerGPU, hipMemcpyHostToDevice, stream[i]));

        dim3 block(256,1,1);
        dim3 grid((NPerGPU+block.x-1)/256,1,1);

        hipLaunchKernelGGL((bit_extract_kernel), dim3(grid), dim3(block), 0, stream[i], d_input[i], d_output[i], NPerGPU);

        CHECK(hipMemcpyAsync(h_output[i], d_output[i], NbytesPerGPU, hipMemcpyDeviceToHost, stream[i]));
    }
    // synchronize streams
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(hipSetDevice(i));
        CHECK(hipStreamSynchronize(stream[i]));
    }

    printf("info: check result\n");
    for (size_t i = 0; i < ngpus; i++)
    {
        for(size_t j = 0; j < NPerGPU; j++)
        {
            uint32_t hgold = ((h_input[i][j] & 0xf00) >> 8);
            if (h_output[i][j] != hgold) {
                fprintf(stderr, "mismatch detected.\n");
                printf("%zu: %08x =? %08x (Ain=%08x)\n", j, h_output[i][j], hgold, h_input[i][j]);
                CHECK(hipErrorUnknown);
            }
        }
    }
    printf("PASSED!\n");

    // Cleanup and shutdown
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(hipSetDevice(i));
        // free  memory
        CHECK(hipFree(d_input[i]));
        CHECK(hipFree(d_output[i]));
        CHECK(hipHostFree(h_input[i]));
        CHECK(hipHostFree(h_output[i]));
		
        CHECK(hipStreamDestroy(stream[i]));
		
        // reset device
        CHECK(hipDeviceReset());
    }

    free(d_input);
    free(d_output);
    free(h_input);
    free(h_output);
    free(stream);
}

