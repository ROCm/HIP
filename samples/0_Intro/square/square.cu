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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(cmd) \
{\
    cudaError_t error  = cmd;\
    if (error != cudaSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}


/* 
 * Square each element in the array A and write to array C.
 */
template <typename T>
__global__ void
vector_square(T *C_d, T *A_d, size_t N)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x ;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}


int main(int argc, char *argv[])
{
    float *A_d, *C_d;
    float *A_h, *C_h;
    size_t N = 1000000;
    size_t Nbytes = N * sizeof(float);

    cudaDeviceProp props;
    CHECK(cudaGetDeviceProperties(&props, 0/*deviceID*/));
    printf ("info: running on device %s\n", props.name);

    printf ("info: allocate host mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
    A_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? cudaErrorMemoryAllocation : cudaSuccess );
    C_h = (float*)malloc(Nbytes);
    CHECK(C_h == 0 ? cudaErrorMemoryAllocation : cudaSuccess );
    // Fill with Phi + i
    for (size_t i=0; i<N; i++) 
    {
        A_h[i] = 1.618f + i; 
    }

    printf ("info: allocate device mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
    CHECK(cudaMalloc(&A_d, Nbytes));
    CHECK(cudaMalloc(&C_d, Nbytes));


    printf ("info: copy Host2Device\n");
    CHECK ( cudaMemcpy(A_d, A_h, Nbytes, cudaMemcpyHostToDevice));

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    printf ("info: launch 'vector_square' kernel\n");
    vector_square <<<blocks, threadsPerBlock>>> (C_d, A_d, N);

    printf ("info: copy Device2Host\n");
    CHECK ( cudaMemcpy(C_h, C_d, Nbytes, cudaMemcpyDeviceToHost));

    printf ("info: check result\n");
    for (size_t i=0; i<N; i++)  {
        if (C_h[i] != A_h[i] * A_h[i]) {
            CHECK(cudaErrorUnknown);
        }
    }
    printf ("PASSED!\n");
}
