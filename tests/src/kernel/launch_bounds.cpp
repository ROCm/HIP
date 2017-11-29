/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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

// Test launch bounds and initialization conditions.

#include "hip/hip_runtime.h"
#include "test_common.h"

int p_blockSize = 256;


__global__
void
__launch_bounds__(256, 2)
myKern(hipLaunchParm lp, int *C, const int *A, int N, int xfactor)
{
    int tid = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);

    if (tid < N) {
        C[tid] = A[tid];
    }
};


void parseMyArguments(int argc, char *argv[])
{
    int more_argc = HipTest::parseStandardArguments(argc, argv, false);
    // parse args for this test:
    for (int i = 1; i < more_argc; i++) {
        const char *arg = argv[i];

        if (!strcmp(arg, "--blockSize")) {
            if (++i >= argc || !HipTest::parseInt(argv[i], &p_blockSize)) {
               failed("Bad peerDevice argument");
            }
        } else {
            failed("Bad argument '%s'", arg);
        }
    };
};




int main(int argc, char *argv[])
{
    parseMyArguments(argc, argv);

    size_t Nbytes = N*sizeof(int);

    int *A_d, *C_d, *A_h, *C_h;
    HIPCHECK ( hipMalloc(&A_d, Nbytes) );
    HIPCHECK ( hipMalloc(&C_d, Nbytes) );

    A_h = (int*)malloc (Nbytes);
    C_h = (int*)malloc (Nbytes);

    for (int i=0; i<N; i++) {
        A_h[i] = i*10;
        C_h[i] = 0x0;
    }

    int blocks = N / p_blockSize;
    printf ("running with N=%zu p_blockSize=%d blocks=%d\n", N, p_blockSize, blocks);

    HIPCHECK ( hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice) );
    HIPCHECK ( hipGetLastError() );

    hipLaunchKernel(myKern, dim3(blocks), dim3(p_blockSize), 0, 0, C_d, A_d, N, 0);

#ifdef __HIP_PLATFORM_NVCC__
    cudaFuncAttributes attrib;
    cudaFuncGetAttributes (&attrib, myKern);
    printf ("binaryVersion = %d\n", attrib.binaryVersion);
    printf ("cacheModeCA = %d\n", attrib.cacheModeCA);
    printf ("constSizeBytes = %zu\n", attrib.constSizeBytes);
    printf ("localSizeBytes = %zud\n", attrib.localSizeBytes);
    printf ("maxThreadsPerBlock = %d\n", attrib.maxThreadsPerBlock);
    printf ("numRegs = %d\n", attrib.numRegs);
    printf ("ptxVersion = %d\n", attrib.ptxVersion);
    printf ("sharedSizeBytes = %zud\n", attrib.sharedSizeBytes);
#endif

    HIPCHECK ( hipDeviceSynchronize() );

    HIPCHECK ( hipGetLastError() );

    HIPCHECK ( hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost) );

    HIPCHECK ( hipDeviceSynchronize() );

    for (int i=0; i<N; i++) {
        int goldVal = i * 10;
        if (C_h[i] != goldVal) {
            failed("mismatch at index:%d computed:%02d, gold:%02d\n", i, (int)C_h[i], (int)goldVal);

        }
    }

    passed();

};
