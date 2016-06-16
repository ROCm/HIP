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
// Simple test for memset.
// Also serves as a template for other tests.

#include "hip_runtime.h"
#include "test_common.h"


__global__ 
void
myKern(hipLaunchParm lp, int *C, const int *A, int N)
{
    int tid = hipThreadIdx_x;

    C[tid] = A[tid];
};



int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, true);

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

    HIPCHECK ( hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice) );

    hipLaunchKernel(myKern, dim3(N), dim3(256), 0, 0, A_d, C_d, N);

    HIPCHECK ( hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost) );

    for (int i=0; i<N; i++) {
        int goldVal = i * 10;
        if (A_h[i] != goldVal) {
            failed("mismatch at index:%d computed:%02x, gold:%02x\n", i, (int)A_h[i], (int)goldVal);

        }
    }

    passed();

};
