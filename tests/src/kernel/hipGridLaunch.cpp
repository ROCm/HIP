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
// Test the Grid_Launch syntax.

/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"


// __device__ maps to __attribute__((hc))
__device__ int foo(int i) { return i + 1; }

//---
// Syntax we would like to support with GRID_LAUNCH enabled:
template <typename T>
__global__ void vectorADD2(T* A_d, T* B_d, T* C_d, size_t N) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = offset; i < N; i += stride) {
        double foo = __hiloint2double(A_d[i], B_d[i]);
        C_d[i] = __double2loint(foo) + __double2hiint(foo);  // A_d[i] + B_d[i] ;
    }
}

int test_gl2(size_t N) {
    size_t Nbytes = N * sizeof(int);

    int *A_d, *B_d, *C_d;
    int *A_h, *B_h, *C_h;

    HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);


    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);


    // Full vadd in one large chunk, to get things started:
    HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(vectorADD2, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, B_d, C_d, N);

    HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    HIPCHECK(hipDeviceSynchronize());

    HipTest::checkVectorADD(A_h, B_h, C_h, N);

    return 0;
}

#if __HIP__
int test_triple_chevron(size_t N) {
    size_t Nbytes = N * sizeof(int);

    int *A_d, *B_d, *C_d;
    int *A_h, *B_h, *C_h;

    HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);


    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);


    // Full vadd in one large chunk, to get things started:
    HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

    vectorADD2<<<dim3(blocks), dim3(threadsPerBlock)>>>(A_d, B_d, C_d, N);

    HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    HIPCHECK(hipDeviceSynchronize());

    HipTest::checkVectorADD(A_h, B_h, C_h, N);

    return 0;
}
#endif

int main(int argc, char* argv[]) {
    HipTest::parseStandardArguments(argc, argv, true);

    test_gl2(N);

#if __HIP__
    test_triple_chevron(N);
#endif

    passed();
}
