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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "hip/hip_ext.h"
#include "test_common.h"

void test(size_t N) {
    size_t Nbytes = N * sizeof(int);
#if defined(__HIP_PLATFORM_HCC__) && GENERIC_GRID_LAUNCH == 1 && defined(__HCC__)
    int *A_d, *B_d, *C_d;
    int *A_h, *B_h, *C_h;

    HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N);


    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

    HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

    hipExtLaunchKernelGGL(HipTest::vectorADD, dim3(blocks),
                          dim3(threadsPerBlock), 0, 0, nullptr, nullptr, 0,
                          static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d, N);

    HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    HIPCHECK(hipDeviceSynchronize());

    HipTest::checkVectorADD(A_h, B_h, C_h, N);
#endif
}

int main(int argc, char* argv[]) {
    HipTest::parseStandardArguments(argc, argv, true);

    test(N);

    passed();
}
