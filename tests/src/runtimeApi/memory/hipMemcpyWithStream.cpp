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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
 * Conformance test for checking functionality of
 * hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t
 * sizeBytes);
 */

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"

int main() {
    size_t Nbytes = N * sizeof(int);
    int numDevices = 0;
    int *A_d, *B_d, *C_d;
    int *A_h, *B_h, *C_h;

    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
    HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));

    HIPCHECK(hipMemcpyWithStream(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));
    HIPCHECK(hipMemcpyWithStream(B_d, B_h, Nbytes, hipMemcpyHostToDevice,stream));
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, stream,
                        static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d, N);
    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
    HipTest::checkVectorADD(A_h, B_h, C_h, N);

    HIPCHECK(hipStreamDestroy(stream));
    passed();
}
