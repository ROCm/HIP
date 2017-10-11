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
 * hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t sizeBytes);
 */

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "test_common.h"
int main()
{

    float *A_d, *B_d, *C_d ;
    float *A_h, *B_h, *C_h ;
    size_t	numW = 321;
    size_t	numH = 211;
    size_t width = numW * sizeof(float);
    size_t sizeElements = width * numH;
    size_t pitch_A, pitch_B, pitch_C;
    int numDevices = 0;

    hipStream_t s;

    HIPCHECK(hipSetDevice(0)); 
    hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
    HipTest::initArrays2DPitch(&A_d, &B_d, &C_d, &pitch_A, &pitch_B, &pitch_C, numW, numH);
    HipTest::initArraysForHost(&A_h, &B_h, &C_h, numW*numH, 0);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numW*numH);
 
    HIPCHECK(hipStreamCreate(&s));
 
    HIPCHECK (hipMemcpy2DAsync (A_d, pitch_A, A_h, width, width, numH, hipMemcpyHostToDevice, s) );
    HIPCHECK (hipMemcpy2DAsync (B_d, pitch_B, B_h, width, width, numH, hipMemcpyHostToDevice, s) );
 
    hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, B_d, C_d, (pitch_C/sizeof(float))*numH);
 
    HIPCHECK (hipMemcpy2DAsync (C_h, width, C_d, pitch_C, width, numH, hipMemcpyDeviceToHost, s) );
    HIPCHECK ( hipDeviceSynchronize() );
    HIPCHECK ( hipStreamSynchronize(s) )
 
    HipTest::checkVectorADD(A_h, B_h, C_h, numW*numH);
 
    passed();
}

