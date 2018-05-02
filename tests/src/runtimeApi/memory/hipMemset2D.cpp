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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

bool testhipMemset2D(int memsetval,int p_gpuDevice)
{
    size_t numH = 256;
    size_t numW = 256;
    size_t pitch_A;
    size_t width = numW * sizeof(char);
    size_t sizeElements = width * numH;
    size_t elements = numW* numH;


    printf ("testhipMemset2D memsetval=%2x device=%d\n", memsetval, p_gpuDevice);
    char *A_d;
    char *A_h;
    bool testResult = true;

    HIPCHECK ( hipMallocPitch((void**)&A_d, &pitch_A, width , numH) );
    A_h = (char*)malloc(sizeElements);
    HIPASSERT(A_h != NULL);
    for (size_t i=0; i<elements; i++) {
        A_h[i] = 1;
    }
    HIPCHECK ( hipMemset2D(A_d, pitch_A, memsetval, numW, numH) );
    HIPCHECK ( hipMemcpy2D(A_h, width, A_d, pitch_A, numW, numH, hipMemcpyDeviceToHost));

    for (int i=0; i<elements; i++) {
        if (A_h[i] != memsetval) {
            testResult = false;
            printf("testhipMemset2D mismatch at index:%d computed:%02x, memsetval:%02x\n", i, (int)A_h[i], (int)memsetval);
            break;
        }
    }
    hipFree(A_d);
    free(A_h);
    return testResult;
}

bool testhipMemset2DAsync(int memsetval,int p_gpuDevice)
{
    size_t numH = 256;
    size_t numW = 256;
    size_t pitch_A;
    size_t width = numW * sizeof(char);
    size_t sizeElements = width * numH;
    size_t elements = numW* numH;


    printf ("testhipMemset2DAsync memsetval=%2x device=%d\n", memsetval, p_gpuDevice);
    char *A_d;
    char *A_h;
    bool testResult = true;

    HIPCHECK ( hipMallocPitch((void**)&A_d, &pitch_A, width , numH) );
    A_h = (char*)malloc(sizeElements);
    HIPASSERT(A_h != NULL);
    for (size_t i=0; i<elements; i++) {
        A_h[i] = 1;
    }
    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK ( hipMemset2DAsync(A_d, pitch_A, memsetval, numW, numH, stream) );
    HIPCHECK ( hipMemcpy2D(A_h, width, A_d, pitch_A, numW, numH, hipMemcpyDeviceToHost));

    for (int i=0; i<elements; i++) {
        if (A_h[i] != memsetval) {
            testResult = false;
            printf("testhipMemset2DAsync mismatch at index:%d computed:%02x, memsetval:%02x\n", i, (int)A_h[i], (int)memsetval);
            break;
        }
    }
    hipFree(A_d);
    HIPCHECK(hipStreamDestroy(stream));
    free(A_h);
    return testResult;
}

int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, true);
    bool testResult = false;
    HIPCHECK(hipSetDevice(p_gpuDevice));

    testResult = testhipMemset2D(memsetval, p_gpuDevice);
    testResult = testhipMemset2DAsync(memsetval, p_gpuDevice);
    passed();

}
