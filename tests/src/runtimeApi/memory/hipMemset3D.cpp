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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

bool testhipMemset3D(int memsetval,int p_gpuDevice)
{
    size_t numH = 256;
    size_t numW = 256;
    size_t depth = 10;
    size_t width = numW * sizeof(char);
    size_t sizeElements = width * numH * depth;
    size_t elements = numW* numH* depth;


	printf ("testhipMemset3D memsetval=%2x device=%d\n", memsetval, p_gpuDevice);
    char *A_h;
    bool testResult = true;
    hipExtent extent = make_hipExtent(width, numH, depth);
    hipPitchedPtr devPitchedPtr;

    HIPCHECK(hipMalloc3D(&devPitchedPtr, extent));
	A_h = (char*)malloc(sizeElements);
	HIPASSERT(A_h != NULL);
	for (size_t i=0; i<elements; i++) {
        A_h[i] = 1;
    }
	HIPCHECK ( hipMemset3D( devPitchedPtr, memsetval, extent) );
	hipMemcpy3DParms myparms = {0};
	myparms.srcPos = make_hipPos(0,0,0);
	myparms.dstPos = make_hipPos(0,0,0);
	myparms.dstPtr = make_hipPitchedPtr(A_h, width , numW, numH);
	myparms.srcPtr = devPitchedPtr;
	myparms.extent = extent;
#ifdef __HIP_PLATFORM_NVCC__
	myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
    myparms.kind = hipMemcpyDeviceToHost;
#endif
    HIPCHECK(hipMemcpy3D(&myparms));

    for (int i=0; i<elements; i++) {
        if (A_h[i] != memsetval) {
			testResult = false;
            printf("mismatch at index:%d computed:%02x, memsetval:%02x\n", i, (int)A_h[i], (int)memsetval);
            break;
        }
    }
    HIPCHECK(hipFree(devPitchedPtr.ptr));
    free(A_h);
    return testResult;
}

int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, true);
    bool testResult = false;
    HIPCHECK(hipSetDevice(p_gpuDevice));
    testResult = testhipMemset3D(memsetval, p_gpuDevice);
    if (testResult) {
        passed();
    } else {
        exit(EXIT_FAILURE);
    }
}
