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
#include "hip_runtime.h"
#include "test_common.h"


//---
// Test simple H2D copies and back.
// Designed to stress a small number of simple smoke tests
void simpleTest1()
{
    printf ("test: %s\n", __func__);
    size_t Nbytes = N*sizeof(int);
    printf ("N=%zu Nbytes=%6.2fMB\n", N, Nbytes/1024.0/1024.0);

    int *A_d, *B_d, *C_d;
    int *A_h, *B_h, *C_h;

    HipTest::initArrays (&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

    printf ("A_d=%p B_d=%p C_d=%p  A_h=%p B_h=%p C_h=%p\n", A_d, B_d, C_d, A_h, B_d, C_h);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

    HIPCHECK ( hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIPCHECK ( hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

    hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, B_d, C_d, N);

    HIPCHECK ( hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    HIPCHECK (hipDeviceSynchronize());

    HipTest::checkVectorADD(A_h, B_h, C_h, N);

    HipTest::freeArrays (A_d, B_d, C_d, A_h, B_h, C_h, false);
    HIPCHECK (hipDeviceReset());

    printf ("  %s success\n", __func__);
}


template <typename T>
void simpleTest2(size_t numElements, bool usePinnedHost)
{
    size_t sizeElements = numElements * sizeof(T);
    size_t alignment = 4096;
    printf ("test: %s<%s> numElements=%zu sizeElements=%zu bytes\n", __func__, TYPENAME(T), numElements, sizeElements);

    T *A_d, *A_h1, *A_h2;

    if (usePinnedHost) {
        HIPCHECK ( hipHostMalloc((void**)&A_h1, sizeElements, hipHostMallocDefault) );
        HIPCHECK ( hipHostMalloc((void**)&A_h2, sizeElements, hipHostMallocDefault) );
    } else {
        A_h1 = (T*)aligned_alloc(alignment, sizeElements);
        HIPASSERT(A_h1);
        A_h2 = (T*)aligned_alloc(alignment, sizeElements);
        HIPASSERT(A_h1);
    }

    // Alloc device array:
    HIPCHECK ( hipMalloc(&A_d, sizeElements) );


    for (size_t i=0; i<numElements; i++) {
        A_h1[i] = 3.14f+ 1000*i;
        A_h2[i] = 12345678.0 + i; // init output with something distincctive, to ensure we replace it.
    }

    HIPCHECK(hipMemcpy(A_d, A_h1, sizeElements, hipMemcpyHostToDevice));
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(A_h2, A_d, sizeElements, hipMemcpyDeviceToHost));
    HIPCHECK(hipDeviceSynchronize());

    for (size_t i=0; i<numElements; i++) {
        HIPASSERT(A_h1[i] == A_h2[i]);
    }

    HIPCHECK(hipFree(A_d));
    if (usePinnedHost) {
        HIPCHECK(hipHostFree(A_h1));
        HIPCHECK(hipHostFree(A_h2));
    } else {
        free(A_h1);
        free(A_h2);
    }
}
int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, true);

    printf ("info: set device to %d, tests=%x\n", p_gpuDevice, p_tests);
    HIPCHECK(hipSetDevice(p_gpuDevice));


    if (p_tests & 0x1) {
        printf ("\n\n=== tests&1\n");
        HIPCHECK ( hipDeviceReset() );
        simpleTest1();
        printf ("===\n\n\n");
    }

    if (p_tests & 0x2) {
        printf ("\n\n=== tests&2 (copy pin-pong, pinned host)\n");
        simpleTest2<float>(N, true/*usePinnedHost*/);
        simpleTest2<char>(N, true/*usePinnedHost*/);
    }

    if (p_tests & 0x4) {
        printf ("\n\n=== tests&2 (copy pin-pong, unpinned host)\n");
        simpleTest2<char>(N, false/*usePinnedHost*/);
        simpleTest2<float>(N, false/*usePinnedHost*/);
    }

    passed();
};
