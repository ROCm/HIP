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


// Test simple H2D copies and back.
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


// Test many different kinds of memory copies:

template <typename T>
void memcpyKind(bool usePinnedHost, bool useHostToHost, bool useMemkindDefault)
{
    printf ("test: %s\n", __func__);


    T *A_d, *B_d, *C_d;
    T *A_h, *B_h, *C_h;

    size_t Nbytes = N*sizeof(T);

    HipTest::initArrays (&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, usePinnedHost);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

    T *A_hh = NULL;
    T *B_hh = NULL;
    T *C_dd = NULL;

    // Allocate some extra arrays:
     
    HIPCHECK ( hipMalloc(&C_dd, Nbytes) );


    if (useHostToHost) {
        if (usePinnedHost) {
            HIPCHECK ( hipMallocHost(&A_hh, Nbytes) );
            HIPCHECK ( hipMallocHost(&B_hh, Nbytes) );
        } else {
            A_hh = (T*)malloc(Nbytes);
            B_hh = (T*)malloc(Nbytes);
        }


        // Do some extra host copies here to mix things up:
        HIPCHECK ( hipMemcpy(A_hh, A_h, Nbytes, useMemkindDefault? hipMemcpyDefault : hipMemcpyHostToHost));
        HIPCHECK ( hipMemcpy(B_hh, B_h, Nbytes, useMemkindDefault? hipMemcpyDefault : hipMemcpyHostToHost));


        HIPCHECK ( hipMemcpy(A_d, A_hh, Nbytes, useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
        HIPCHECK ( hipMemcpy(B_d, B_hh, Nbytes, useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
    } else {
        HIPCHECK ( hipMemcpy(A_d, A_h, Nbytes, useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
        HIPCHECK ( hipMemcpy(B_d, B_h, Nbytes, useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
    }

    hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, B_d, C_d, N);

#if 0
    // Do some extra host copies here to mix things up:
    HIPCHECK ( hipMemcpy(C_dd, C_d,  Nbytes, useMemkindDefault? hipMemcpyDefault : hipMemcpyHostToHost));

    //Destroy the original C_d:
    HIPCHECK ( hipMemset(C_d, 0x5A, Nbytes));

    HIPCHECK ( hipMemcpy(C_h, C_dd, Nbytes, useMemkindDefault? hipMemcpyDefault:hipMemcpyDeviceToHost));
#else
    HIPCHECK ( hipMemcpy(C_h, C_d, Nbytes, useMemkindDefault? hipMemcpyDefault:hipMemcpyDeviceToHost));
#endif

    HIPCHECK ( hipDeviceSynchronize() );
    HipTest::checkVectorADD(A_h, B_h, C_h, N);

    HipTest::freeArrays (A_d, B_d, C_d, A_h, B_h, C_h, usePinnedHost);
    HIPCHECK ( hipDeviceReset() );

    printf ("  %s success\n", __func__);
}



int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, true);


    simpleTest1();

    memcpyKind<float>(false, false, false);
    memcpyKind<float>(true, false, false);
    //memcpyKind<float>(true);

    passed();

}
