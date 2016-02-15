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


void printSep()
{
    printf ("======================================================================================\n");
}

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
void memcpytest2(size_t numElements, bool usePinnedHost, bool useHostToHost, bool useDeviceToDevice, bool useMemkindDefault)
{
    size_t sizeElements = numElements * sizeof(T);
    printf ("test: %s<%s> size=%lu (%6.2fMB) usePinnedHost:%d, useHostToHost:%d, useDeviceToDevice:%d, useMemkindDefault:%d\n", 
            __func__, 
            typeid(T).name(),
            sizeElements, sizeElements/1024.0/1024.0,
            usePinnedHost, useHostToHost, useDeviceToDevice, useMemkindDefault);


    T *A_d, *B_d, *C_d;
    T *A_h, *B_h, *C_h;


    HipTest::initArrays (&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, numElements, usePinnedHost);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

    T *A_hh = NULL;
    T *B_hh = NULL;
    T *C_dd = NULL;



    if (useHostToHost) {
        if (usePinnedHost) {
            HIPCHECK ( hipMallocHost(&A_hh, sizeElements) );
            HIPCHECK ( hipMallocHost(&B_hh, sizeElements) );
        } else {
            A_hh = (T*)malloc(sizeElements);
            B_hh = (T*)malloc(sizeElements);
        }


        // Do some extra host-to-host copies here to mix things up:
        HIPCHECK ( hipMemcpy(A_hh, A_h, sizeElements, useMemkindDefault? hipMemcpyDefault : hipMemcpyHostToHost));
        HIPCHECK ( hipMemcpy(B_hh, B_h, sizeElements, useMemkindDefault? hipMemcpyDefault : hipMemcpyHostToHost));


        HIPCHECK ( hipMemcpy(A_d, A_hh, sizeElements, useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
        HIPCHECK ( hipMemcpy(B_d, B_hh, sizeElements, useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
    } else {
        HIPCHECK ( hipMemcpy(A_d, A_h, sizeElements, useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
        HIPCHECK ( hipMemcpy(B_d, B_h, sizeElements, useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
    }

    hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, B_d, C_d, numElements);

    if (useDeviceToDevice) {
        HIPCHECK ( hipMalloc(&C_dd, sizeElements) );

        // Do an extra device-to-device copies here to mix things up:
        HIPCHECK ( hipMemcpy(C_dd, C_d,  sizeElements, useMemkindDefault? hipMemcpyDefault : hipMemcpyDeviceToDevice));

        //Destroy the original C_d:
        HIPCHECK ( hipMemset(C_d, 0x5A, sizeElements));

        HIPCHECK ( hipMemcpy(C_h, C_dd, sizeElements, useMemkindDefault? hipMemcpyDefault:hipMemcpyDeviceToHost));
    } else {
        HIPCHECK ( hipMemcpy(C_h, C_d, sizeElements, useMemkindDefault? hipMemcpyDefault:hipMemcpyDeviceToHost));
    }

    HIPCHECK ( hipDeviceSynchronize() );
    HipTest::checkVectorADD(A_h, B_h, C_h, numElements);

    HipTest::freeArrays (A_d, B_d, C_d, A_h, B_h, C_h, usePinnedHost);

    printf ("  %s success\n", __func__);
}


template<typename T>
void memcpytest2_loop(size_t numElements)
{
    printSep();

    for (int usePinnedHost =0; usePinnedHost<=1; usePinnedHost++) {
#define USE_HOST_2_HOST
#ifdef USE_HOST_2_HOST
        for (int useHostToHost =0; useHostToHost<=1; useHostToHost++) {  // TODO
#else
        for (int useHostToHost =0; useHostToHost<=0; useHostToHost++) {  // TODO
#endif
            for (int useDeviceToDevice =0; useDeviceToDevice<=1; useDeviceToDevice++) {
                for (int useMemkindDefault =0; useMemkindDefault<=1; useMemkindDefault++) {
                    memcpytest2<T>(numElements, usePinnedHost, useHostToHost, useDeviceToDevice, useMemkindDefault);
                }
            }
        }
    }
}


template<typename T>
void memcpytest2_sizes(size_t maxElem=0, size_t offset=0)
{
    printSep();
    printf ("test: %s<%s>\n", __func__,  typeid(T).name());

    int deviceId;
    HIPCHECK(hipGetDevice(&deviceId));

    size_t free, total;
    HIPCHECK(hipMemGetInfo(&free, &total));

    if (maxElem == 0) {
        maxElem = free/sizeof(T)/5;
    }

    printf ("  device#%d: hipMemGetInfo: free=%zu (%4.2fMB) total=%zu (%4.2fMB)    maxSize=%6.1fMB offset=%lu\n", 
            deviceId, free, (float)(free/1024.0/1024.0), total, (float)(total/1024.0/1024.0), maxElem*sizeof(T)/1024.0/1024.0, offset);

    for (size_t elem=64; elem+offset<=maxElem; elem*=2) {
        HIPCHECK ( hipDeviceReset() );
        memcpytest2<T>(elem+offset, 0, 1, 1, 0);  // unpinned host
        HIPCHECK ( hipDeviceReset() );
        memcpytest2<T>(elem+offset, 1, 1, 1, 0);  // pinned host
    }
}


template<typename T>
void multiThread_1(bool serialize, bool usePinnedHost)
{
    printSep();
    printf ("test: %s<%s> serialize=%d usePinnedHost=%d\n", __func__,  typeid(T).name(), serialize, usePinnedHost);
    std::thread t1 (memcpytest2<T>,N, usePinnedHost,0,0,0);
    if (serialize) {
        t1.join();
    }

    
    std::thread t2 (memcpytest2<T>,N, usePinnedHost,0,0,0);
    if (serialize) {
        t2.join();
    }

    if (!serialize) {
        t1.join();
        t2.join();
    }
}



int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, true);

    printf ("info: set device to %d\n", p_gpuDevice);
    HIPCHECK(hipSetDevice(p_gpuDevice));


    if (p_tests & 0x1) {
        HIPCHECK ( hipDeviceReset() );
        simpleTest1();
    }

    if (p_tests & 0x2) {
        HIPCHECK ( hipDeviceReset() );
        memcpytest2_loop<float>(N);
        memcpytest2_loop<double>(N);
        memcpytest2_loop<char>(N);
        memcpytest2_loop<int>(N);
    }

    if (p_tests & 0x4) {
        HIPCHECK ( hipDeviceReset() );
        printSep();
        memcpytest2_sizes<float>(0,0);
        printSep();
        memcpytest2_sizes<float>(0,64);
        printSep();
        memcpytest2_sizes<float>(1024*1024, 13);
        printSep();
        memcpytest2_sizes<float>(1024*1024, 50);
    }

    if (p_tests & 0x8) {
        HIPCHECK ( hipDeviceReset() );
        printSep();
        multiThread_1<float>(true, true);
        multiThread_1<float>(false, true);
        multiThread_1<float>(false, false); // TODO
    }

    passed();

}
