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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * RUN_NAMED: %t hipMemcpy-modes --tests 0x1
 * RUN_NAMED: %t hipMemcpy-size --tests 0x6
 * RUN_NAMED: %t hipMemcpy-multithreaded --tests 0x8
 * HIT_END
 */

#include "hip_runtime.h"
#include "hip_runtime.h"
#include "test_common.h"


void printSep()
{
    printf ("======================================================================================\n");
}




//---
// Test many different kinds of memory copies.
// The subroutine allocates memory , copies to device, runs a vector add kernel, copies back, and checks the result.
//
// IN: numElements  controls the number of elements used for allocations.
// IN: usePinnedHost : If true, allocate host with hipHostMalloc and is pinned ; else allocate host memory with malloc.
// IN: useHostToHost : If true, add an extra host-to-host copy.
// IN: useDeviceToDevice : If true, add an extra deviceto-device copy after result is produced.
// IN: useMemkindDefault : If true, use memkinddefault (runtime figures out direction).  if false, use explicit memcpy direction.
//
template <typename T>
void memcpytest2(size_t numElements, bool usePinnedHost, bool useHostToHost, bool useDeviceToDevice, bool useMemkindDefault)
{
    size_t sizeElements = numElements * sizeof(T);
    printf ("test: %s<%s> size=%lu (%6.2fMB) usePinnedHost:%d, useHostToHost:%d, useDeviceToDevice:%d, useMemkindDefault:%d\n", 
            __func__, 
            TYPENAME(T),
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
            HIPCHECK ( hipHostMalloc((void**)&A_hh, sizeElements, hipHostMallocDefault) );
            HIPCHECK ( hipHostMalloc((void**)&B_hh, sizeElements, hipHostMallocDefault) );
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


//---
//Try all the 16 possible combinations to memcpytest2 - usePinnedHost, useHostToHost, useDeviceToDevice, useMemkindDefault
template<typename T>
void memcpytest2_for_type(size_t numElements)
{
    printSep();

    for (int usePinnedHost =0; usePinnedHost<=1; usePinnedHost++) {
        for (int useHostToHost =0; useHostToHost<=1; useHostToHost++) {  // TODO
            for (int useDeviceToDevice =0; useDeviceToDevice<=1; useDeviceToDevice++) {
                for (int useMemkindDefault =0; useMemkindDefault<=1; useMemkindDefault++) {
                    memcpytest2<T>(numElements, usePinnedHost, useHostToHost, useDeviceToDevice, useMemkindDefault);
                }
            }
        }
    }
}


//---
//Try many different sizes to memory copy.
template<typename T>
void memcpytest2_sizes(size_t maxElem=0, size_t offset=0)
{
    printSep();
    printf ("test: %s<%s>\n", __func__,  TYPENAME(T));

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


//---
//Create multiple threads to stress multi-thread locking behavior in the allocation/deallocation/tracking logic:
template<typename T>
void multiThread_1(bool serialize, bool usePinnedHost)
{
    printSep();
    printf ("test: %s<%s> serialize=%d usePinnedHost=%d\n", __func__,  TYPENAME(T), serialize, usePinnedHost);
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
        printf ("\n\n=== tests&1 (types and different memcpy kinds (H2D, D2H, H2H, D2D)\n");
        HIPCHECK ( hipDeviceReset() );
        memcpytest2_for_type<float>(N);
        memcpytest2_for_type<double>(N);
        memcpytest2_for_type<char>(N);
        memcpytest2_for_type<int>(N);
        printf ("===\n\n\n");
    }


    if (p_tests & 0x2) {
        // Some tests around the 64MB boundary which have historically shown issues:
        printf ("\n\n=== tests&0x2 (64MB boundary)\n");
#if 0
        // These all pass:
        memcpytest2<float>(15*1024*1024, 1, 0, 0, 0);  
        memcpytest2<float>(16*1024*1024, 1, 0, 0, 0);  
        memcpytest2<float>(16*1024*1024+16*1024,  1, 0, 0, 0);  
#endif
        // Just over 64MB:
        memcpytest2<float>(16*1024*1024+512*1024,  1, 0, 0, 0);  
        memcpytest2<float>(17*1024*1024+1024,  1, 0, 0, 0);  
        memcpytest2<float>(32*1024*1024, 1, 0, 0, 0);  
        memcpytest2<float>(32*1024*1024, 0, 0, 0, 0);  
        memcpytest2<float>(32*1024*1024, 1, 1, 1, 0);  
        memcpytest2<float>(32*1024*1024, 1, 1, 1, 0);  
    }


    if (p_tests & 0x4) {
        printf ("\n\n=== tests&4 (test sizes and offsets)\n");
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
        printf ("\n\n=== tests&8\n");
        HIPCHECK ( hipDeviceReset() );
        printSep();

        // Simplest cases: serialize the threads, and also used pinned memory:
        // This verifies that the sub-calls to memcpytest2 are correct.
        multiThread_1<float>(true, true); 

        // Serialize, but use unpinned memory to stress the unpinned memory xfer path.
        multiThread_1<float>(true, false);

        // Remove serialization, so two threads are performing memory copies in parallel.
        multiThread_1<float>(false, true);

        // Remove serialization, and use unpinned.
        multiThread_1<float>(false, false); // TODO
        printf ("===\n\n\n");
    }


    passed();

}
