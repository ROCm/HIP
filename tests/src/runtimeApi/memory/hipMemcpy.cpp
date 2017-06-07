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
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * RUN_NAMED: %t hipMemcpy-modes --tests 0x1
 * RUN_NAMED: %t hipMemcpy-size --tests 0x6
 * RUN_NAMED: %t hipMemcpy-dev-offsets --tests 0x10
 * RUN_NAMED: %t hipMemcpy-host-offsets --tests 0x20
 * RUN_NAMED: %t hipMemcpy-multithreaded --tests 0x8
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"
#include "test_common.h"


void printSep()
{
    printf ("======================================================================================\n");
}

//-------
template<typename T>
class DeviceMemory
{
public:
    DeviceMemory(size_t numElements);
    ~DeviceMemory();

    T *A_d() const { return _A_d + _offset; };
    T *B_d() const { return _B_d + _offset; };
    T *C_d() const { return _C_d + _offset; };
    T *C_dd() const { return _C_dd + _offset; };

    size_t maxNumElements() const { return _maxNumElements; };


    void offset(int offset) { _offset = offset; };
    int offset() const { return _offset; };
    
private:
    T * _A_d;
    T*  _B_d;
    T*  _C_d;
    T*  _C_dd;


    size_t _maxNumElements;
    int _offset;
};

template<typename T>
DeviceMemory<T>::DeviceMemory(size_t numElements)
    : _maxNumElements(numElements), 
      _offset(0)
{
    T ** np = nullptr;
    HipTest::initArrays (&_A_d, &_B_d, &_C_d, np, np, np, numElements, 0);


    size_t sizeElements = numElements * sizeof(T);


    HIPCHECK ( hipMalloc(&_C_dd, sizeElements) );
}


template<typename T>
DeviceMemory<T>::~DeviceMemory ()
{
    T * np = nullptr;
    HipTest::freeArrays (_A_d, _B_d, _C_d, np, np, np, 0);

    HIPCHECK (hipFree(_C_dd));
    
    _C_dd = NULL;
};



//-------
template<typename T>
class HostMemory
{
public:
    HostMemory(size_t numElements, bool usePinnedHost);
    void reset(size_t numElements, bool full=false) ;
    ~HostMemory();


    T *A_h() const { return _A_h + _offset; };
    T *B_h() const { return _B_h + _offset; };
    T *C_h() const { return _C_h + _offset; };



    size_t maxNumElements() const { return _maxNumElements; };

    void offset(int offset) { _offset = offset; };
    int offset() const { return _offset; };
public:

    // Host arrays, secondary copy
    T * A_hh;
    T*  B_hh;

    bool   _usePinnedHost; 
private:
    size_t _maxNumElements;

    int _offset;

    // Host arrays
    T * _A_h;
    T*  _B_h;
    T*  _C_h;
};

template<typename T>
HostMemory<T>::HostMemory(size_t numElements, bool usePinnedHost)
    : _maxNumElements(numElements),
      _usePinnedHost(usePinnedHost),
      _offset(0)
{
    T ** np = nullptr;
    HipTest::initArrays (np, np, np, &_A_h, &_B_h, &_C_h, numElements, usePinnedHost);

    A_hh = NULL;
    B_hh = NULL;


    size_t sizeElements = numElements * sizeof(T);

    if (usePinnedHost) {
        HIPCHECK ( hipHostMalloc((void**)&A_hh, sizeElements, hipHostMallocDefault) );
        HIPCHECK ( hipHostMalloc((void**)&B_hh, sizeElements, hipHostMallocDefault) );
    } else {
        A_hh = (T*)malloc(sizeElements);
        B_hh = (T*)malloc(sizeElements);
    }

}


template<typename T>
void
HostMemory<T>::reset(size_t numElements, bool full) 
{
    // Initialize the host data:
    for (size_t i=0; i<numElements; i++) {
        (A_hh)[i] = 1097.0 + i; 
        (B_hh)[i] = 1492.0 + i; // Phi

        if (full) {
            (_A_h)[i] = 3.146f + i; // Pi
            (_B_h)[i] = 1.618f + i; // Phi
        }
    }
}

template<typename T>
HostMemory<T>::~HostMemory ()
{
    HipTest::freeArraysForHost (_A_h, _B_h, _C_h, _usePinnedHost);

    if (_usePinnedHost) {
        HIPCHECK (hipHostFree(A_hh));
        HIPCHECK (hipHostFree(B_hh));

    } else {
        free(A_hh);
        free(B_hh);
    }
    T *A_hh = NULL;
    T *B_hh = NULL;

};



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
void memcpytest2(DeviceMemory<T> *dmem, HostMemory<T> *hmem, size_t numElements, bool useHostToHost, bool useDeviceToDevice, bool useMemkindDefault)
{
    size_t sizeElements = numElements * sizeof(T);
    printf ("test: %s<%s> size=%lu (%6.2fMB) usePinnedHost:%d, useHostToHost:%d, useDeviceToDevice:%d, useMemkindDefault:%d, offsets:dev:%+d host:+%d\n", 
            __func__, 
            TYPENAME(T),
            sizeElements, sizeElements/1024.0/1024.0,
            hmem->_usePinnedHost, useHostToHost, useDeviceToDevice, useMemkindDefault,
            dmem->offset(), hmem->offset()
            );


    hmem->reset(numElements);
    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

    assert (numElements <= dmem->maxNumElements());
    assert (numElements <= hmem->maxNumElements());



    if (useHostToHost) {
        // Do some extra host-to-host copies here to mix things up:
        HIPCHECK ( hipMemcpy(hmem->A_hh, hmem->A_h(), sizeElements, useMemkindDefault? hipMemcpyDefault : hipMemcpyHostToHost));
        HIPCHECK ( hipMemcpy(hmem->B_hh, hmem->B_h(), sizeElements, useMemkindDefault? hipMemcpyDefault : hipMemcpyHostToHost));


        HIPCHECK ( hipMemcpy(dmem->A_d(), hmem->A_hh, sizeElements, useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
        HIPCHECK ( hipMemcpy(dmem->B_d(), hmem->B_hh, sizeElements, useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
    } else {
        HIPCHECK ( hipMemcpy(dmem->A_d(), hmem->A_h(), sizeElements, useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
        HIPCHECK ( hipMemcpy(dmem->B_d(), hmem->B_h(), sizeElements, useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice));
    }

    hipLaunchKernel(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, dmem->A_d(), dmem->B_d(), dmem->C_d(), numElements);

    if (useDeviceToDevice) {
        // Do an extra device-to-device copy here to mix things up:
        HIPCHECK ( hipMemcpy(dmem->C_dd(), dmem->C_d(),  sizeElements, useMemkindDefault? hipMemcpyDefault : hipMemcpyDeviceToDevice));

        //Destroy the original dmem->C_d():
        HIPCHECK ( hipMemset(dmem->C_d(), 0x5A, sizeElements));

        HIPCHECK ( hipMemcpy(hmem->C_h(), dmem->C_dd(), sizeElements, useMemkindDefault? hipMemcpyDefault:hipMemcpyDeviceToHost));
    } else {
        HIPCHECK ( hipMemcpy(hmem->C_h(), dmem->C_d(), sizeElements, useMemkindDefault? hipMemcpyDefault:hipMemcpyDeviceToHost));
    }

    HIPCHECK ( hipDeviceSynchronize() );
    HipTest::checkVectorADD(hmem->A_h(), hmem->B_h(), hmem->C_h(), numElements);



    printf ("  %s success\n", __func__);
}


//---
//Try all the 16 possible combinations to memcpytest2 - usePinnedHost, useHostToHost, useDeviceToDevice, useMemkindDefault
template<typename T>
void memcpytest2_for_type(size_t numElements)
{
    printSep();

    DeviceMemory<T> memD(numElements); 
    HostMemory<T> memU(numElements, 0/*usePinnedHost*/); 
    HostMemory<T> memP(numElements, 1/*usePinnedHost*/);

    for (int usePinnedHost =0; usePinnedHost<=1; usePinnedHost++) {
        for (int useHostToHost =0; useHostToHost<=1; useHostToHost++) {  // TODO
            for (int useDeviceToDevice =0; useDeviceToDevice<=1; useDeviceToDevice++) {
                for (int useMemkindDefault =0; useMemkindDefault<=1; useMemkindDefault++) {
                    memcpytest2<T>(&memD, usePinnedHost ? &memP : &memU, numElements, useHostToHost, useDeviceToDevice, useMemkindDefault);
                }
            }
        }
    }
}


//---
//Try many different sizes to memory copy.
template<typename T>
void memcpytest2_sizes(size_t maxElem=0)
{
    printSep();
    printf ("test: %s<%s>\n", __func__,  TYPENAME(T));

    int deviceId;
    HIPCHECK(hipGetDevice(&deviceId));

    size_t free, total;
    HIPCHECK(hipMemGetInfo(&free, &total));

    if (maxElem == 0) {
        maxElem = free/sizeof(T)/8;
    }

    printf ("  device#%d: hipMemGetInfo: free=%zu (%4.2fMB) total=%zu (%4.2fMB)    maxSize=%6.1fMB\n", 
            deviceId, free, (float)(free/1024.0/1024.0), total, (float)(total/1024.0/1024.0), maxElem*sizeof(T)/1024.0/1024.0);
    HIPCHECK ( hipDeviceReset() );
    DeviceMemory<T> memD(maxElem); 
    HostMemory<T> memU(maxElem, 0/*usePinnedHost*/); 
    HostMemory<T> memP(maxElem, 1/*usePinnedHost*/);

    for (size_t elem=1; elem<=maxElem; elem*=2) {
        memcpytest2<T>(&memD, &memU, elem, 1, 1, 0);  // unpinned host
        memcpytest2<T>(&memD, &memP, elem, 1, 1, 0);  // pinned host
    }
}


//---
//Try many different sizes to memory copy.
template<typename T>
void memcpytest2_offsets(size_t maxElem, bool devOffsets, bool hostOffsets)
{
    printSep();
    printf ("test: %s<%s>\n", __func__,  TYPENAME(T));

    int deviceId;
    HIPCHECK(hipGetDevice(&deviceId));

    size_t free, total;
    HIPCHECK(hipMemGetInfo(&free, &total));


    printf ("  device#%d: hipMemGetInfo: free=%zu (%4.2fMB) total=%zu (%4.2fMB)    maxSize=%6.1fMB\n", 
            deviceId, free, (float)(free/1024.0/1024.0), total, (float)(total/1024.0/1024.0), maxElem*sizeof(T)/1024.0/1024.0);
    HIPCHECK ( hipDeviceReset() );
    DeviceMemory<T> memD(maxElem); 
    HostMemory<T> memU(maxElem, 0/*usePinnedHost*/); 
    HostMemory<T> memP(maxElem, 1/*usePinnedHost*/);

    size_t elem = maxElem / 2;

    for (int offset=0; offset < 512; offset++) {
        assert (elem + offset < maxElem);
        if (devOffsets) {
            memD.offset(offset);
        }
        if (hostOffsets) {
            memU.offset(offset);
            memP.offset(offset);
        }
        memcpytest2<T>(&memD, &memU, elem, 1, 1, 0);  // unpinned host
        memcpytest2<T>(&memD, &memP, elem, 1, 1, 0);  // pinned host
    }

    for (int offset=512; offset < elem; offset*=2) {
        assert (elem + offset < maxElem);
        if (devOffsets) {
            memD.offset(offset);
        }
        if (hostOffsets) {
            memU.offset(offset);
            memP.offset(offset);
        }
        memcpytest2<T>(&memD, &memU, elem, 1, 1, 0);  // unpinned host
        memcpytest2<T>(&memD, &memP, elem, 1, 1, 0);  // pinned host
    }
}


//---
//Create multiple threads to stress multi-thread locking behavior in the allocation/deallocation/tracking logic:
template<typename T>
void multiThread_1(bool serialize, bool usePinnedHost)
{
    printSep();
    printf ("test: %s<%s> serialize=%d usePinnedHost=%d\n", __func__,  TYPENAME(T), serialize, usePinnedHost);
    DeviceMemory<T> memD(N); 
    HostMemory<T> mem1(N, usePinnedHost); 
    HostMemory<T> mem2(N, usePinnedHost); 

    std::thread t1 (memcpytest2<T>, &memD, &mem1, N, 0,0,0);
    if (serialize) {
        t1.join();
    }

    
    std::thread t2 (memcpytest2<T>,&memD,  &mem2, N, 0,0,0);
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
        // Some tests around the 64KB boundary which have historically shown issues:
        printf ("\n\n=== tests&0x2 (64KB boundary)\n");
        size_t maxElem = 32*1024*1024;
        DeviceMemory<float> memD(maxElem); 
        HostMemory<float> memU(maxElem, 0/*usePinnedHost*/); 
        HostMemory<float> memP(maxElem, 0/*usePinnedHost*/); 
        // These all pass:
        memcpytest2<float>(&memD, &memP, 15*1024*1024, 0, 0, 0);  
        memcpytest2<float>(&memD, &memP, 16*1024*1024, 0, 0, 0);  
        memcpytest2<float>(&memD, &memP, 16*1024*1024+16*1024,  0, 0, 0);  

        // Just over 64MB:
        memcpytest2<float>(&memD, &memP, 16*1024*1024+512*1024,  0, 0, 0);  
        memcpytest2<float>(&memD, &memP, 17*1024*1024+1024,  0, 0, 0);  
        memcpytest2<float>(&memD, &memP, 32*1024*1024, 0, 0, 0);  
        memcpytest2<float>(&memD, &memU, 32*1024*1024, 0, 0, 0);  
        memcpytest2<float>(&memD, &memP, 32*1024*1024, 1, 1, 0);  
        memcpytest2<float>(&memD, &memP, 32*1024*1024, 1, 1, 0);  


    }



    if (p_tests & 0x4) {
        printf ("\n\n=== tests&4 (test sizes)\n");
        HIPCHECK ( hipDeviceReset() );
        memcpytest2_sizes<float>(0);
        printSep();
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


    if (p_tests & 0x10) {
        printf ("\n\n=== tests&0x10 (test device offsets)\n");
        HIPCHECK ( hipDeviceReset() );
        size_t maxSize = 256*1024;
        memcpytest2_offsets<char>  (maxSize, true, false);
        memcpytest2_offsets<float> (maxSize, true, false);
        memcpytest2_offsets<double>(maxSize, true, false);
    }


    if (p_tests & 0x20) {
        printf ("\n\n=== tests&0x10 (test device offsets)\n");
        HIPCHECK ( hipDeviceReset() );
        size_t maxSize = 256*1024;
        memcpytest2_offsets<char>  (maxSize, false, true);
        memcpytest2_offsets<float> (maxSize, false, true);
        memcpytest2_offsets<double>(maxSize, false, true);
    }



    passed();

}
