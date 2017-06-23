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


// Test pointer tracking logic: allocate memory and retrieve stats with hipPointerGetAttributes

#include "hip/hip_runtime.h"
#include "test_common.h"
#include <vector>

#ifdef __HIP_PLATFORM_HCC__
#include "hc_am.hpp"

#endif


size_t Nbytes = 0;

//=================================================================================================
// Utility Functions:
//=================================================================================================

bool operator==(const hipPointerAttribute_t &lhs, const hipPointerAttribute_t &rhs)
{
    return ((lhs.hostPointer == rhs.hostPointer) &&
            (lhs.devicePointer == rhs.devicePointer) &&
            (lhs.memoryType == rhs.memoryType) &&
            (lhs.device == rhs.device) &&
            (lhs.allocationFlags == rhs.allocationFlags)
            ) ;

};


bool operator!=(const hipPointerAttribute_t &lhs, const hipPointerAttribute_t &rhs)
{
    return ! (lhs == rhs);
}


const char *memoryTypeToString(hipMemoryType memoryType)
{
    switch (memoryType) {
        case hipMemoryTypeHost   : return "[Host]";
        case hipMemoryTypeDevice : return "[Device]";
        default:                   return "[Unknown]";
    };
}


void resetAttribs(hipPointerAttribute_t *attribs)
{
    attribs->hostPointer = (void*) (-1);
    attribs->devicePointer = (void*) (-1);
    attribs->memoryType = hipMemoryTypeHost;
    attribs->device = -2;
    attribs->isManaged = -1;
    attribs->allocationFlags = 0xffff;
};


void printAttribs(const hipPointerAttribute_t *attribs)
{
    printf ("hostPointer:%p devicePointer:%p  memoryType:%s deviceId:%d isManaged:%d allocationFlags:%u\n",
            attribs->hostPointer,
            attribs->devicePointer,
            memoryTypeToString(attribs->memoryType),
            attribs->device,
            attribs->isManaged,
            attribs->allocationFlags
            );
};


inline int zrand(int max)
{
    return rand() % max;
}


//=================================================================================================
// Functions to run tests
//=================================================================================================
//--
//Run through a couple simple cases to test lookups and host pointer arithmetic:
void testSimple()
{
    printf ("\n");
    printf ("===========================================================================\n");
    printf ("Simple Tests\n");
    printf ("===========================================================================\n");

    char *A_d;
    char *A_Pinned_h;
    char *A_OSAlloc_h;
    hipError_t e;

    HIPCHECK ( hipMalloc(&A_d, Nbytes) );
    HIPCHECK ( hipHostMalloc((void**)&A_Pinned_h, Nbytes, hipHostMallocDefault) );
    A_OSAlloc_h = (char*)malloc(Nbytes);

    size_t free, total;
    HIPCHECK(hipMemGetInfo(&free, &total));
    printf ("hipMemGetInfo: free=%zu (%4.2f) Nbytes=%lu total=%zu (%4.2f)\n", free, (float)(free/1024.0/1024.0), Nbytes, total, (float)(total/1024.0/1024.0));
    HIPASSERT(free + Nbytes <= total);


    hipPointerAttribute_t attribs;
    hipPointerAttribute_t attribs2;

    // Device memory
    printf ("\nDevice memory (hipMalloc)\n");
    HIPCHECK( hipPointerGetAttributes(&attribs, A_d));
    printf("getAttr:%-20s", "A_d"); printAttribs(&attribs);

    // Check pointer arithmetic cases:
    resetAttribs(&attribs2);
    HIPCHECK( hipPointerGetAttributes(&attribs2, A_d+100));
    printf("getAttr:%-20s", "A_d+100"); printAttribs(&attribs2);
    HIPASSERT((char*)attribs.devicePointer+100 == (char*)attribs2.devicePointer);

    // Corner case at end of array:
    resetAttribs(&attribs2);
    HIPCHECK( hipPointerGetAttributes(&attribs2, A_d+Nbytes-1));
    printf("getAttr:%-20s", "A_d+Nbytes-1"); printAttribs(&attribs2);
    HIPASSERT((char*)attribs.devicePointer+Nbytes-1 == (char*)attribs2.devicePointer);

    // Pointer just beyond array - must be invalid or at least a different pointer
    resetAttribs(&attribs2);
    e = hipPointerGetAttributes(&attribs2, A_d+Nbytes+1);
    printf("getAttr:%-20s err=%d (%s), neg-test expected\n", "A_d+NBytes", e, hipGetErrorString(e));
    if (e != hipErrorInvalidValue) {
        // We might have strayed into another pointer area.
        printf("getAttr:%-20s", "A_d+NBytes"); printAttribs(&attribs2);
        HIPASSERT((char*)attribs.devicePointer != (char*)attribs2.devicePointer);
    }


    resetAttribs(&attribs2);
    e = hipPointerGetAttributes(&attribs2, A_d+Nbytes);
    if (e != hipErrorInvalidValue) {
        printf("%-20s", "A_d+Nbytes"); printAttribs(&attribs2);
        HIPASSERT(attribs.devicePointer != attribs2.devicePointer);
    }

    hipFree(A_d);
    e = hipPointerGetAttributes(&attribs, A_d);
    HIPASSERT(e == hipErrorUnknown); // Just freed the pointer, this should return an error.


    // Device-visible host memory
    printf ("\nDevice-visible host memory (hipHostMalloc)\n");
    HIPCHECK( hipPointerGetAttributes(&attribs, A_Pinned_h));
    printf("getAttr:%-20s", "A_pinned_h"); printAttribs(&attribs);

    resetAttribs(&attribs2);
    HIPCHECK( hipPointerGetAttributes(&attribs2, A_Pinned_h+Nbytes/2));
    printf("getAttr:%-20s", "A_pinned_h+NBytes/2"); printAttribs(&attribs2);
    HIPASSERT((char*)attribs.hostPointer+Nbytes/2 == (char*)attribs2.hostPointer);


    hipHostFree(A_Pinned_h);
    e = hipPointerGetAttributes(&attribs, A_Pinned_h);
    HIPASSERT(e == hipErrorUnknown); // Just freed the pointer, this should return an error.
    printf("getAttr:%-20s err=%d (%s), neg-test expected\n", "A_d+NBytes", e, hipGetErrorString(e));


    // OS memory
    printf ("\nOS-allocated memory (malloc)\n");
    e = hipPointerGetAttributes(&attribs, A_OSAlloc_h);
    printf("getAttr:%-20s err=%d (%s), neg-test expected\n", "A_OSAlloc_h", e, hipGetErrorString(e));
    HIPASSERT(e == hipErrorUnknown); // OS-allocated pointers should return hipErrorUnknown.
}

//---
//Reset the memory tracker (remove allocations from all known devices):
//This frees any memory allocated through the runtime.
//The routine will not release any
void resetTracker ()
{
    if (p_verbose & 0x1) {
        printf ("info: reset tracker for all devices in platform\n");
    }

    int numDevices;
    HIPCHECK(hipGetDeviceCount(&numDevices));

    // Clean up:
    for (int i=0; i<numDevices; i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipDeviceReset());
    };
}


// Store the hipPointer attrib and some extra info so can later compare the looked-up info against the reference expectation
struct SuperPointerAttribute {
    void *                  _pointer;
    size_t                  _sizeBytes;
    hipPointerAttribute_t   _attrib;
};


//---
//Support function to check result against a reference:
void checkPointer(SuperPointerAttribute &ref, int major, int minor, void *pointer)
{
    hipPointerAttribute_t attribs;
    resetAttribs(&attribs);

    hipError_t e = hipPointerGetAttributes(&attribs, pointer);
    if ((e != hipSuccess) || (attribs != ref._attrib)) {
        printf("Test %d.%d (err=%d)\n", major, minor, e);
        HIPCHECK(e);
        printf("  ref    ::  "); printAttribs(&ref._attrib);
        printf("  getattr::  "); printAttribs(&attribs);

        HIPASSERT(attribs != ref._attrib);
    } else {
        if (p_verbose & 0x1) {
            printf("#%4d.%d GOOD:%p getattr ::  ",major, minor, pointer); printAttribs(&attribs);
        }
    }
}


//---
//Test that allocates memory across all 4 devices withing the specified size range (minSize...maxSize).
//Then does lookups to make sure the info reported by the tracker matches expecations
//Then deallocates it all.
//
//Multiple threads can call this funtion and in fact we do this in the testMultiThreaded_1 test.
void clusterAllocs(int numAllocs, size_t minSize, size_t maxSize)
{
    printf ("  clusterAllocs numAllocs=%d size=%lu..%lu\n", numAllocs, minSize, maxSize);
    std::vector <SuperPointerAttribute> reference(numAllocs);

    HIPASSERT(minSize > 0);
    HIPASSERT(maxSize >= minSize);

    int numDevices;
    HIPCHECK(hipGetDeviceCount(&numDevices));

    //---
    //Populate with device and host allocations.
    size_t totalDeviceAllocated[numDevices];
    for (int i =0; i<numDevices; i++) {
        totalDeviceAllocated[i] = 0;
    }
    for (int i=0; i<numAllocs; i++) {
        bool isDevice = rand() & 0x1;
        reference[i]._sizeBytes = zrand(maxSize-minSize) + minSize;

        reference[i]._attrib.device = zrand(numDevices);
        HIPCHECK(hipSetDevice(reference[i]._attrib.device));
        reference[i]._attrib.isManaged = 0;

        void * ptr;
        if (isDevice) {
            totalDeviceAllocated[reference[i]._attrib.device] += reference[i]._sizeBytes;
            HIPCHECK(hipMalloc((void**)&ptr, reference[i]._sizeBytes));
            reference[i]._attrib.memoryType    = hipMemoryTypeDevice;
            reference[i]._attrib.devicePointer = ptr;
            reference[i]._attrib.hostPointer   = NULL;
            reference[i]._attrib.allocationFlags = 0; // TODO-randomize these.
        } else {
            HIPCHECK(hipHostMalloc((void**)&ptr, reference[i]._sizeBytes, hipHostMallocDefault));
            reference[i]._attrib.memoryType    = hipMemoryTypeHost;
            reference[i]._attrib.devicePointer = ptr;
            reference[i]._attrib.hostPointer   = ptr;
            reference[i]._attrib.allocationFlags = 0; // TODO-randomize these.
        }
        reference[i]._pointer = ptr;
    }

#ifdef __HIP_PLATFORM_HCC__
    if (p_verbose & 0x2) {
        printf ("Tracker after insertions:\n");
        hc::am_memtracker_print();
    }
#endif


    for (int i =0; i<numDevices; i++) {
        size_t free, total;
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipMemGetInfo(&free, &total));
        printf ("  device#%d: hipMemGetInfo: free=%zu (%4.2fMB) totalDevice=%lu (%4.2fMB) total=%zu (%4.2fMB)\n",
                i, free, (float)(free/1024.0/1024.0), totalDeviceAllocated[i], (float)(totalDeviceAllocated[i])/1024.0/1024.0, total, (float)(total/1024.0/1024.0));
        HIPASSERT(free + totalDeviceAllocated[i] <= total);
    }


    // Now look up each pointer we inserted and verify we can find it:
    for (int i=0; i<numAllocs; i++) {
        SuperPointerAttribute &ref = reference[i];
        checkPointer(ref, i, 0, ref._pointer);
        checkPointer(ref, i, 1, (char *)ref._pointer + ref._sizeBytes/2);
        if (ref._sizeBytes > 1) {
            checkPointer(ref, i, 2, (char *)ref._pointer + ref._sizeBytes-1);
        }

        if (ref._attrib.memoryType == hipMemoryTypeDevice) {
            hipFree(ref._pointer);
        } else {
            hipHostFree(ref._pointer);
        }

    }

#ifdef __HIP_PLATFORM_HCC__
    if (p_verbose & 0x2) {
        printf ("Tracker after cleanup:\n");
        hc::am_memtracker_print();
    }
#endif
}


//---
// Multi-threaded test with many simul allocs.
// IN : serialize will force the test to run in serial fashion.
// Seems like this does not hit MT corner cases in the tracker very often - testMultiThreaded_2 below seems more effective.
void testMultiThreaded_1(bool serialize=false)
{
    printf ("\n===========================================================================\n");
    printf ("MultiThreaded_1\n");
    if (serialize) printf ("[SERIALIZE]\n");
    printf ("===========================================================================\n");
    std::thread t1(clusterAllocs, 1000, 101, 1000);
    if (serialize) t1.join();

    std::thread t2(clusterAllocs, 1000,  11,  100);
    if (serialize) t2.join();

    std::thread t3(clusterAllocs, 1000,   5,  10);
    if (serialize) t3.join();

    std::thread t4(clusterAllocs, 1000,   1,   4);
    if (serialize) t4.join();

    if (!serialize) {
        t1.join();
        t2.join();
        t3.join();
        t4.join();
    }

    resetTracker();
}


///================================================================================================

//---
//Repeatedly query a single entry:
void thread_query(void *ptr, const hipPointerAttribute_t *refAttrib)
{
    int count = 0;

    for (int count=0; count< 1000000; count++) {
        hipPointerAttribute_t a;
        hipError_t e = hipPointerGetAttributes(&a, ptr);
        if ((e != hipSuccess) || (a!= *refAttrib)) {
            printf("Test %d (err=%d)\n", count, e);
            HIPCHECK(e);

            printf("  ref    ::  "); printAttribs(refAttrib);
            printf("  getattr::  "); printAttribs(&a);
        }
    }
}


#ifdef __HIP_PLATFORM_HCC__
//---
// Add pointers to tracker very quickly, then remove them quickly:
enum Dir {Up, Down};
void thread_noise_generator(int iters, size_t numBuffers, Dir addDir, Dir removeDir)
{
    const size_t bufferSize = 16;
    size_t maxSize = numBuffers*bufferSize;
    HIPASSERT((maxSize % bufferSize) == 0); // loop logic assumes this is true


    for (int i=0; i<iters; i++) {
        char * basePtr = (char*)malloc(maxSize);

        auto acc = hc::accelerator();

        if (addDir == Up) {
            for (char *p = basePtr; p<basePtr + maxSize; p+=bufferSize) {
                hc::AmPointerInfo info(p, p, bufferSize, acc, false, false);
                hc::am_memtracker_add(p, info);
            }
        } else if (addDir == Down) {
            for (char *p = basePtr+maxSize-bufferSize; p>=0; p-=bufferSize) {
                hc::AmPointerInfo info(p, p, bufferSize, acc, false, false);
                hc::am_memtracker_add(p, info);
            }
        }

        if (removeDir == Up) {
            for (char *p = basePtr; p<basePtr + maxSize; p+=bufferSize) {
                hc::am_memtracker_remove(p);
            }
        } else if (removeDir == Down) {
            for (char *p = basePtr+maxSize-bufferSize; p>=0; p-=bufferSize)  {
                hc::am_memtracker_remove(p);
            }
        }
    }
}


//---
//Multi-thread test that is effective at catching locking errors in the alloc/dealloc/tracker.
//The query thread repeately requests information on the same block of memory.
//Meanwhile, the thread_noise_generator registers a large number of blocks, and
//then unregisters them.   This causes a large amount of rebalancing in the tree
//structure and will generate errors unless the locks in the tracker are preventing reading
//while writing.
void testMultiThreaded_2()
{
    std::atomic<int> inflight(2);

    printf ("\n===========================================================================\n");
    printf ("MultiThreaded_2\n");
    printf ("===========================================================================\n");

    hipSetDevice(0);
    hipDeviceReset();

    // Create some entries in the tracker:
    for (int i=0; i<1000; i++) {
        void *C_d;
        HIPCHECK(hipMalloc(&C_d, 32));
    }


    // Allocate a pointer that we will repeatedly lookup:
    void *A_d;
    HIPCHECK(hipMalloc(&A_d, 10000));
    hipPointerAttribute_t attrib1;
    HIPCHECK(hipPointerGetAttributes(&attrib1, A_d));
    std::thread t1(thread_query, A_d, &attrib1);

    std::thread t2(thread_noise_generator, 10000, 1000, Up, Up);

    t1.join();
    t2.join();

    hipSetDevice(0);
    hipDeviceReset();
}
#endif



int main(int argc, char *argv[])
{
    N= 1000000;
    HipTest::parseStandardArguments(argc, argv, true);


    Nbytes = N*sizeof(char);

    printf ("N=%zu (%6.2f MB) device=%d\n", N, Nbytes/(1024.0*1024.0), p_gpuDevice);


    if (p_tests & 0x01) {
        printf ("info: set device to %d\n", p_gpuDevice);
        HIPCHECK(hipSetDevice(p_gpuDevice));
        testSimple();
    }

    if (p_tests & 0x02) {
        srand(0x100);
        printf ("\n===========================================================================\n");
        clusterAllocs(100, 1024*1, 1024*1024);
        resetTracker();
    }

    if (p_tests & 0x04) {
        srand(0x200);
        printf ("\n===========================================================================\n");
        clusterAllocs(1000, 1, 10); //  Many tiny allocations;
        resetTracker();
    }

    if (p_tests & 0x08) {
        srand(0x300);
        testMultiThreaded_1(true);
        testMultiThreaded_1(false);
    }


#ifdef __HIP_PLATFORM_HCC__
    if (p_tests & 0x10) {
        srand(0x400);
        testMultiThreaded_2();
        resetTracker();
    }
#endif

    printf ("\n");
    passed();
}
