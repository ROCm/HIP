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

#include "hip_runtime.h"
#include "test_common.h"

#ifdef __HIP_PLATFORM_HCC__
#include "hcc_detail/AM.h"
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


void printAttribs(hipPointerAttribute_t *attribs) 
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
// Functins to run tests
//=================================================================================================
//
//Run through a couple simple cases to test lookups and hostd pointer arithmetic:
void simpleTests() 
{
    char *A_d;
    char *A_Pinned_h;
    char *A_OSAlloc_h;
    hipError_t e;

    HIPCHECK ( hipMalloc(&A_d, Nbytes) );
    HIPCHECK ( hipMallocHost(&A_Pinned_h, Nbytes) );
    A_OSAlloc_h = (char*)malloc(Nbytes);


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
    HIPASSERT(attribs == attribs2);

    // Corner case at end of array:
    resetAttribs(&attribs2);
    HIPCHECK( hipPointerGetAttributes(&attribs2, A_d+Nbytes-1));
    printf("getAttr:%-20s", "A_d+NBytes-1"); printAttribs(&attribs2);
    HIPASSERT(attribs == attribs2);

    // Pointer just beyond array - must be invalid or at least a different pointer
    resetAttribs(&attribs2);
    e = hipPointerGetAttributes(&attribs2, A_d+Nbytes+1);
    printf("getAttr:%-20s err=%d (%s), neg-test expected\n", "A_d+NBytes", e, hipGetErrorString(e)); 
    if (e != hipErrorInvalidValue) {
        // We might have strayed into another pointer area.
        printf("getAttr:%-20s", "A_d+NBytes"); printAttribs(&attribs2);
        HIPASSERT(attribs.devicePointer != attribs2.devicePointer);
    }


    resetAttribs(&attribs2);
    e = hipPointerGetAttributes(&attribs2, A_d+Nbytes);
    if (e != hipErrorInvalidValue) {
        printf("%-20s", "A_d+Nbytes"); printAttribs(&attribs2);
        HIPASSERT(attribs.devicePointer != attribs2.devicePointer);
    }

    hipFree(A_d);
    e = hipPointerGetAttributes(&attribs, A_d);
    HIPASSERT(e == hipErrorInvalidValue); // Just freed the pointer, this should return an error.


    // Device-visible host memory
    printf ("\nDevice-visible host memory (hipMallocHost)\n");
    HIPCHECK( hipPointerGetAttributes(&attribs, A_Pinned_h));
    printf("getAttr:%-20s", "A_pinned_h"); printAttribs(&attribs);

    resetAttribs(&attribs2);
    HIPCHECK( hipPointerGetAttributes(&attribs2, A_Pinned_h+Nbytes/2));
    printf("getAttr:%-20s", "A_pinned_h+NBytes/2"); printAttribs(&attribs2);
    HIPASSERT(attribs == attribs2);


    hipFreeHost(A_Pinned_h);
    e = hipPointerGetAttributes(&attribs, A_Pinned_h);
    HIPASSERT(e == hipErrorInvalidValue); // Just freed the pointer, this should return an error.
    printf("getAttr:%-20s err=%d (%s), neg-test expected\n", "A_d+NBytes", e, hipGetErrorString(e)); 


    // OS memory
    printf ("\nOS-allocated memory (malloc)\n");
    e = hipPointerGetAttributes(&attribs, A_OSAlloc_h);
    printf("getAttr:%-20s err=%d (%s), neg-test expected\n", "A_OSAlloc_h", e, hipGetErrorString(e)); 
    HIPASSERT(e == hipErrorInvalidValue); // OS-allocated pointers should return hipErrorInvalidValue.
}




struct SuperPointerAttribute {
    void *                  _pointer;
    size_t                  _sizeBytes;
    hipPointerAttribute_t   _attrib;
};



void checkPointer(SuperPointerAttribute &ref, int major, int minor, void *pointer)
{
    hipPointerAttribute_t attribs;
    resetAttribs(&attribs);

    HIPCHECK(hipPointerGetAttributes(&attribs, pointer));
    if (attribs != ref._attrib) {
        printf("Test %d.%d", major, minor);
        printf("  ref    ::  "); printAttribs(&ref._attrib);
        printf("  getattr::  "); printAttribs(&attribs);
        
        HIPASSERT(attribs == ref._attrib);
    } else {
        if (p_verbose & 0x1) {
            printf("#%4d.%d GOOD:%p getattr ::  ",major, minor, pointer); printAttribs(&attribs);
        }
    }
}


void clusterAllocs(int numAllocs, size_t minSize, size_t maxSize)
{
    printf ("===========================================================================\n");
    printf ("clusterAllocs numAllocs=%d size=%lu..%lu\n", numAllocs, minSize, maxSize);
    printf ("===========================================================================\n");
    std::vector <SuperPointerAttribute> reference(numAllocs);

    HIPASSERT(minSize > 0);
    HIPASSERT(maxSize >= minSize);

    int numDevices;
    HIPCHECK(hipGetDeviceCount(&numDevices));

    //---
    //Populate with device and host allocations.
    for (int i=0; i<numAllocs; i++) {
        bool isDevice = rand() & 0x1;
        reference[i]._sizeBytes = zrand(maxSize-minSize) + minSize;

        reference[i]._attrib.device = zrand(numDevices);
        HIPCHECK(hipSetDevice(reference[i]._attrib.device));
        reference[i]._attrib.isManaged = 0;

        void * ptr;
        if (isDevice) {
            HIPCHECK(hipMalloc(&ptr, reference[i]._sizeBytes));
            reference[i]._attrib.memoryType    = hipMemoryTypeDevice;
            reference[i]._attrib.devicePointer = ptr;
            reference[i]._attrib.hostPointer   = NULL;
            reference[i]._attrib.allocationFlags = 0; // TODO-randomize these.
        } else {
            HIPCHECK(hipMallocHost(&ptr, reference[i]._sizeBytes));
            reference[i]._attrib.memoryType    = hipMemoryTypeHost;
            reference[i]._attrib.devicePointer = ptr;
            reference[i]._attrib.hostPointer   = ptr;
            reference[i]._attrib.allocationFlags = 1; // TODO-randomize these.
        }
        reference[i]._pointer = ptr;
    }

#ifdef __HIP_PLATFORM_HCC__
    if (p_verbose & 0x2) {
        hc::AM_print_tracker();
    }
#endif


    // Now look up each pointer we inserted and verify we can find it:
    for (int i=0; i<numAllocs; i++) {
        SuperPointerAttribute &ref = reference[i];
        checkPointer(ref, i, 0, ref._pointer);
        checkPointer(ref, i, 1, (char *)ref._pointer + ref._sizeBytes/2);
        if (ref._sizeBytes > 1) {
            checkPointer(ref, i, 2, (char *)ref._pointer + ref._sizeBytes-1);
        }

    }
}


void testMultiThreaded()
{
    std::thread t1(clusterAllocs, 1000, 101, 1000);
    std::thread t2(clusterAllocs, 1000,  11,  100);
    std::thread t3(clusterAllocs, 1000,   5,  10);
    std::thread t4(clusterAllocs, 1000,   1,   4);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
}


int main(int argc, char *argv[])
{

    N= 1000000;
    HipTest::parseStandardArguments(argc, argv, true);

    HIPCHECK(hipSetDevice(p_gpuDevice));

    Nbytes = N*sizeof(char);

    printf ("N=%zu (%6.2f MB) device=%d\n", N, Nbytes/(1024.0*1024.0), p_gpuDevice);


    if (p_tests & 0x1) {
        simpleTests();
    }

    if (p_tests & 0x2) {
        srand(0x100);
        clusterAllocs(100, 1024*1, 1024*1024);
    }

    if (p_tests & 0x4) {
        srand(0x200);
        clusterAllocs(1000, 1, 10); //  Many tiny allocations;
    }

    if (p_tests & 0x8) {
        testMultiThreaded();
    }

    printf ("\n");
    passed();
}
