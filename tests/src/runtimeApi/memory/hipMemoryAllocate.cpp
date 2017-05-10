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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include"test_common.h"

#define NUM_ELEMENTS 1024*1024*64
#define SIZE NUM_ELEMENTS*sizeof(int)

int p_count = 4;


void multiGpuHostAlloc(int allocDevice)
{

    int numDevices;
    HIPCHECK(hipGetDeviceCount(&numDevices));
    assert(numDevices > 1);

    printf ("info: trying multiGpuHostAlloc with allocDevice=%d numDevices=%d\n", allocDevice, numDevices);


    HIPCHECK(hipSetDevice(allocDevice));

    int *Ah, *Ch;
    hipHostMalloc((void**)&Ah, SIZE);
    hipHostMalloc((void**)&Ch, SIZE);

    const int init = -1;
    for (size_t i=0; i<NUM_ELEMENTS; i++) {
        Ah[i] = init;
        Ch[i] = -2;
    }

    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, NUM_ELEMENTS);

    // The host memory allocations should be visible on all of the devices - verify by launching a kernel here that reads those devices:
    for (int i=0; i<numDevices; i++)  {
        HIPCHECK(hipSetDevice(i));
        hipLaunchKernelGGL(HipTest::addCountReverse , dim3(blocks), dim3(threadsPerBlock), 0, 0/*_stream*/,    Ah, Ch, NUM_ELEMENTS, p_count);
        HIPCHECK(hipDeviceSynchronize());
    };


    int expected = init + p_count;
    for (size_t i=0; i<NUM_ELEMENTS; i++) {
        if (Ch[i] != expected) {
            failed("for Ch[%zu] (%d)  !=  expected(%d)\n", i, Ch[i], expected);
        }
    }
}


int main(int argc, char *argv[])
{
    int more_argc = HipTest::parseStandardArguments(argc, argv, false);
    //assert(more_argc == 0);

    {
        float *Ad, *B, *Bd, *Bm, *C, *Cd, *ptr_0;
        B = (float*)malloc(SIZE);
        hipMalloc((void**)&Ad, SIZE);
        hipHostMalloc((void**)&B, SIZE);
        hipHostMalloc((void**)&Bd, SIZE, hipHostMallocDefault);
        hipHostMalloc((void**)&Bm, SIZE, hipHostMallocMapped);
        hipHostMalloc((void**)&C, SIZE, hipHostMallocMapped);

        hipHostGetDevicePointer((void**)&Cd, C, 0/*flags*/);

        HIPCHECK_API(hipMalloc((void**)&ptr_0,0), hipSuccess);

        HIPCHECK_API(hipFree(Ad) ,  hipSuccess);
        HIPCHECK_API(hipHostFree(Ad) , hipErrorInvalidValue);

        HIPCHECK_API(hipFree(B)  , hipErrorInvalidDevicePointer); // try to hipFree on malloced memory
        HIPCHECK_API(hipFree(Bd) , hipErrorInvalidDevicePointer);
        HIPCHECK_API(hipFree(Bm) , hipErrorInvalidDevicePointer);
        HIPCHECK_API(hipFree(ptr_0) , hipSuccess);
        HIPCHECK_API(hipHostFree(Bd) , hipSuccess);
        HIPCHECK_API(hipHostFree(Bm) , hipSuccess);

        HIPCHECK_API(hipFree(C) , hipErrorInvalidDevicePointer);
        HIPCHECK_API(hipHostFree(C) , hipSuccess);


        HIPCHECK_API(hipFree(NULL) , hipSuccess);
        HIPCHECK_API(hipHostFree(NULL) , hipSuccess);

        {
            // Some negative testing - request a too-big allocation and verify it fails:
            // Someday when we support virtual memory may need to refactor these:
            size_t tooBig = 128LL*1024*1024*1024*1024;  // 128 TB;
            void *p;
            HIPCHECK_API ( hipMalloc(&p, tooBig),  hipErrorMemoryAllocation );
            HIPCHECK_API ( hipHostMalloc(&p, tooBig),  hipErrorMemoryAllocation );
        }
    }


    {
        int numDevices;
        HIPCHECK(hipGetDeviceCount(&numDevices));
        assert(numDevices > 1);

        multiGpuHostAlloc(0);
        multiGpuHostAlloc(1);
    }

    passed();
}
