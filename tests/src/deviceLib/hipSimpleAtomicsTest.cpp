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
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


// Includes HIP Runtime
#include "hip/hip_runtime.h"
#include <test_common.h>

#define EXIT_WAIVED	2

const char *sampleName = "hipSimpleAtomicsTest";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);


#define min(a,b) (a) < (b) ? (a) : (b)
#define max(a,b) (a) > (b) ? (a) : (b)

int computeGold(int *gpuData, const int len)
{
    int val = 0;

    for (int i = 0; i < len; ++i)
    {
        val += 10;
    }

    if (val != gpuData[0])
    {
        printf("atomicAdd failed\n");
        return false;
    }

    val = 0;

    for (int i = 0; i < len; ++i)
    {
        val -= 10;
    }

    if (val != gpuData[1])
    {
        printf("atomicSub failed\n");
        return false;
    }

    bool found = false;

    for (int i = 0; i < len; ++i)
    {
        // third element should be a member of [0, len)
        if (i == gpuData[2])
        {
            found = true;
            break;
        }
    }

    if (!found)
    {
        printf("atomicExch failed\n");
        return false;
    }

    val = -(1 << 8);

    for (int i = 0; i < len; ++i)
    {
        // fourth element should be len-1
        val = max(val, i);
    }

    if (val != gpuData[3])
    {
        printf("atomicMax failed\n");
        return false;
    }

    val = 1 << 8;

    for (int i = 0; i < len; ++i)
    {
        val = min(val, i);
    }

    if (val != gpuData[4])
    {
        printf("atomicMin failed\n");
        return false;
    }

    int limit = 17;
    val = 0;

    for (int i = 0; i < len; ++i)
    {
        val = (val >= limit) ? 0 : val+1;
    }

    if (val != gpuData[5])
    {
        printf("atomicInc failed\n");
        return false;
    }

    limit = 137;
    val = 0;

    for (int i = 0; i < len; ++i)
    {
        val = ((val == 0) || (val > limit)) ? limit : val-1;
    }

    if (val != gpuData[6])
    {
        printf("atomicDec failed\n");
        return false;
    }

    found = false;

    for (int i = 0; i < len; ++i)
    {
        // eighth element should be a member of [0, len)
        if (i == gpuData[7])
        {
            found = true;
            break;
        }
    }

    if (!found)
    {
        printf("atomicCAS failed\n");
        return false;
    }

    val = 0xff;

    for (int i = 0; i < len; ++i)
    {
        // 9th element should be 1
        val &= (2 * i + 7);
    }

    if (val != gpuData[8])
    {
        printf("atomicAnd failed\n");
        return false;
    }

    val = 0;

    for (int i = 0; i < len; ++i)
    {
        // 10th element should be 0xff
        val |= (1 << i);
    }

    if (val != gpuData[9])
    {
        printf("atomicOr failed\n");
        return false;
    }

    val = 0xff;

    for (int i = 0; i < len; ++i)
    {
        // 11th element should be 0xff
        val ^= i;
    }

    if (val != gpuData[10])
    {
        printf("atomicXor failed\n");
        return false;
    }

    return true;
}

__global__ void testKernel(hipLaunchParm lp,int *g_odata)
{
    // access thread id
    const unsigned int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    // Test various atomic instructions

    // Arithmetic atomic instructions

    // Atomic addition
    atomicAdd(&g_odata[0], 10);

    // Atomic subtraction (final should be 0)
    atomicSub(&g_odata[1], 10);

    // Atomic exchange
    atomicExch(&g_odata[2], tid);

    // Atomic maximum
    atomicMax(&g_odata[3], tid);

    // Atomic minimum
    atomicMin(&g_odata[4], tid);

    // Atomic increment (modulo 17+1)
    atomicInc((unsigned int *)&g_odata[5], 17);
 
    // Atomic decrement
    atomicDec((unsigned int *)&g_odata[6], 137);

    // Atomic compare-and-swap
    atomicCAS(&g_odata[7], tid-1, tid);

    // Bitwise atomic instructions

    // Atomic AND
    atomicAnd(&g_odata[8], 2*tid+7);

    // Atomic OR
    atomicOr(&g_odata[9], 1 << tid);

    // Atomic XOR
    atomicXor(&g_odata[10], tid);
}


int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);

    runTest(argc, argv);

    hipDeviceReset();
    printf("%s completed, returned %s\n",
           sampleName,
           testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void runTest(int argc, char **argv)
{
    hipDeviceProp_t deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev = 0;

    hipGetDeviceProperties(&deviceProp, dev);

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, "
           "SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = (deviceProp.major * 0x10 + deviceProp.minor);

    unsigned int numThreads = 256;
    unsigned int numBlocks = 64;
    unsigned int numData = 11;
    unsigned int memSize = sizeof(int) * numData;

    //allocate mem for the result on host side
    int *hOData = (int *) malloc(memSize);

    //initialize the memory
    for (unsigned int i = 0; i < numData; i++)
        hOData[i] = 0;

    //To make the AND and XOR tests generate something other than 0...
    hOData[8] = hOData[10] = 0xff;

    // allocate device memory for result
    int *dOData;
    hipMalloc((void **) &dOData, memSize);
    // copy host memory to device to initialize to zero
    hipMemcpy(dOData, hOData, memSize,hipMemcpyHostToDevice);

    // execute the kernel
    hipLaunchKernel(testKernel, dim3(numBlocks), dim3(numThreads), 0, 0, dOData);

    //Copy result from device to host
    hipMemcpy(hOData,dOData, memSize,hipMemcpyDeviceToHost);

    // Compute reference solution
    testResult = computeGold(hOData, numThreads * numBlocks);

    // Cleanup memory
    free(hOData);
    hipFree(dOData);

    passed();
}
