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
 * RUN: %t
 * HIT_END
 */

#include <vector>
#include"test_common.h"

#define LEN 1024*1024
#define SIZE LEN*sizeof(float)

__global__ void Add(float *Ad, float *Bd, float *Cd){
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    Cd[tx] = Ad[tx] + Bd[tx];
}


__global__ void Set(int *Ad, int val){
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    Ad[tx] = val;
}


#define SYNC_EVENT 0
#define SYNC_STREAM 1
#define SYNC_DEVICE 2

std::vector<std::string> syncMsg = {"event", "stream", "device"};

void CheckHostPointer(int numElements, int *ptr, unsigned eventFlags, int syncMethod, std::string msg)
{
    std::cerr << "test: CheckHostPointer " << msg 
              //<< " HIP_COHERENT_HOST_ALLOC=" << HIP_COHERENT_HOST_ALLOC
              //<< " HIP_EVENT_SYS_RELEASE=" <<   HIP_EVENT_SYS_RELEASE
              << " eventFlags = " << std::hex << eventFlags 
              << ((eventFlags & hipEventReleaseToDevice) ?  " hipEventReleaseToDevice" : "")  
              << ((eventFlags & hipEventReleaseToSystem) ? " hipEventReleaseToSystem" : "") 
              << " ptr=" << ptr 
              << " syncMethod=" << syncMsg[syncMethod] << "\n";

    hipStream_t s;
    hipEvent_t e;

    // Init:
    HIPCHECK(hipStreamCreate(&s));
    HIPCHECK(hipEventCreateWithFlags(&e, eventFlags))
    dim3 dimBlock(64,1,1);
    dim3 dimGrid(numElements/dimBlock.x,1,1);

    const int expected = 13;

    // Init array to know state:
    hipLaunchKernelGGL(Set, dimGrid, dimBlock, 0, 0x0, ptr, -42);
    HIPCHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(Set, dimGrid, dimBlock, 0, s, ptr, expected);
    HIPCHECK(hipEventRecord(e, s));

    // Host waits for event :
    switch (syncMethod) {
        case SYNC_EVENT:
            HIPCHECK(hipEventSynchronize(e));
            break;
        case SYNC_STREAM:
            HIPCHECK(hipStreamSynchronize(s));
            break;
        case SYNC_DEVICE:
            HIPCHECK(hipDeviceSynchronize());
            break;
        default:
            assert(0);
    };
            
    for (int i=0; i<numElements; i++) {
        if (ptr[i] != expected) {
            printf ("mismatch at %d: %d != %d\n", i, ptr[i], expected);
            assert(ptr[i] == expected);
        }
    }

    HIPCHECK(hipStreamDestroy(s));
    HIPCHECK(hipEventDestroy(e));
};

int main(){


    hipDeviceProp_t prop;
    int device;
    HIPCHECK(hipGetDevice(&device));
    HIPCHECK(hipGetDeviceProperties(&prop, device));
    if(prop.canMapHostMemory != 1){
        std::cout<<"Exiting..."<<std::endl;
        failed("Does support HostPinned Memory");
    }


    {
        float *A, *B, *C;
        float *Ad, *Bd, *Cd;
        HIPCHECK(hipHostMalloc((void**)&A, SIZE, hipHostMallocWriteCombined | hipHostMallocMapped));
        HIPCHECK(hipHostMalloc((void**)&B, SIZE, hipHostMallocDefault));
        HIPCHECK(hipHostMalloc((void**)&C, SIZE, hipHostMallocMapped));

        HIPCHECK(hipHostGetDevicePointer((void**)&Ad, A, 0));
        HIPCHECK(hipHostGetDevicePointer((void**)&Cd, C, 0));

        for(int i=0;i<LEN;i++){
            A[i] = 1.0f;
            B[i] = 2.0f;
        }

        HIPCHECK(hipMalloc((void**)&Bd, SIZE));
        HIPCHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));

        dim3 dimGrid(LEN/512,1,1);
        dim3 dimBlock(512,1,1);

        hipLaunchKernelGGL(Add, dimGrid, dimBlock, 0, 0, Ad, Bd, Cd);

        HIPCHECK(hipDeviceSynchronize());

        HIPCHECK(hipHostFree(A));
        HIPCHECK(hipHostFree(B));
        HIPCHECK(hipHostFree(C));
    }

    {
        int numElements = 1024*16;
        size_t sizeBytes = numElements * sizeof (int);

#ifdef __HIP_PLATFORM_HCC__
        { 
            // Stimulate error condition:
            int *A = &numElements;
            HIPCHECK_API(hipHostMalloc((void**)&A, sizeBytes, hipHostMallocCoherent|hipHostMallocNonCoherent), hipErrorInvalidValue);

            assert (A == 0);
        }
#endif


        {
            int *A = nullptr;
            HIPCHECK(hipHostMalloc((void**)&A, sizeBytes, hipHostMallocNonCoherent));
            const char *ptrType = "non-coherent"; // TODO
            CheckHostPointer(numElements, A, hipEventReleaseToSystem, SYNC_DEVICE,   ptrType);
            CheckHostPointer(numElements, A, hipEventReleaseToSystem, SYNC_STREAM,  ptrType);
            CheckHostPointer(numElements, A, hipEventReleaseToSystem, SYNC_EVENT,   ptrType);

            // agent-scope releases don't provide host visibility, don't use them here:
        }

        if (1) { 
            int *A = nullptr;
            HIPCHECK(hipHostMalloc((void**)&A, sizeBytes, hipHostMallocCoherent));
            const char *ptrType = "coherent";
            CheckHostPointer(numElements, A, hipEventReleaseToDevice, SYNC_DEVICE,   ptrType);
            CheckHostPointer(numElements, A, hipEventReleaseToDevice, SYNC_STREAM,  ptrType);
            CheckHostPointer(numElements, A, hipEventReleaseToDevice, SYNC_EVENT,   ptrType);

            CheckHostPointer(numElements, A, hipEventReleaseToSystem, SYNC_DEVICE,   ptrType);
            CheckHostPointer(numElements, A, hipEventReleaseToSystem, SYNC_STREAM,  ptrType);
            CheckHostPointer(numElements, A, hipEventReleaseToSystem, SYNC_EVENT,   ptrType);
        }


        // Check defaults:
        if (1) { 
            int *A = nullptr;
            HIPCHECK(hipHostMalloc((void**)&A, sizeBytes));
            const char *ptrType = "default";
            CheckHostPointer(numElements, A, 0, SYNC_DEVICE,   ptrType);
            CheckHostPointer(numElements, A, 0, SYNC_STREAM,  ptrType);
            CheckHostPointer(numElements, A, 0, SYNC_EVENT,   ptrType);
            
            CheckHostPointer(numElements, A, 0, SYNC_DEVICE,   ptrType);
            CheckHostPointer(numElements, A, 0, SYNC_STREAM,  ptrType);
            CheckHostPointer(numElements, A, 0, SYNC_EVENT,   ptrType);
        }




    }
        
    passed();

}
