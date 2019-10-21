/*
 Copyright (c) 2019-present Advanced Micro Devices, Inc. All rights reserved.
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
 * TEST: %t
 * HIT_END
*/
#include "test_common.h"
#include <mutex>
#include <condition_variable>
#include <stdlib.h>


//Globals
const int workloadCount = 1000000;
std::mutex gMutx;
bool callbackCompleted = false;
std::condition_variable condVar;

// Device function
__global__ void increment(int *data,int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i < N)
        data[i] = 1 + data[i];
}

struct USER_DATA
{
    int* result;     // Data received from device
    int* copyOfOriginalData; // Copy of initial data which will be used for validation
};

// Callback
void callback(hipStream_t event, hipError_t status, void *userData)
{
    USER_DATA *data = (USER_DATA *)userData;

    if(!(data == nullptr || data->result == nullptr || data->copyOfOriginalData == nullptr))
    {
        for(int i=0;i<workloadCount;i++)
        {
            if(data->result[i] != data->copyOfOriginalData[i]+1)
            {
               std::cout<<"Error value : "<<data->result[i]<<"| Expected value :"<<data->copyOfOriginalData[i]+1<<std::endl;
               break;
            }
        }
    }
    callbackCompleted = true;
    condVar.notify_all();
}

int main()
{
    int *hData = nullptr;
    int *dData = nullptr;
    int *hResultData = nullptr;
    int devCount = 0;
    int size = workloadCount * sizeof(int);

    // query device count
    HIPCHECK(hipGetDeviceCount(&devCount));

    // Allocate 
    // Host allocation
    hData = (int *)malloc(size);
    hResultData = (int *)malloc(size);

    if(hData == nullptr || hResultData == nullptr)
    {
        HIPCHECK(hipErrorInvalidValue);
    }
    
    // Initialize host data
    for(int i =0; i<workloadCount; i++)
    {
        hData[i] = rand()%workloadCount;
    }

    // Device allocation
    HIPCHECK(hipMalloc(&dData,size));

    HIPCHECK(hipMemcpyAsync(dData,hData,size,hipMemcpyHostToDevice,0));

    dim3 block(256);
    dim3 grid((workloadCount + block.x-1) / block.x);
    hipLaunchKernelGGL(increment, grid, block, 0, 0, dData, workloadCount);

    HIPCHECK(hipMemcpyAsync(hResultData, dData, size, hipMemcpyDeviceToHost, 0));

    hipStream_t stream;
    USER_DATA *inputParam = (USER_DATA*)malloc(sizeof(USER_DATA));
    
    if(inputParam == nullptr) return 0;

    inputParam->result = hResultData;
    inputParam->copyOfOriginalData = hData;

    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK(hipStreamAddCallback(stream,callback,inputParam,0));

    // Wait for stream add callback to complete
    std::unique_lock<std::mutex> l(gMutx);
    
    while(!callbackCompleted)
        condVar.wait(l);

    // Will destroy device memory hence explicite hipFree is not needed
    HIPCHECK(hipDeviceReset());

    free(hData);
    free(hResultData);
    passed();
}
