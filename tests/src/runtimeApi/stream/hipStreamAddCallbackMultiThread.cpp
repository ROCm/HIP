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
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
*/

#include <thread>
#include <unistd.h>
#include <mutex>
#include <condition_variable>
#include "test_common.h"

// Will indicate completion of callback added as part of hipStreamAddCallback
struct signal
{
    int completedThreads;
    std::mutex mu;
    std::condition_variable cv;
};

struct workload
{
    int _workloadId;  
    int _deviceID;

    int *copyOf_hData;  // copy of host data which will be used to validation
    int *hData;         // will contain host data
    int *dData;         // device data will be stored 
    hipStream_t _stream;// stream on which data will be processed

    bool success;       // start will be stored 
};

// Global data
int numWorkloads          = 8;
const int perWorkloadSize = 1000000;
signal completionSignal;

// Kernel run on device
__global__ void increment(int *data, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        data[i] = data[i]+1;
}

/*
 * Method validates processed data array with saved copy and notifies via conditional variable to all waiting threads
*/
void Analyze(hipStream_t event, hipError_t status, void *data)
{
    HIPCHECK(status);
	
    workload *W = (workload *) data;
    
    if(W != NULL)
    {
        HIPCHECK(hipSetDevice(W->_deviceID));

        W->success = true;

        for (int i=0; i< perWorkloadSize; ++i)
        {
                W->success &= (W->copyOf_hData[i] == (W->hData[i]+1));
                if(!W->success)
                {
                std::cout<<"\nExpected Data :"<<(W->hData[i]+1)<<" Current Data :"<<W->copyOf_hData[i]<<std::endl;
                break;
                }
        }
    }

    std::lock_guard<std::mutex> guard(completionSignal.mu);
    completionSignal.completedThreads += 1;

    completionSignal.cv.notify_all();
}

/*
 * Thread routine to launch workloads into separate stream 
 */
void LaunchWorkload(void *args)
{
    workload *W = (workload *) args;

    if (W == nullptr) return;

    std::srand(std::time(nullptr));

    HIPCHECK(hipSetDevice(W->_deviceID));

    // Allocate memory
    HIPCHECK(hipMalloc(&W->dData,perWorkloadSize*sizeof(int)));

    size_t s = perWorkloadSize*sizeof(int);
    HIPCHECK(hipMemset(W->dData,0,s))

    W->hData = (int *) malloc(perWorkloadSize*sizeof(int));
    W->copyOf_hData = (int *) malloc(perWorkloadSize*sizeof(int));

    // Initialize host array
    for(int i =0;i<perWorkloadSize;i++)
    {
        W->hData[i] = W->copyOf_hData[i] = std::rand() % perWorkloadSize;
    }

    HIPCHECK(hipStreamCreate(&W->_stream));

    dim3 block(256);
    dim3 grid((perWorkloadSize + block.x-1) / block.x);

    HIPCHECK(hipMemcpyAsync(W->dData,W->hData,perWorkloadSize*sizeof(int),hipMemcpyHostToDevice,W->_stream));

    hipLaunchKernelGGL(increment, grid, block,0, W->_stream,W->dData, perWorkloadSize);
 
    HIPCHECK(hipMemcpyAsync(W->copyOf_hData,W->dData,perWorkloadSize*sizeof(int),hipMemcpyDeviceToHost,W->_stream));

    HIPCHECK(hipStreamAddCallback(W->_stream, Analyze, W,0))
}

int main(int argc, char* argv[]) {

    int numDevice = 0;
	
    HIPCHECK(hipGetDeviceCount(&numDevice));
	
    std::thread workerThread[numWorkloads];
	
    workload *workloads;
    workloads = (workload *) malloc(numWorkloads * sizeof(workload));
        
    for(int i =0;i<numWorkloads; i++)
    {
        workloads[i]._workloadId = i;
        workloads[i]._deviceID = i%numDevice;

        // launch threads
        workerThread[i] = std::thread(LaunchWorkload,&workloads[i]);
    }
	
    // should wait for joining all threads
    for(int i =0; i<numWorkloads; i++)
    {
        workerThread[i].join();
    }
	
    // wait util all callbacks are done
    std::unique_lock<std::mutex> l(completionSignal.mu);
    while(completionSignal.completedThreads != numWorkloads)
    {
        completionSignal.cv.wait(l);
    }

    //clean-up
    hipDeviceReset();
    free(workloads);

    passed();
    return 0;
}
