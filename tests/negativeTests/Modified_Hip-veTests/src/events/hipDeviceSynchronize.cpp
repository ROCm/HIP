#include <iostream>
#include <string.h>
#include <string>
#include <math.h>
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdlib.h>
using namespace std;
/*
 * A simple example of adding inter-stream dependencies using
 * cudaStreamWaitEvent. This code launches 4 kernels in each of n_streams
 * streams. An event is recorded at the completion of each stream (kernelEvent).
 * cudaStreamWaitEvent is then called on that event and the last stream
 * (streams[n_streams - 1]) to force all computation in the final stream to only
 * execute when all other streams have completed.
 */

#define N 300000
#define NSTREAM 4
#define CHECK(status,l) \
 str=hipGetErrorString(status);\
if(strcmp(str.c_str(),"hipSuccess"))\
	{\
	 cout<<"Api failed at the line:"<<l<<endl;\
	cout<<str<<endl;\
	};

__global__ void kernel_1(hipLaunchParm lp)
{
//printf("Am in first kernel\n");
//cout<<"Am in first kernel"<<endl;
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }

}

__global__ void kernel_2(hipLaunchParm lp)
{
//printf("Am in second kernel\n");
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_3(hipLaunchParm lp)
{
//printf("Am in third kernel\n");
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_4(hipLaunchParm lp)
{
//printf("Am in fourth kernel\n");
    double sum = 0.0;
while(10)
{
    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}}

int main(int argc, char **argv)
{
    int n_streams = NSTREAM;
    int isize = 1;
    int iblock = 1;
    int bigcase = 0;
    string str;
    // get argument from command line
    if (argc > 1) n_streams = atoi(argv[1]);

    if (argc > 2) bigcase = atoi(argv[2]);

    float elapsed_time;

    // set up max connectioin
    char * iname = "DEVICE_MAX_CONNECTIONS";
    setenv (iname, "32", 1);
    char *ivalue =  getenv (iname);
    cout<<iname<<" = "<< ivalue<<endl;

    int dev = 0;
    hipDeviceProp_t deviceProp;
    CHECK(hipGetDeviceProperties(&deviceProp, dev),__LINE__);
    cout<<"> Using Device"<< dev<<":"<< deviceProp.name<<" with num_streams "<<n_streams<<endl;
    CHECK(hipSetDevice(dev),__LINE__);

    // check if device support hyper-q
/*    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
    {
        if (deviceProp.concurrentKernels == 0)
        {
            cout<<"> GPU does not support concurrent kernel execution (SM 3.5 "
                    "or higher required)\n"<<endl;
            cout<<"> CUDA kernel runs will be serialized\n"<<endl;
        }
        else
        {
            cout<<"> GPU does not support HyperQ"<<endl;
            cout<<"> CUDA kernel runs will have limited concurrency"<<endl;
        }
    }
*/
    cout<<"> Compute Capability"<<deviceProp.major<<"."<<deviceProp.minor<<"  hardware with  multi-processors"<<deviceProp.multiProcessorCount<<endl;

    // Allocate and initialize an array of stream handles
    hipStream_t *streams = (hipStream_t *) malloc(n_streams * sizeof(
                                hipStream_t));

    for (int i = 0 ; i < n_streams ; i++)
    {
        CHECK(hipStreamCreate(&(streams[i])),__LINE__);
    }

    // run kernel with more threads
    if (bigcase == 1)
    {
        iblock = 512;
        isize = 1 << 12;
    }

    // set up execution configuration
    dim3 block (iblock);
    dim3 grid  (isize / iblock);
    cout<<"> grid "<<grid.x<<" block "<<block.x<<endl;

    // creat events
    hipEvent_t start, stop;
    CHECK(hipEventCreate(&start),__LINE__);
    CHECK(hipEventCreate(&stop),__LINE__);



    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
hipLaunchKernel(kernel_1,grid,block, 0, streams[i]);
hipLaunchKernel(kernel_2,grid,block, 0, streams[i]);
hipLaunchKernel(kernel_3,grid,block, 0, streams[i]);
hipLaunchKernel(kernel_4,grid,block, 0, streams[i]);
    }

    // record stop event
    CHECK(hipEventRecord(stop, 0),__LINE__);
    string out;
//	out=hipGetErrorString(hipEventSynchronize(stop));
//    cout<<"out value is:"<<endl;
//out=hipGetErrorString(hipDeviceSynchronize());

out=hipGetErrorString(hipDeviceSynchronize(NULL));
cout<<"The output of hipDeviceSynchronize() is:"<<out<<endl;
    // calculate elapsed time
    //CHECK(hipEventElapsedTime(&elapsed_time, start, stop),__LINE__);
//    cout<<"Measured time for parallel execution = "<<elapsed_time / 1000.0f<<endl;

    // release all stream
    for (int i = 0 ; i < n_streams ; i++)
    {
        CHECK(hipStreamDestroy(streams[i]),__LINE__);
    }

    free(streams);

    // reset device
    CHECK(hipDeviceReset(),__LINE__);

    return 0;
}
