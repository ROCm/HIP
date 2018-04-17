#include<iostream>
#include<unistd.h>
#include<string>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
using namespace std;

int N=10000;
__global__ void kernel_1(hipLaunchParm lp)
{
//printf("Am in first kernel\n");
//cout<<"Am in first kernel"<<endl;
//while(10)
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
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

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}


int main()
{
hipError_t status;
hipEvent_t start,stop;
hipStream_t stream;
string out;
int isize=1,iblock=1;
iblock=512;
isize= 1<<12;

dim3 block (iblock);
dim3 grid (isize/iblock);



out=hipGetErrorString(hipEventCreateWithFlags(&start,hipEventBlockingSync));
cout<<"The output of hipEventCreate(&start):"<<out<<endl;
out=hipGetErrorString(hipEventCreate(&stop));
cout<<"The output of hipEventCreateWithFlags(&stop,hipEventBlockingSync):"<<out<<endl;
out=hipGetErrorString(hipStreamCreate(&stream));
cout<<"The output of hipEventStream(&stream):"<<out<<endl;
out=hipGetErrorString(hipEventRecord(start,stream));
cout<<"The output of hipEventRecord(start,stream):"<<out<<endl;

hipLaunchKernel(kernel_1,grid,block, 0, stream);
hipLaunchKernel(kernel_2,grid,block, 0, stream);
hipLaunchKernel(kernel_3,grid,block, 0, stream);
hipLaunchKernel(kernel_4,grid,block, 0, stream);
cout<<"Hi am after the kernel launch."<<endl;


out=hipGetErrorString(hipEventRecord(stop,stream));
cout<<"The output of hipEventRecord(stop,stream):"<<out<<endl;

float ms;
out=hipGetErrorString(hipEventElapsedTime(NULL,start,stop));
cout<<"The output of hipEventElapsedTime() is: "<<out<<endl;
cout<<"The value of ms obtained is: "<<ms<<endl;
}
