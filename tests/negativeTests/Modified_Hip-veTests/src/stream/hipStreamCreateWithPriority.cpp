#include<iostream>
#include<string>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
using namespace std;

int main()
{
string out;
int i,lp,gp;
hipStream_t stream[5];

for(i=0;i<5;i++)
{
out=hipGetErrorString(hipStreamCreateWithPriority(&stream[i], hipStreamDefault,-i));
cout<<"The output of hipStreamCreateWithPriority() is:"<<out<<endl;
}


out=hipGetErrorString(hipDeviceGetStreamPriorityRange(&lp,&gp));
cout<<"The output of hipDeviceGetStreamPriorityRange() is: "<<out<<endl;
cout<<"The value in least priority is: "<<lp<<endl;
cout<<"The value in greatest priority is: "<<gp<<endl;

}

