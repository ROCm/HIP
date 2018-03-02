#include<iostream>
#include<string>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
using namespace std;

int main()
{

void *ptr;
string str;
void **dptr;
//str=hipGetErrorString(hipHostMalloc(0x0,0,0));
//cout<<"The output of hipHostMalloc(0x0,0,0) is: "<<str<<endl;


str=hipGetErrorString(hipHostMalloc((void**)&ptr,1,hipHostMallocMapped));
cout<<"The output of hipHostMalloc((void**)&ptr,1,hipHostMallocMapped) is: "<<str<<endl;
str=hipGetErrorString(hipHostGetDevicePointer(dptr,ptr,0));
cout<<"the output of hipHostGetDevicePointer(dptr,ptr,0) is: "<<str<<endl;


/*
str=hipGetErrorString(hipHostMalloc((void**)&ptr,1,hipHostMallocWriteCombined));
cout<<"The output of hipHostMalloc((void**)&ptr,1,hipHostMallocWriteCombined) is: "<<str<<endl;

str=hipGetErrorString(hipHostMalloc((void**)&ptr,1,hipHostMallocDefault));
cout<<"The output of hipHostMalloc((void**)&ptr,1,hipHostMallocDefault) is: "<<str<<endl;


*/

}

