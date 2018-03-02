#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
using namespace std;

int main()
{
void *ptr;
string str;
unsigned int flagsptr;

str=hipGetErrorString(hipHostMalloc((void**)&ptr,1,hipHostMallocMapped));
cout<<"The output of hipHostMalloc((void**)&ptr,1,hipHostMallocMapped) is: "<<str<<endl;

str=hipGetErrorString(hipHostGetFlags(&flagsptr,ptr));
cout<<"The output of hipHostGetFlags(&flagsptr,ptr) is: "<<str<<endl;
cout<<"The value of flagsptr is: "<<flagsptr<<endl;

str=hipGetErrorString(hipHostMalloc((void**)&ptr,1,hipHostMallocWriteCombined));
cout<<"The output of hipHostMalloc((void**)&ptr,1,hipHostMallocWriteCombined) is: "<<str<<endl;

str=hipGetErrorString(hipHostGetFlags(&flagsptr,ptr));
cout<<"The output of hipHostGetFlags(&flagsptr,ptr) is: "<<str<<endl;
cout<<"The value of flagsptr is: "<<flagsptr<<endl;

str=hipGetErrorString(hipHostMalloc((void**)&ptr,1,hipHostMallocDefault));
cout<<"The output of hipHostMalloc((void**)&ptr,1,hipHostMallocDefault) is: "<<str<<endl;

str=hipGetErrorString(hipHostGetFlags(&flagsptr,ptr));
cout<<"The output of hipHostGetFlags(&flagsptr,ptr) is: "<<str<<endl;
cout<<"The value of flagsptr is: "<<flagsptr<<endl;


}
