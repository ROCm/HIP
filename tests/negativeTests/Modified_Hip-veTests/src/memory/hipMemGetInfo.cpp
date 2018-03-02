#include <iostream>
#include <string>
#include <hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

using namespace std;

int main()
{
string out;
size_t free,total;
float ptr_f,ptr_t;
int int_f,int_t;
//out=hipGetErrorString(hipMemGetInfo(&free,&total));

out=hipGetErrorString(hipMemGetInfo(&int_f,&int_t));
//out=hipGetErrorString(hipMemGetInfo(&ptr_f,&ptr_t));
//out=hipGetErrorString(hipMemGetInfo(NULL,&total));

//out=hipGetErrorString(hipMemGetInfo(0,&total));
cout<<"The output of hipMemGetInfo() is: "<<out<<endl;

cout<<"The value of free is: "<<free<<endl;
cout<<"The value of total is: "<<total<<endl;




}
