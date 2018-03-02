#include<iostream>
#include<string>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

using namespace std;

int main()
{
string str;
str=hipGetErrorString(hipMalloc(NULL,NULL));
cout<<"the output of hipGetErrorString(hipMalloc(NULL,NULL) is: "<<str<<endl;



}
