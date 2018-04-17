#include<iostream>
#include<string>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

using namespace std;

int main()
{

void *ptr,*ptr_m;
string out;
hipPointerAttribute_t attr;

out=hipGetErrorString(hipMalloc(&ptr,50));
cout<<"The output of hipMalloc() is: "<<out<<endl;

//out=hipGetErrorString(hipPointerGetAttributes(&attr,ptr));
//cout<<"The output of hipPointerGetAttributes() is: "<<out<<endl;


//out=hipGetErrorString(hipPointerGetAttributes(NULL,ptr));
//cout<<"The output of hipPointerGetAttributes() is: "<<out<<endl;


//out=hipGetErrorString(hipPointerGetAttributes(&attr,NULL));
//cout<<"The output of hipPointerGetAttributes() is: "<<out<<endl;

out=hipGetErrorString(hipPointerGetAttributes(&attr,0x0));
cout<<"The output of hipPointerGetAttributes() is: "<<out<<endl;

}
