#include<iostream>
#include<string>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

using namespace std;

int main()
{
string out;
void *dst;
int val[10];
hipStream_t stream;

out=hipGetErrorString(hipMalloc(&dst,80));
cout<<"The value of hipMalloc() is: "<<out<<endl;

//out=hipGetErrorString(hipMemsetAsync(dst,99,80,0));

//out=hipGetErrorString(hipMemsetAsync(NULL,99,80,0));

//out=hipGetErrorString(hipMemsetAsync(0,99,80,0));

//out=hipGetErrorString(hipMemsetAsync(dst,NULL,80,0));

//providing out of bound memory value

//out=hipGetErrorString(hipMemsetAsync(dst,99,10000,0));

out=hipGetErrorString(hipStreamCreate(&stream));
cout<<"The output of hipStreamCreate() is: "<<out<<endl;

out=hipGetErrorString(hipMemsetAsync(dst,99,80,stream));
cout<<"The output of hipMemsetAsync() is: "<<out<<endl;


out=hipGetErrorString(hipMemcpy(val,dst,sizeof(val),hipMemcpyDeviceToHost));
cout<<"The output of hipMemcpy() is: "<<out<<endl;

cout<<"The content from gpu is: "<<val[0]<<endl;

}
