#include <iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include<string>

using namespace std;

int main()
{
void *ptr0,*ptr1;
string out;
char str[100]="Hi, I am Ellesemere. What is ur name?";
char output[100];
out=hipGetErrorString(hipSetDevice(0));
cout<<"The value of hipSetDevice() is: "<<out<<endl;

out=hipGetErrorString(hipMalloc(&ptr0,100));
cout<<"The output of hipMalloc() is: "<<out<<endl;


//out=hipGetErrorString(hipMemcpyToSymbolAsync(ptr0,str,100,0, hipMemcpyHostToDevice,0));
//cout<<"The output of hipMemcpyToSymbolAsync() is: "<<out<<endl;

//out=hipGetErrorString(hipMemcpyToSymbolAsync(NULL,str,100,0, hipMemcpyHostToDevice,0));
//cout<<"The output of hipMemcpyToSymbolAsync() is: "<<out<<endl;

//out=hipGetErrorString(hipMemcpyToSymbolAsync(0,str,100,0, hipMemcpyHostToDevice,0));
//cout<<"The output of hipMemcpyToSymbolAsync() is: "<<out<<endl;

//out=hipGetErrorString(hipMemcpyToSymbol(ptr0,str,00,0, hipMemcpyHostToDevice));
//cout<<"The output of hipMemcpyToSymbol() is: "<<out<<endl;
hipStream_t stream;
out=hipGetErrorString(hipStreamCreate(&stream));
cout<<"The output of hipStreamCreate() is: "<<out<<endl;

out=hipGetErrorString(hipMemcpyToSymbolAsync(ptr0,str,100,0, hipMemcpyHostToDevice,stream));
cout<<"The output of hipMemcpyToSymbolAsync() is: "<<out<<endl;

}
