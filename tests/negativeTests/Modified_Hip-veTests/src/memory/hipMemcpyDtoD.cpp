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

out=hipGetErrorString(hipSetDevice(1));
cout<<"The output of hipSetDevice() is: "<<out<<endl;

out=hipGetErrorString(hipMalloc(&ptr1,100));
cout<<"The output of hipMalloc() is: "<<out<<endl;

out=hipGetErrorString(hipMemcpy(ptr0,str,100, hipMemcpyHostToDevice));
cout<<"The output of hipMemcpy() is: "<<out<<endl;


out=hipGetErrorString(hipSetDevice(0));
cout<<"The value of hipSetDevice() is: "<<out<<endl;


out=hipGetErrorString(hipMemcpyDtoD(1,0,100));
cout<<"The output of hipMemcpyDtoD() is: "<<out<<endl;



/*out=hipGetErrorString(hipMemcpy(ptr1,ptr0,100, hipMemcpyDeviceToDevice));
cout<<"The output of hipMemcpy() is: "<<out<<endl;


//out=hipGetErrorString(hipMemcpy(output,ptr1,100, hipMemcpyDeviceToHost));
//cout<<"The output of hipMemcpy() is: "<<out<<endl;
//cout<<"The content in the second gpu is: "<<output<<endl;

out=hipGetErrorString(hipMemcpy(output,ptr1,100, hipMemcpyDeviceToDevice));
cout<<"The output of hipMemcpy() is: "<<out<<endl;

cout<<"The content in the second gpu is: "<<output<<endl;
//out=hipGetErrorString(hipMemcpy(NULL,str,100, hipMemcpyHostToDevice));
//cout<<"The output of hipMemcpy() is: "<<out<<endl;

//out=hipGetErrorString(hipMemcpy(0,str,100, hipMemcpyHostToDevice));
//cout<<"The output of hipMemcpy() is: "<<out<<endl;

//out=hipGetErrorString(hipMemcpy(ptr,NULL,100, hipMemcpyHostToDevice));
//cout<<"The output of hipMemcpy() is: "<<out<<endl;

//out=hipGetErrorString(hipMemcpy(ptr,str,0, hipMemcpyHostToDevice));
//cout<<"The output of hipMemcpy() is: "<<out<<endl;

//out=hipGetErrorString(hipMemcpy(ptr,str,100, NULL));
//cout<<"The output of hipMemcpy() is: "<<out<<endl;
*/
}
