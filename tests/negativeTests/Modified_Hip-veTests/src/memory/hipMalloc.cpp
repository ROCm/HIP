#include<iostream>
#include<string>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

using namespace std;

int main()
{
void *ptr;
string str;

str=hipGetErrorString(hipMalloc((void**)&ptr,100));
cout<<"the output of hipGetErrorString(hipMalloc((void**)&ptr,100) is: "<<str<<endl;
/*str=hipGetErrorString(hipMalloc((void**)&ptr,0));
cout<<"the output of hipGetErrorString(hipMalloc((void**)&ptr,0) is: "<<str<<endl;str=hipGetErrorString(hipMalloc((void**)ptr,100));
cout<<"the output of hipGetErrorString(hipMalloc((void**)ptr,100) is: "<<str<<endl;

str=hipGetErrorString(hipMalloc(NULL,NULL));
cout<<"the output of hipGetErrorString(hipMalloc(NULL,NULL) is: "<<str<<endl;
*/


}
