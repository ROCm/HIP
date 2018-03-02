#include<iostream>
#include<string>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

using namespace std;

int main()
{
int x[10],y[10],z[10],a,p,q;
a=sizeof(x);
string str;
void **dptr;
//cout<<"The size of x[10] is: "<<y<<endl;

str=hipGetErrorString(hipHostRegister((void*)x,a,hipHostRegisterDefault));
cout<<"the output of hipHostRegister((void*)x,a,hipHostRegisterDefault) is: "<<str<<endl;
str=hipGetErrorString(hipHostGetDevicePointer(dptr,(void*)x,0));
cout<<"the output of hipHostGetDevicePointer(x,a,hipHostRegisterDefault) is: "<<str<<endl;

}
