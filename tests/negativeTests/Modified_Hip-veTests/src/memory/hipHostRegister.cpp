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
str=hipGetErrorString(hipHostGetDevicePointer((void**)dptr,(void*)x,0));
cout<<"the output of hipHostGetDevicePointer((void*)x,a,hipHostRegisterDefault) is: "<<str<<endl;


a=sizeof(y);

str=hipGetErrorString(hipHostRegister((void*)y,a,hipHostRegisterPortable));
cout<<"the output of hipHostRegister((void*)y,a,hipHostRegisterPortable) is: "<<str<<endl;
a=sizeof(z);

str=hipGetErrorString(hipHostRegister((void*)z,a,hipHostRegisterMapped));
cout<<"the output of hipHostRegister((void*)z,a,hipHostRegisterMapped) is: "<<str<<endl;

str=hipGetErrorString(hipHostRegister(NULL,a,hipHostRegisterDefault));
cout<<"the output of hipHostRegister(NULL,a,hipHostRegisterDefault) is: "<<str<<endl;
str=hipGetErrorString(hipHostRegister((void*)a,0,hipHostRegisterDefault));
cout<<"the output of hipHostRegister((void*)x,0,hipHostRegisterDefault) is: "<<str<<endl;

str=hipGetErrorString(hipHostRegister((void*)p,NULL,hipHostRegisterDefault));
cout<<"the output of hipHostRegister((void*)p,NULL,hipHostRegisterDefault) is: "<<str<<endl;

str=hipGetErrorString(hipHostRegister((void*)q,a,NULL));
cout<<"the output of hipHostRegister((void*)q,a,NULL) is: "<<str<<endl;
str=hipGetErrorString(hipHostRegister(NULL,NULL,NULL));
cout<<"the output of hipHostRegister(NULL,NULL,NULL) is: "<<str<<endl;
}
