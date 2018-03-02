#include<iostream>
#include<string>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

using namespace std;
int main()
{
int count,d1[1000],d2[1000],i,j=1000,*d1ptr=d1,*d2ptr=d2,size=4000;
void *ptr0,*ptr01,*ptr1,*ptr_to_cpy;
hipError_t status;
string out;
cout<<"size of int is: "<<sizeof(int)<<endl;
for(i=0;i<1000;i++)
d1[i]=i;

for(i=0;i<1000;i++)
{
d2[i]=j;
j++;
}


cout<<"d1"<<d1<<endl;
cout<<"d2"<<d2<<endl;
out=hipGetErrorString(hipGetDeviceCount(&count));
cout<<"The output of hipGetDeviceCount() is: "<<out<<endl;
cout<<"The count value obtained is: "<<count<<endl;

out=hipGetErrorString(hipSetDevice(0));
cout<<"The output of hipSetDevice(0) is: "<<out<<endl;

out=hipGetErrorString(hipMalloc(&ptr0,size));
cout<<"The output of hipMalloc(&ptr0,size) is: "<<out<<endl;

out=hipGetErrorString(hipMalloc(&ptr01,size));
cout<<"The output of hipMalloc(&ptr01,size) is: "<<out<<endl;

//cout<<"*ptr:"<<*ptr<<" ptr:"<<ptr<<" &ptr:"<<&ptr<<endl;
cout<<"ptr:"<<ptr0<<endl;
out=hipGetErrorString(hipMemcpy(ptr0,d1,size,hipMemcpyHostToDevice));
cout<<"The output of hipMemcpy(ptr0,d1,size,hipMemcpyHostToDevice) is: "<<out<<endl;


out=hipGetErrorString(hipSetDevice(1));
cout<<"The output of hipSetDevice(1) is: "<<out<<endl;

out=hipGetErrorString(hipMalloc(&ptr1,size));
cout<<"The output of hipMalloc(&ptr1,size) is: "<<out<<endl;
//cout<<"*ptr:"<<*ptr<<" ptr:"<<ptr<<" &ptr:"<<&ptr<<endl;
cout<<"ptr:"<<ptr1<<endl;

out=hipGetErrorString(hipMemcpy(ptr1,d2,size,hipMemcpyHostToDevice));
cout<<"The output of hipMemcpy(ptr1,d2,size,hipMemcpyHostToDevice) is: "<<out<<endl;


out=hipGetErrorString(hipSetDevice(0));
cout<<"The output of hipSetDevice(0) is: "<<out<<endl;

out=hipGetErrorString(hipMalloc(&ptr_to_cpy,size));
cout<<"The output of hipMalloc(&ptr_to_cpy,size) is: "<<out<<endl;

//out=hipGetErrorString(hipMemcpyPeer(ptr_to_cpy,0,ptr1,1,4000));

//out=hipGetErrorString(hipMemcpyPeer(NULL,0,ptr1,1,4000));

out=hipGetErrorString(hipMemcpyPeer(ptr_to_cpy,0,ptr0,0,4000));

//out=hipGetErrorString(hipMemcpyPeer(ptr_to_cpy,0,NULL,1,4000));

cout<<"The output of hipMemcpyPeer(ptr_to_cpy,0,ptr0,0,4000) is: "<<out<<endl;

}
