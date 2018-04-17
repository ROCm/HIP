#include<iostream>
#include<string>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

using namespace std;
int main()
{
int count;
hipError_t status;
string out;

//out=hipGetErrorString(hipGetDeviceCount(&count));
//cout<<"The output of hipGetDeviceCount() is: "<<out<<endl;
//cout<<"The count value obtained is: "<<count<<endl;
out=hipGetErrorString(hipSetDevice(0));
cout<<"The output of hipSetDevice(0) is: "<<out<<endl;

int access_status;
out=hipGetErrorString(hipDeviceCanAccessPeer(&access_status,0,1));
cout<<"The output of hipDeviceCanAccessPeer(&access_status,0,1) is: "<<out<<endl;
cout<<"The output of access_status is:"<<access_status<<endl;

out=hipGetErrorString(hipDeviceEnablePeerAccess(1,0));
cout<<"The output of hipDeviceEnablePeerAccess(1,0) is: "<<out<<endl;
//out=hipGetErrorString(hipDeviceDisablePeerAccess(1));
//cout<<"The output of hipDeviceDisablePeerAccess(1) is: "<<out<<endl;

out=hipGetErrorString(hipDeviceCanAccessPeer(&access_status,0,1));
cout<<"The output of hipDeviceCanAccessPeer(&access_status,0,1) is: "<<out<<endl;
cout<<"The output of access_status is:"<<access_status<<endl;

//out=hipGetErrorString(hipDeviceEnablePeerAccess(1,0));
//cout<<"The output of hipDeviceEnablePeerAccess(1) is: "<<out<<endl;
//cout<<"The count value obtained is: "<<count<<endl;



}
