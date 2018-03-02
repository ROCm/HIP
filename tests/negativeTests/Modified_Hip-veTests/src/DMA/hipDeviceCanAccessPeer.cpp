#include <iostream>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

using namespace std;

int main()
{
int did1[4],did2,access,count;
hipError_t status;
string out;

out=hipGetErrorString(hipGetDeviceCount(&count));
cout<<"The output of hipGetDeviceCount() is: "<<out<<endl;
cout<<"The count value obtained is: "<<count<<endl;
int access_status;
out=hipGetErrorString(hipDeviceCanAccessPeer(&access_status,NULL,1));
cout<<"The output of hipDeviceCanAccessPeer() is: "<<out<<endl;
cout<<"The output of access_status is:"<<access_status<<endl;



}
