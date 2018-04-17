#include <iostream>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
using namespace std;

int main()
{
hipError_t strr;
string strr_out;
hipEvent_t evt;

strr_out=hipGetErrorString(hipEventCreateWithFlags(&evt,NULL));
cout<<"The strr_out value for pEventCreateWithFlags(&evt,NULL)) is:"<<strr_out<<endl;
strr_out=hipGetErrorString(hipEventCreateWithFlags(&evt,0));
cout<<"The strr_out value for hipEventCreateWithFlags(&evt,0)) is:"<<strr_out<<endl;
strr_out=hipGetErrorString(hipEventCreateWithFlags(&evt,2112));
cout<<"The strr_out value for hipEventCreateWithFlags(&evt,2112)) is:"<<strr_out<<endl;  }
