#include <iostream>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

using namespace std;

int main()
{
hipError_t status;
hipEvent_t evt,evt1;
string strr_out;
hipStream_t stream,stream1;
strr_out=hipGetErrorString(hipEventCreate(&evt));
cout<<strr_out<<endl;

//strr_out=hipGetErrorString(hipEventRecord(evt1,NULL));
//cout<<strr_out<<endl;

strr_out=hipGetErrorString(hipStreamCreate(&stream));
cout<<strr_out<<endl;

strr_out=hipGetErrorString(hipEventRecord(evt,2323));
cout<<strr_out<<endl;

}
