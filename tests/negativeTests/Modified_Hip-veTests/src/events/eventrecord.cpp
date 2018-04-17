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
cout<<"The output of hipEventCreate() is: "<<strr_out<<endl;

strr_out=hipGetErrorString(hipEventRecord(evt,stream1));
cout<<"The output of hipEventRecord() is:"<<strr_out<<endl;

}
