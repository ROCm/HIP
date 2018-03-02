#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

using namespace std;

int main()
{
hipError_t status;
hipArray_t array;
/*hipChannelFormatDesc desc;
desc.x=1;
desc.y=1;
desc.z=1;
desc.w=1;*/

hipChannelFormatDesc channelDesc = hipCreateChannelDesc(32, 0, 0, 0,hipChannelFormatKindFloat );





hipExtent extent;
extent.depth=100;
extent.height=100;
extent.width=100;
//status=hipMalloc3DArray(&array,&channelDesc,extent,0);

status=hipMalloc3DArray(0,&channelDesc,extent,0);
//status=hipMalloc3DArray(NULL,&channelDesc,extent,0);
cout<<"The output of hipMalloc3DArray() api is: "<<hipGetErrorString(status)<<endl;




}
