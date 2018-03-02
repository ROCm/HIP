#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <string>

using namespace std;
int main()
{

hipEvent_t evt,evt1;
string out;

out=hipGetErrorString(hipEventCreate(&evt));
cout<<"The output of hipEventCreate() is: "<<out<<endl;

out=hipGetErrorString(hipEventDestroy(evt1));
cout<<"The output of hipEventDestroy() is: "<<out<<endl;


}
