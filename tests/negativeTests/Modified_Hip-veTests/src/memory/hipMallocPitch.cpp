#include <iostream>
#include <string.h>
#include <unistd.h>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
using namespace std;

int main()
{
string out;
size_t pitch;
void *ptr;
int *int_ptr;
//out=hipGetErrorString(hipMallocPitch(&ptr, &pitch, 12,24));

out=hipGetErrorString(hipMallocPitch(&ptr, &pitch, 12,0));
//out=hipGetErrorString(hipMallocPitch(&ptr, &pitch, 0,24));
//out=hipGetErrorString(hipMallocPitch(&ptr, &pitch, 12,NULL));
//out=hipGetErrorString(hipMallocPitch(&ptr, &pitch, NULL,24));
//out=hipGetErrorString(hipMallocPitch(&ptr, NULL, 12,24));
//out=hipGetErrorString(hipMallocPitch(NULL, &pitch, 12,24));

//out=hipGetErrorString(hipMallocPitch(0, &pitch, 12,24));
cout<<"The output of hipMallocPitch() is: "<<out<<endl;
int_ptr=(int *)(ptr);
cout<<"The value of *int_ptr is: "<<(*int_ptr)<<endl;
cout<<"The value of int_ptr is: "<<int_ptr<<endl;
cout<<"The value of pitch is: "<<pitch<<endl;


}
