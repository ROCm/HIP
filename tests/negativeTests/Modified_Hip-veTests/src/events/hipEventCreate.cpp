#include <iostream>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
using namespace std;

int main(){
  
hipError_t strr;
string strr_out;
hipEvent_t evt;

strr_out=hipGetErrorString(hipEventCreate(NULL));
cout<<"The strr_out value is:"<<strr_out<<endl;
}
