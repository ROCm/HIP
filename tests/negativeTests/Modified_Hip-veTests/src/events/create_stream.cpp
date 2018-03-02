#include <iostream>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
using namespace std;

int main(){
  
hipError_t strr;
hipStream_t stream;
strr=hipStreamCreate(&stream);
string strr_out=hipGetErrorString(strr);
cout<<"the output of hipStreamCreate(&stream) is: "<<strr_out<<endl;
strr=hipStreamCreate(NULL);//This one crashes
cout<<"the output of hipStreamCreate(NULL) is: "<<strr_out<<endl;
strr=hipStreamCreate(0);
cout<<"the output of hipStreamCreate(0) is: "<<strr_out<<endl;

hipStream_t stream1;
//strr=hipStreamCreate(stream1); This gives compilation error
}
