#include <iostream>
#include <string>
#include <string.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

using namespace std;

int main(){
string strr;
	hipStream_t stream;

strr=hipGetErrorString(hipStreamCreateWithFlags(&stream, -1));
cout<<"The output of hipStreamCreateWithFlags(&stream, -1) is "<<strr<<endl;

        return(0);
  }
