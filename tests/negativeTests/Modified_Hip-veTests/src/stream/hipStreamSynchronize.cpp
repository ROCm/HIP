#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <sys/wait.h>
#include<iostream>
#include<signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include "stream.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

using namespace std;
GLOBAL();
SIGNAL();

int main(){
  LOCAL_HIP_STREAM_CREATE();
  SIGNAL_REGISTER();
     //           file.open(str);
string strr;
//	hipError_t strr;
	hipStream_t stream;
strr=hipGetErrorString(hipStreamCreate(&stream));
cout<<"The output of hipStreamCreate(&stream) is "<<strr<<endl;

strr=hipGetErrorString(hipStreamSynchronize(stream));
cout<<"The output of hipStreamSynchronize(stream) is "<<strr<<endl;
delete stream;
strr=hipGetErrorString(hipStreamSynchronize(stream));
cout<<"The output of hipStreamSynchronize(stream) after delete stream  is "<<strr<<endl;

strr=hipGetErrorString(hipStreamSynchronize(NULL));
cout<<"The output of hipStreamSynchronize(NULL) is "<<strr<<endl;
strr=hipGetErrorString(hipStreamSynchronize(0));
cout<<"The output of hipStreamSynchronize(0) is "<<strr<<endl;
//file.close();
        return(0);
  }
