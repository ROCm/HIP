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

strr=hipGetErrorString(hipStreamDestroy(stream));
cout<<"The output of hipStreamDestroy(stream) is "<<strr<<endl;

strr=hipGetErrorString(hipStreamDestroy(NULL));
cout<<"The output of hipStreamDestroy(NULL) is "<<strr<<endl;


//file.close();
        return(0);
  }
