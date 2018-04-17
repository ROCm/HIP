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
strr=hipGetErrorString(hipStreamCreateWithFlags(&stream, hipStreamDefault));
cout<<"The output of hipStreamCreateWithFlags(&stream, hipStreamDefault) is "<<strr<<endl;

strr=hipGetErrorString(hipStreamCreateWithFlags(&stream, NULL));
cout<<"The output of hipStreamCreateWithFlags(&stream, NULL) is "<<strr<<endl;

strr=hipGetErrorString(hipStreamCreateWithFlags(&stream, 0));
cout<<"The output of hipStreamCreateWithFlags(&stream, 0) is "<<strr<<endl;

strr=hipGetErrorString(hipStreamCreateWithFlags(&stream, 16));
cout<<"The output of hipStreamCreateWithFlags(&stream, 16) is "<<strr<<endl;

strr=hipGetErrorString(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
cout<<"The output of hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) is "<<strr<<endl;

strr=hipGetErrorString(hipStreamCreateWithFlags(&stream, 1000));
cout<<"The output of hipStreamCreateWithFlags(&stream, 1000) is "<<strr<<endl;


strr=hipGetErrorString(hipStreamCreateWithFlags(&stream, -1));
cout<<"The output of hipStreamCreateWithFlags(&stream, -1) is "<<strr<<endl;

//file.close();
        return(0);
  }
