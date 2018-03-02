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
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "stream.h"

using namespace std;
GLOBAL();
SIGNAL();

int main(){
  LOCAL_HIP_STREAM_CREATE();
  SIGNAL_REGISTER();
hipError_t strr;
hipStream_t stream;
//strr=hipStreamCreate(NULL);
//string strr_out=hipGetErrorString(strr);
//string strr_out=hipGetErrorString(hipStreamCreate(&stream));
//cout<<"the strr_out is: "<<strr_out<<endl;


  while(i<3)
{
 if((childpid = fork()) == -1)
        {
                perror("fork");
                exit(1);
        }

        if(childpid == 0)
        {
                file.open(str);
                close(fd[0]);
        lineno=__LINE__;
         if(i==0){
        CHECK_ERRORS(hipStreamCreate(NULL),__LINE__);
        
        exit(0);}
     hipStream_t *stream=new hipStream_t;
//	hipStream_t stream;
        lineno=__LINE__;
        if(i==1){
        CHECK_ERRORS(hipStreamCreate(stream),__LINE__);
		
        exit(0);}
	delete stream;
 	lineno=__LINE__;
        if(i==2){
      CHECK_ERRORS(hipStreamCreate(stream),__LINE__);

        exit(0);
	}

		
		}
        else
        {
                close(fd[1]);
                waitpid(childpid,NULL,WUNTRACED);
                c=1;
                lineno=1;
        }
i++;
                }
file.close();
        return(0);
  }
