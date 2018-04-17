/*
 __host__ __device__ hipError_t hipDeviceGetLimit ( size_t* pValue, hipLimit limit ) 
*/

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
#include "common_prototype3.h"

using namespace std;
GLOBAL();
SIGNAL();

int main(){
  LOCAL_GET_LIMIT();
SIGNAL_REGISTER();


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
        CHECK_ERRORS(hipDeviceGetLimit(NULL,hipLimitMallocHeapSize),__LINE__);
        // Output --> hipErrorInvalidValue
        exit(0);}

        lineno=__LINE__;
        if(i==1){
//        CHECK_ERRORS(hipDeviceGetLimit(&pValue,hipLimitMallocHeapSize),__LINE__);
		//compilation error 
        exit(0);}
        lineno=__LINE__;
        if(i==2){
        CHECK_ERRORS(hipDeviceGetLimit(pValue,hipLimitMallocHeapSize),__LINE__);
        exit(0);}
        }
        else
        {
                close(fd[1]);
                waitpid(childpid,NULL,WUNTRACED);
                c=1;
                lineno=1;
        }
sleep(1);
i++;
                }
file.close();
        return(0);
  }

// CHECK_ERRORS(); 
// CHECK_ERRORS(); 
// CHECK_ERRORS(); // 

