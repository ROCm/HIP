/*
__host__ __device__ hipError_t hipGetDevice ( int* device ) 
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

LOCAL_GET_DEVICE();
  SIGNAL_REGISTER();
  
  while(i<4)
{
 if((childpid = fork()) == -1)
        {
                perror("fork");
                exit(1);
        }

        if(childpid == 0)
        {
                file.open(str);
//              cout<<"Am in child process."<<endl;;
                close(fd[0]);
        lineno=__LINE__;
        if(i==0){
        CHECK_ERRORS(hipGetDevice(NULL),__LINE__);
        //Output --> invalid argument
        exit(0);}

        lineno=__LINE__;
        if(i==1){
//        CHECK_ERRORS(hipGetDevice(-1),__LINE__);
		// compilation error, as expected
        exit(0);}
        lineno=__LINE__;
        if(i==2){
        CHECK_ERRORS(hipGetDevice(device),__LINE__);
        exit(0);}

        lineno=__LINE__;
        if(i==3){
        CHECK_ERRORS(hipGetDevice(device_ptr),__LINE__);
		//Output --> No crash or error.
        exit(0);}
		
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
