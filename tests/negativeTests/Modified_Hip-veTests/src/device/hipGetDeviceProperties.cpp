/*
 __host__ hipError_t hipGetDeviceProperties ( hipDeviceProp* prop, int  device ) 
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
 
LOCAL_GET_DEVICE_PROPERTIES(); 
SIGNAL_REGISTER();

while(i<12)
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
        CHECK_ERRORS(hipGetDeviceProperties(NULL,NULL),__LINE__);
        //compliation error
        exit(0);}

        lineno=__LINE__;
        if(i==1){
//        CHECK_ERRORS(hipGetDeviceProperties(prop,device),__LINE__);
		 // invalid argument error
        exit(0);}
        lineno=__LINE__;
        if(i==2){
        CHECK_ERRORS(hipGetDeviceProperties(NULL,device),__LINE__);
        exit(0);}

        lineno=__LINE__;
        if(i==3){
  //      CHECK_ERRORS(hipGetDeviceProperties(prop,NULL),__LINE__);
		// compilation error
        exit(0);}

        lineno=__LINE__;
        if(i==4){
    //    CHECK_ERRORS(hipGetDeviceProperties(prop,0),__LINE__);
		//compilation error
        exit(0);}

        lineno=__LINE__;
        if(i==5){
        CHECK_ERRORS(hipGetDevice(&device),__LINE__);
		exit(0);}
        lineno=__LINE__;
        if(i==6){
        CHECK_ERRORS(hipGetDeviceProperties(&prop,device),__LINE__);		
        exit(0);}

        lineno=__LINE__;
        if(i==7){
        CHECK_ERRORS(hipGetDeviceProperties(&prop,NULL),__LINE__);
		// This does not return any error, even if the device is NULL
        exit(0);}
		
		lineno=__LINE__;
        if(i==8){
        CHECK_ERRORS(hipGetDeviceProperties(&prop,0),__LINE__);
		// This does not return any error, even if the device is NULL
        exit(0);}
		
		lineno=__LINE__;
        if(i==9){
        CHECK_ERRORS(hipGetDeviceProperties(&prop,-1),__LINE__);
		// Output --> Error: Invalid ordinal
        exit(0);}
		
		lineno=__LINE__;
        if(i==10){
        CHECK_ERRORS(hipGetDeviceProperties(&prop,65536),__LINE__);
        exit(0);}
		
		lineno=__LINE__;
        if(i==11){
        CHECK_ERRORS(hipGetDeviceProperties(&prop,10),__LINE__);
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













