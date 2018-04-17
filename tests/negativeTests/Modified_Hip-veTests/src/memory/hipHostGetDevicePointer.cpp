/*
hipHostGetDevicePointer (void **devPtr, void *hstPtr, unsigned int flags)
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
#include "memory.h"


using namespace std;
GLOBAL();
SIGNAL();

int main(){
	
LOCAL_HOST_GET_DEVICE_POINTER();	
SIGNAL_REGISTER();
	
	
while(i<15)
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
//        CHECK_ERRORS(hipHostGetDevicePointer(NULL,NULL,NULL),__LINE__);
        
        exit(0);}

        lineno=__LINE__;
        if(i==1){
        CHECK_ERRORS(hipHostGetDevicePointer(NULL,NULL,1),__LINE__);
		
        exit(0);}
        lineno=__LINE__;
        if(i==2){
//        CHECK_ERRORS(hipHostGetDevicePointer(NULL,NULL, 65536),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==3){
//        CHECK_ERRORS(hipHostGetDevicePointer(NULL,NULL,0),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==4){
//        CHECK_ERRORS(hipHostGetDevicePointer(NULL,NULL,-1),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==5){
//        CHECK_ERRORS(hipHostGetDevicePointer(0,0,0),__LINE__);
		 
		exit(0);}
        lineno=__LINE__;
        if(i==6){
//        CHECK_ERRORS(hipHostGetDevicePointer(0x0,0x0,0),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==7){
//        CHECK_ERRORS(hipHostGetDevicePointer(0x0,0x0,0x0),__LINE__);
		
        exit(0);}
		void *voidPtr = NULL ;
		void **devPtr = &voidPtr;
		lineno=__LINE__;
        if(i==8){
//        CHECK_ERRORS(hipHostGetDevicePointer(devPtr,NULL,1),__LINE__);
		
        exit(0);}
		 void *hstPtr = NULL ;
		lineno=__LINE__;
        if(i==9){
//        CHECK_ERRORS(hipHostGetDevicePointer(devPtr,hstPtr,0),__LINE__);
		
        exit(0);}
		 float *fPtr = new float;
		float **fDevPtr = &fPtr;
		lineno=__LINE__;
        if(i==10){
//        CHECK_ERRORS(hipHostGetDevicePointer((void **)fDevPtr,NULL,1),__LINE__);
		
        exit(0);}
		
		lineno=__LINE__;
        if(i==11){
//        CHECK_ERRORS(hipHostGetDevicePointer((void **)fDevPtr,hstPtr,1),__LINE__);
		
        exit(0);}
		delete fPtr;
		lineno=__LINE__;
        if(i==12){
//        CHECK_ERRORS(hipHostGetDevicePointer((void **)fDevPtr,hstPtr,1),__LINE__);
		
        exit(0);}
		delete fDevPtr;
		lineno=__LINE__;
        if(i==13){
//        CHECK_ERRORS(hipHostGetDevicePointer((void **)fDevPtr,hstPtr,1),__LINE__);
		
        exit(0);}
		 delete hstPtr;
		lineno=__LINE__;
        if(i==14){
       // CHECK_ERRORS(hipHostGetDevicePointer((void **)fDevPtr,hstPtr,1),__LINE__);
		
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
