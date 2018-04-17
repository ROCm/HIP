/*
 __host__ hipError_t hipIpcGetMemHandle ( hipIpcMemHandle_t* handle, void* devPtr ) 
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
 
LOCAL_IPCGET_MEMHANDLE();
SIGNAL_REGISTER();
 
 
 while(i<8)
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
        CHECK_ERRORS(hipIpcGetMemHandle(NULL,NULL),__LINE__);
        
        exit(0);}

        lineno=__LINE__;
        if(i==1){
        CHECK_ERRORS(hipIpcGetMemHandle(handle,NULL),__LINE__);
		
        exit(0);}
        lineno=__LINE__;
        if(i==2){
        CHECK_ERRORS(hipIpcGetMemHandle(NULL,devPtr),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==3){
        CHECK_ERRORS(hipIpcGetMemHandle(handle,devPtr),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==4){
        CHECK_ERRORS(hipIpcGetMemHandle(handle_ptr,devPtr),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==5){
//        CHECK_ERRORS(hipIpcGetMemHandle(&handle,&devPtr),__LINE__);
		  //compilation error
		exit(0);}
        lineno=__LINE__;
        if(i==6){
  //      CHECK_ERRORS(hipIpcGetMemHandle(hipHandle(-1),devPtr),__LINE__);
		//undeclared hipHandle
        exit(0);}

        lineno=__LINE__;
        if(i==7){
    //    CHECK_ERRORS(hipIpcGetMemHandle(hipHandle(0),devPtr),__LINE__);
		//undeclared hipHandle
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

