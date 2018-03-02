/*
 __host__ hipError_t hipIpcOpenMemHandle ( void** devPtr, hipIpcMemHandle_t handle, unsigned int  flags )
flags --> hipIpcMemLazyEnablePeerAccess 
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
cout<<"Currently all the scenarios are giving compilation error in this "<<__FILE__<<
" . hence disabled all the test from compilation."<<endl;
 LOCAL_IPC_OPEN_MEM_HANDLE();
 SIGNAL_REGISTER();
 
 
while(i<5)
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
//        CHECK_ERRORS(hipIpcOpenMemHandle(NULL,handle,hipIpcMemLazyEnablePeerAccess),__LINE__);
        
        exit(0);}

        lineno=__LINE__;
        if(i==1){
  //      CHECK_ERRORS(hipIpcOpenMemHandle(devPtr,handle,hipIpcMemLazyEnablePeerAccess),__LINE__);
		
        exit(0);}
        lineno=__LINE__;
        if(i==2){
    //    CHECK_ERRORS(hipIpcOpenMemHandle(NULL,NULL,NULL),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==3){
      //  CHECK_ERRORS(hipIpcOpenMemHandle(NULL,handle,NULL),__LINE__);
		//Compilation error
        exit(0);}

        lineno=__LINE__;
        if(i==4){
//        CHECK_ERRORS(hipIpcOpenMemHandle(devPtr,handle,hipIpcMemLazyEnablePeerAccess),__LINE__);
		
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
