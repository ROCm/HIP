/*
 __host__ hipError_t hipIpcOpenEventHandle ( hipEvent_t* event, hipIpcEventHandle_t handle ) 
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

LOCAL_IPC_OPEN_EVENT_HANDLE();
SIGNAL_REGISTER();
cout<<"Running of this test "<<__FILE__<<" will give the following error"<<endl<<endl;
cout<<"hipIpcOpenEventHandle_modified.cpp:50:22: error: use of undeclared identifier 'hipIpcOpenEventHandle'; did you mean 'hipIpcGetMemHandle'?"<<endl;
cout<<"There fore all the test are disable for now"<<endl;
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
                close(fd[0]);
        lineno=__LINE__;
        if(i==0){
//        CHECK_ERRORS(hipIpcOpenEventHandle(NULL,NULL),__LINE__);
        //compilation error: no suitable constructor exists to convert from "long" to "cudaIpcEventHandle_st"
        exit(0);}
		event = new hipEvent_t;
		delete event;
        lineno=__LINE__;
        if(i==1){
  //      CHECK_ERRORS(hipIpcOpenEventHandle(event,handle),__LINE__);
		//Output --> unknown error
        exit(0);}
		hipEvent_t eventOne;
		hipEventCreate(&eventOne);
        lineno=__LINE__;
        if(i==2){
    //    CHECK_ERRORS(hipIpcOpenEventHandle(&eventOne,handle),__LINE__);
		
        exit(0);}
		hipEvent_t eventTwo;
		hipEventCreate(&eventTwo);
		delete eventTwo;
        lineno=__LINE__;
        if(i==3){
      //  CHECK_ERRORS(hipIpcOpenEventHandle(&eventTwo,handle),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==4){
//        CHECK_ERRORS(hipIpcOpenEventHandle(&eventTwo,hipIpcEventHandle_t(-1)),__LINE__);
		// compilation error
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

//Note: I cud not find much info abt hipIpcEventHandle_t or any documentation on hipIpcOpenEventHandle. So unable to test further
