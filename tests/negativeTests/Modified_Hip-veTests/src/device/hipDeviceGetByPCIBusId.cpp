/*
__host__ hipError_t hipDeviceGetByPCIBusId ( int* device, const char* pciBusId ) 

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

LOCAL_PCIBUSID();
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
        CHECK_ERRORS(hipDeviceGetByPCIBusId(NULL,NULL),__LINE__);
        //Crash
        exit(0);}

        lineno=__LINE__;
        if(i==1){
        CHECK_ERRORS(hipDeviceGetByPCIBusId(device,NULL),__LINE__);
		//Crash
        exit(0);}
        lineno=__LINE__;
        if(i==2){
        CHECK_ERRORS(hipDeviceGetByPCIBusId(device_ptr,NULL),__LINE__);
		//Crash
        exit(0);}

        lineno=__LINE__;
        if(i==3){
//        CHECK_ERRORS(hipDeviceGetByPCIBusId(NULL,pciBusId),__LINE__);
		// compilation error: argument of type "const char *" is incompatible with parameter of type "const int *"
        exit(0);}

        lineno=__LINE__;
        if(i==4){
  //      CHECK_ERRORS(hipDeviceGetByPCIBusId(device,pciBusId),__LINE__);
		// compilation error:
        exit(0);}

        lineno=__LINE__;
        if(i==5){
    //    CHECK_ERRORS(hipDeviceGetByPCIBusId(device,pciBusId_ptr),__LINE__);
		 // compilation error:
		exit(0);}
        lineno=__LINE__;
        if(i==6){
      //  CHECK_ERRORS(hipDeviceGetByPCIBusId(&device,NULL),__LINE__);
		// compilation error:
        exit(0);}

        lineno=__LINE__;
        if(i==7){
 //       CHECK_ERRORS(hipDeviceGetByPCIBusId(&device_ptr,NULL),__LINE__);
		// compilation error:
        exit(0);}
        }
        else
        {
                close(fd[1]);
                waitpid(childpid,NULL,WUNTRACED);
                c=1;
                lineno=1;
        }
//sleep(1);
i++;
                }
file.close();
        return(0);
  }





 
// CHECK_ERRORS();
// CHECK_ERRORS();
// CHECK_ERRORS();

// CHECK_ERRORS(); 
// CHECK_ERRORS(); 
// CHECK_ERRORS(); 
// CHECK_ERRORS();
// CHECK_ERRORS();// same as above

