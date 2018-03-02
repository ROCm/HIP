/*
  __host__ hipError_t hipDeviceGetPCIBusId ( char* pciBusId, int  len, int  device ) 
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

LOCAL_GET_PCI_BUS_ID();
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
                close(fd[0]);
        lineno=__LINE__;
        if(i==0){
        CHECK_ERRORS(hipDeviceGetPCIBusId(NULL,NULL,NULL),__LINE__);
        //crash
        exit(0);}

        lineno=__LINE__;
        if(i==1){
        CHECK_ERRORS(hipDeviceGetPCIBusId(NULL,len,NULL),__LINE__);
		// crash
        exit(0);}
        lineno=__LINE__;
        if(i==2){
        CHECK_ERRORS(hipDeviceGetPCIBusId(NULL,NULL,device),__LINE__);
		//Output --> Invalid argument
        exit(0);}

        lineno=__LINE__;
        if(i==3){
        CHECK_ERRORS(hipDeviceGetPCIBusId(pciBusId,NULL,NULL),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==4){
        CHECK_ERRORS(hipDeviceGetPCIBusId(pciBusId,len,device),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==5){
        CHECK_ERRORS(hipDeviceGetPCIBusId(pciBusId,-1,device),__LINE__);
		 
		exit(0);}
        lineno=__LINE__;
        if(i==6){
        CHECK_ERRORS(hipDeviceGetPCIBusId(pciBusId,len,-1),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==7){
        CHECK_ERRORS(hipDeviceGetPCIBusId(pciBusId,-1,-1),__LINE__);
		
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


