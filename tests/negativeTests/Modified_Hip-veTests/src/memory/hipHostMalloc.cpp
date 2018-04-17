/*
hipHostMalloc (void **ptr, size_t size, unsigned int flags)
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
LOCAL_HIP_HOST_MALLOC();	
SIGNAL_REGISTER();	
	
	
while(i<14)
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
        CHECK_ERRORS(hipHostMalloc(0x0,0,0),__LINE__);
        
        exit(0);}
		 void *Vptr;
		void **ptr = &Vptr;
        lineno=__LINE__;
        if(i==1){
        CHECK_ERRORS(hipHostMalloc(NULL,1,0),__LINE__);
		
        exit(0);}
        lineno=__LINE__;
        if(i==2){
        CHECK_ERRORS(hipHostMalloc(ptr,SIZE_MAX,0),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==3){
        CHECK_ERRORS(hipHostMalloc(ptr,1),__LINE__);
		
        exit(0);}
        lineno=__LINE__;
        if(i==4){
        CHECK_ERRORS(hipHostMalloc(ptr,1,0),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==5){
        CHECK_ERRORS(hipHostMalloc(ptr,1,hipHostMallocDefault),__LINE__);
		 
		exit(0);}
        lineno=__LINE__;
        if(i==6){
        CHECK_ERRORS(hipHostMalloc(ptr,1,hipHostMallocPortable),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==7){
        CHECK_ERRORS(hipHostMalloc(ptr,1,hipHostMallocMapped),__LINE__);
		
        exit(0);}
		
		lineno=__LINE__;
        if(i==8){
        CHECK_ERRORS(hipHostMalloc(ptr,1,hipHostMallocWriteCombined),__LINE__);
		
        exit(0);}
		
		lineno=__LINE__;
        if(i==9){
        CHECK_ERRORS(hipHostMalloc(ptr,1,hipHostMallocDefault),__LINE__);
		
        exit(0);}
		
		lineno=__LINE__;
        if(i==10){
        CHECK_ERRORS(hipHostMalloc(ptr,0,hipHostMallocDefault),__LINE__);
		
        exit(0);}
		
		lineno=__LINE__;
        if(i==11){
        CHECK_ERRORS(hipHostMalloc(ptr,NULL,hipHostMallocDefault),__LINE__);
		
        exit(0);}
		
		lineno=__LINE__;
        if(i==12){
        CHECK_ERRORS(hipHostMalloc(ptr,0x0,hipHostMallocDefault),__LINE__);
		
        exit(0);}
		
		lineno=__LINE__;
        if(i==13){
        CHECK_ERRORS(hipHostMalloc(ptr,1000,hipHostMallocDefault),__LINE__);
		
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
