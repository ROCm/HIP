/*
hipHostUnregister (void *hostPtr)
*/

#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <sys/wait.h>
#include <iostream>
#include <signal.h>
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
LOCAL_HIP_HOST_UNREGISTER();
SIGNAL_REGISTER();
	
	
while(i<6)
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
        CHECK_ERRORS(hipHostUnregister(NULL),__LINE__);
                exit(0);}
				
		float *hostPtr = new float;
        lineno=__LINE__;
        if(i==1){
        CHECK_ERRORS(hipHostUnregister(hostPtr),__LINE__);
		        exit(0);}
				
		delete hostPtr;
		        lineno=__LINE__;
        if(i==2){
        CHECK_ERRORS(hipHostUnregister(hostPtr),__LINE__);
		        exit(0);}
		int *hPtr = NULL;
		
        lineno=__LINE__;
        if(i==3){
        CHECK_ERRORS(hipHostUnregister(hPtr),__LINE__);
		 exit(0);}
		 
		 int p; 
        lineno=__LINE__;
        if(i==4){
        CHECK_ERRORS(hipHostUnregister(&p),__LINE__);
		
        exit(0);}
		float fPtr; 
		lineno=__LINE__;
        if(i==5){
        CHECK_ERRORS(hipHostUnregister(&fPtr),__LINE__);
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