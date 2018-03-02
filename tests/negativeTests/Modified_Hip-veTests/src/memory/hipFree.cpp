/*
 hipFree (void *ptr)
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
	LOCAL_HIP_FREE();
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
                close(fd[0]);
        lineno=__LINE__;
        if(i==0){
        CHECK_ERRORS(hipFree(NULL),__LINE__);
        //invalid device ptr
        exit(0);}

        lineno=__LINE__;
        if(i==1){
        CHECK_ERRORS(hipFree(ptr),__LINE__);
		// same as above
        exit(0);}
		delete ptr;
        lineno=__LINE__;
        if(i==2){
        CHECK_ERRORS(hipFree(ptr),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==3){
        CHECK_ERRORS(hipFree(ptr_i),__LINE__);
		
        exit(0);}

        lineno=__LINE__;
        if(i==4){
        CHECK_ERRORS(hipFree(&p),__LINE__);
		
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
	
