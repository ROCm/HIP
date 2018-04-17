/*
__host__ hipError_t hipSetDevice ( int  device ) 
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

 LOCAL_SET_DEVICE();
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
        CHECK_ERRORS(hipSetDevice(NULL),__LINE__);
        //Does not print any error despite NULL as input
        exit(0);}

        lineno=__LINE__;
        if(i==1){
        CHECK_ERRORS(hipSetDevice(-1),__LINE__);
		//Output --> Invalid device ordinal
        exit(0);}
        lineno=__LINE__;
        if(i==2){
        CHECK_ERRORS(hipSetDevice(65536),__LINE__);
		//same as above
        exit(0);}

        lineno=__LINE__;
        if(i==3){
        CHECK_ERRORS(hipSetDevice(10),__LINE__);
		//same as above
        exit(0);}

        lineno=__LINE__;
        if(i==4){
        CHECK_ERRORS(hipSetDevice(device),__LINE__);
		
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