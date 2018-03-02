/*
 __host__ hipError_t hipDeviceGetStreamPriorityRange ( int* leastPriority, int* greatestPriority ) 
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
LOCAL_STREAM_PRIORITY_RANGE();
SIGNAL_REGISTER();


while(i<1)
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
        CHECK_ERRORS(hipDeviceGetStreamPriorityRange(NULL,NULL),__LINE__);
        
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
