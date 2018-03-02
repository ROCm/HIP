/*
__host__ hipError_t hipDeviceSetLimit ( hipLimit limit, size_t value ) 
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
LOCAL_SET_LIMIT();
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
                close(fd[0]);
        lineno=__LINE__;
        if(i==0){
        CHECK_ERRORS(hipDeviceSetLimit(NULL,NULL),__LINE__);
        
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


 
