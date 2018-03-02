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
#include "stream.h"

using namespace std;
GLOBAL();
SIGNAL();


int main()
{
LOCAL_HIP_DEVICE_RESET();
SIGNAL_REGISTER();
 
 
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

 CHECK_ERRORS(hipGetDevice(&deviceId),__LINE__);
 
lineno=__LINE__;

 CHECK_ERRORS(hipGetDeviceProperties(&props, deviceId),__LINE__);
 printf ("info: running on device #%d %s\n", deviceId, props.name);


lineno=__LINE__;

CHECK_ERRORS(hipStreamCreate(&stream),__LINE__);
lineno=__LINE__;

CHECK_ERRORS(hipStreamQuery(stream),__LINE__);
lineno=__LINE__;

CHECK_ERRORS(hipDeviceReset(),__LINE__);
lineno=__LINE__;

CHECK_ERRORS(hipStreamQuery(stream),__LINE__);


}
        else
        {
                close(fd[1]);
                waitpid(childpid,NULL,WUNTRACED);
                c=1;
                lineno=1;
        }
file.close();
        return(0);
}
