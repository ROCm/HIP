/*
__host__ __device__ hipError_t hipDeviceGetAttribute ( int* value, hipDeviceAttr attr, int  device ) 
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

LOCAL_ATTRIBUTE();
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
//        CHECK_ERRORS(hipDeviceGetAttribute(NULL,NULL,NULL),__LINE__);
        //CHECK_ERRORS(hipChooseDevice(NULL, NULL),__LINE__);
		//Compilation error: argument of type "long" is incompatible with parameter of type "hipDeviceAttribute_t"
        exit(0);}

        lineno=__LINE__;
        if(i==1){
        CHECK_ERRORS(hipDeviceGetAttribute(NULL,attr,device),__LINE__);
		//Output--> Error: hipErrorInvalidValue at 15 in hipDeviceGetAttribute.cpp
        exit(0);}
        lineno=__LINE__;
        if(i==2){
        CHECK_ERRORS(hipDeviceGetAttribute(NULL,attr,NULL),__LINE__);
		//same as above
        exit(0);}

        lineno=__LINE__;
        if(i==3){
  //      CHECK_ERRORS(hipDeviceGetAttribute(value,NULL,device),__LINE__);
		//Compilation error ; same as #1 above
        exit(0);}

        lineno=__LINE__;
        if(i==4){
        CHECK_ERRORS(hipDeviceGetAttribute(value,attr,NULL),__LINE__);
		//  crashes on repeated runs
        exit(0);}

        lineno=__LINE__;
        if(i==5){
        CHECK_ERRORS(hipDeviceGetAttribute(value,attr,device),__LINE__);
		// crashes on repeated runs 
		exit(0);}
        lineno=__LINE__;
        if(i==6){
        CHECK_ERRORS(hipDeviceGetAttribute(attr_value,attr,device),__LINE__);
		//crashes on repeated runs
        exit(0);}

        lineno=__LINE__;
        if(i==7){
    //    CHECK_ERRORS(hipDeviceGetAttribute(attr_value,1,0),__LINE__);
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
//sleep(1);
i++;
                }
file.close();
        return(0);
  }

//  hipDeviceAttribute_t attr;
//  int device;
//  int *value = new int;
//  delete value;
//  int *attr_value;
  

// CHECK_ERRORS();
// CHECK_ERRORS(); 
// CHECK_ERRORS(); 
// CHECK_ERRORS(); 
// CHECK_ERRORS(); 
// CHECK_ERRORS();  
// CHECK_ERRORS(); 
// return 0;
//}
