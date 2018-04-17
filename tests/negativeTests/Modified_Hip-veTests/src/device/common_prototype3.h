#ifndef HIP_NEGATIVE_TESTS_INCLUDE_COMMON_H
#define HIP_NEGATIVE_TESTS_INCLUDE_COMMON_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <fstream>
#include <string>




#define CHECK_ERRORS(status,lineval) \
  { \
c=1;\
for(;c<(lineval + 1);c++)\
{\
getline(file,line);\
if(c==lineval)\
{\
string res=hipGetErrorString(status),out="Result: ";\
out+=res;\
out+=line;\
cout<<out<<endl<<endl;\
}\
}\
}
#endif


#define GLOBAL()\
int fd[2],nbytes,lineno=0,c=1,i=0;\
pid_t childpid;\
char readbuffer[80];\
int err=pipe(fd);\
string line;



#define LOCAL()\
memset (readbuffer, 0, sizeof(readbuffer));\
  hipDeviceProp_t prop;\
  hipDeviceProp_t *ptr = new hipDeviceProp_t;\
  delete ptr;\
  int *device_ptr = new int;\
  delete device_ptr;\
  int device,l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_ATTRIBUTE()\
memset (readbuffer, 0, sizeof(readbuffer));\
  hipDeviceAttribute_t attr;\
  int device,l=0,count=0;\
  int *value=new int;\
  delete value;\
  int *attr_value;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_PCIBUSID()\
memset (readbuffer, 0, sizeof(readbuffer));\
int *device_ptr=new int;\
delete device_ptr;\
int *device;\
char *pciBusId_ptr=new char;\
delete pciBusId_ptr;\
const char *pciBusId;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_CACHE_CONFIG()\
memset (readbuffer, 0, sizeof(readbuffer));\
  hipFuncCache_t *cacheConfig=NULL;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_GET_LIMIT()\
memset (readbuffer, 0, sizeof(readbuffer));\
size_t *pValue;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;



#define LOCAL_P2P_ATTRIBUTE()\
memset(readbuffer,0,sizeof(readbuffer));\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;

#define LOCAL_GET_PCI_BUS_ID()\
memset(readbuffer,0,sizeof(readbuffer));\
char pciBusId[64];\
char *pciBusId_ptr=new char;\
  int l=0,count=0,len,device;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_STREAM_PRIORITY_RANGE()\
memset(readbuffer,0,sizeof(readbuffer));\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;

#define LOCAL_SET_LIMIT()\
memset(readbuffer,0,sizeof(readbuffer));\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_DEVICE_COUNT()\
memset(readbuffer,0,sizeof(readbuffer));\
int *count1;\
int *count_ptr=new int;\
delete count_ptr;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;




#define LOCAL_GET_DEVICE()\
memset(readbuffer,0,sizeof(readbuffer));\
int *device;\
int *device_ptr=new int;\
delete device_ptr;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;



#define LOCAL_GET_DEVICE_PROPERTIES()\
memset(readbuffer,0,sizeof(readbuffer));\
int device;\
hipDeviceProp_t prop;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_IPCCLOSE_MEMHANDLE()\
memset(readbuffer,0,sizeof(readbuffer));\
void *devPtr;\
int *dev_ptr=new int;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_IPCGET_MEMHANDLE()\
memset(readbuffer,0,sizeof(readbuffer));\
hipIpcMemHandle_t *handle;\
hipIpcMemHandle_t *handle_ptr=new hipIpcMemHandle_t;\
delete handle_ptr;\
void *devPtr;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_IPC_OPEN_EVENT_HANDLE()\
memset(readbuffer,0,sizeof(readbuffer));\
hipEvent_t *event;\
hipIpcEventHandle_t handle;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_IPC_OPEN_MEM_HANDLE()\
memset(readbuffer,0,sizeof(readbuffer));\
hipIpcEventHandle_t handle;\
void **devPtr;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_SET_DEVICE()\
memset(readbuffer,0,sizeof(readbuffer));\
  int device;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;

#define LOCAL_SET_DEVICE_FLAG()\
memset(readbuffer,0,sizeof(readbuffer));\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_SET_VALID_DEVICES()\
memset(readbuffer,0,sizeof(readbuffer));\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;


#define LOCAL_GET_DEVICE_FLAGS()\
memset(readbuffer,0,sizeof(readbuffer));\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;

#define LOCAL_HIP_FREE()\
memset(readbuffer,0,sizeof(readbuffer));\
  float *ptr = new float;\
  int *ptr_i=NULL,p;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;



#define SIGNAL_REGISTER()\
struct sigaction act;\
memset(&act, 0, sizeof(act));\
   act.sa_handler = sig_handler;\
   sigaction(SIGINT,  &act, 0);\
   sigaction(SIGTERM, &act, 0);\
sigaction(SIGHUP, &act, 0);\
sigaction(SIGQUIT, &act, 0);\
sigaction(SIGILL, &act, 0);\
sigaction(SIGTRAP, &act, 0);\
sigaction(SIGABRT, &act, 0);\
sigaction(SIGBUS, &act, 0);\
sigaction(SIGFPE, &act, 0);\
sigaction(SIGUSR1, &act, 0);\
sigaction(SIGSEGV, &act, 0);\
sigaction(SIGUSR2, &act, 0);\
sigaction(SIGPIPE, &act, 0);\
sigaction(SIGALRM, &act, 0);\
sigaction(SIGSTKFLT, &act, 0);\
sigaction(SIGCONT, &act, 0);\
sigaction(SIGSTOP, &act, 0);\
sigaction(SIGTSTP, &act, 0);\
sigaction(SIGTTIN, &act, 0);\
sigaction(SIGTTOU, &act, 0);\
sigaction(SIGURG, &act, 0);\
sigaction(SIGXCPU, &act, 0);\
sigaction(SIGXFSZ, &act, 0);\
sigaction(SIGVTALRM, &act, 0);\
sigaction(SIGPROF, &act, 0);\
sigaction(SIGWINCH, &act, 0);\
sigaction(SIGIO, &act, 0);\
sigaction(SIGPWR, &act, 0);\
sigaction(SIGSYS, &act, 0);


#define SIGNAL() \
 void sig_handler(int signo)\
{\
std::string strsig;\
switch(signo){\
                case SIGTERM:\
                        strsig="received SIGTERM";\
                        break;\
                case SIGHUP  :\
                        strsig="received SIGHUP";\
                        break;\
                case SIGINT  :\
                        strsig="received SIGINT";\
                        break;\
                case SIGQUIT  :\
                        strsig="received SIGQUIT";\
                        break;\
                case SIGILL  :\
                        strsig="received SIGILL";\
                        break;\
                case SIGTRAP  :\
                        strsig="received SIGTRAP";\
                        break;\
                case SIGABRT  :\
                        strsig="received SIGABRT";\
                        break;\
                case SIGBUS  :\
                        strsig="received SIGBUS";\
                        break;\
                case SIGFPE  :\
                        strsig="received SIGFPE";\
                        break;\
                case SIGUSR1  :\
                        strsig="received SIGUSR1";\
                        break;\
                case SIGSEGV  :\
                        strsig="received SIGSEGV";\
                        break;\
                case SIGUSR2  :\
                        strsig="received SIGUSR2";\
                        break;\
                case SIGPIPE  :\
                        strsig="received SIGPIPE";\
                        break;\
                case SIGALRM  :\
			strsig="received SIGALRM";\
                        break;\
                case SIGSTKFLT  :\
                        strsig="received SIGSTKFLT";\
                        break;\
                case SIGCONT  :\
                        strsig="received SIGCONT";\
                        break;\
                case SIGSTOP  :\
                        strsig="received SIGSTOP";\
                        break;\
                case SIGTSTP  :\
                        strsig="received SIGTSTP";\
                        break;\
                case SIGTTIN  :\
                        strsig="received SIGTTIN";\
                        break;\
                case SIGTTOU  :\
                        strsig="received SIGTTOU";\
                        break;\
                case SIGURG  :\
                        strsig="received SIGURG";\
                        break;\
                case SIGXCPU  :\
                        strsig="received SIGXCPU";\
                        break;\
                case SIGXFSZ  :\
                        strsig="received SIGXFSZ";\
                        break;\
                case SIGPROF  :\
                        strsig="received SIGPROF";\
                        break;\
                case SIGWINCH  :\
                        strsig="received SIGWINCH";\
                        break;\
                case SIGIO  :\
                        strsig="received SIGIO";\
                        break;\
                case SIGPWR  :\
                        strsig="received SIGPWR";\
                        break;\
                case SIGSYS  :\
			strsig="received SIGSYS";\
                        break;\
                default:\
                        printf("No signal caught\n");\
                        	}\
ifstream file;\
  string str,str1;\
  str="../../src/device/";\
  str=str + __FILE__;\
file.open(str);\
c=1;\
lineno=lineno + 2;\
for(;c<(lineno + 1);c++)\
{\
getline(file,line);\
if(c==lineno)\
{\
string out="Result: ";\
out+=strsig;\
out+=line;\
cout<<out<<endl<<endl;\
int len=out.size();\
char send[100];\
strcpy(send,out.c_str());\
file.close();\
}\
}\
exit(0);\
}



