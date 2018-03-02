#ifndef HIP_NEGATIVE_TESTS_INCLUDE_COMMON_H
#define HIP_NEGATIVE_TESTS_INCLUDE_COMMON_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <fstream>
#include <string>


#ifdef WINDOWS
    #include <direct.h>
    #define GetCurrentDir _getcwd
#else
    #include <unistd.h>
    #define GetCurrentDir getcwd
 #endif

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

#define LOCAL_HIP_FREE()\
memset(readbuffer,0,sizeof(readbuffer));\
  float *ptr = new float;\
  int *ptr_i=NULL,p;\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
char Pwd[FILENAME_MAX];\
GetCurrentDir(Pwd,sizeof(Pwd));\
str=Pwd;\
  str= str +  "/" + "../../src/memory/" + __FILE__;


#define LOCAL_HOST_FREE()\
memset(readbuffer,0,sizeof(readbuffer));\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
char Pwd[FILENAME_MAX];\
GetCurrentDir(Pwd,sizeof(Pwd));\
str=Pwd;\
  str= str +  "/" + "../../src/memory/" + __FILE__;



#define LOCAL_HOST_GET_DEVICE_POINTER()\
memset(readbuffer,0,sizeof(readbuffer));\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
char Pwd[FILENAME_MAX];\
GetCurrentDir(Pwd,sizeof(Pwd));\
str=Pwd;\
  str= str +  "/" + "../../src/memory/" + __FILE__;


 #define LOCAL_HIP_HOST_MALLOC()\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
char Pwd[FILENAME_MAX];\
GetCurrentDir(Pwd,sizeof(Pwd));\
str=Pwd;\
  str= str +  "/" + "../../src/memory/" + __FILE__;

 #define LOCAL_HIP_HOST_UNREGISTER()\
  int l=0,count=0;\
  ifstream file;\
  string str,str1;\
char Pwd[FILENAME_MAX];\
GetCurrentDir(Pwd,sizeof(Pwd));\
str=Pwd;\
  str= str +  "/" + "../../src/memory/" + __FILE__;

 #define LOCAL_HIP_STREAM_CREATE()\
  int l=0,count=0;\
 hipStream_t *stream=new hipStream_t;\
delete stream;\
  ifstream file;\
  string str,str1;\
char Pwd[FILENAME_MAX];\
GetCurrentDir(Pwd,sizeof(Pwd));\
str=Pwd;\
  str= str +  "/" + "../../src/memory/" + __FILE__;



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
char Pwd[FILENAME_MAX];\
GetCurrentDir(Pwd,sizeof(Pwd));\
str=Pwd;\
str= str + "/" + "../../src/memory/" + __FILE__;\
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



