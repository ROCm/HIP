/* 
 test : hipCopyHammer_D2D 
 Desc : This test is part of hipHammerTests written to test stress/corner/edge cases.
        host buffer used is un-pinned.  
        hipCopyHammer_D2D does repeated deviceX->deviceY transfers each of size ~500KB and more.
 Author : Phani 
*/
#include <iostream>
#include <hip/hip_runtime.h>
#include <assert.h>
#include <string.h>
using namespace std;

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"

#define HIPCHECK(error) \
{\
    hipError_t localError = error; \
    if (localError != hipSuccess) { \
        printf("%serror: '%s'(%d) from %s at %s:%d%s\n", \
        KRED, hipGetErrorString(localError), localError,\
        #error,__FILE__, __LINE__, KNRM); \
    }\
}


int main(){
    
    int gpus;
    int thisGpu, thatGpu,canAccessPeer;
    char *A_d0, *A_d1,G0[100]="Hello, how are you?",G1[100],*Gvp0,*Gvp1,Out[100];
    char *A_h;
    size_t hSize =  500 *  1024 * 1024 * sizeof(char);
    hipStream_t s,s1; 
    string out;
    thisGpu = 0;
    thatGpu = (thisGpu + 1);

    HIPCHECK(hipGetDeviceCount(&gpus));
    printf ("thisGpu=%d thatGpu=%d (Total no. of gpu =%d)\n", thisGpu, thatGpu,gpus);
    

  //  DeviceToDevice src:A_d0, dst:A_d1

    HIPCHECK(hipSetDevice(thisGpu));
//    HIPCHECK(hipMalloc(&A_d0, hSize));
out=hipGetErrorString(hipMalloc(&Gvp0,100*sizeof(char)));
cout<<"The output of hipMalloc() is:"<<out<<endl;
//    HIPCHECK(hipMemset(A_d0, 0x13, hSize));
out=hipGetErrorString(hipMemcpyHtoD(Gvp0,G0,100*sizeof(char)));
    hipStreamCreate(&s);

    HIPCHECK(hipSetDevice(3));
//    HIPCHECK(hipMalloc(&A_d1, hSize));
out=hipGetErrorString(hipMalloc(&Gvp1,100*sizeof(char)));
cout<<"The output of hipMalloc() is:"<<out<<endl;
//    HIPCHECK(hipMemset(A_d1, 0x0, hSize));
    hipStreamCreate(&s1);
    
    HIPCHECK(hipSetDevice(thisGpu));
   // HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer,thisGpu,thatGpu));
  //  assert(canAccessPeer);
    out=hipGetErrorString(hipDeviceCanAccessPeer(&canAccessPeer,thisGpu,thatGpu));
    cout<<"The output of hipDeviceCanAccessPeer is:"<<out<<endl;
    cout<<"the value of canAccessPeer is:"<<canAccessPeer<<endl;
    //HIPCHECK(hipDeviceEnablePeerAccess(thatGpu,0));
    
    out=hipGetErrorString(hipDeviceEnablePeerAccess(thatGpu,0));
    cout<<"The output of hipDeviceEnablePeerAccess() is:"<<out<<endl;
   
    out=hipGetErrorString(hipDeviceCanAccessPeer(&canAccessPeer,thisGpu,thatGpu));
    cout<<"The output of hipDeviceCanAccessPeer is:"<<out<<endl;
    cout<<"the value of canAccessPeer is:"<<canAccessPeer<<endl;
    //since the test is taking too long with 65536, changed iter to 2048
   for(int iter=0;iter<20;iter++){
    hipMemcpyAsync(Gvp1,Gvp0,100*sizeof(char),hipMemcpyDeviceToDevice,s1);
    printf("Async copied from DeviceToDevice: Gvp0->Gvp1:%d\n",iter);
    hipStreamSynchronize(s);
    hipDeviceSynchronize(); 
   }
out=hipGetErrorString(hipSetDevice(4));
out=hipGetErrorString(hipMemcpyDtoH(Out,Gvp1,100*sizeof(char)));
cout<<"The output of hipMemcpyDtoH() is:"<<out<<endl;
cout<<"The output from the second gpu is:"<<Out<<endl;


/*
    HIPCHECK(hipSetDevice(thatGpu));
    HIPCHECK(hipDeviceCanAccessPeer(&canAccessPeer,thatGpu,thisGpu));
    assert(canAccessPeer);
    HIPCHECK(hipDeviceEnablePeerAccess(thisGpu,0));

   for(int iter=0;iter<2048;iter++){
    hipMemcpyAsync(A_d0,A_d1,hSize,hipMemcpyDeviceToDevice,s1);
    printf("Async copied from DeviceToDevice: A_d1->A_d0:%d\n",iter);
    hipStreamSynchronize(s1);
    hipDeviceSynchronize(); 
   }
    

  //  DeviceToDevice src:A_d0, dst:A_d1
   for(int iter=0;iter<20;iter++){
    hipMemcpyAsync(A_d1,A_d0,hSize,hipMemcpyDeviceToDevice,s);
    printf("Async copied from DeviceToDevice: A_d0->A_d1:%d\n",iter);
    hipStreamSynchronize(s);
    hipDeviceSynchronize(); 

    hipMemcpyAsync(A_d0,A_d1,hSize,hipMemcpyDeviceToDevice,s1);
    printf("Async copied from DeviceToDevice: A_d1->A_d0:%d\n",iter);
    hipStreamSynchronize(s1);
    hipDeviceSynchronize(); 
   }*/
    hipFree(A_d0);
    hipFree(A_d1);
    hipFree(Gvp0);
    hipFree(Gvp1);
    hipDeviceReset();

  return 0;
}
