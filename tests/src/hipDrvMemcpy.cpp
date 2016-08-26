#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

#define LEN 1024
#define SIZE LEN<<2

int main(){
  int *A, *B, *C;
  hipDeviceptr Ad, Bd;
  A = new int[LEN];
  B = new int[LEN];
  C = new int[LEN];
  for(int i=0;i<LEN;i++){
    A[i] = i;
  }
  hipMalloc((void**)&Ad, SIZE);
  hipMalloc((void**)&Bd, SIZE);
  hipMemcpyHtoD(Ad, A, SIZE);
  hipMemcpyDtoD(Bd, Ad, SIZE);
  hipMemcpyDtoH(B, Bd, SIZE);
  hipMemcpyHtoH(C, B,SIZE);
  for(int i=0;i<16;i++){
    std::cout<<A[i]<<" "<<C[i]<<std::endl;
  }
}
