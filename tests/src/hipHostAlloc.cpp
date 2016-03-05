#include<iostream>
#include"test_common.h"

__global__ void Add(hipLaunchParm lp, float *Ad, float *Bd, float *Cd){
int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
Cd[tx] = Ad[tx] + Bd[tx];
}

int main(){
float *A, *B, *C;
float *Ad, *Bd, *Cd;

hipDeviceProp_t prop;
int device;
hipGetDevice(&device);
hipDeviceGetProperties(&prop, device);
if(prop.canMapHostMemory != 1){
std::cout<<"Exiting.."<<std::endl;
}
}
