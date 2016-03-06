#include"test_common.h"
#include<malloc.h>

#define LEN 1024*1024
#define SIZE LEN*sizeof(float)

__global__ void Add(hipLaunchParm lp, float *Ad, float *Bd, float *Cd){
int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
Cd[tx] = Ad[tx] + Bd[tx];
}

int main(){
float *A, *B, *C, *D;
float *Ad, *Bd, *Cd, *Dd;
unsigned int FlagA, FlagB, FlagC;
FlagA = hipHostAllocWriteCombined | hipHostAllocMapped;
FlagB = hipHostAllocWriteCombined | hipHostAllocMapped;
FlagC = hipHostAllocMapped;
hipDeviceProp_t prop;
int device;
HIPCHECK(hipGetDevice(&device));
HIPCHECK(hipGetDeviceProperties(&prop, device));
if(prop.canMapHostMemory != 1){
std::cout<<"Exiting..."<<std::endl;
}
HIPCHECK(hipHostAlloc((void**)&A, SIZE, hipHostAllocWriteCombined | hipHostAllocMapped));
HIPCHECK(hipHostAlloc((void**)&B, SIZE, hipHostAllocWriteCombined | hipHostAllocMapped));
HIPCHECK(hipHostAlloc((void**)&C, SIZE, hipHostAllocMapped));

HIPCHECK(hipHostAlloc((void**)&D, SIZE, hipHostAllocDefault));

unsigned int flagA, flagB, flagC;

HIPCHECK(hipHostGetDevicePointer((void**)&Ad, A, 0));
HIPCHECK(hipHostGetDevicePointer((void**)&Bd, B, 0));
HIPCHECK(hipHostGetDevicePointer((void**)&Cd, C, 0));
HIPCHECK(hipHostGetDevicePointer((void**)&Dd, D, 0));
HIPCHECK(hipHostGetFlags(&flagA, A));
HIPCHECK(hipHostGetFlags(&flagB, B));
HIPCHECK(hipHostGetFlags(&flagC, C));

for(int i=0;i<LEN;i++){
A[i] = 1.0f;
B[i] = 2.0f;
}

dim3 dimGrid(LEN/512,1,1);
dim3 dimBlock(512,1,1);

hipLaunchKernel(HIP_KERNEL_NAME(Add), dimGrid, dimBlock, 0, 0, Ad, Bd, Cd);

HIPCHECK(hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost));
HIPASSERT(C[10] == 3.0f);
HIPASSERT(flagA == FlagA);
HIPASSERT(flagB == FlagB);
HIPASSERT(flagC == FlagC);
passed();

}
