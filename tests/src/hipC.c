#include"hip_runtime.h"
#include<stdio.h>

#define ITER 1<<20
#define SIZE 1024*1024*sizeof(int)

__global__ void Iter(hipLaunchParm lp, int *Ad){
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if(tx == 0){
        for(int i=0;i<ITER;i++){
            Ad[tx] += 1;
        }
    }
}

int main(){
    int A=0, *Ad;
    hipMalloc((void**)&Ad, SIZE);
    hipMemcpy(Ad, &A, SIZE, hipMemcpyHostToDevice);
    dim3 dimGrid, dimBlock;
    dimGrid.x = 1, dimGrid.y =1, dimGrid.z = 1;
    dimBlock.x = 1, dimBlock.y = 1, dimGrid.z = 1;
    hipLaunchKernel(HIP_KERNEL_NAME(Iter), dimGrid, dimBlock, 0, 0, Ad);
    hipMemcpy(&A, Ad, SIZE, hipMemcpyDeviceToHost);
}
