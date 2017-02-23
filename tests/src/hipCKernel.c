#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

__global__ void Kernel(hipLaunchParm lp, float *Ad){
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    Ad[tx] += Ad[tx-1];
}

int main(){
    dim3 dimBlock;
    dim3 dimGrid;
    dimGrid.x = 1;
    dimGrid.y = 1;
    dimGrid.z = 1;
    dimBlock.x = 1;
    dimBlock.y = 1;
    dimBlock.z = 1;
    float *A;
    hipLaunchKernel(HIP_KERNEL_NAME(Kernel), dimGrid, dimBlock, 0, 0, A);
}
