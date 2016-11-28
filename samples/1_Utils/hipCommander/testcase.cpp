#include <hip/hip_runtime.h>

static const int BLOCKSIZEX=32;
static const int BLOCKSIZEY=16;

__global__ void fails(hipLaunchParm lp, float* pErrorI)
{
    if(pErrorI!=0)
    {
        pErrorI[0]=1;
    }
}

int main()
{
    dim3 blocks(1,1);
    dim3 threads(BLOCKSIZEX,BLOCKSIZEY);
    float error;

    hipLaunchKernel(HIP_KERNEL_NAME(fails), blocks, threads, 0, 0, &error);
}
