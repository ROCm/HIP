#include<hip_runtime.h>

__global__ void hello_world(hipLaunchParm lp, float *a, float *b)
{
    int tx = hipThreadIdx_x;
    b[tx] = a[tx];
}

