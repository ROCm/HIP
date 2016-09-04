#include "hip/hip_runtime.h"

extern "C" __global__ void hello_world(hipLaunchParm lp, float *a, float *b)
{
    int tx = hipThreadIdx_x;
    b[tx] = a[tx];
}

