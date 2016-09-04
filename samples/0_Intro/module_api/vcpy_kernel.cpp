#include<hip_runtime.h>

#ifdef __HIP_PLATFORM_NVCC__
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

EXTERN_C __global__ void hello_world(hipLaunchParm lp, float *a, float *b)
{
    int tx = hipThreadIdx_x;
    b[tx] = a[tx];
}

