#include <hip/hip_runtime.h>


    
extern "C" __global__ void
memcpyIntKernel(hipLaunchParm lp, int *dst, const int * src, size_t numElements)
{
    int gid = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int stride = hipBlockDim_x * hipGridDim_x ;
    for (size_t i= gid; i< numElements; i+=stride){
       dst[i] = src[i];
    }
};
