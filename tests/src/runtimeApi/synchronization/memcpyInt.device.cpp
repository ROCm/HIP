#include <hip/hip_runtime.h>


extern "C" __global__ void memcpyIntKernel(hipLaunchParm lp, int* dst, const int* src,
                                           size_t numElements) {
    int gid = (blockIdx.x * blockDim.x + threadIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (size_t i = gid; i < numElements; i += stride) {
        dst[i] = src[i];
    }
};
