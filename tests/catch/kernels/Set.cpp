#include <kernels.hh>

__global__ void Set(int* Ad, int val) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    Ad[tx] = val;
}