#include"hip/hip_runtime.h"


extern "C" __global__ void mataddK(int* A, int* B, int* C, int N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;
        C[ROW * N + COL] = A[ROW*N + COL] + B[ROW*N + COL];
}
