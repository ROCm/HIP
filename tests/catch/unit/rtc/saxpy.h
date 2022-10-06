#include "test_header1.h"
#include "test_header2.h"

extern "C"
__global__
void saxpy(real a, realptr x, realptr y, realptr out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a * x[tid] + y[tid];
    }
}
