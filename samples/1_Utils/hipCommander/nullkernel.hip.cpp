#include "hip/hip_runtime.h"

extern "C" __global__ void NullKernel(float* Ad) {
    if (Ad) {
        Ad[0] = 42;
    }
}
