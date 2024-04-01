#include <iostream>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#define LEN 1024
#define SIZE LEN << 2

int main() {
    int *A, *B;
    hipDeviceptr_t Ad, Bd;
    A = new int[LEN];
    B = new int[LEN];

    for (int i = 0; i < LEN; i++) {
        A[i] = i;
    }

    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);

    hipMemcpyHtoD(Ad, A, SIZE);
    hipMemcpyDtoD(Bd, Ad, SIZE);
    hipMemcpyDtoH(B, Bd, SIZE);

    for (int i = 0; i < 16; i++) {
        std::cout << A[i] << " " << B[i] << std::endl;
    }

    int *Ah, *Bh;
    hipHostMalloc(&Ah, SIZE, 0);
    hipHostMalloc(&Bh, SIZE, 0);
    memcpy(Ah, A, SIZE);
    hipStream_t stream;
    hipStreamCreate(&stream);

    hipMemcpyHtoDAsync(Ad, Ah, SIZE, stream);
    hipStreamSynchronize(stream);
    hipMemcpyDtoDAsync(Bd, Ad, SIZE, stream);
    hipStreamSynchronize(stream);
    hipMemcpyDtoHAsync(Bh, Bd, SIZE, stream);
    hipStreamSynchronize(stream);

    std::cout << Ah[10] << " " << Bh[10] << std::endl;
}
