#include <hip_runtime.h>

__global__ void vadd_hip(hipLaunchParm lp, const float *a, const float *b, float *c, int N)
{
    int idx = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}


int main(int argc, char *argv[])
{
    int sizeElements = 1000000;
    size_t sizeBytes = sizeElements * sizeof(float);

    // Allocate host memory
    float *A_h = (float*)malloc(sizeBytes);
    float *B_h = (float*)malloc(sizeBytes);
    float *C_h = (float*)malloc(sizeBytes);

    // Allocate device memory:
    float *A_d, *B_d, *C_d;
    hipMalloc(&A_d, sizeBytes);
    hipMalloc(&B_d, sizeBytes);
    hipMalloc(&C_d, sizeBytes);

    // Initialize host data
    for (int i=0; i<sizeElements; i++) {
        A_h[i] = 1.618f * i; 
        B_h[i] = 3.142f * i;
    }

    hipMemcpy(A_d, A_h, sizeBytes, hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h, sizeBytes, hipMemcpyHostToDevice);

    // Launch kernel onto default accelerator:
    int blockSize = 256;  // pick arbitrary block size
    int blocks = (sizeElements+blockSize-1)/blockSize; // round up to launch enough blocks
    hipLaunchKernel(vadd_hip, dim3(blocks), dim3(blockSize), 0, 0, A_d, B_d, C_d, sizeElements);

    hipMemcpy(C_h, C_d, sizeBytes, hipMemcpyDeviceToHost);

    for (int i=0; i<sizeElements; i++) {
        float ref= 1.618f * i + 3.142f * i;
        if (C_h[i] != ref) {
            printf ("error:%d computed=%6.2f, reference=%6.2f\n", i, C_h[i], ref);
        }
    };
}
