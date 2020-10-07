#include <hip/hip_runtime.h>
#include <math.h>
#include "test_common.h"

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp CLANG_OPTIONS -g -O0
 * TEST: %t -N 256M
 * HIT_END
 */

__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, true);

    printf("info: set device to %d\n", p_gpuDevice);
    HIPCHECK(hipSetDevice(p_gpuDevice));

    int numElements = (N < (64 * 1024 * 1024)) ? 64 * 1024 * 1024 : N;
    bool testResult = true;
    float *A, *B;

    HIPCHECK(hipMallocManaged(&A, numElements*sizeof(float)));
    HIPCHECK(hipMallocManaged(&B, numElements*sizeof(float)));

    for (int i = 0; i < numElements; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    hipDevice_t device = hipCpuDeviceId; 

    HIPCHECK(hipMemAdvise(A, numElements*sizeof(float), hipMemAdviseSetReadMostly, device));
    HIPCHECK(hipMemPrefetchAsync(A, numElements*sizeof(float), 0));
    HIPCHECK(hipMemPrefetchAsync(B, numElements*sizeof(float), 0));
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemRangeGetAttribute(&device, sizeof(device), hipMemRangeAttributeLastPrefetchLocation, A, numElements*sizeof(float)));
    if (device != p_gpuDevice) {
      printf("hipMemRangeGetAttribute error, device = %d!\n", device);
    }
    uint32_t read_only = 0xf;
    HIPCHECK(hipMemRangeGetAttribute(&read_only, sizeof(read_only), hipMemRangeAttributeReadMostly, A, numElements*sizeof(float)));
    if (read_only != 1) {
      printf("hipMemRangeGetAttribute error, read_only = %d!\n", read_only);
    }

    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(blockSize, 1, 1);
    hipEvent_t event0, event1;
    HIPCHECK(hipEventCreate(&event0));
    HIPCHECK(hipEventCreate(&event1));
    HIPCHECK(hipEventRecord(event0, 0));
    hipLaunchKernelGGL(add, dimGrid, dimBlock, 0, 0, numElements, A, B);
    HIPCHECK(hipEventRecord(event1, 0));
    HIPCHECK(hipDeviceSynchronize());
    float time = 0.0f;
    HIPCHECK(hipEventElapsedTime(&time, event0, event1));
    printf("Time %.3f ms\n", time);

    float maxError = 0.0f;
    HIPCHECK(hipMemPrefetchAsync(B, numElements*sizeof(float), hipCpuDeviceId));
    HIPCHECK(hipDeviceSynchronize());
    device = p_gpuDevice;
    HIPCHECK(hipMemRangeGetAttribute(&device, sizeof(device), hipMemRangeAttributeLastPrefetchLocation, A, numElements*sizeof(float)));
    if (device != hipCpuDeviceId) {
      printf("hipMemRangeGetAttribute error (CPU device is expected), device = %d!\n", device);
    }
    for (int i = 0; i < numElements; i++)
        maxError = fmax(maxError, fabs(B[i]-3.0f));

    HIPCHECK(hipFree(A));
    HIPCHECK(hipFree(B));
    if(maxError == 0.0f)
        passed();
    failed("Output Mismatch\n");
}
