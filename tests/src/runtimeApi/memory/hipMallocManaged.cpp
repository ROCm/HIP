#include <hip/hip_runtime.h>
#include <math.h>
#include "test_common.h"

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
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
    int numElements = N;
    bool testResult = true;
    float *A, *B;

    hipMallocManaged(&A, numElements*sizeof(float));
    hipMallocManaged(&B, numElements*sizeof(float));

    for (int i = 0; i < numElements; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(blockSize, 1, 1);
    hipLaunchKernelGGL(add, dimGrid, dimBlock, 0, 0, numElements, A, B);

    hipDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < numElements; i++)
        maxError = fmax(maxError, fabs(B[i]-3.0f));

    hipFree(A);
    hipFree(B);
    if(maxError == 0.0f)
        passed();
    failed("Output Mismatch\n");
}
