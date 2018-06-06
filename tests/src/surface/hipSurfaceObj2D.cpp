/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */
#include <stdio.h>

#include <hip/hip_runtime.h>
#include "test_common.h"

__global__ void tex2DKernel(hipSurfaceObject_t surfaceObject, hipSurfaceObject_t outputSurfObj,
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float data;
    surf2Dread(&data, surfaceObject, x * 4, y, hipBoundaryModeZero);
    surf2Dwrite(data, outputSurfObj, x * 4, y, hipBoundaryModeZero);
}

int runTest(int argc, char** argv);

int main(int argc, char** argv) {
    int testResult = runTest(argc, argv);

    if (testResult) {
        passed();
    } else {
        exit(EXIT_FAILURE);
    }
}

int runTest(int argc, char** argv) {
    int testResult = 1;
    unsigned int width = 256;
    unsigned int height = 256;
    unsigned int size = width * height * sizeof(float);
    float* hData = (float*)malloc(size);
    memset(hData, 0, size);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            hData[i * width + j] = i * width + j;
        }
    }
    printf("hData: ");
    for (int i = 0; i < 64; i++) {
        printf("%f  ", hData[i]);
    }
    printf("\n");

    hipChannelFormatDesc channelDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
    hipArray *hipArray, *hipOutArray;
    hipMallocArray(&hipArray, &channelDesc, width, height);

    hipMemcpyToArray(hipArray, 0, 0, hData, size, hipMemcpyHostToDevice);

    struct hipResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = hipResourceTypeArray;
    resDesc.res.array.array = hipArray;
    // Create surface object
    hipSurfaceObject_t surfaceObject = 0;
    hipCreateSurfaceObject(&surfaceObject, &resDesc);

    hipMallocArray(&hipOutArray, &channelDesc, width, height);
    struct hipResourceDesc resOutDesc;
    memset(&resOutDesc, 0, sizeof(resOutDesc));
    resOutDesc.resType = hipResourceTypeArray;
    resOutDesc.res.array.array = hipOutArray;
    hipSurfaceObject_t outSurfaceObject = 0;
    hipCreateSurfaceObject(&outSurfaceObject, &resOutDesc);

    float* dData = NULL;
    hipMalloc((void**)&dData, size);

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    hipLaunchKernelGGL(tex2DKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, surfaceObject,
                       outSurfaceObject, width, height);

    hipDeviceSynchronize();

    float* hOutputData = (float*)malloc(size);
    memset(hOutputData, 0, size);
    hipMemcpyFromArray(hOutputData, hipOutArray, 0, 0, size, hipMemcpyDeviceToHost);

    printf("dData: ");
    for (int i = 0; i < 64; i++) {
        printf("%f  ", hOutputData[i]);
    }
    printf("\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (hData[i * width + j] != hOutputData[i * width + j]) {
                printf("Difference [ %d %d ]:%f ----%f\n", i, j, hData[i * width + j],
                       hOutputData[i * width + j]);
                testResult = 0;
                break;
            }
        }
    }
    hipDestroySurfaceObject(surfaceObject);
    hipDestroySurfaceObject(outSurfaceObject);
    hipFree(dData);
    hipFreeArray(hipArray);
    hipFreeArray(hipOutArray);
    return testResult;
}
