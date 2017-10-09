/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <hip/hip_runtime.h>
#include "test_common.h"

bool testResult = true;

__global__ void tex2DKernel(float* outputData,
                             hipTextureObject_t textureObject,
                             int width,
                             int height)
{
    int x = hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y*hipBlockDim_y + hipThreadIdx_y;
    outputData[y*width + x] = tex2D<float>(textureObject, x, y);
}

void runTest(int argc, char **argv);

int main(int argc, char **argv)
{
    runTest(argc, argv);

    if(testResult) {
        passed();
    } else {
        exit(EXIT_FAILURE);
    }

}

void runTest(int argc, char **argv)
{
    unsigned int width = 256;
    unsigned int height = 256;
    unsigned int size = width * height * sizeof(float);
    float* hData = (float*) malloc(size);
    memset(hData, 0, size);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            hData[i*width+j] = i*width+j;
        }
    }
    printf("hData: ");
    for (int i = 0; i < 64; i++) {
        printf("%f  ", hData[i]);
    }
    printf("\n");

    hipChannelFormatDesc channelDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
    hipArray *hipArray;
    hipMallocArray(&hipArray, &channelDesc, width, height);

    hipMemcpyToArray(hipArray, 0, 0, hData, size, hipMemcpyHostToDevice);

    struct hipResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = hipResourceTypeArray;
	resDesc.res.array.array = hipArray;

	// Specify texture object parameters
	struct hipTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = hipAddressModeWrap;
	texDesc.addressMode[1] = hipAddressModeWrap;
	texDesc.filterMode = hipFilterModePoint;
	texDesc.readMode = hipReadModeElementType;
	texDesc.normalizedCoords = 0;

	// Create texture object
	hipTextureObject_t textureObject = 0;
    hipCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL);

    float* dData = NULL;
    hipMalloc((void **) &dData, size);

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    hipLaunchKernelGGL(tex2DKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, dData, textureObject, width, height);

    hipDeviceSynchronize();

    float *hOutputData = (float *) malloc(size);
    memset(hOutputData, 0,  size);
    hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost);

    printf("dData: ");
    for (int i = 0; i < 64; i++) {
        printf("%f  ", hOutputData[i]);
    }
    printf("\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (hData[i*width+j] != hOutputData[i*width+j]) {
                printf("Difference [ %d %d ]:%f ----%f\n",i, j, hData[i*width+j] , hOutputData[i*width+j]);
                testResult = false;
                break;
            }
        }
    }
    hipDestroyTextureObject(textureObject);
    hipFree(dData);
    hipFreeArray(hipArray);
}
