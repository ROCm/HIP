/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <hip/hip_runtime.h>
#include "test_common.h"
#include "hipTextureHelper.hpp"


template<bool normalizedCoords>
__global__ void tex3DKernel(float *outputData, hipTextureObject_t textureObject,
                            int width, int height, int depth, float offsetX,
                            float offsetY, float offsetZ) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  outputData[z * width * depth + y * width + x] = tex3D<float>(textureObject,
                        normalizedCoords ? (x + offsetX) / width : x + offsetX,
                        normalizedCoords ? (y + offsetY) / height : y + offsetY,
                        normalizedCoords ? (z + offsetZ) / depth : z + offsetZ);
}

template<hipTextureAddressMode addressMode, hipTextureFilterMode filterMode, bool normalizedCoords>
bool runTest(const int width, const int height, const int depth, const float offsetX, const float offsetY, const float offsetZ) {
  printf("%s(addressMode=%d, filterMode=%d, normalizedCoords=%d, width=%d, height=%d, depth=%d, offsetX=%f, offsetY=%f, offsetZ=%f)\n",
      __FUNCTION__, addressMode, filterMode, normalizedCoords, width, height,
      depth, offsetX, offsetY, offsetZ);
  bool testResult = true;
  unsigned int size = width * height * depth * sizeof(float);
  float *hData = (float*) malloc(size);
  memset(hData, 0, size);

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        int index = i * width * depth + j * width + k;
        hData[index] = index;
      }
    }
  }

  // Allocate array and copy image data
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<float>();
  hipArray *arr;

  HIPCHECK(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width, height, depth), hipArrayDefault));
  hipMemcpy3DParms myparms = {0};
  myparms.srcPos = make_hipPos(0,0,0);
  myparms.dstPos = make_hipPos(0,0,0);
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(float), width, height);
  myparms.dstArray = arr;
  myparms.extent = make_hipExtent(width, height, depth);
  myparms.kind = hipMemcpyHostToDevice;

  HIPCHECK(hipMemcpy3D(&myparms));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = arr;

  // Specify texture object parameters
  hipTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = addressMode;
  texDesc.addressMode[1] = addressMode;
  texDesc.addressMode[2] = addressMode;
  texDesc.filterMode = filterMode;
  texDesc.readMode = hipReadModeElementType;
  texDesc.normalizedCoords = normalizedCoords;

  // Create texture object
  hipTextureObject_t textureObject = 0;
  hipCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL);

  float *dData = NULL;
  hipMalloc((void**) &dData, size);
  hipMemset(dData, 0, size);
  dim3 dimBlock(8, 8, 8); // 512 threads
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y -1)/ dimBlock.y,
               (depth + dimBlock.z - 1) / dimBlock.z);

  hipLaunchKernelGGL(tex3DKernel<normalizedCoords>, dimGrid, dimBlock, 0, 0, dData,
                     textureObject, width, height, depth, offsetX, offsetY, offsetZ);

  hipDeviceSynchronize();

  float *hOutputData = (float*) malloc(size);
  memset(hOutputData, 0, size);
  hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost);
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        int index = i * width * depth + j * width + k;
        float expectedValue = getExpectedValue<addressMode, filterMode>(
            width, height, depth, offsetX + k, offsetY + j, offsetZ + i, hData);

        if (!hipTextureSamplingVerify<float, filterMode>(hOutputData[index], expectedValue)) {
          printf("mismatched [ %d %d %d]:%f ----%f\n", k, j, i, hOutputData[index], expectedValue);
          testResult = false;
          goto line1;
        }
      }
    }
  }
line1:
  hipDestroyTextureObject(textureObject);
  hipFree(dData);
  hipFreeArray(arr);
  free(hData);
  free(hOutputData);
  printf("%s %s\n", __FUNCTION__, testResult ? "succeeded":"failed");
  return testResult;
}

int main(int argc, char **argv) {
  bool testResult = true;

  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModePoint, false>(256, 256, 256, -3.9, 6.1, 9.5);
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModePoint, false>(256, 256, 256, 4.4, -7.0, 5.3);

  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModePoint, false>(256, 256, 256, -8.5, 2.9, 5.8);
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModePoint, false>(256, 256, 256, 12.5, 6.7, 11.4);

  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModeLinear, false>(256, 256, 256, -0.4, -0.4, -0.4);
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModeLinear, false>(256, 256, 256, 4, 14.6, -0.3);

  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModeLinear, false>(256, 256, 256, 6.9, 7.4, 0.4);
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModeLinear, false>(256, 256, 256, 12.5, 23.7, 0.34);

  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModePoint, true>(256, 256, 256, -3, 8.9, -4);
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModePoint, true>(256, 256, 256, 4, -0.1, 8.2);

  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModePoint, true>(256, 256, 256, -8.5, 15.9, 0.1);
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModePoint, true>(256, 256, 256, 12.5, -17.9, -0.35);

  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModeLinear, true>(256, 256, 256, -3, 5.8, 0.89);
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModeLinear, true>(256, 256, 256, 4, 9.1, 2.08);

  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModeLinear, true>(256, 256, 256, -8.5, 6.6, 3.67);
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModeLinear, true>(256, 256, 256, 12.5, 0.01, -9.9);

  if (testResult) {
    passed();
  } else {
    exit (EXIT_FAILURE);
  }
}
