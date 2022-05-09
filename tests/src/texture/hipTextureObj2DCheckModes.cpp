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
__global__ void tex2DKernel(float *outputData, hipTextureObject_t textureObject,
                            int width, int height, float offsetX,
                            float offsetY) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D<float>(textureObject,
                                           normalizedCoords ? (x + offsetX) / width : x + offsetX,
                                           normalizedCoords ? (y + offsetY) / height : y + offsetY);
#endif
}

template<hipTextureAddressMode addressMode, hipTextureFilterMode filterMode, bool normalizedCoords>
bool runTest(const int width, const int height, const float offsetX, const float offsetY) {
  printf("%s(addressMode=%d, filterMode=%d, normalizedCoords=%d, width=%d, height=%d, offsetX=%f, offsetY=%f)\n",
         __FUNCTION__, addressMode, filterMode, normalizedCoords, width, height, offsetX, offsetY);
  bool testResult = true;
  unsigned int size = width * height * sizeof(float);
  float *hData = (float*) malloc(size);
  memset(hData, 0, size);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      hData[index] = index;
    }
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(
      32, 0, 0, 0, hipChannelFormatKindFloat);
  hipArray *hipArray;
  hipMallocArray(&hipArray, &channelDesc, width, height);

  HIPCHECK(hipMemcpy2DToArray(hipArray, 0, 0, hData, width * sizeof(float), width * sizeof(float), height, hipMemcpyHostToDevice));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = hipArray;

  // Specify texture object parameters
  hipTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = addressMode;
  texDesc.addressMode[1] = addressMode;
  texDesc.filterMode = filterMode;
  texDesc.readMode = hipReadModeElementType;
  texDesc.normalizedCoords = normalizedCoords;

  // Create texture object
  hipTextureObject_t textureObject = 0;
  hipCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL);

  float *dData = NULL;
  hipMalloc((void**) &dData, size);

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y -1)/ dimBlock.y, 1);

  hipLaunchKernelGGL(tex2DKernel<normalizedCoords>, dimGrid, dimBlock, 0, 0, dData,
                     textureObject, width, height, offsetX, offsetY);

  hipDeviceSynchronize();

  float *hOutputData = (float*) malloc(size);
  memset(hOutputData, 0, size);
  hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      float expectedValue = getExpectedValue<addressMode, filterMode>(width, height,
                                                    offsetX + j, offsetY + i, hData);
      if (!hipTextureSamplingVerify<float, filterMode>(hOutputData[index], expectedValue)) {
        printf("mismatched [ %d %d ]:%f ----%f\n", j, i, hOutputData[index], expectedValue);
        testResult = false;
        goto line1;
      }
    }
  }
line1:
  hipDestroyTextureObject(textureObject);
  hipFree(dData);
  hipFreeArray(hipArray);
  free(hData);
  free(hOutputData);
  printf("%s %s\n", __FUNCTION__, testResult ? "succeeded":"failed");
  return testResult;
}

int main(int argc, char **argv) {
  checkImageSupport();

  bool testResult = true;

  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModePoint, false>(256, 256, -3.9, 6.1);
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModePoint, false>(256, 256, 4.4, -7.0);

  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModePoint, false>(256, 256, -8.5, 2.9);
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModePoint, false>(256, 256, 12.5, 6.7);

  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModeLinear, false>(256, 256, -0.4, -0.4);
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModeLinear, false>(256, 256, 4, 14.6);

  // The following two cases have quite big deviation on Cpu and Gpu in 2D, so comment them out temporarily.
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModeLinear, false>(256, 256, -0.4, 0.4);
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModeLinear, false>(256, 256, 12.5, 23.7);

  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModePoint, true>(256, 256, -3, 8.9);
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModePoint, true>(256, 256, 4, -0.1);

  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModePoint, true>(256, 256, -8.5, 15.9);
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModePoint, true>(256, 256, 12.5, -17.9);

  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModeLinear, true>(256, 256, -3, 5.8);
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModeLinear, true>(256, 256, 4, 9.1);

  // The following two cases have quite big deviation on Cpu and Gpu in 2D, so comment them out temporarily.
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModeLinear, true>(256, 256, -8.5, 6.6);
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModeLinear, true>(256, 256, 12.5, 0.01);

  if (testResult) {
    passed();
  } else {
    exit (EXIT_FAILURE);
  }
}
