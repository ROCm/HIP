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
__global__ void tex1DKernel(float *outputData, hipTextureObject_t textureObject,
                            int width, float offsetX) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  outputData[x] = tex1D<float>(textureObject, normalizedCoords ? (x + offsetX) / width : x + offsetX);
}

template<hipTextureAddressMode addressMode, hipTextureFilterMode filterMode, bool normalizedCoords>
bool runTest(const int width, const float offsetX) {
  printf("%s(addressMode=%d, filterMode=%d, normalizedCoords=%d, width=%d, offsetX=%f)\n", __FUNCTION__,
         addressMode, filterMode, normalizedCoords, width, offsetX);
  bool testResult = true;
  unsigned int size = width * sizeof(float);
  float *hData = (float*) malloc(size);
  memset(hData, 0, size);
  for (int j = 0; j < width; j++) {
    hData[j] = j;
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(
      32, 0, 0, 0, hipChannelFormatKindFloat);
  hipArray *hipArray;
  hipMallocArray(&hipArray, &channelDesc, width);

  HIPCHECK(hipMemcpy2DToArray(hipArray, 0, 0, hData, width * sizeof(float), width * sizeof(float), 1, hipMemcpyHostToDevice));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = hipArray;

  // Specify texture object parameters
  hipTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = addressMode;
  texDesc.filterMode = filterMode;
  texDesc.readMode = hipReadModeElementType;
  texDesc.normalizedCoords = normalizedCoords;

  // Create texture object
  hipTextureObject_t textureObject = 0;
  hipCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL);

  float *dData = NULL;
  hipMalloc((void**) &dData, size);

  dim3 dimBlock(16, 1, 1);
  dim3 dimGrid((width + dimBlock.x - 1)/ dimBlock.x, 1, 1);

  hipLaunchKernelGGL(tex1DKernel<normalizedCoords>, dimGrid, dimBlock, 0, 0, dData,
                     textureObject, width, offsetX);

  hipDeviceSynchronize();

  float *hOutputData = (float*) malloc(size);
  memset(hOutputData, 0, size);
  hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost);

  for (int j = 0; j < width; j++) {
    float expectedValue = getExpectedValue<addressMode, filterMode>(width, offsetX + j, hData);
    if (!hipTextureSamplingVerify<float, filterMode>(hOutputData[j], expectedValue)) {
      printf("mismatched [ %d ]:%f ----%f\n", j, hOutputData[j], expectedValue);
      testResult = false;
      break;
    }
  }

  hipDestroyTextureObject(textureObject);
  hipFree(dData);
  hipFreeArray(hipArray);
  free(hData);
  free(hOutputData);
  printf("%s %s\n", __FUNCTION__, testResult ? "succeeded":"failed");
  return testResult;
}

int main(int argc, char **argv) {
  bool testResult = true;
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModePoint, false>(256, -3);
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModePoint, false>(256, 4);

  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModePoint, false>(256, -8.5);
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModePoint, false>(256, 12.5);

  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModeLinear, false>(256, -3);
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModeLinear, false>(256, 4);

  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModeLinear, false>(256, -8.5);
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModeLinear, false>(256, 12.5);

  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModePoint, true>(256, -3);
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModePoint, true>(256, 4);

  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModePoint, true>(256, -8.5);
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModePoint, true>(256, 12.5);

  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModeLinear, true>(256, -3);
  testResult = testResult && runTest<hipAddressModeClamp, hipFilterModeLinear, true>(256, 4);

  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModeLinear, true>(256, -8.5);
  testResult = testResult && runTest<hipAddressModeBorder, hipFilterModeLinear, true>(256, 12.5);

  if (testResult) {
    passed();
  } else {
    exit (EXIT_FAILURE);
  }
}
