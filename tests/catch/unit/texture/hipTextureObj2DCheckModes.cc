/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_texture_helper.hh>

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
static void runTest(const int width, const int height, const float offsetX, const float offsetY) {
  //printf("%s(addressMode=%d, filterMode=%d, normalizedCoords=%d, width=%d, height=%d, offsetX=%f, offsetY=%f)\n",
  //     __FUNCTION__, addressMode, filterMode, normalizedCoords, width, height, offsetX, offsetY);
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
  HIP_CHECK(hipMallocArray(&hipArray, &channelDesc, width, height));

  HIP_CHECK(hipMemcpy2DToArray(hipArray, 0, 0, hData, width * sizeof(float), width * sizeof(float), height, hipMemcpyHostToDevice));

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
  HIP_CHECK(hipCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL));

  float *dData = nullptr;
  HIP_CHECK(hipMalloc((void**) &dData, size));

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y -1)/ dimBlock.y, 1);

  hipLaunchKernelGGL(tex2DKernel<normalizedCoords>, dimGrid, dimBlock, 0, 0, dData,
                     textureObject, width, height, offsetX, offsetY);

  HIP_CHECK(hipDeviceSynchronize());

  float *hOutputData = (float*) malloc(size);
  memset(hOutputData, 0, size);
  HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));

  bool result = true;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      float expectedValue = getExpectedValue<float, addressMode, filterMode>(width, height,
                                                    offsetX + j, offsetY + i, hData);
      if (!hipTextureSamplingVerify<float, filterMode>(hOutputData[index], expectedValue)) {
        INFO("Mismatch at (" << offsetX + j << ", " << offsetY + i << "):" <<
             hOutputData[index] << " expected:" << expectedValue);
        result = false;
        goto line1;
      }
    }
  }
line1:
  HIP_CHECK(hipDestroyTextureObject(textureObject));
  HIP_CHECK(hipFree(dData));
  HIP_CHECK(hipFreeArray(hipArray));
  free(hData);
  free(hOutputData);
  REQUIRE(result);
}

TEST_CASE("Unit_hipTextureObj2DCheckModes") {
  CHECK_IMAGE_SUPPORT

#ifdef _WIN32
  INFO("Unit_hipTextureObj2DCheckModes skipped on Windows");
  return;
#endif
  SECTION("hipAddressModeClamp, hipFilterModePoint, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, false>(256, 256, -3.9, 6.1);
    runTest<hipAddressModeClamp, hipFilterModePoint, false>(256, 256, 4.4, -7.0);
  }

  SECTION("hipAddressModeBorder, hipFilterModePoint, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, false>(256, 256, -8.5, 2.9);
    runTest<hipAddressModeBorder, hipFilterModePoint, false>(256, 256, 12.5, 6.7);
  }

  SECTION("hipAddressModeClamp, hipFilterModeLinear, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, false>(256, 256, -0.4, -0.4);
    runTest<hipAddressModeClamp, hipFilterModeLinear, false>(256, 256, 4, 14.6);
  }

  SECTION("hipAddressModeBorder, hipFilterModeLinear, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, false>(256, 256, -0.4, 0.4);
    runTest<hipAddressModeBorder, hipFilterModeLinear, false>(256, 256, 12.5, 23.7);
  }

  SECTION("hipAddressModeClamp, hipFilterModePoint, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, true>(256, 256, -3, 8.9);
    runTest<hipAddressModeClamp, hipFilterModePoint, true>(256, 256, 4, -0.1);
  }

  SECTION("hipAddressModeBorder, hipFilterModePoint, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, true>(256, 256, -8.5, 15.9);
    runTest<hipAddressModeBorder, hipFilterModePoint, true>(256, 256, 12.5, -17.9);
  }

  SECTION("hipAddressModeClamp, hipFilterModeLinear, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, true>(256, 256, -3, 5.8);
    runTest<hipAddressModeClamp, hipFilterModeLinear, true>(256, 256, 4, 9.1);
  }

  SECTION("hipAddressModeBorder, hipFilterModeLinear, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, true>(256, 256, -8.5, 6.6);
    runTest<hipAddressModeBorder, hipFilterModeLinear, true>(256, 256, 12.5, 0.01);
  }
}
