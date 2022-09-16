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
__global__ void tex2DRGBAKernel(float4 *outputData, hipTextureObject_t textureObject,
                            int width, int height, float offsetX,
                            float offsetY) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D<float4>(textureObject,
                                           normalizedCoords ? (x + offsetX) / width : x + offsetX,
                                           normalizedCoords ? (y + offsetY) / height : y + offsetY);
#endif
}

template<hipTextureAddressMode addressMode, hipTextureFilterMode filterMode, bool normalizedCoords, bool sRGB = false>
static void runTest(const int width, const int height, const float offsetX, const float offsetY) {
  //printf("%s(addressMode=%d, filterMode=%d, normalizedCoords=%d, width=%d, height=%d, offsetX=%f, offsetY=%f)\n",
  //     __FUNCTION__, addressMode, filterMode, normalizedCoords, width, height, offsetX, offsetY);
  constexpr float uCharMax = UCHAR_MAX;
  unsigned int size = width * height * sizeof(uchar4);
  uchar4 *hData = (uchar4*) malloc(size);
  memset(hData, 0, size);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      hData[index].x = static_cast<unsigned char>(j);
      hData[index].y = static_cast<unsigned char>(i);
      hData[index].z = static_cast<unsigned char>(index);
      hData[index].w = static_cast<unsigned char>(i + j);
    }
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<uchar4>();
  hipArray *hipArray;
  HIP_CHECK(hipMallocArray(&hipArray, &channelDesc, width, height));

  HIP_CHECK(hipMemcpy2DToArray(hipArray, 0, 0, hData, width * sizeof(uchar4), width * sizeof(uchar4), height, hipMemcpyHostToDevice));

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
  texDesc.readMode = hipReadModeNormalizedFloat;
  texDesc.normalizedCoords = normalizedCoords;
  texDesc.sRGB = sRGB ? 1 : 0;

  // Create texture object
  hipTextureObject_t textureObject = 0;
  HIP_CHECK(hipCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL));

  float4 *dData = nullptr;
  size = width * height * sizeof(float4);
  HIP_CHECK(hipMalloc((void**) &dData, size));

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y -1)/ dimBlock.y, 1);

  hipLaunchKernelGGL(tex2DRGBAKernel<normalizedCoords>, dimGrid, dimBlock, 0, 0, dData,
                     textureObject, width, height, offsetX, offsetY);

  HIP_CHECK(hipDeviceSynchronize());

  float4 *hInputData = (float4*) malloc(size);  // CPU expected values
  float4 *hOutputData = (float4*) malloc(size); // GPU output values
  memset(hInputData, 0, size);
  memset(hOutputData, 0, size);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      hInputData[index].x = hData[index].x / uCharMax;
      hInputData[index].y = hData[index].y / uCharMax;
      hInputData[index].z = hData[index].z / uCharMax;
      hInputData[index].w = hData[index].w / uCharMax;
    }
  }
  HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));

  bool result = true;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
        //printf("(%d, %d): hInputData=(%f, %f, %f, %f), hOutputData=(%f, %f, %f, %f)\n", i, j,
        //     hInputData[index].x, hInputData[index].y, hInputData[index].z, hInputData[index].w,
        //     hOutputData[index].x, hOutputData[index].y, hOutputData[index].z, hOutputData[index].w);

      float4 cpuExpected = getExpectedValue<float4, addressMode, filterMode, sRGB>(width, height,
                                                    offsetX + j, offsetY + i, hInputData);
      float4 gpuOutput = hOutputData[index];
      if (sRGB) {
        // CTS will map to sRGP before comparison, so we do so
        cpuExpected = hipSRGBMap(cpuExpected);
        gpuOutput = hipSRGBMap(gpuOutput);
      }
      // Convert from [0, 1] back to [0, 255]
      gpuOutput *= uCharMax;
      cpuExpected *= uCharMax;
      if (!hipTextureSamplingVerify<float4, filterMode, sRGB>(gpuOutput, cpuExpected)) {
        WARN("Mismatch at (" << offsetX + j << ", " << offsetY + i << ") GPU output : " <<
             gpuOutput.x << ", " <<
             gpuOutput.y << ", " <<
             gpuOutput.z << ", " <<
             gpuOutput.w << ", " <<
             " CPU expected: " <<
             cpuExpected.x << ", " <<
             cpuExpected.y << ", " <<
             cpuExpected.z << ", " <<
             cpuExpected.w << "\n");
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
  free(hInputData);
  REQUIRE(result);
}

TEST_CASE("Unit_hipTextureObj2DCheckRGBAModes") {
  CHECK_IMAGE_SUPPORT

  SECTION("RGBA 2D hipAddressModeClamp, hipFilterModePoint, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, false>(256, 256, -3.9, 6.1);
    runTest<hipAddressModeClamp, hipFilterModePoint, false>(256, 256, 4.4, -7.0);
  }

  SECTION("RGBA 2D hipAddressModeBorder, hipFilterModePoint, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, false>(256, 256, -8.5, 2.9);
    runTest<hipAddressModeBorder, hipFilterModePoint, false>(256, 256, 12.5, 6.7);
  }

  SECTION("RGBA 2D hipAddressModeClamp, hipFilterModeLinear, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, false>(256, 256, -0.4, -0.4);
    runTest<hipAddressModeClamp, hipFilterModeLinear, false>(256, 256, 4, 14.6);
  }

  SECTION("RGBA 2D hipAddressModeBorder, hipFilterModeLinear, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, false>(256, 256, -0.4, 0.4);
    runTest<hipAddressModeBorder, hipFilterModeLinear, false>(256, 256, 12.5, 23.7);
  }

  SECTION("RGBA 2D hipAddressModeClamp, hipFilterModePoint, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, true>(256, 256, -3, 8.9);
    runTest<hipAddressModeClamp, hipFilterModePoint, true>(256, 256, 4, -0.1);
  }

  SECTION("RGBA 2D hipAddressModeBorder, hipFilterModePoint, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, true>(256, 256, -8.5, 15.9);
    runTest<hipAddressModeBorder, hipFilterModePoint, true>(256, 256, 12.5, -17.9);
  }

  SECTION("RGBA 2D hipAddressModeClamp, hipFilterModeLinear, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, true>(256, 256, -3, 5.8);
    runTest<hipAddressModeClamp, hipFilterModeLinear, true>(256, 256, 4, 9.1);
  }

  SECTION("RGBA 2D hipAddressModeBorder, hipFilterModeLinear, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, true>(256, 256, -8.5, 6.6);
    runTest<hipAddressModeBorder, hipFilterModeLinear, true>(256, 256, 12.5, 0.01);
  }
}


TEST_CASE("Unit_hipTextureObj2DCheckSRGBAModes") {
  CHECK_IMAGE_SUPPORT

  SECTION("SRGBA 2D hipAddressModeClamp, hipFilterModePoint, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, false, true>(256, 256, -3.9, 6.1);
    runTest<hipAddressModeClamp, hipFilterModePoint, false, true>(256, 256, 4.4, -7.0);
  }

  SECTION("SRGBA 2D hipAddressModeBorder, hipFilterModePoint, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, false, true>(256, 256, -8.5, 2.9);
    runTest<hipAddressModeBorder, hipFilterModePoint, false, true>(256, 256, 12.5, 6.7);
  }

  SECTION("SRGBA 2D hipAddressModeClamp, hipFilterModeLinear, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, false, true>(256, 256, -0.4, -0.4);
    runTest<hipAddressModeClamp, hipFilterModeLinear, false, true>(256, 256, 4, 14.6);
  }

  SECTION("SRGBA 2D hipAddressModeBorder, hipFilterModeLinear, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, false, true>(256, 256, -0.4, 0.4);
    runTest<hipAddressModeBorder, hipFilterModeLinear, false, true>(256, 256, 12.5, 23.7);
  }

  SECTION("SRGBA 2D hipAddressModeClamp, hipFilterModePoint, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, true, true>(256, 256, -3, 8.9);
    runTest<hipAddressModeClamp, hipFilterModePoint, true, true>(256, 256, 4, -0.1);
  }

  SECTION("SRGBA 2D hipAddressModeBorder, hipFilterModePoint, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, true, true>(256, 256, -8.5, 15.9);
    runTest<hipAddressModeBorder, hipFilterModePoint, true, true>(256, 256, 12.5, -17.9);
  }

  SECTION("SRGBA 2D hipAddressModeClamp, hipFilterModeLinear, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, true, true>(256, 256, -3, 5.8);
    runTest<hipAddressModeClamp, hipFilterModeLinear, true, true>(256, 256, 4, 9.1);
  }

  SECTION("SRGBA 2D hipAddressModeBorder, hipFilterModeLinear, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, true, true>(256, 256, -8.5, 6.6);
    runTest<hipAddressModeBorder, hipFilterModeLinear, true, true>(256, 256, 12.5, 0.01);
  }
}
