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
__global__ void tex1DRGBAKernel(float4 *outputData, hipTextureObject_t textureObject,
                            int width, float offsetX) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  outputData[x] = tex1D<float4>(textureObject,
                                normalizedCoords ? (x + offsetX) / width : x + offsetX);
#endif
}

__global__ void tex1DRGBAKernelFetch(float4 *outputData, hipTextureObject_t textureObject, float offsetX) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  outputData[x] = tex1Dfetch<float4>(textureObject, int(x + offsetX));
#endif
}

template<hipTextureAddressMode addressMode, hipTextureFilterMode filterMode, hipResourceType resType,
                  bool normalizedCoords, bool sRGB = false>
static void runTest(const int width, const float offsetX = 0) {
  constexpr float uCharMax = UCHAR_MAX;
  unsigned int size = width * sizeof(uchar4);
  uchar4 *hData = (uchar4*) malloc(size);
  memset(hData, 0, size);
  for (int j = 0; j < width; j++) {
    hData[j].x = static_cast<unsigned char>(j);
    hData[j].y = static_cast<unsigned char>(j + 10);
    hData[j].z = static_cast<unsigned char>(j + 20);
    hData[j].w = static_cast<unsigned char>(j + 30);
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<uchar4>();
  uchar4 *hipBuff = nullptr;
  hipArray *hipArray = nullptr;
  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));

  if (resType == hipResourceTypeArray) {
    HIP_CHECK(hipMallocArray(&hipArray, &channelDesc, width));
    HIP_CHECK(
        hipMemcpy2DToArray(hipArray, 0, 0, hData, size, size, 1,
                           hipMemcpyHostToDevice));
    resDesc.resType = hipResourceTypeArray;  // Will call tex1D in kernel
    resDesc.res.array.array = hipArray;
  } else if (resType == hipResourceTypeLinear) {
    if (normalizedCoords || filterMode == hipFilterModeLinear
        || addressMode == hipAddressModeWrap
        || addressMode == hipAddressModeMirror) {
      free(hData);
      FAIL("One or more unexpected parameters for hipResourceTypeLinear");
    }
    HIP_CHECK(hipMalloc((void** ) &hipBuff, size));
    HIP_CHECK(hipMemcpy(hipBuff, hData, size, hipMemcpyHostToDevice));
    resDesc.resType = hipResourceTypeLinear; // Will call tex1Dfetch in kernel
    resDesc.res.linear.devPtr = hipBuff;
    resDesc.res.linear.sizeInBytes = size;
    resDesc.res.linear.desc = channelDesc;
  } else FAIL("Unexpected resource type " << resType);

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
  auto ret = hipCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL);
#if HT_AMD
  if(ret == hipErrorInvalidValue && resType == hipResourceTypeLinear) {
    free(hData);
    HIP_CHECK(hipFree(hipBuff));
    HipTest::HIP_SKIP_TEST("sRGB is not supported for hipResourceTypeLinear type on AMD devices");
    return;
  }
#endif
  HIP_CHECK(ret);

  float4 *dData = nullptr;
  size = width * sizeof(float4);
  HIP_CHECK(hipMalloc((void**) &dData, size));

  dim3 dimBlock(16, 1, 1);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, 1, 1);

  if (resType == hipResourceTypeArray) {
    hipLaunchKernelGGL(tex1DRGBAKernel<normalizedCoords>, dimGrid, dimBlock,
                       0, 0, dData, textureObject, width, offsetX);
    HIP_CHECK(hipGetLastError()); 
  } else {
    hipLaunchKernelGGL(tex1DRGBAKernelFetch, dimGrid, dimBlock,
                       0, 0, dData, textureObject, offsetX);
    HIP_CHECK(hipGetLastError()); 
  }

  HIP_CHECK(hipDeviceSynchronize());
  size = width * sizeof(float4);
  float4 *hInputData = (float4*) malloc(size);  // CPU expected values
  float4 *hOutputData = (float4*) malloc(size); // GPU output values
  memset(hInputData, 0, size);
  memset(hOutputData, 0, size);

  for (int j = 0; j < width; j++) {
    hInputData[j].x = hData[j].x / uCharMax;
    hInputData[j].y = hData[j].y / uCharMax;
    hInputData[j].z = hData[j].z / uCharMax;
    hInputData[j].w = hData[j].w / uCharMax;
  }

  HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));

  bool result = true;

  for (int j = 0; j < width; j++) {
    float4 cpuExpected =
        getExpectedValue<float4, addressMode, filterMode, sRGB>(width, offsetX + j, hInputData);
    float4 gpuOutput = hOutputData[j];
    if (sRGB) {
      // CTS will map to sRGP before comparison, so we do so
      cpuExpected = hipSRGBMap(cpuExpected);
      gpuOutput = hipSRGBMap(gpuOutput);
    }
    // Convert from [0, 1] back to [0, 255]
    gpuOutput *= uCharMax;
    cpuExpected *= uCharMax;
    if (!hipTextureSamplingVerify<float4, filterMode, sRGB>(gpuOutput,
                                                            cpuExpected)) {
      WARN(
          "Mismatch at (" << offsetX + j << ") GPU output : "
          << gpuOutput.x << ", " << gpuOutput.y << ", " << gpuOutput.z << ", " << gpuOutput.w << ", " <<
          " CPU expected: "
          << cpuExpected.x << ", " << cpuExpected.y << ", " << cpuExpected.z << ", " << cpuExpected.w << "\n");
      result = false;
      goto line1;
    }
  }

line1:
  HIP_CHECK(hipDestroyTextureObject(textureObject));
  HIP_CHECK(hipFree(dData));
  if (hipArray) HIP_CHECK(hipFreeArray(hipArray));
  if (hipBuff) HIP_CHECK(hipFree(hipBuff));
  free(hData);
  free(hOutputData);
  free(hInputData);
  REQUIRE(result);
}

TEST_CASE("Unit_hipTextureObj1DCheckRGBAModes - array") {
  CHECK_IMAGE_SUPPORT

  SECTION("RGBA 1D hipAddressModeClamp, hipFilterModePoint, hipResourceTypeArray, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, hipResourceTypeArray, false>(255, -3.9);
    runTest<hipAddressModeClamp, hipFilterModePoint, hipResourceTypeArray, false>(255, 4.4);
  }

  SECTION("RGBA 1D hipAddressModeBorder, hipFilterModePoint, hipResourceTypeArray, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, hipResourceTypeArray, false>(255, -8.5);
    runTest<hipAddressModeBorder, hipFilterModePoint, hipResourceTypeArray, false>(255, 12.5);
  }

  SECTION("RGBA 1D hipAddressModeClamp, hipFilterModeLinear, hipResourceTypeArray, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, hipResourceTypeArray, false>(255, -0.41);
    runTest<hipAddressModeClamp, hipFilterModeLinear, hipResourceTypeArray, false>(255, 4);
  }

#if HT_AMD
  // nvidia RTX2070 has problem in this mode
  SECTION("RGBA 1D hipAddressModeBorder, hipFilterModeLinear, hipResourceTypeArray, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, hipResourceTypeArray, false>(255, 0);
    runTest<hipAddressModeBorder, hipFilterModeLinear, hipResourceTypeArray, false>(255, 12.1);
  }
#endif

  SECTION("RGBA 1D hipAddressModeClamp, hipFilterModePoint, hipResourceTypeArray, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, hipResourceTypeArray, true>(255, -3.1);
    runTest<hipAddressModeClamp, hipFilterModePoint, hipResourceTypeArray, true>(255, 4.2);
  }

  SECTION("RGBA 1D hipAddressModeBorder, hipFilterModePoint, hipResourceTypeArray, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, hipResourceTypeArray, true>(255, -8.15);
    runTest<hipAddressModeBorder, hipFilterModePoint, hipResourceTypeArray, true>(255, 12.35);
  }

  SECTION("RGBA 1D hipAddressModeClamp, hipFilterModeLinear, hipResourceTypeArray, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, hipResourceTypeArray, true>(255, -3.1);
    runTest<hipAddressModeClamp, hipFilterModeLinear, hipResourceTypeArray, true>(255, 4.2);
  }

#if HT_AMD
  // nvidia RTX2070 has problem in this mode
  SECTION("RGBA 1D hipAddressModeBorder, hipFilterModeLinear, hipResourceTypeArray, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, hipResourceTypeArray, true>(255, 0);
    runTest<hipAddressModeBorder, hipFilterModeLinear, hipResourceTypeArray, true>(255, -6.7);
  }
#endif
}


TEST_CASE("Unit_hipTextureObj1DCheckSRGBAModes - array") {
  CHECK_IMAGE_SUPPORT

  SECTION("SRGBA 1D hipAddressModeClamp, hipFilterModePoint, hipResourceTypeArray, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, hipResourceTypeArray, false, true>(255, -3.9);
    runTest<hipAddressModeClamp, hipFilterModePoint, hipResourceTypeArray, false, true>(255, 4.4);
  }

  SECTION("SRGBA 1D hipAddressModeBorder, hipFilterModePoint, hipResourceTypeArray, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, hipResourceTypeArray, false, true>(255, -8.5);
    runTest<hipAddressModeBorder, hipFilterModePoint, hipResourceTypeArray, false, true>(255, 12.5);
  }

  SECTION("SRGBA 1D hipAddressModeClamp, hipFilterModeLinear, hipResourceTypeArray, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, hipResourceTypeArray, false, true>(255, -0.4);
    runTest<hipAddressModeClamp, hipFilterModeLinear, hipResourceTypeArray, false, true>(255, 4);
  }

#if HT_AMD
  // nvidia RTX2070 has problem in this mode
  SECTION("SRGBA 1D hipAddressModeBorder, hipFilterModeLinear, hipResourceTypeArray, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, hipResourceTypeArray, false, true>(255, 0);
    runTest<hipAddressModeBorder, hipFilterModeLinear, hipResourceTypeArray, false, true>(255, 12.5);
  }
#endif

  SECTION("SRGBA 1D hipAddressModeClamp, hipFilterModePoint, hipResourceTypeArray, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, hipResourceTypeArray, true, true>(255, -1.3);
    runTest<hipAddressModeClamp, hipFilterModePoint, hipResourceTypeArray, true, true>(255, 4.1);
  }

  SECTION("SRGBA 1D hipAddressModeBorder, hipFilterModePoint, hipResourceTypeArray, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, hipResourceTypeArray, true, true>(255, -8.5);
    runTest<hipAddressModeBorder, hipFilterModePoint, hipResourceTypeArray, true, true>(255, 12.5);
  }

  SECTION("SRGBA 1D hipAddressModeClamp, hipFilterModeLinear, hipResourceTypeArray, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, hipResourceTypeArray, true, true>(255, -3);
    runTest<hipAddressModeClamp, hipFilterModeLinear, hipResourceTypeArray, true, true>(255, 4);
  }
#if HT_AMD
  // nvidia RTX2070 has problem in this mode
  SECTION("SRGBA 1D hipAddressModeBorder, hipFilterModeLinear, hipResourceTypeArray, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, hipResourceTypeArray, true, true>(255, 0);
    runTest<hipAddressModeBorder, hipFilterModeLinear, hipResourceTypeArray, true, true>(255, 12.35);
  }
#endif
}

TEST_CASE("Unit_hipTextureObj1DCheckRGBAModes - buffer") {
  CHECK_IMAGE_SUPPORT

  SECTION("RGBA 1D hipAddressModeClamp, hipFilterModePoint, hipResourceTypeLinear, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, hipResourceTypeLinear, false, false>(255);
 }

  SECTION("RGBA 1D hipAddressModeBorder, hipFilterModePoint, hipResourceTypeLinear, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, hipResourceTypeLinear, false, false>(255);
  }
}

TEST_CASE("Unit_hipTextureObj1DCheckSRGBAModes - buffer") {
  CHECK_IMAGE_SUPPORT

  SECTION("SRGBA 1D hipAddressModeClamp, hipFilterModePoint, hipResourceTypeLinear, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, hipResourceTypeLinear, false, true>(255);
  }

  SECTION("SRGBA 1D hipAddressModeBorder, hipFilterModePoint, hipResourceTypeLinear, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, hipResourceTypeLinear, false, true>(255);
  }
}
