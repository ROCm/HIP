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

#define SIZE          10
#define EPSILON       0.00001
#define THRESH_HOLD   0.01  // For filter mode

static float getNormalizedValue(const float value,
                                const hipChannelFormatDesc& desc) {
  if ((desc.x == 8) && (desc.f == hipChannelFormatKindSigned))
    return (value / SCHAR_MAX);
  if ((desc.x == 8) && (desc.f == hipChannelFormatKindUnsigned))
    return (value / UCHAR_MAX);
  if ((desc.x == 16) && (desc.f == hipChannelFormatKindSigned))
    return (value / SHRT_MAX);
  if ((desc.x == 16) && (desc.f == hipChannelFormatKindUnsigned))
    return (value / USHRT_MAX);
  return value;
}

texture<char, hipTextureType1D, hipReadModeNormalizedFloat>            texc;
texture<unsigned char, hipTextureType1D, hipReadModeNormalizedFloat>   texuc;

template<typename T>
__global__ void normalizedValTextureTest(unsigned int numElements,
                                         float* pDst) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  unsigned int elementID = hipThreadIdx_x;
  if (elementID >= numElements)
    return;
  float coord = elementID/static_cast<float>(numElements);
  if (std::is_same<T, char>::value)
    pDst[elementID] = tex1D(texc, coord);
  else if (std::is_same<T, unsigned char>::value)
    pDst[elementID] = tex1D(texuc, coord);
#endif
}

static void textureVerifyFilterModePoint(float *hOutputData,
                                         float *expected, int size) {
  for (int i = 0; i < size; i++) {
    if ((hOutputData[i] == expected[i])
        || (i >= 1 && hOutputData[i] == expected[i - 1]) ||  // round down
        (i < (size - 1) && hOutputData[i] == expected[i + 1])) {  // round up
      continue;
    }
      INFO("Mismatch at output[" << i << "]:" << hOutputData[i] <<
           " expected[" << i << "]:" << expected[i]);
    if (i >= 1) {
      INFO(", expected[" << i - 1 << "]:" << expected[i - 1]);
    }
    if (i < (size - 1)) {
      INFO(", expected[" << i + 1 << "]:" << expected[i + 1]);
    }
    REQUIRE(false);
  }
}

static void textureVerifyFilterModeLinear(float *hOutputData,
                                          float *expected,  int size) {
  for (int i = 0; i < size; i++) {
    float mean = (fabs(expected[i]) + fabs(hOutputData[i])) / 2;
    float ratio = fabs(expected[i] - hOutputData[i]) / (mean + EPSILON);
    if (ratio > THRESH_HOLD) {
      INFO("Mismatch found at output[" << i << "]:" << hOutputData[i] <<
           " expected[" << i << "]:" << expected[i] << ", ratio:" << ratio);
      REQUIRE(false);
    }
  }
}

template<hipTextureFilterMode fMode = hipFilterModePoint>
static void textureVerify(float *hOutputData, float *expected, size_t size) {
  if (fMode == hipFilterModePoint) {
    textureVerifyFilterModePoint(hOutputData, expected, size);
  } else if (fMode == hipFilterModeLinear) {
    textureVerifyFilterModeLinear(hOutputData, expected, size);
  }
}

template<typename T, hipTextureFilterMode fMode = hipFilterModePoint>
static void textureTest(texture<T, hipTextureType1D,
                        hipReadModeNormalizedFloat> *tex) {
  hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
  hipArray_t dData;
  HIP_CHECK(hipMallocArray(&dData, &desc, SIZE, 1, hipArrayDefault));

  T hData[] = {65, 66, 67, 68, 69, 70, 71, 72, 73, 74};
  HIP_CHECK(hipMemcpy2DToArray(dData, 0, 0, hData, sizeof(T) * SIZE,
            sizeof(T) * SIZE, 1, hipMemcpyHostToDevice));

  tex->normalized = true;
  tex->channelDesc = desc;
  tex->filterMode = fMode;
  HIP_CHECK(hipBindTextureToArray(tex, dData, &desc));

  float *dOutputData = NULL;
  HIP_CHECK(hipMalloc(&dOutputData, sizeof(float) * SIZE));
  REQUIRE(dOutputData != nullptr);

  hipLaunchKernelGGL(normalizedValTextureTest<T>, dim3(1, 1, 1),
                     dim3(SIZE, 1, 1), 0, 0, SIZE, dOutputData);

  float *hOutputData = new float[SIZE];
  REQUIRE(hOutputData != nullptr);
  HIP_CHECK(hipMemcpy(hOutputData, dOutputData, (sizeof(float) * SIZE),
                     hipMemcpyDeviceToHost));

  float expected[SIZE];
  for (int i = 0; i < SIZE; i++) {
    expected[i] = getNormalizedValue(static_cast<float>(hData[i]), desc);
  }
  textureVerify<fMode>(hOutputData, expected, SIZE);

  HIP_CHECK(hipFreeArray(dData));
  HIP_CHECK(hipFree(dOutputData));
  delete [] hOutputData;
}

template<hipTextureFilterMode fMode = hipFilterModePoint>
static void runTest_hipTextureFilterMode() {
  textureTest<char, fMode>(&texc);
  textureTest<unsigned char, fMode>(&texuc);
}

TEST_CASE("Unit_hipNormalizedFloatValueTex_CheckModes") {
#if HT_AMD
  int imageSupport{};
  HIP_CHECK(hipDeviceGetAttribute(&imageSupport,
                           hipDeviceAttributeImageSupport, 0));
  if (!imageSupport) {
    INFO("Texture is not supported on the device. Test is skipped");
    return;
  }
  hipDeviceProp_t props;
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  INFO("Device :: " << props.name);
  INFO("Arch - AMD GPU :: " << props.gcnArch);
#endif

  SECTION("hipNormalizedFloatValueTexture for hipFilterModePoint") {
    runTest_hipTextureFilterMode<hipFilterModePoint>();
  }
  SECTION("hipNormalizedFloatValueTexture for hipFilterModeLinear") {
    runTest_hipTextureFilterMode<hipFilterModeLinear>();
  }
}
