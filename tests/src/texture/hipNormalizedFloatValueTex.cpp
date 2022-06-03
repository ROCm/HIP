/*
Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * // Test hipFilterModePoint
 * TEST: %t --textureFilterMode 0
 * // Test hipFilterModeLinear
 * TEST: %t --textureFilterMode 1
 * HIT_END
 */

#include "test_common.h"
#include <math.h>
#define SIZE 10
#include "hipTextureHelper.hpp"

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

texture<char, hipTextureType1D, hipReadModeNormalizedFloat> texc;

texture<unsigned char, hipTextureType1D, hipReadModeNormalizedFloat> texuc;

texture<short, hipTextureType1D, hipReadModeNormalizedFloat> texs;

texture<unsigned short, hipTextureType1D, hipReadModeNormalizedFloat> texus;


template<typename T>
__global__ void normalizedValTextureTest(unsigned int numElements, float* pDst)
{
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
    unsigned int elementID = hipThreadIdx_x;
    if(elementID >= numElements)
        return;
    float coord =(float) elementID/numElements;
    if(std::is_same<T, char>::value)
        pDst[elementID] = tex1D(texc, coord);
    else if(std::is_same<T, unsigned char>::value)
        pDst[elementID] = tex1D(texuc, coord);
    else if(std::is_same<T, short>::value)
        pDst[elementID] = tex1D(texs, coord);
    else if(std::is_same<T, unsigned short>::value)
        pDst[elementID] = tex1D(texus, coord);
#endif
}

bool textureVerifyFilterModePoint(float *hOutputData, float *expected, size_t size) {
    bool testResult = true;
    for (int i = 0; i < size; i++) {
      if ((hOutputData[i] == expected[i])
          || (i >= 1 && hOutputData[i] == expected[i - 1]) ||  // round down
          (i < (size - 1) && hOutputData[i] == expected[i + 1]))  // round up
          {
        continue;
      }
      printf("mismatch at output[%d]:%f expected[%d]:%f", i, hOutputData[i], i,
             expected[i]);
      if (i >= 1) {
        printf(", expected[%d]:%f", i - 1, expected[i - 1]);
      }
      if (i < (size - 1)) {
        printf(", expected[%d]:%f", i + 1, expected[i + 1]);
      }
      printf("\n");
      testResult = false;
      break;
    }
    return testResult;
}

bool textureVerifyFilterModeLinear(float *hOutputData, float *expected,  size_t size) {
    bool testResult = true;
    for (int i = 0; i < size; i++) {
      float mean = (fabs(expected[i]) + fabs(hOutputData[i])) / 2;
      float ratio = fabs(expected[i] - hOutputData[i]) / (mean + HIP_SAMPLING_VERIFY_EPSILON);
      if (ratio > HIP_SAMPLING_VERIFY_RELATIVE_THRESHOLD) {
        printf("mismatch at output[%d]:%f expected[%d]:%f, ratio:%f\n", i,
               hOutputData[i], i, expected[i], ratio);
        testResult = false;
        break;
      }
    }
    return testResult;
}

template<hipTextureFilterMode fMode = hipFilterModePoint>
bool textureVerify(float *hOutputData, float *expected, size_t size) {
    bool testResult = true;
    if (fMode == hipFilterModePoint) {
      testResult = textureVerifyFilterModePoint(hOutputData, expected, size);
    } else if (fMode == hipFilterModeLinear) {
      testResult = textureVerifyFilterModeLinear(hOutputData, expected, size);
    }
    return testResult;
}

template<typename T, hipTextureFilterMode fMode = hipFilterModePoint>
bool textureTest(texture<T, hipTextureType1D, hipReadModeNormalizedFloat> *tex)
{
    hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
    hipArray_t dData;
    HIPCHECK(hipMallocArray(&dData, &desc, SIZE));

    T hData[] = {65, 66, 67, 68, 69, 70, 71, 72, 73, 74};
    HIPCHECK(hipMemcpy2DToArray(dData, 0, 0, hData, sizeof(T)*SIZE, sizeof(T)*SIZE, 1, hipMemcpyHostToDevice));

    tex->addressMode[0] = hipAddressModeClamp;
    tex->normalized = true;
    tex->channelDesc = desc;
    tex->filterMode = fMode;
    HIPCHECK(hipBindTextureToArray(tex, dData, &desc));

    float *dOutputData = NULL;
    HIPCHECK(hipMalloc((void **) &dOutputData, sizeof(float)*SIZE));

    hipLaunchKernelGGL(normalizedValTextureTest<T>, dim3(1,1,1), dim3(SIZE,1,1), 0, 0, SIZE, dOutputData);

    float *hOutputData = new float[SIZE];
    HIPCHECK(hipMemcpy(hOutputData, dOutputData, (sizeof(float)*SIZE), hipMemcpyDeviceToHost));

    float expected[SIZE];
    for(int i = 0; i < SIZE; i++) {
      expected[i] = getNormalizedValue(float(hData[i]), desc);
    }
    bool testResult = textureVerify<fMode>(hOutputData, expected, SIZE);

    HIPCHECK(hipFreeArray(dData));
    HIPCHECK(hipFree(dOutputData));
    delete [] hOutputData;
    return testResult;
}

template<hipTextureFilterMode fMode = hipFilterModePoint>
bool runTest() {
    bool status = true;
    status &= textureTest<char, fMode>(&texc);
    status &= textureTest<unsigned char, fMode>(&texuc);
    status &= textureTest<short, fMode>(&texs);
    status &= textureTest<unsigned short, fMode>(&texus);
    return status;
}

int main(int argc, char** argv)
{
    HipTest::parseStandardArguments(argc, argv, true);
    checkImageSupport();

    int device = p_gpuDevice;
    bool status = false;
    HIPCHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    HIPCHECK(hipGetDeviceProperties(&props, device));
    std::cout << "Device :: " << props.name << std::endl;
    #ifdef __HIP_PLATFORM_AMD__
    std::cout << "Arch - AMD GPU :: " << props.gcnArch << std::endl;
    #endif

    if(textureFilterMode == 0) {
      printf("Test hipFilterModePoint\n");
      status = runTest<hipFilterModePoint>();
    } else if(textureFilterMode == 1) {
      printf("Test hipFilterModeLinear\n");
      printf("THRESH_HOLD:%f, EPSILON:%f\n", HIP_SAMPLING_VERIFY_RELATIVE_THRESHOLD,
             HIP_SAMPLING_VERIFY_EPSILON);
      status = runTest<hipFilterModeLinear>();
    } else {
      printf("Wrong argument!\n");
      printf("hipNormalizedFloatValueTex --textureFilterMode 0 for hipFilterModePoint\n");
      printf("hipNormalizedFloatValueTex --textureFilterMode 1 for hipFilterModeLinear\n");
    }

    if(status){
        passed();
    }
    else{
        failed("checks failed!");
    }
}
