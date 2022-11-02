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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>


class TextureObjectTestWrapper {
 private:
  float* mHostData;
  bool mOmmitDestroy;

 public:
  hipTextureObject_t mTextureObject = 0;
  hipResourceDesc mResDesc;
  hipTextureDesc mTexDesc;
  hipChannelFormatDesc mChannelDesc;
  hipResourceViewDesc mResViewDesc;
  hipArray* mArray;
  size_t mSize; /* size in bytes*/
  int mWidth;   /* width in elements */

  TextureObjectTestWrapper(bool useResourceViewDescriptor, bool ommitDestroy = false)
      : mOmmitDestroy(ommitDestroy), mWidth(128) {
    int i;
    mSize = mWidth * sizeof(float);

    mHostData = (float*)malloc(mSize);
    memset(mHostData, 0, mSize);

    for (i = 0; i < mWidth; i++) {
      mHostData[i] = i;
    }

    mChannelDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
    hipMallocArray(&mArray, &mChannelDesc, mWidth);

    HIP_CHECK(hipMemcpy2DToArray(mArray, 0, 0, mHostData, mSize, mSize, 1, hipMemcpyHostToDevice));

    memset(&mResDesc, 0, sizeof(mResDesc));
    mResDesc.resType = hipResourceTypeArray;
    mResDesc.res.array.array = mArray;

    memset(&mTexDesc, 0, sizeof(mTexDesc));
    mTexDesc.addressMode[0] = hipAddressModeClamp;
    mTexDesc.filterMode = hipFilterModePoint;
    mTexDesc.readMode = hipReadModeElementType;
    mTexDesc.normalizedCoords = false;

    memset(&mResViewDesc, 0, sizeof(mResViewDesc));

    if (useResourceViewDescriptor) {
#if HT_AMD
      mResViewDesc.format = hipResViewFormatFloat1;
      mResViewDesc.width = mSize;
#else
      std::cout << "Resource View Descriptors are not supported on NVIDIA currently" << std::endl;
      useResourceViewDescriptor = false;
#endif
    }


    HIP_CHECK(hipCreateTextureObject(&mTextureObject, &mResDesc, &mTexDesc,
                                     useResourceViewDescriptor ? &mResViewDesc : nullptr));
  }

  ~TextureObjectTestWrapper() {
    if (!mOmmitDestroy) {
      HIP_CHECK(hipDestroyTextureObject(mTextureObject));
    }
    HIP_CHECK(hipFreeArray(mArray));
    free(mHostData);
  }
};

/* hipGetTextureObjectResourceDesc tests */

TEST_CASE("Unit_hipGetTextureObjectResourceDesc_positive") {
  CHECK_IMAGE_SUPPORT

  TextureObjectTestWrapper texObjWrapper(false);

  hipResourceDesc checkDesc;
  memset(&checkDesc, 0, sizeof(checkDesc));

  HIP_CHECK(hipGetTextureObjectResourceDesc(&checkDesc, texObjWrapper.mTextureObject));

  REQUIRE(checkDesc.resType == texObjWrapper.mResDesc.resType);
  REQUIRE(checkDesc.res.array.array == texObjWrapper.mResDesc.res.array.array);
}


TEST_CASE("Unit_hipGetTextureObjectResourceDesc_negative") {
  CHECK_IMAGE_SUPPORT

  TextureObjectTestWrapper texObjWrapper(false);

  hipResourceDesc checkDesc;
  memset(&checkDesc, 0, sizeof(checkDesc));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipGetTextureObjectResourceDesc(nullptr, texObjWrapper.mTextureObject),
                    hipErrorInvalidValue);
  }

  SECTION("texture is invalid") {
    HIP_CHECK_ERROR(hipGetTextureObjectResourceDesc(&checkDesc, static_cast<hipTextureObject_t>(0)),
                    hipErrorInvalidValue);
  }
}

/* hipGetTextureObjectResourceViewDesc tests */


TEST_CASE("Unit_hipGetTextureObjectResourceViewDesc_positive") {
  CHECK_IMAGE_SUPPORT
#if HT_AMD
  TextureObjectTestWrapper texObjWrapper(true);

  hipResourceViewDesc checkDesc;
  memset(&checkDesc, 0, sizeof(checkDesc));

  HIP_CHECK(hipGetTextureObjectResourceViewDesc(&checkDesc, texObjWrapper.mTextureObject));

  REQUIRE(checkDesc.format == texObjWrapper.mResViewDesc.format);
  REQUIRE(checkDesc.width == texObjWrapper.mResViewDesc.width);

#else
  HipTest::HIP_SKIP_TEST("Skipping on NVIDIA platform");
#endif
}


TEST_CASE("Unit_hipGetTextureObjectResourceViewDesc_negative") {
  CHECK_IMAGE_SUPPORT
#if HT_AMD
  TextureObjectTestWrapper texObjWrapper(true);

  hipResourceViewDesc checkDesc;
  memset(&checkDesc, 0, sizeof(checkDesc));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipGetTextureObjectResourceViewDesc(nullptr, texObjWrapper.mTextureObject),
                    hipErrorInvalidValue);
  }

  SECTION("texture is invalid") {
    HIP_CHECK_ERROR(
        hipGetTextureObjectResourceViewDesc(&checkDesc, static_cast<hipTextureObject_t>(0)),
        hipErrorInvalidValue);
  }

#else
  HipTest::HIP_SKIP_TEST("Skipping on NVIDIA platform");
#endif
}


/* hipGetTextureObjectTextureDesc tests */


TEST_CASE("Unit_hipGetTextureObjectTextureDesc_positive") {
  CHECK_IMAGE_SUPPORT
#if HT_AMD
  TextureObjectTestWrapper texObjWrapper(false);

  hipTextureDesc checkDesc;
  memset(&checkDesc, 0, sizeof(checkDesc));

  HIP_CHECK(hipGetTextureObjectTextureDesc(&checkDesc, texObjWrapper.mTextureObject));

  REQUIRE(checkDesc.addressMode[0] == texObjWrapper.mTexDesc.addressMode[0]);
  REQUIRE(checkDesc.filterMode == texObjWrapper.mTexDesc.filterMode);
  REQUIRE(checkDesc.readMode == texObjWrapper.mTexDesc.readMode);
  REQUIRE(checkDesc.normalizedCoords == texObjWrapper.mTexDesc.normalizedCoords);

#else
  HipTest::HIP_SKIP_TEST("Skipping on NVIDIA platform");
#endif
}


TEST_CASE("Unit_hipGetTextureObjectTextureDesc_negative") {
  CHECK_IMAGE_SUPPORT
#if HT_AMD
  TextureObjectTestWrapper texObjWrapper(false);

  hipTextureDesc checkDesc;
  memset(&checkDesc, 0, sizeof(checkDesc));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipGetTextureObjectTextureDesc(nullptr, texObjWrapper.mTextureObject),
                    hipErrorInvalidValue);
  }

  SECTION("texture is invalid") {
    HIP_CHECK_ERROR(hipGetTextureObjectTextureDesc(&checkDesc, static_cast<hipTextureObject_t>(0)),
                    hipErrorInvalidValue);
  }

#else
  HipTest::HIP_SKIP_TEST("Skipping on NVIDIA platform");
#endif
}

/* hipDestroyTextureObject test */

TEST_CASE("Unit_hipDestroyTextureObject_positive") {
  CHECK_IMAGE_SUPPORT

  TextureObjectTestWrapper texObjWrapper(false, true);
  REQUIRE(hipDestroyTextureObject(texObjWrapper.mTextureObject) == hipSuccess);
}
