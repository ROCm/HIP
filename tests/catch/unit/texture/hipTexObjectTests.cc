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


class TexObjectTestWrapper {
 private:
  float* mHostData;
  bool mOmmitDestroy;

 public:
  hipTextureObject_t mTextureObject = 0;
  HIP_RESOURCE_DESC mResDesc;
  HIP_TEXTURE_DESC mTexDesc;
  HIP_RESOURCE_VIEW_DESC mResViewDesc;
  HIP_ARRAY_DESCRIPTOR mArrayDesc;
  hiparray mArray;
  size_t mSize; /* size in bytes*/
  int mWidth;   /* width in elements */

  TexObjectTestWrapper(bool useResourceViewDescriptor, bool ommitDestroy = false)
      : mOmmitDestroy(ommitDestroy), mWidth(128) {
    int i;
    mSize = mWidth * sizeof(float);

    mHostData = (float*)malloc(mSize);
    memset(mHostData, 0, mSize);

    for (i = 0; i < mWidth; i++) {
      mHostData[i] = i;
    }

    memset(&mArrayDesc, 0, sizeof(mArrayDesc));
    mArrayDesc.Format = HIP_AD_FORMAT_FLOAT;
    mArrayDesc.NumChannels = 1;
    mArrayDesc.Width = mWidth;
    mArrayDesc.Height = 0;

    HIP_CHECK(hipArrayCreate(&mArray, &mArrayDesc));
    HIP_CHECK(hipMemcpyHtoA(reinterpret_cast<hipArray*>(mArray), 0, mHostData, mSize));

    memset(&mResDesc, 0, sizeof(mResDesc));
    mResDesc.resType = HIP_RESOURCE_TYPE_ARRAY;
    mResDesc.res.array.hArray = mArray;
    mResDesc.flags = 0;

    memset(&mTexDesc, 0, sizeof(mTexDesc));
    mTexDesc.filterMode = HIP_TR_FILTER_MODE_POINT;
    mTexDesc.flags = 0;

    memset(&mResViewDesc, 0, sizeof(mResViewDesc));


    if (useResourceViewDescriptor) {
#if HT_AMD
      mResViewDesc.format = HIP_RES_VIEW_FORMAT_FLOAT_1X32;
      mResViewDesc.width = mSize;
#else
      std::cout << "Resource View Descriptors are not supported on NVIDIA currently" << std::endl;
      useResourceViewDescriptor = false;
#endif
    }


    HIP_CHECK(hipTexObjectCreate(&mTextureObject, &mResDesc, &mTexDesc,
                                 useResourceViewDescriptor ? &mResViewDesc : nullptr));
  }

  ~TexObjectTestWrapper() {
    if (!mOmmitDestroy) {
      HIP_CHECK(hipTexObjectDestroy(mTextureObject));
    }
    HIP_CHECK(hipArrayDestroy(mArray));
    free(mHostData);
  }
};

/* hipTexObjectGetResourceDesc tests */

TEST_CASE("Unit_hipGetTexObjectResourceDesc_positive") {
  CHECK_IMAGE_SUPPORT

  TexObjectTestWrapper texObjWrapper(false);

  HIP_RESOURCE_DESC checkDesc;
  memset(&checkDesc, 0, sizeof(checkDesc));

  HIP_CHECK(hipTexObjectGetResourceDesc(&checkDesc, texObjWrapper.mTextureObject));

  REQUIRE(checkDesc.resType == texObjWrapper.mResDesc.resType);
  REQUIRE(checkDesc.res.array.hArray == texObjWrapper.mResDesc.res.array.hArray);
}


TEST_CASE("Unit_hipGetTexObjectResourceDesc_negative") {
  CHECK_IMAGE_SUPPORT

  TexObjectTestWrapper texObjWrapper(false);

  HIP_RESOURCE_DESC checkDesc;
  memset(&checkDesc, 0, sizeof(checkDesc));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipTexObjectGetResourceDesc(nullptr, texObjWrapper.mTextureObject),
                    hipErrorInvalidValue);
  }

  SECTION("texture is invalid") {
    HIP_CHECK_ERROR(hipTexObjectGetResourceDesc(&checkDesc, static_cast<hipTextureObject_t>(0)),
                    hipErrorInvalidValue);
  }
}


/* hipTexObjectGetResourceViewDesc tests */

TEST_CASE("Unit_hipGetTexObjectResourceViewDesc_positive") {
  CHECK_IMAGE_SUPPORT
#if HT_AMD
  TexObjectTestWrapper texObjWrapper(true);

  HIP_RESOURCE_VIEW_DESC checkDesc;
  memset(&checkDesc, 0, sizeof(checkDesc));

  HIP_CHECK(hipTexObjectGetResourceViewDesc(&checkDesc, texObjWrapper.mTextureObject));

  REQUIRE(checkDesc.format == texObjWrapper.mResViewDesc.format);
  REQUIRE(checkDesc.width == texObjWrapper.mResViewDesc.width);

#else
  HipTest::HIP_SKIP_TEST("Skipping on NVIDIA platform");
#endif
}


TEST_CASE("Unit_hipGetTexObjectResourceViewDesc_negative") {
  CHECK_IMAGE_SUPPORT
#if HT_AMD
  TexObjectTestWrapper texObjWrapper(true);

  HIP_RESOURCE_VIEW_DESC checkDesc;
  memset(&checkDesc, 0, sizeof(checkDesc));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipTexObjectGetResourceViewDesc(nullptr, texObjWrapper.mTextureObject),
                    hipErrorInvalidValue);
  }

  SECTION("texture is invalid") {
    HIP_CHECK_ERROR(hipTexObjectGetResourceViewDesc(&checkDesc, static_cast<hipTextureObject_t>(0)),
                    hipErrorInvalidValue);
  }

#else
  HipTest::HIP_SKIP_TEST("Skipping on NVIDIA platform");
#endif
}

/* hipTexObjectGetTextureDesc tests */


TEST_CASE("Unit_hipGetTexObjectTextureDesc_positive") {
  CHECK_IMAGE_SUPPORT

  TexObjectTestWrapper texObjWrapper(false);

  HIP_TEXTURE_DESC checkDesc;
  memset(&checkDesc, 0, sizeof(checkDesc));

  HIP_CHECK(hipTexObjectGetTextureDesc(&checkDesc, texObjWrapper.mTextureObject));

  REQUIRE(checkDesc.filterMode == texObjWrapper.mTexDesc.filterMode);
  REQUIRE(checkDesc.flags == texObjWrapper.mTexDesc.flags);
}


TEST_CASE("Unit_hipGetTexObjectTextureDesc_negative") {
  CHECK_IMAGE_SUPPORT

  TexObjectTestWrapper texObjWrapper(false);

  HIP_TEXTURE_DESC checkDesc;
  memset(&checkDesc, 0, sizeof(checkDesc));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipTexObjectGetTextureDesc(nullptr, texObjWrapper.mTextureObject),
                    hipErrorInvalidValue);
  }

  SECTION("texture is invalid") {
    HIP_CHECK_ERROR(hipTexObjectGetTextureDesc(&checkDesc, static_cast<hipTextureObject_t>(0)),
                    hipErrorInvalidValue);
  }
}

/* hipTexObjectDestroy test */

TEST_CASE("Unit_hipTexObjectDestroy_positive") {
  CHECK_IMAGE_SUPPORT

  TexObjectTestWrapper texObjWrapper(false, true);
  REQUIRE(hipTexObjectDestroy(texObjWrapper.mTextureObject) == hipSuccess);
}
