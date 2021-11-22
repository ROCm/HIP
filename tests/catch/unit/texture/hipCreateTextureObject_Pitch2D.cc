/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#define UNALIGN_OFFSET 1
#define SIZE_H 20
#define SIZE_W 30
#define N 512


/*
 * Validates Pitch2D Resource texture object with negative and functional tests
 */
TEST_CASE("Unit_hipCreateTextureObject_Pitch2DResource") {
  hipError_t ret;
  hipResourceDesc resDesc;
  hipTextureDesc texDesc;
  hipTextureObject_t texObj;
  hipDeviceProp_t devProp;
  size_t devPitchA;
  float *devPtrA;

  /** Initialization */
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devPtrA), &devPitchA,
                                             SIZE_W*sizeof(float), SIZE_H));
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  memset(&resDesc, 0, sizeof(resDesc));
  memset(&texDesc, 0, sizeof(texDesc));
  resDesc.resType = hipResourceTypePitch2D;

  /** Sections */
  SECTION("hipResourceTypePitch2D and devPtr(nullptr)") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = nullptr;
    resDesc.res.pitch2D.height = SIZE_H;
    resDesc.res.pitch2D.width = SIZE_W;
    resDesc.res.pitch2D.pitchInBytes = devPitchA;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypePitch2D and devicePtr un-aligned") {
    if (devProp.textureAlignment > UNALIGN_OFFSET) {
      // Populate resource descriptor
      resDesc.res.pitch2D.devPtr = reinterpret_cast<char *>(devPtrA)
                                                        + UNALIGN_OFFSET;
      resDesc.res.pitch2D.height = SIZE_H;
      resDesc.res.pitch2D.width = SIZE_W;
      resDesc.res.pitch2D.pitchInBytes = devPitchA;
      resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

      // Populate texture descriptor
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.readMode = hipReadModeElementType;
      ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
      REQUIRE(ret != hipSuccess);
    }
  }

  SECTION("hipResourceTypePitch2D and pitch(un-aligned)") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = devPtrA;
    resDesc.res.pitch2D.height = SIZE_H;
    resDesc.res.pitch2D.width = SIZE_W;
    resDesc.res.pitch2D.pitchInBytes = UNALIGN_OFFSET;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypePitch2D and height(0)") {
    if ((TestContext::get()).isAmd()) {
      // Populate resource descriptor
      resDesc.res.pitch2D.devPtr = devPtrA;
      resDesc.res.pitch2D.height = 0;
      resDesc.res.pitch2D.width = SIZE_W;
      resDesc.res.pitch2D.pitchInBytes = devPitchA;
      resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

      // Populate texture descriptor
      texDesc.readMode = hipReadModeElementType;

      ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
      REQUIRE(ret != hipSuccess);
    } else {
      // Test expected to return error with height(0).
      WARN("Resourcetype Pitch2D/height(0) skipped on nvidia");
    }
  }

  SECTION("hipResourceTypePitch2D and height(0)/devptr(nullptr)") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = nullptr;
    resDesc.res.pitch2D.height = 0;
    resDesc.res.pitch2D.width = SIZE_W;
    resDesc.res.pitch2D.pitchInBytes = devPitchA;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypePitch2D and height(max(size_t))") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = devPtrA;
    resDesc.res.pitch2D.height = std::numeric_limits<std::size_t>::max();
    resDesc.res.pitch2D.width = SIZE_W;
    resDesc.res.pitch2D.pitchInBytes = devPitchA;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypePitch2D and width(0)") {
    if ((TestContext::get()).isAmd()) {
      // Populate resource descriptor
      resDesc.resType = hipResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = devPtrA;
      resDesc.res.pitch2D.height = SIZE_H;
      resDesc.res.pitch2D.width = 0;
      resDesc.res.pitch2D.pitchInBytes = devPitchA;
      resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

      // Populate texture descriptor
      texDesc.readMode = hipReadModeElementType;

      ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
      REQUIRE(ret != hipSuccess);
    } else {
      // api expected to return failure when width(0) is passed.
      WARN("ResourceType Pitch2D/width(0) skipped on nvidia");
    }
  }

  SECTION("hipResourceTypePitch2D and width(0)/devPtr(nullptr)") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = nullptr;
    resDesc.res.pitch2D.height = SIZE_H;
    resDesc.res.pitch2D.width = 0;
    resDesc.res.pitch2D.pitchInBytes = devPitchA;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypePitch2D and width(max(size_t))") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = devPtrA;
    resDesc.res.pitch2D.height = SIZE_H;
    resDesc.res.pitch2D.width = std::numeric_limits<std::size_t>::max();
    resDesc.res.pitch2D.pitchInBytes = devPitchA;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypePitch2D and pitch(max(size_t))") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = devPtrA;
    resDesc.res.pitch2D.height = SIZE_H;
    resDesc.res.pitch2D.width = SIZE_W;
    resDesc.res.pitch2D.pitchInBytes = std::numeric_limits<std::size_t>::max();
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  /** De-Initialization */
  HIP_CHECK(hipFree(devPtrA));
}

