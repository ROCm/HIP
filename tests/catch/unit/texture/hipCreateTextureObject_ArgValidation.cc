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

#define N 512

/*
 * Validate argument list of texture object api.
 */
TEST_CASE("Unit_hipCreateTextureObject_ArgValidation") {
  checkImageSupport();

  float *texBuf;
  hipError_t ret;
  constexpr int xsize = 32;
  hipResourceDesc resDesc;
  hipTextureDesc texDesc;
  hipTextureObject_t texObj;

  /** Initialization */
  HIP_CHECK(hipMalloc(&texBuf, N * sizeof(float)));
  // Populate resource descriptor
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeLinear;
  resDesc.res.linear.devPtr = texBuf;
  resDesc.res.linear.desc = hipCreateChannelDesc(xsize, 0, 0, 0,
                       hipChannelFormatKindFloat);
  resDesc.res.linear.sizeInBytes = N * sizeof(float);

  // Populate texture descriptor
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = hipReadModeElementType;


  /** Sections */
  SECTION("TextureObject as nullptr") {
    ret = hipCreateTextureObject(nullptr, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Resouce Descriptor as nullptr") {
    ret = hipCreateTextureObject(&texObj, nullptr, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Texture Descriptor as nullptr") {
    if ((TestContext::get()).isAmd()) {
      ret = hipCreateTextureObject(&texObj, &resDesc, nullptr, nullptr);
      REQUIRE(ret != hipSuccess);
    } else {
      // API expected to return failure. Test skipped
      // on nvidia as api returns success and would lead
      // to unexpected behavior with app.
      WARN("Texture Desc(nullptr) skipped on nvidia");
    }
  }

  SECTION("Destroy TextureObject with nullptr") {
    ret = hipDestroyTextureObject((hipTextureObject_t)nullptr);
    // api to return success and no crash seen.
    REQUIRE(ret == hipSuccess);
  }

  /** De-Initialization */
  HIP_CHECK(hipFree(texBuf));
}
