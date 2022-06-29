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

/*
 * Validates Array Resource texture object with negative/functional tests.
 */
TEST_CASE("Unit_hipCreateTextureObject_ArrayResource") {
  checkImageSupport();

  hipError_t ret;
  hipResourceDesc resDesc;
  hipTextureDesc texDesc;
  hipTextureObject_t texObj;

  /* set resource type as hipResourceTypeArray and array(nullptr) */
  // Populate resource descriptor
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = nullptr;

  // Populate texture descriptor
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = hipReadModeElementType;

  ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
  REQUIRE(ret != hipSuccess);
}

/*
 * Validates MipMappedArray Resource texture object
 * with negative/functional tests.
 */
TEST_CASE("Unit_hipCreateTextureObject_MmArrayResource") {
  checkImageSupport();

  hipError_t ret;
  hipResourceDesc resDesc;
  hipTextureDesc texDesc;
  hipTextureObject_t texObj;

  /* set resource type as hipResourceTypeMipmappedArray and mipmap(nullptr) */
  // Populate resource descriptor
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeMipmappedArray;
  resDesc.res.mipmap.mipmap = nullptr;

  // Populate texture descriptor
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = hipReadModeElementType;

  ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
  REQUIRE(ret != hipSuccess);
}
