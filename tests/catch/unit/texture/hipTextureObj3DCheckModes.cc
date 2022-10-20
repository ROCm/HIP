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

bool isGfx90a = false;

template<bool normalizedCoords>
__global__ void tex3DKernel(float *outputData, hipTextureObject_t textureObject,
                            int width, int height, int depth, float offsetX,
                            float offsetY, float offsetZ) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  outputData[z * width * height + y * width + x] = tex3D<float>(textureObject,
                        normalizedCoords ? (x + offsetX) / width : x + offsetX,
                        normalizedCoords ? (y + offsetY) / height : y + offsetY,
                        normalizedCoords ? (z + offsetZ) / depth : z + offsetZ);
#endif
}

template<hipTextureAddressMode addressMode, hipTextureFilterMode filterMode, bool normalizedCoords>
static void runTest(const int width, const int height, const int depth, const float offsetX, const float offsetY,
             const float offsetZ) {
  //printf("%s(addressMode=%d, filterMode=%d, normalizedCoords=%d, width=%d, height=%d, depth=%d, offsetX=%f, offsetY=%f, offsetZ=%f)\n",
  //    __FUNCTION__, addressMode, filterMode, normalizedCoords, width, height,
  //    depth, offsetX, offsetY, offsetZ);
  bool result = true;
  unsigned int size = width * height * depth * sizeof(float);
  float *hData = (float*) malloc(size);
  memset(hData, 0, size);

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        int index = i * width * height + j * width + k;
        hData[index] = index;
      }
    }
  }

  // Allocate array and copy image data
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<float>();
  hipArray *arr;

  HIP_CHECK(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width, height, depth), hipArrayDefault));
  hipMemcpy3DParms myparms;
  memset(&myparms, 0, sizeof(myparms));
  myparms.srcPos = make_hipPos(0,0,0);
  myparms.dstPos = make_hipPos(0,0,0);
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(float), width, height);
  myparms.dstArray = arr;
  myparms.extent = make_hipExtent(width, height, depth);
  myparms.kind = hipMemcpyHostToDevice;

  HIP_CHECK(hipMemcpy3D(&myparms));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = arr;

  // Specify texture object parameters
  hipTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = addressMode;
  texDesc.addressMode[1] = addressMode;
  texDesc.addressMode[2] = addressMode;
  texDesc.filterMode = filterMode;
  texDesc.readMode = hipReadModeElementType;
  texDesc.normalizedCoords = normalizedCoords;

  // Create texture object
  hipTextureObject_t textureObject = 0;
  hipError_t res = hipCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL);
  if (res != hipSuccess) {
    HIP_CHECK(hipFreeArray(arr));
    free(hData);
    if (res == hipErrorNotSupported && isGfx90a) {
      printf("gfx90a doesn't support 3D linear filter! Skipped!\n");
    } else {
      result = false;
    }
    REQUIRE(result);
    return;
  }

  float *dData = nullptr;
  HIP_CHECK(hipMalloc((void**) &dData, size));
  HIP_CHECK(hipMemset(dData, 0, size));
  dim3 dimBlock(8, 8, 8); // 512 threads
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y -1)/ dimBlock.y,
               (depth + dimBlock.z - 1) / dimBlock.z);

  hipLaunchKernelGGL(tex3DKernel<normalizedCoords>, dimGrid, dimBlock, 0, 0, dData,
                     textureObject, width, height, depth, offsetX, offsetY, offsetZ);
  HIP_CHECK(hipGetLastError()); 

  HIP_CHECK(hipDeviceSynchronize());

  float *hOutputData = (float*) malloc(size);
  memset(hOutputData, 0, size);
  HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        int index = i * width * height + j * width + k;
        float expectedValue = getExpectedValue<float, addressMode, filterMode>(
            width, height, depth, offsetX + k, offsetY + j, offsetZ + i, hData);

        if (!hipTextureSamplingVerify<float, filterMode>(hOutputData[index], expectedValue)) {
          INFO("Mismatch at (" << offsetX + k << ", " << offsetY + j << ", " << offsetZ + i << "):" <<
               hOutputData[index] << " expected:" << expectedValue);
          result = false;
          goto line1;
        }
      }
    }
  }
line1:
  HIP_CHECK(hipDestroyTextureObject(textureObject));
  free(hOutputData);
  HIP_CHECK(hipFree(dData));
  HIP_CHECK(hipFreeArray(arr));
  free(hData);
  REQUIRE(result);

}

TEST_CASE("Unit_hipTextureObj3DCheckModes") {
  CHECK_IMAGE_SUPPORT

  int device = 0;
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, device));
  if (!strncmp(props.gcnArchName, "gfx90a", strlen("gfx90a"))) {
    isGfx90a = true;
  }

  SECTION("hipAddressModeClamp, hipFilterModePoint, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, false>
      (256, 256, 256, -3.9, 6.1, 9.5);
    runTest<hipAddressModeClamp, hipFilterModePoint, false>
      (256, 256, 256, 4.4, -7.0, 5.3);
  }

  SECTION("hipAddressModeBorder, hipFilterModePoint, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, false>
      (256, 256, 256, -8.5, 2.9, 5.8);
    runTest<hipAddressModeBorder, hipFilterModePoint, false>
      (256, 256, 256, 12.5, 6.7, 11.4);
  }

  SECTION("hipAddressModeClamp, hipFilterModeLinear, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, false>
      (256, 256, 256, -0.4, -0.4, -0.4);
    runTest<hipAddressModeClamp, hipFilterModeLinear, false>
      (256, 256, 256, 4, 14.6, -0.3);
  }

  SECTION("hipAddressModeBorder, hipFilterModeLinear, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, false>
      (256, 256, 256, 6.9, 7.4, 0.4);
    runTest<hipAddressModeBorder, hipFilterModeLinear, false>
      (256, 256, 256, 12.5, 23.7, 0.34);
  }

  SECTION("hipAddressModeClamp, hipFilterModePoint, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, true>
      (256, 256, 256, -3, 8.9, -4);
    runTest<hipAddressModeClamp, hipFilterModePoint, true>
      (256, 256, 256, 4, -0.1, 8.2);
  }

  SECTION("hipAddressModeBorder, hipFilterModePoint, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, true>
      (256, 256, 256, -8.5, 15.9, 0.1);
    runTest<hipAddressModeBorder, hipFilterModePoint, true>
      (256, 256, 256, 12.5, -17.9, -0.35);
  }

  SECTION("hipAddressModeClamp, hipFilterModeLinear, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, true>
      (256, 256, 256, -3, 5.8, 0.89);
    runTest<hipAddressModeClamp, hipFilterModeLinear, true>
      (256, 256, 256, 4, 9.1, 2.08);
  }

  SECTION("hipAddressModeBorder, hipFilterModeLinear, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, true>
      (256, 256, 256, -8.5, 6.6, 3.67);
    runTest<hipAddressModeBorder, hipFilterModeLinear, true>
      (256, 256, 256, 12.5, 0.01, -9.9);
  }
}
