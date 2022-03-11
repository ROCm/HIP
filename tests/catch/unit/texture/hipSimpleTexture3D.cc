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

// Texture reference for 3D texture
texture<float, hipTextureType3D, hipReadModeElementType> texf;
texture<int, hipTextureType3D, hipReadModeElementType>   texi;
texture<char, hipTextureType3D, hipReadModeElementType>  texc;

template <typename T>
__global__ void simpleKernel3DArray(T* outputData, int width,
                                    int height, int depth) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        if (std::is_same<T, float>::value)
          outputData[i*width*height + j*width + k] = tex3D(texf, k, j, i);
        else if (std::is_same<T, int>::value)
          outputData[i*width*height + j*width + k] = tex3D(texi, k, j, i);
        else if (std::is_same<T, char>::value)
          outputData[i*width*height + j*width + k] = tex3D(texc, k, j, i);
      }
    }
  }
#endif
}

template <typename T>
static void runSimpleTexture3D_Check(int width, int height, int depth,
            texture<T, hipTextureType3D, hipReadModeElementType> *tex) {
  unsigned int size = width * height * depth * sizeof(T);
  T* hData = reinterpret_cast<T*>(malloc(size));
  REQUIRE(hData != nullptr);
  memset(hData, 0, size);

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        hData[i*width*height + j*width +k] = i*width*height + j*width + k;
      }
    }
  }

  // Allocate array and copy image data
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<T>();
  hipArray *arr;

  HIP_CHECK(hipMalloc3DArray(&arr, &channelDesc,
            make_hipExtent(width, height, depth), hipArrayDefault));
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T), width, height);
  myparms.dstArray = arr;
  myparms.extent = make_hipExtent(width, height, depth);
  myparms.kind = hipMemcpyHostToDevice;

  HIP_CHECK(hipMemcpy3D(&myparms));

  // set texture parameters
  tex->addressMode[0] = hipAddressModeWrap;
  tex->addressMode[1] = hipAddressModeWrap;
  tex->filterMode = hipFilterModePoint;
  tex->normalized = false;

  // Bind the array to the texture
  HIP_CHECK(hipBindTextureToArray(*tex, arr, channelDesc));

  // Allocate device memory for result
  T* dData = nullptr;
  HIP_CHECK(hipMalloc(&dData, size));
  REQUIRE(dData != nullptr);

  hipLaunchKernelGGL(simpleKernel3DArray, dim3(1, 1, 1), dim3(1, 1, 1),
                     0, 0, dData, width, height, depth);
  HIP_CHECK(hipDeviceSynchronize());

  // Allocate mem for the result on host side
  T *hOutputData = reinterpret_cast<T*>(malloc(size));
  REQUIRE(hOutputData != nullptr);
  memset(hOutputData, 0,  size);

  // copy result from device to host
  HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));
  HipTest::checkArray(hData, hOutputData, width, height, depth);

  HIP_CHECK(hipFree(dData));
  HIP_CHECK(hipFreeArray(arr));
  free(hData);
  free(hOutputData);
}

TEST_CASE("Unit_hipSimpleTexture3D_Check_DataTypes") {
#if HT_AMD
  int imageSupport{};
  HIP_CHECK(hipDeviceGetAttribute(&imageSupport,
                           hipDeviceAttributeImageSupport, 0));
  if (!imageSupport) {
    INFO("Texture is not supported on the device. Test is skipped");
    return;
  }
#endif
  for ( int i = 1; i < 25; i++ ) {
    runSimpleTexture3D_Check<float>(i, i, i, &texf);
    runSimpleTexture3D_Check<int>(i+1, i, i, &texi);
    runSimpleTexture3D_Check<char>(i, i+1, i, &texc);
  }
}
