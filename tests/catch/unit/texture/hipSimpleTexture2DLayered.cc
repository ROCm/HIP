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

typedef float T;

// Texture reference for 2D Layered texture
texture<float, hipTextureType2DLayered> tex2DL;

__global__ void simpleKernelLayeredArray(T* outputData,
                                         int width, int height, int layer) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[layer * width * height + y * width + x] = tex2DLayered(tex2DL,
                                                              x, y, layer);
}

TEST_CASE("Unit_hipSimpleTexture2DLayered_Check") {
  constexpr int SIZE = 512;
  constexpr int num_layers = 5;
  constexpr unsigned int width = SIZE;
  constexpr unsigned int height = SIZE;
  constexpr unsigned int size = width * height * num_layers * sizeof(T);

  T* hData = reinterpret_cast<T*>(malloc(size));
  REQUIRE(hData != nullptr);
  memset(hData, 0, size);

  for (unsigned int layer = 0; layer < num_layers; layer++) {
    for (int i = 0; i < static_cast<int>(width * height); i++) {
      hData[layer * width * height + i] = i;
    }
  }
  hipChannelFormatDesc channelDesc;
  // Allocate array and copy image data
  channelDesc = hipCreateChannelDesc(sizeof(T)*8, 0, 0, 0,
                                     hipChannelFormatKindFloat);
  hipArray *arr;

  HIP_CHECK(hipMalloc3DArray(&arr, &channelDesc,
               make_hipExtent(width, height, num_layers), hipArrayLayered));
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T), width, height);
  myparms.dstArray = arr;
  myparms.extent = make_hipExtent(width , height, num_layers);
  // myparms.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&myparms));

  // set texture parameters
  tex2DL.addressMode[0] = hipAddressModeWrap;
  tex2DL.addressMode[1] = hipAddressModeWrap;
  tex2DL.filterMode = hipFilterModePoint;
  tex2DL.normalized = false;

  // Bind the array to the texture
  HIP_CHECK(hipBindTextureToArray(tex2DL, arr, channelDesc));

  // Allocate device memory for result
  T* dData = nullptr;
  HIP_CHECK(hipMalloc(&dData, size));
  REQUIRE(dData != nullptr);

  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
  for (unsigned int layer = 0; layer < num_layers; layer++)
    hipLaunchKernelGGL(simpleKernelLayeredArray, dimGrid, dimBlock, 0, 0,
                       dData, width, height, layer);
  HIP_CHECK(hipDeviceSynchronize());

  // Allocate mem for the result on host side
  T *hOutputData = reinterpret_cast<T*>(malloc(size));
  REQUIRE(hOutputData != nullptr);
  memset(hOutputData, 0,  size);

  // copy result from device to host
  HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));
  HipTest::checkArray(hData, hOutputData, width, height, num_layers);

  HIP_CHECK(hipFree(dData));
  HIP_CHECK(hipFreeArray(arr));
  free(hData);
  free(hOutputData);
}
