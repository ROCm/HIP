/*
Copyright (c) 2019-present Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include <hip/hip_runtime.h>
#include "test_common.h"

// Height Width Vector
std::vector<unsigned int> hw_vector = {2048, 1024, 512, 256, 64};
std::vector<unsigned int> mip_vector = {8, 4, 2, 1};

__global__ void tex2DKernel(float* outputData, hipTextureObject_t textureObject, int width,
                            int height, float level) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2DLod<float>(textureObject, x, y, level);
}

bool runMipMapTest(unsigned int width, unsigned int height, unsigned int mipmap_level) {
  bool testResult = true;

  printf("Width: %u Height: %u mip: %u \n", width, height, mipmap_level);

  // Create new width & height to be tested
  unsigned int orig_width = width;
  unsigned int orig_height = height;
  width /= pow(2, mipmap_level);
  height /= pow(2, mipmap_level);
  unsigned int size = width * height * sizeof(float);


  float* hData = (float*)malloc(size);
  memset(hData, 0, size);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      hData[i * width + j] = i * width + j;
    }
  }
  printf("hData: ");
  for (int i = 0; i < 64; i++) {
    printf("%f  ", hData[i]);
    if (i % width == 0) {
      printf("\n");
    }
  }
  printf("\n");

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
  HIP_ARRAY3D_DESCRIPTOR mipmapped_array_desc;
  memset(&mipmapped_array_desc, 0x00, sizeof(HIP_ARRAY3D_DESCRIPTOR));
  mipmapped_array_desc.Width = orig_width;
  mipmapped_array_desc.Height = orig_height;
  mipmapped_array_desc.Depth = 0;
  mipmapped_array_desc.Format = HIP_AD_FORMAT_FLOAT;
  mipmapped_array_desc.NumChannels = ((channelDesc.x != 0) + (channelDesc.y != 0)
                                      + (channelDesc.z != 0) + (channelDesc.w != 0));
  mipmapped_array_desc.Flags = 0;


  hipMipmappedArray* mip_array_ptr;
  hipMipmappedArrayCreate(&mip_array_ptr, &mipmapped_array_desc, 2 * mipmap_level);

  hipArray *hipArray = nullptr;
  HIPCHECK(hipMipmappedArrayGetLevel(&hipArray, mip_array_ptr, mipmap_level));
  HIPCHECK(hipMemcpyToArray(hipArray, 0, 0, hData, size, hipMemcpyHostToDevice));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = hipArray;

  // Specify texture object parameters
  hipTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = hipAddressModeWrap;
  texDesc.addressMode[1] = hipAddressModeWrap;
  texDesc.filterMode = hipFilterModePoint;
  texDesc.readMode = hipReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture object
  hipTextureObject_t textureObject = 0;
  hipCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL);

  float* dData = NULL;
  hipMalloc((void**)&dData, size);

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  hipLaunchKernelGGL(tex2DKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, dData, textureObject,
                     width, height, (2 * mipmap_level));

  hipDeviceSynchronize();

  float* hOutputData = (float*)malloc(size);
  memset(hOutputData, 0, size);
  hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost);

  printf("dData: ");
  for (int i = 0; i < 64; i++) {
    printf("%f  ", hOutputData[i]);
    if (i % width == 0) {
      printf("\n");
    }
  }
  printf("\n");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (hData[i * width + j] != hOutputData[i * width + j]) {
        printf("Difference [ %d %d ]:%f ----%f\n", i, j, hData[i * width + j],
                hOutputData[i * width + j]);
        testResult = false;
        break;
      }
    }
  }
  hipDestroyTextureObject(textureObject);
  hipFree(dData);
  hipFreeArray(hipArray);
  return testResult;
}


bool runTest(int argc, char** argv) {
  bool testResult = true;

  for (auto& hw: hw_vector) {
    for (auto& mip: mip_vector) {
      if ((hw / static_cast<int>(pow (2,(mip * 2)))) > 0) {
        testResult |= runMipMapTest(hw, hw, mip);
      }
    }
  }

  printf("\n");
  return testResult;
}

int main(int argc, char** argv) {
  bool testResult = true;

#ifdef _WIN32
  testResult = runTest(argc, argv);
#else
  std::cout<<"Mipmaps are Supported only on windows, skipping the test"<<std::endl;
#endif

  if (testResult) {
      passed();
  } else {
      exit(EXIT_FAILURE);
  }
}

