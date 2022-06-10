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

// Height Width Vector
std::vector<unsigned int> hw_vector = {2048, 1024, 512, 256, 64};
std::vector<unsigned int> mip_vector = {8, 4, 2, 1};

// MipMap is currently supported only on windows
#if (defined(_WIN32) && !defined(__HIP_NO_IMAGE_SUPPORT))
__global__ void tex2DKernel(float* outputData, hipTextureObject_t textureObject, int width,
                            float level) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2DLod<float>(textureObject, x, y, level);
}

static void runMipMapTest(unsigned int width, unsigned int height, unsigned int mipmap_level) {
  INFO("Width: " << width << "Height: " << height << "mip: " << mipmap_level);

  // Create new width & height to be tested
  unsigned int orig_width = width;
  unsigned int orig_height = height;
  unsigned int i, j;
  width /= pow(2, mipmap_level);
  height /= pow(2, mipmap_level);
  unsigned int size = width * height * sizeof(float);

  float* hData = reinterpret_cast<float*>(malloc(size));
  REQUIRE(hData != nullptr);
  memset(hData, 0, size);
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      hData[i * width + j] = i * width + j;
    }
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
  HIP_ARRAY3D_DESCRIPTOR mipmapped_array_desc;
  memset(&mipmapped_array_desc, 0x00, sizeof(HIP_ARRAY3D_DESCRIPTOR));
  mipmapped_array_desc.Width = orig_width;
  mipmapped_array_desc.Height = orig_height;
  mipmapped_array_desc.Depth = 0;
  mipmapped_array_desc.Format = HIP_AD_FORMAT_FLOAT;
  mipmapped_array_desc.NumChannels =
      ((channelDesc.x != 0) + (channelDesc.y != 0) + (channelDesc.z != 0) + (channelDesc.w != 0));
  mipmapped_array_desc.Flags = 0;

  hipMipmappedArray* mip_array_ptr;
  HIP_CHECK(hipMipmappedArrayCreate(&mip_array_ptr, &mipmapped_array_desc, 2 * mipmap_level));

  hipArray* hipArray = nullptr;
  HIP_CHECK(hipMipmappedArrayGetLevel(&hipArray, mip_array_ptr, mipmap_level));
  HIP_CHECK(hipMemcpyToArray(hipArray, 0, 0, hData, size, hipMemcpyHostToDevice));

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
  HIP_CHECK(hipCreateTextureObject(&textureObject, &resDesc, &texDesc, nullptr));

  float* dData = nullptr;
  HIP_CHECK(hipMalloc(&dData, size));
  REQUIRE(dData != nullptr);

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  hipLaunchKernelGGL(tex2DKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, dData, textureObject, width,
                     (2 * mipmap_level));
  hipDeviceSynchronize();

  float* hOutputData = reinterpret_cast<float*>(malloc(size));
  REQUIRE(hOutputData != nullptr);
  memset(hOutputData, 0, size);
  HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));

  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      if (hData[i * width + j] != hOutputData[i * width + j]) {
        INFO("Difference found at [ " << i << j << " ]: " << hData[i * width + j]
                                      << hOutputData[i * width + j]);
        REQUIRE(false);
      }
    }
  }
  HIP_CHECK(hipDestroyTextureObject(textureObject));
  HIP_CHECK(hipFree(dData));
  HIP_CHECK(hipFreeArray(hipArray));
  free(hData);
}
#endif

TEST_CASE("Unit_hipTextureMipmapObj2D_Check") {
#if HT_AMD
  int imageSupport{};
  HIP_CHECK(hipDeviceGetAttribute(&imageSupport, hipDeviceAttributeImageSupport, 0));
  if (!imageSupport) {
    INFO("Texture is not supported on the device. Test is skipped");
    return;
  }
#endif
#ifdef _WIN32
  for (auto& hw : hw_vector) {
    for (auto& mip : mip_vector) {
      if ((hw / static_cast<int>(pow(2, (mip * 2)))) > 0) {
        runMipMapTest(hw, hw, mip);
      }
    }
  }
#else
  SUCCEED("Mipmaps are Supported only on windows, skipping the test.");
#endif
}
