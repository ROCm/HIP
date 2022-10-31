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
/*
Testcase Scenarios :
Unit_hipTextureMipmapRef2D_Positive_Check - Test correct execution of hipBindTextureToMipmappedArray api for diffrent mipmapped array sizes and number of levels
Unit_hipTextureMipmapRef2D_Negative_Parameters - Test unsuccessful execution of hipBindTextureToMipmappedArray api when parameters are invalid
*/
#include <hip_test_common.hh>

texture<float, 2, hipReadModeElementType> texRef;

// MipMap is currently supported only on windows
#if (defined(_WIN32) && !defined(__HIP_NO_IMAGE_SUPPORT))
__global__ void tex2DKernel(float* outputData, int width, float level) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2DLod<float>(texRef, x, y, level);
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

  // Allocate memory for Mipmapped array and set data to mipmap_level
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<float>();

  hipMipmappedArray* mip_array_ptr;
  HIP_CHECK(hipMallocMipmappedArray(&mip_array_ptr, &channelDesc, make_hipExtent(orig_width, orig_height, 0), 2 * mipmap_level, hipArrayDefault));

  hipArray* hipArray = nullptr;
  HIP_CHECK(hipMipmappedArrayGetLevel(&hipArray, mip_array_ptr, mipmap_level));
  HIP_CHECK(hipMemcpy2DToArray(hipArray, 0, 0, hData, width * sizeof(float), width * sizeof(float), height, hipMemcpyHostToDevice));

  // Set texture parameters
  texRef.addressMode[0] = hipAddressModeWrap;
  texRef.addressMode[1] = hipAddressModeWrap;
  texRef.filterMode = hipFilterModePoint;
  texRef.normalized = 0;

  // Bind the array to the texture
  HIP_CHECK(hipBindTextureToMipmappedArray(&texRef, mip_array_ptr, &channelDesc));

  // Allocate device memory for result
  float* dData = nullptr;
  HIP_CHECK(hipMalloc(&dData, size));
  REQUIRE(dData != nullptr);

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  hipLaunchKernelGGL(tex2DKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, dData, width, mipmap_level);
  HIP_CHECK(hipGetLastError()); 
  HIP_CHECK(hipDeviceSynchronize());

  // Allocate memory on host and copy result from device to host
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
  HIP_CHECK(hipUnbindTexture(texRef));
  HIP_CHECK(hipFree(dData));
  HIP_CHECK(hipFreeArray(hipArray));
  HIP_CHECK(hipFreeMipmappedArray(mip_array_ptr));
  free(hData);
}
#endif

TEST_CASE("Unit_hipTextureMipmapRef2D_Positive_Check") {
  CHECK_IMAGE_SUPPORT
  // Height Width Vector
  std::vector<unsigned int> hw_vec = {2048, 1024, 512, 256, 64};
  std::vector<unsigned int> mip_vec = {8, 4, 2, 1};
#ifdef _WIN32
  for (auto& hw : hw_vec) {
    for (auto& mip : mip_vec) {
      if ((hw / static_cast<int>(pow(2, (mip * 2)))) > 0) {
        runMipMapTest(hw, hw, mip);
      }
    }
  }
#else
  SUCCEED("Mipmaps are Supported only on windows, skipping the test.");
#endif
}

TEST_CASE("Unit_hipTextureMipmapRef2D_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT

#ifdef _WIN32
  unsigned int width = 64;
  unsigned int height = 64;
  unsigned int mipmap_level = 1;

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<float>();

  hipMipmappedArray* mip_array_ptr;
  HIP_CHECK(hipMallocMipmappedArray(&mip_array_ptr, &channelDesc, make_hipExtent(width, height, 0), mipmap_level, hipArrayDefault));

  texRef.addressMode[0] = hipAddressModeWrap;
  texRef.addressMode[1] = hipAddressModeWrap;
  texRef.filterMode = hipFilterModePoint;
  texRef.normalized = 0;

  SECTION("textureReference is nullptr") {
    ret = hipBindTextureToMipmappedArray(nullptr, mip_array_ptr, &channelDesc)
    REQUIRE(ret != hipSuccess);
  }

  SECTION("MipmappedArray is nullptr") {
    ret = hipBindTextureToMipmappedArray(&texRef, nullptr, &channelDesc)
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Channel descriptor is nullptr") {
    ret = hipBindTextureToMipmappedArray(&texRef, mip_array_ptr, nullptr)
    REQUIRE(ret != hipSuccess);
  }

  HIP_CHECK(hipFreeMipmappedArray(mip_array_ptr));
#else
  SUCCEED("Mipmaps are Supported only on windows, skipping the test.");
#endif
}
