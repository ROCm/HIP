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

#pragma once

#include <hip_test_common.hh>

constexpr size_t BlockSize = 16;

// read from a texture using normalized coordinates
constexpr size_t ChannelToRead = 1;
template <typename T>
__global__ void readFromTexture(T* output, hipTextureObject_t texObj, size_t width, size_t height,
                                bool textureGather) {
  #if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  // Calculate normalized texture coordinates
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const float u = x / (float)width;

  // Read from texture and write to global memory
  if (height == 0) {
    output[x] = tex1D<T>(texObj, u);
  } else {
    const float v = y / (float)height;
    if (textureGather) {
      // tex2Dgather not supported on __gfx90a__
      #if !defined(__gfx90a__)
      output[y * width + x] = tex2Dgather<T>(texObj, u, v, ChannelToRead);
      #else
      #warning("tex2Dgather not supported on gfx90a");
      #endif
    } else {
      output[y * width + x] = tex2D<T>(texObj, u, v);
    }
  }
  #endif
}

template <typename T> void checkDataIsAscending(const std::vector<T>& hostData) {
  bool allMatch = true;
  size_t i = 0;
  for (; i < hostData.size(); ++i) {
    allMatch = allMatch && hostData[i] == static_cast<T>(i);
    if (!allMatch) break;
  }
  INFO("hostData[" << i << "] == " << static_cast<T>(hostData[i]));
  REQUIRE(allMatch);
}

inline size_t getFreeMem() {
  size_t free = 0, total = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  return free;
}

struct Sizes {
  int max1D;
  std::array<int, 2> max2D;
  std::array<int, 3> max3D;

  Sizes(unsigned int flag) {
    int device;
    HIP_CHECK(hipGetDevice(&device));
    static_assert(
        hipArrayDefault == 0,
        "hipArrayDefault is assumed to be equivalent to 0 for the following switch statment");
#if HT_NVIDIA
    static_assert(hipArraySurfaceLoadStore == CUDA_ARRAY3D_SURFACE_LDST,
                  "hipArraySurface is assumed to be equivalent to CUDA_ARRAY3D_SURFACE_LDST for "
                  "the following switch statment");
#endif
    switch (flag) {
      case hipArrayDefault: {  // 0
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, device));
        max1D = prop.maxTexture1D;
        max2D = {prop.maxTexture2D[0], prop.maxTexture2D[1]};
        max3D = {prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]};
        return;
      }
      case hipArraySurfaceLoadStore: {  // CUDA_ARRAY3D_SURFACE_LDST
        int value;
        HIP_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSurface1D, device));
        max1D = value;
        HIP_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSurface2D, device));
        max2D = {value, value};
        HIP_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSurface3D, device));
        max3D = {value, value, value};
        return;
      }
      default: {
        INFO("Array flag not supported");
        REQUIRE(false);
        return;
      }
    }
  }
};

inline const char* channelFormatString(hipChannelFormatKind formatKind) noexcept {
  switch (formatKind) {
    case hipChannelFormatKindFloat:
      return "float";
    case hipChannelFormatKindSigned:
      return "signed";
    case hipChannelFormatKindUnsigned:
      return "unsigned";
    default:
      return "error";
  }
}

// All the possible formats for channel data in an array.
static const std::vector<hipArray_Format> driverFormats{
    HIP_AD_FORMAT_UNSIGNED_INT8, HIP_AD_FORMAT_UNSIGNED_INT16, HIP_AD_FORMAT_UNSIGNED_INT32,
    HIP_AD_FORMAT_SIGNED_INT8,   HIP_AD_FORMAT_SIGNED_INT16,   HIP_AD_FORMAT_SIGNED_INT32,
    HIP_AD_FORMAT_HALF,          HIP_AD_FORMAT_FLOAT};

// Helpful for printing errors
inline const char* formatToString(hipArray_Format f) {
  switch (f) {
    case HIP_AD_FORMAT_UNSIGNED_INT8:
      return "Unsigned Int 8";
    case HIP_AD_FORMAT_UNSIGNED_INT16:
      return "Unsigned Int 16";
    case HIP_AD_FORMAT_UNSIGNED_INT32:
      return "Unsigned Int 32";
    case HIP_AD_FORMAT_SIGNED_INT8:
      return "Signed Int 8";
    case HIP_AD_FORMAT_SIGNED_INT16:
      return "Signed Int 16";
    case HIP_AD_FORMAT_SIGNED_INT32:
      return "Signed Int 32";
    case HIP_AD_FORMAT_HALF:
      return "Float 16";
    case HIP_AD_FORMAT_FLOAT:
      return "Float 32";
    default:
      return "not found";
  }
}
