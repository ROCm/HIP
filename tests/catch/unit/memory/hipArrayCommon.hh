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

template <class T, size_t N, hipArray_Format Format> struct type_and_size_and_format {
  using type = T;
  static constexpr size_t size = N;
  static constexpr hipArray_Format format = Format;
};

// Create a map of type to scalar type, vector size and scalar type format enum.
// This is useful for creating simpler function that depend on the vector size.
template <typename T> struct vector_info;
template <>
struct vector_info<int> : type_and_size_and_format<int, 1, HIP_AD_FORMAT_SIGNED_INT32> {};
template <> struct vector_info<float> : type_and_size_and_format<float, 1, HIP_AD_FORMAT_FLOAT> {};
template <>
struct vector_info<short> : type_and_size_and_format<short, 1, HIP_AD_FORMAT_SIGNED_INT16> {};
template <>
struct vector_info<char> : type_and_size_and_format<char, 1, HIP_AD_FORMAT_SIGNED_INT8> {};
template <>
struct vector_info<unsigned int>
    : type_and_size_and_format<unsigned int, 1, HIP_AD_FORMAT_UNSIGNED_INT32> {};
template <>
struct vector_info<unsigned short>
    : type_and_size_and_format<unsigned short, 1, HIP_AD_FORMAT_UNSIGNED_INT16> {};
template <>
struct vector_info<unsigned char>
    : type_and_size_and_format<unsigned char, 1, HIP_AD_FORMAT_UNSIGNED_INT8> {};

template <>
struct vector_info<int2> : type_and_size_and_format<int, 2, HIP_AD_FORMAT_SIGNED_INT32> {};
template <> struct vector_info<float2> : type_and_size_and_format<float, 2, HIP_AD_FORMAT_FLOAT> {};
template <>
struct vector_info<short2> : type_and_size_and_format<short, 2, HIP_AD_FORMAT_SIGNED_INT16> {};
template <>
struct vector_info<char2> : type_and_size_and_format<char, 2, HIP_AD_FORMAT_SIGNED_INT8> {};
template <>
struct vector_info<uint2>
    : type_and_size_and_format<unsigned int, 2, HIP_AD_FORMAT_UNSIGNED_INT32> {};
template <>
struct vector_info<ushort2>
    : type_and_size_and_format<unsigned short, 2, HIP_AD_FORMAT_UNSIGNED_INT16> {};
template <>
struct vector_info<uchar2>
    : type_and_size_and_format<unsigned char, 2, HIP_AD_FORMAT_UNSIGNED_INT8> {};

template <>
struct vector_info<int4> : type_and_size_and_format<int, 4, HIP_AD_FORMAT_SIGNED_INT32> {};
template <> struct vector_info<float4> : type_and_size_and_format<float, 4, HIP_AD_FORMAT_FLOAT> {};
template <>
struct vector_info<short4> : type_and_size_and_format<short, 4, HIP_AD_FORMAT_SIGNED_INT16> {};
template <>
struct vector_info<char4> : type_and_size_and_format<char, 4, HIP_AD_FORMAT_SIGNED_INT8> {};
template <>
struct vector_info<uint4>
    : type_and_size_and_format<unsigned int, 4, HIP_AD_FORMAT_UNSIGNED_INT32> {};
template <>
struct vector_info<ushort4>
    : type_and_size_and_format<unsigned short, 4, HIP_AD_FORMAT_UNSIGNED_INT16> {};
template <>
struct vector_info<uchar4>
    : type_and_size_and_format<unsigned char, 4, HIP_AD_FORMAT_UNSIGNED_INT8> {};

// read from a texture using normalized coordinates
constexpr size_t ChannelToRead = 1;
template <typename T>
__global__ void readFromTexture(T* output, hipTextureObject_t texObj, size_t width, size_t height,
                                bool textureGather) {
  // Calculate normalized texture coordinates
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const float u = x / (float)width;

  // Read from texture and write to global memory
  if (height == 0) {
    output[x] = tex1D<T>(texObj, u);
  } else {
    const float v = y / (float)height;
    output[y * width + x] =
        textureGather ? tex2Dgather<T>(texObj, u, v, ChannelToRead) : tex2D<T>(texObj, u, v);
  }
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
