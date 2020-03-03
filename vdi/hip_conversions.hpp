/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hcc_detail/driver_types.h>
#include <hip/hcc_detail/texture_types.h>

// HIP_MEMCPY3D is currently broken.
// TODO remove this struct once the headers will be fixed.
struct _HIP_MEMCPY3D {
  unsigned int srcXInBytes;
  unsigned int srcY;
  unsigned int srcZ;
  unsigned int srcLOD;
  hipMemoryType srcMemoryType;
  const void* srcHost;
  hipDeviceptr_t srcDevice;
  hipArray_t srcArray;
  unsigned int srcPitch;
  unsigned int srcHeight;

  unsigned int dstXInBytes;
  unsigned int dstY;
  unsigned int dstZ;
  unsigned int dstLOD;
  hipMemoryType dstMemoryType;
  void* dstHost;
  hipDeviceptr_t dstDevice;
  hipArray_t dstArray;
  unsigned int dstPitch;
  unsigned int dstHeight;

  unsigned int WidthInBytes;
  unsigned int Height;
  unsigned int Depth;
};

namespace hip
{
inline
cl_channel_type getCLChannelType(const hipArray_Format hipFormat,
                                 const hipTextureReadMode hipReadMode) {
  if (hipReadMode == hipReadModeElementType) {
    switch (hipFormat) {
      case HIP_AD_FORMAT_UNSIGNED_INT8:
        return CL_UNSIGNED_INT8;
      case HIP_AD_FORMAT_SIGNED_INT8:
        return CL_SIGNED_INT8;
      case HIP_AD_FORMAT_UNSIGNED_INT16:
        return CL_UNSIGNED_INT16;
      case HIP_AD_FORMAT_SIGNED_INT16:
        return CL_SIGNED_INT16;
      case HIP_AD_FORMAT_UNSIGNED_INT32:
        return CL_UNSIGNED_INT32;
      case HIP_AD_FORMAT_SIGNED_INT32:
        return CL_SIGNED_INT32;
      case HIP_AD_FORMAT_HALF:
        return CL_HALF_FLOAT;
      case HIP_AD_FORMAT_FLOAT:
        return CL_FLOAT;
    }
  } else if (hipReadMode == hipReadModeNormalizedFloat) {
    switch (hipFormat) {
      case HIP_AD_FORMAT_UNSIGNED_INT8:
        return CL_UNORM_INT8;
      case HIP_AD_FORMAT_SIGNED_INT8:
        return CL_SNORM_INT8;
      case HIP_AD_FORMAT_UNSIGNED_INT16:
        return CL_UNORM_INT16;
      case HIP_AD_FORMAT_SIGNED_INT16:
        return CL_SNORM_INT16;
      case HIP_AD_FORMAT_UNSIGNED_INT32:
        return CL_UNSIGNED_INT32;
      case HIP_AD_FORMAT_SIGNED_INT32:
        return CL_SIGNED_INT32;
      case HIP_AD_FORMAT_HALF:
        return CL_HALF_FLOAT;
      case HIP_AD_FORMAT_FLOAT:
        return CL_FLOAT;
    }
  }

  ShouldNotReachHere();

  return {};
}

inline
cl_channel_order getCLChannelOrder(const unsigned int hipNumChannels) {
  switch (hipNumChannels) {
    case 1:
      return CL_R;
    case 2:
      return CL_RG;
    case 4:
      return CL_RGBA;
  }

  ShouldNotReachHere();

  return {};
}

inline
cl_mem_object_type getCLMemObjectType(const unsigned int hipWidth,
                                      const unsigned int hipHeight,
                                      const unsigned int hipDepth,
                                      const unsigned int flags) {
  if (flags == hipArrayDefault) {
    if ((hipWidth != 0) && (hipHeight == 0) && (hipDepth == 0)) {
      return CL_MEM_OBJECT_IMAGE1D;
    } else if ((hipWidth != 0) && (hipHeight != 0) && (hipDepth == 0)) {
      return CL_MEM_OBJECT_IMAGE2D;
    } else if ((hipWidth != 0) && (hipHeight != 0) && (hipDepth != 0)) {
      return CL_MEM_OBJECT_IMAGE3D;
    }
  } else if (flags == hipArrayLayered) {
    if ((hipWidth != 0) && (hipHeight == 0) && (hipDepth != 0)) {
      return CL_MEM_OBJECT_IMAGE1D_ARRAY;
    } else if ((hipWidth != 0) && (hipHeight != 0) && (hipDepth != 0)) {
      return CL_MEM_OBJECT_IMAGE2D_ARRAY;
    }
  }

  ShouldNotReachHere();

  return {};
}

inline
cl_addressing_mode getCLAddressingMode(const hipTextureAddressMode hipAddressMode) {
  switch (hipAddressMode) {
    case hipAddressModeWrap:
      return CL_ADDRESS_REPEAT;
    case hipAddressModeClamp:
      return CL_ADDRESS_CLAMP;
    case hipAddressModeMirror:
      return CL_ADDRESS_MIRRORED_REPEAT;
    case hipAddressModeBorder:
      return CL_ADDRESS_CLAMP_TO_EDGE;
  }

  ShouldNotReachHere();

  return {};
}

inline
cl_filter_mode getCLFilterMode(const hipTextureFilterMode hipFilterMode) {
  switch (hipFilterMode) {
    case hipFilterModePoint:
      return CL_FILTER_NEAREST;
    case hipFilterModeLinear:
      return CL_FILTER_LINEAR;
  }

  ShouldNotReachHere();

  return {};
}

inline
cl_mem_object_type getCLMemObjectType(const hipResourceType hipResType) {
  switch (hipResType) {
    case hipResourceTypeLinear:
      return CL_MEM_OBJECT_IMAGE1D;
    case hipResourceTypePitch2D:
      return CL_MEM_OBJECT_IMAGE2D;
    default:
      break;
  }

  ShouldNotReachHere();

  return {};
}

inline
size_t getElementSize(const hipArray_Format arrayFormat) {
  switch (arrayFormat) {
    case HIP_AD_FORMAT_UNSIGNED_INT8:
    case HIP_AD_FORMAT_SIGNED_INT8:
      return 1;
    case HIP_AD_FORMAT_UNSIGNED_INT16:
    case HIP_AD_FORMAT_SIGNED_INT16:
    case HIP_AD_FORMAT_HALF:
      return 2;
    case HIP_AD_FORMAT_UNSIGNED_INT32:
    case HIP_AD_FORMAT_SIGNED_INT32:
    case HIP_AD_FORMAT_FLOAT:
      return 4;
  }

  ShouldNotReachHere();

  return {};
}

inline
hipChannelFormatDesc getChannelFormatDesc(int numChannels,
                                          hipArray_Format arrayFormat) {
  switch (arrayFormat) {
    case HIP_AD_FORMAT_UNSIGNED_INT8:
      switch (numChannels) {
        case 1:
          return {8, 0, 0, 0, hipChannelFormatKindUnsigned};
        case 2:
          return {8, 8, 0, 0, hipChannelFormatKindUnsigned};
        case 4:
          return {8, 8, 8, 8, hipChannelFormatKindUnsigned};
      }
    case HIP_AD_FORMAT_SIGNED_INT8:
      switch (numChannels) {
        case 1:
          return {8, 0, 0, 0, hipChannelFormatKindSigned};
        case 2:
          return {8, 8, 0, 0, hipChannelFormatKindSigned};
        case 4:
          return {8, 8, 8, 8, hipChannelFormatKindSigned};
      }
    case HIP_AD_FORMAT_UNSIGNED_INT16:
      switch (numChannels) {
        case 1:
          return {16, 0, 0, 0, hipChannelFormatKindUnsigned};
        case 2:
          return {16, 16, 0, 0, hipChannelFormatKindUnsigned};
        case 4:
          return {16, 16, 16, 16, hipChannelFormatKindUnsigned};
      }
    case HIP_AD_FORMAT_SIGNED_INT16:
      switch (numChannels) {
        case 1:
          return {16, 0, 0, 0, hipChannelFormatKindSigned};
        case 2:
          return {16, 16, 0, 0, hipChannelFormatKindSigned};
        case 4:
          return {16, 16, 16, 16, hipChannelFormatKindSigned};
      }
    case HIP_AD_FORMAT_UNSIGNED_INT32:
      switch (numChannels) {
        case 1:
          return {32, 0, 0, 0, hipChannelFormatKindUnsigned};
        case 2:
          return {32, 32, 0, 0, hipChannelFormatKindUnsigned};
        case 4:
          return {32, 32, 32, 32, hipChannelFormatKindUnsigned};
      }
    case HIP_AD_FORMAT_SIGNED_INT32:
      switch (numChannels) {
        case 1:
          return {32, 0, 0, 0, hipChannelFormatKindSigned};
        case 2:
          return {32, 32, 0, 0, hipChannelFormatKindSigned};
        case 4:
          return {32, 32, 32, 32, hipChannelFormatKindSigned};
      }
    case HIP_AD_FORMAT_HALF:
      switch (numChannels) {
        case 1:
          return {16, 0, 0, 0, hipChannelFormatKindFloat};
        case 2:
          return {16, 16, 0, 0, hipChannelFormatKindFloat};
        case 4:
          return {16, 16, 16, 16, hipChannelFormatKindFloat};
      }
    case HIP_AD_FORMAT_FLOAT:
      switch (numChannels) {
        case 1:
          return {32, 0, 0, 0, hipChannelFormatKindFloat};
        case 2:
          return {32, 32, 0, 0, hipChannelFormatKindFloat};
        case 4:
          return {32, 32, 32, 32, hipChannelFormatKindFloat};
      }
  }

  ShouldNotReachHere();

  return {};
}

inline
unsigned int getNumChannels(const hipChannelFormatDesc& desc) {
  return ((desc.x != 0) + (desc.y != 0) + (desc.z != 0) + (desc.w != 0));
}

inline
hipArray_Format getArrayFormat(const hipChannelFormatDesc& desc) {
  switch (desc.f) {
    case hipChannelFormatKindUnsigned:
      switch (desc.x) {
        case 8:
          return HIP_AD_FORMAT_UNSIGNED_INT8;
        case 16:
          return HIP_AD_FORMAT_UNSIGNED_INT16;
        case 32:
          return HIP_AD_FORMAT_UNSIGNED_INT32;
      }
    case hipChannelFormatKindSigned:
      switch (desc.x) {
        case 8:
          return HIP_AD_FORMAT_SIGNED_INT8;
        case 16:
          return HIP_AD_FORMAT_SIGNED_INT16;
        case 32:
          return HIP_AD_FORMAT_SIGNED_INT32;
      }
    case hipChannelFormatKindFloat:
      switch (desc.x) {
        case 16:
          return HIP_AD_FORMAT_HALF;
        case 32:
          return HIP_AD_FORMAT_FLOAT;
      }
    default:
      break;
  }

  ShouldNotReachHere();

  return {};
}

inline
int getNumChannels(const hipResourceViewFormat hipFormat) {
  switch (hipFormat) {
    case hipResViewFormatUnsignedChar1:
    case hipResViewFormatSignedChar1:
    case hipResViewFormatUnsignedShort1:
    case hipResViewFormatSignedShort1:
    case hipResViewFormatUnsignedInt1:
    case hipResViewFormatHalf1:
    case hipResViewFormatFloat1:
      return 1;
    case hipResViewFormatUnsignedChar2:
    case hipResViewFormatSignedChar2:
    case hipResViewFormatUnsignedShort2:
    case hipResViewFormatSignedShort2:
    case hipResViewFormatUnsignedInt2:
    case hipResViewFormatHalf2:
    case hipResViewFormatFloat2:
      return 2;
    case hipResViewFormatUnsignedChar4:
    case hipResViewFormatSignedChar4:
    case hipResViewFormatUnsignedShort4:
    case hipResViewFormatSignedShort4:
    case hipResViewFormatUnsignedInt4:
    case hipResViewFormatHalf4:
    case hipResViewFormatFloat4:
      return 4;
    default:
      break;
  }

  ShouldNotReachHere();

  return {};
}

inline
hipArray_Format getArrayFormat(const hipResourceViewFormat hipFormat) {
  switch (hipFormat) {
    case hipResViewFormatUnsignedChar1:
    case hipResViewFormatUnsignedChar2:
    case hipResViewFormatUnsignedChar4:
      return HIP_AD_FORMAT_UNSIGNED_INT8;
    case hipResViewFormatSignedChar1:
    case hipResViewFormatSignedChar2:
    case hipResViewFormatSignedChar4:
      return HIP_AD_FORMAT_SIGNED_INT8;
    case hipResViewFormatUnsignedShort1:
    case hipResViewFormatUnsignedShort2:
    case hipResViewFormatUnsignedShort4:
      return HIP_AD_FORMAT_UNSIGNED_INT16;
    case hipResViewFormatSignedShort1:
    case hipResViewFormatSignedShort2:
    case hipResViewFormatSignedShort4:
      return HIP_AD_FORMAT_SIGNED_INT16;
    case hipResViewFormatUnsignedInt1:
    case hipResViewFormatUnsignedInt2:
    case hipResViewFormatUnsignedInt4:
      return HIP_AD_FORMAT_UNSIGNED_INT32;
    case hipResViewFormatSignedInt1:
    case hipResViewFormatSignedInt2:
    case hipResViewFormatSignedInt4:
      return HIP_AD_FORMAT_SIGNED_INT32;
    case hipResViewFormatHalf1:
    case hipResViewFormatHalf2:
    case hipResViewFormatHalf4:
      return HIP_AD_FORMAT_HALF;
    case hipResViewFormatFloat1:
    case hipResViewFormatFloat2:
    case hipResViewFormatFloat4:
      return HIP_AD_FORMAT_FLOAT;
    default:
      break;
  }

  ShouldNotReachHere();

  return {};
}

inline
hipResourceViewFormat getResourceViewFormat(const hipChannelFormatDesc& desc) {
  switch (desc.f) {
    case hipChannelFormatKindUnsigned:
      switch (getNumChannels(desc)) {
        case 1:
          switch (desc.x) {
            case 8:
              return hipResViewFormatUnsignedChar1;
            case 16:
              return hipResViewFormatUnsignedShort1;
            case 32:
              return hipResViewFormatUnsignedInt1;
          }
        case 2:
          switch (desc.x) {
            case 8:
              return hipResViewFormatUnsignedChar2;
            case 16:
              return hipResViewFormatUnsignedShort2;
            case 32:
              return hipResViewFormatUnsignedInt2;
          }
        case 4:
          switch (desc.x) {
            case 8:
              return hipResViewFormatUnsignedChar4;
            case 16:
              return hipResViewFormatUnsignedShort4;
            case 32:
              return hipResViewFormatUnsignedInt4;
          }
      }
    case hipChannelFormatKindSigned:
      switch (getNumChannels(desc)) {
        case 1:
          switch (desc.x) {
            case 8:
              return hipResViewFormatSignedChar1;
            case 16:
              return hipResViewFormatSignedShort1;
            case 32:
              return hipResViewFormatSignedInt1;
          }
        case 2:
          switch (desc.x) {
            case 8:
              return hipResViewFormatSignedChar2;
            case 16:
              return hipResViewFormatSignedShort2;
            case 32:
              return hipResViewFormatSignedInt2;
          }
        case 4:
          switch (desc.x) {
            case 8:
              return hipResViewFormatSignedChar4;
            case 16:
              return hipResViewFormatSignedShort4;
            case 32:
              return hipResViewFormatSignedInt4;
          }
      }
    case hipChannelFormatKindFloat:
      switch (getNumChannels(desc)) {
        case 1:
          switch (desc.x) {
            case 16:
              return hipResViewFormatHalf1;
            case 32:
              return hipResViewFormatFloat1;
          }
        case 2:
          switch (desc.x) {
            case 16:
              return hipResViewFormatHalf2;
            case 32:
              return hipResViewFormatFloat2;
          }
        case 4:
          switch (desc.x) {
            case 16:
              return hipResViewFormatHalf4;
            case 32:
              return hipResViewFormatFloat4;
          }
      }
    default:
      break;
  }

  ShouldNotReachHere();

  return {};
}

inline
hipTextureReadMode getReadMode(unsigned int flags) {
  if (flags & HIP_TRSF_READ_AS_INTEGER) {
    return hipReadModeElementType;
  } else {
    return hipReadModeNormalizedFloat;
  }
}

inline
int getNormalizedCoords(unsigned int flags) {
  if (flags & HIP_TRSF_NORMALIZED_COORDINATES) {
    return 1;
  } else {
    return 0;
  }
}

inline
int getSRGB(unsigned int flags) {
  if (flags & HIP_TRSF_SRGB) {
    return 1;
  } else {
    return 0;
  }
}

inline
hipTextureDesc getTextureDesc(const textureReference* texRef,
                              const hipTextureReadMode readMode) {
  hipTextureDesc texDesc = {};
  std::memcpy(texDesc.addressMode, texRef->addressMode, sizeof(texDesc.addressMode));
  texDesc.filterMode = texRef->filterMode;
  texDesc.readMode = readMode;
  texDesc.sRGB = texRef->sRGB;
  texDesc.normalizedCoords = texRef->normalized;
  texDesc.maxAnisotropy = texRef->maxAnisotropy;
  texDesc.mipmapFilterMode = texRef->mipmapFilterMode;
  texDesc.mipmapLevelBias = texRef->mipmapLevelBias;
  texDesc.minMipmapLevelClamp = texRef->minMipmapLevelClamp;
  texDesc.maxMipmapLevelClamp = texRef->maxMipmapLevelClamp;

  return texDesc;
}

inline
hipResourceViewDesc getResourceViewDesc(hipArray_const_t array,
                                        const hipResourceViewFormat format) {
  hipResourceViewDesc resViewDesc = {};
  resViewDesc.format = format;
  resViewDesc.width = array->width;
  resViewDesc.height = array->height;
  resViewDesc.depth = array->depth;
  resViewDesc.firstMipmapLevel = 0;
  resViewDesc.lastMipmapLevel = 0;
  resViewDesc.firstLayer = 0;
  resViewDesc.lastLayer = 0; /* TODO add hipArray::numLayers */

  return resViewDesc;
}

inline
hipResourceViewDesc getResourceViewDesc(hipMipmappedArray_const_t array,
                                        const hipResourceViewFormat format) {
  hipResourceViewDesc resViewDesc = {};
  resViewDesc.format = format;
  resViewDesc.width = array->width;
  resViewDesc.height = array->height;
  resViewDesc.depth = array->depth;
  resViewDesc.firstMipmapLevel = 0;
  resViewDesc.lastMipmapLevel = 0; /* TODO add hipMipmappedArray::numMipLevels */
  resViewDesc.firstLayer = 0;
  resViewDesc.lastLayer = 0; /* TODO add hipArray::numLayers */

  return resViewDesc;
}

inline
std::pair<hipMemoryType, hipMemoryType> getMemoryType(const hipMemcpyKind kind) {
  switch (kind) {
    case hipMemcpyHostToHost:
      return {hipMemoryTypeHost, hipMemoryTypeHost};
    case hipMemcpyHostToDevice:
      return {hipMemoryTypeHost, hipMemoryTypeDevice};
    case hipMemcpyDeviceToHost:
      return {hipMemoryTypeDevice, hipMemoryTypeHost};
    case hipMemcpyDeviceToDevice:
      return {hipMemoryTypeDevice, hipMemoryTypeDevice};
    case hipMemcpyDefault:
      return {hipMemoryTypeUnified, hipMemoryTypeUnified};
  }

  ShouldNotReachHere();

  return {};
}

inline
_HIP_MEMCPY3D getMemcpy3DParms(const hip_Memcpy2D& desc2D) {
  _HIP_MEMCPY3D desc3D = {};

  desc3D.srcXInBytes = desc2D.srcXInBytes;
  desc3D.srcY = desc2D.srcY;
  desc3D.srcZ = 0;
  desc3D.srcLOD = 0;
  desc3D.srcMemoryType = desc2D.srcMemoryType;
  desc3D.srcHost = desc2D.srcHost;
  desc3D.srcDevice = desc2D.srcDevice;
  desc3D.srcArray = desc2D.srcArray;
  desc3D.srcPitch = desc2D.srcPitch;
  desc3D.srcHeight = 0;

  desc3D.dstXInBytes = desc2D.dstXInBytes;
  desc3D.dstY = desc2D.dstY;
  desc3D.dstZ = 0;
  desc3D.dstLOD = 0;
  desc3D.dstMemoryType = desc2D.dstMemoryType;
  desc3D.dstHost = desc2D.dstHost;
  desc3D.dstDevice = desc2D.dstDevice;
  desc3D.dstArray = desc2D.dstArray;
  desc3D.dstPitch = desc2D.dstPitch;
  desc3D.dstHeight = 0;

  desc3D.WidthInBytes = desc2D.WidthInBytes;
  desc3D.Height = desc2D.Height;
  desc3D.Depth = 0;

  return desc3D;
}
};
