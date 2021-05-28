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

#include <hip/amd_detail/amd_driver_types.h>
#include <hip/amd_detail/amd_texture_types.h>

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
cl_channel_order getCLChannelOrder(const unsigned int hipNumChannels,
                                   const int sRGB) {
  switch (hipNumChannels) {
    case 1:
      return CL_R;
    case 2:
      return CL_RG;
    case 4:
      return (sRGB == 1) ? CL_sRGBA : CL_RGBA;
    default:
      break;
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
      return CL_ADDRESS_CLAMP_TO_EDGE;
    case hipAddressModeMirror:
      return CL_ADDRESS_MIRRORED_REPEAT;
    case hipAddressModeBorder:
      return CL_ADDRESS_CLAMP;
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
      return CL_MEM_OBJECT_IMAGE1D_BUFFER;
    case hipResourceTypePitch2D:
      return CL_MEM_OBJECT_IMAGE2D;
    default:
      break;
  }

  ShouldNotReachHere();

  return {};
}

inline
size_t getElementSize(const hipArray_const_t array) {
  switch (array->Format) {
    case HIP_AD_FORMAT_UNSIGNED_INT8:
    case HIP_AD_FORMAT_SIGNED_INT8:
      return 1 * array->NumChannels;
    case HIP_AD_FORMAT_UNSIGNED_INT16:
    case HIP_AD_FORMAT_SIGNED_INT16:
    case HIP_AD_FORMAT_HALF:
      return 2 * array->NumChannels;
    case HIP_AD_FORMAT_UNSIGNED_INT32:
    case HIP_AD_FORMAT_SIGNED_INT32:
    case HIP_AD_FORMAT_FLOAT:
      return 4 * array->NumChannels;
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
    case hipResViewFormatSignedInt1:
    case hipResViewFormatHalf1:
    case hipResViewFormatFloat1:
      return 1;
    case hipResViewFormatUnsignedChar2:
    case hipResViewFormatSignedChar2:
    case hipResViewFormatUnsignedShort2:
    case hipResViewFormatSignedShort2:
    case hipResViewFormatUnsignedInt2:
    case hipResViewFormatSignedInt2:
    case hipResViewFormatHalf2:
    case hipResViewFormatFloat2:
      return 2;
    case hipResViewFormatUnsignedChar4:
    case hipResViewFormatSignedChar4:
    case hipResViewFormatUnsignedShort4:
    case hipResViewFormatSignedShort4:
    case hipResViewFormatUnsignedInt4:
    case hipResViewFormatSignedInt4:
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
hipTextureDesc getTextureDesc(const textureReference* texRef) {
  hipTextureDesc texDesc = {};
  std::memcpy(texDesc.addressMode, texRef->addressMode, sizeof(texDesc.addressMode));
  texDesc.filterMode = texRef->filterMode;
  texDesc.readMode = texRef->readMode;
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
HIP_MEMCPY3D getDrvMemcpy3DDesc(const hip_Memcpy2D& desc2D) {
  HIP_MEMCPY3D desc3D = {};

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
  desc3D.Depth = 1;

  return desc3D;
}

inline
HIP_MEMCPY3D getDrvMemcpy3DDesc(const hipMemcpy3DParms& desc) {
  HIP_MEMCPY3D descDrv = {};

  descDrv.WidthInBytes = desc.extent.width;
  descDrv.Height = desc.extent.height;
  descDrv.Depth = desc.extent.depth;

  descDrv.srcXInBytes = desc.srcPos.x;
  descDrv.srcY = desc.srcPos.y;
  descDrv.srcZ = desc.srcPos.z;
  descDrv.srcLOD = 0;

  descDrv.dstXInBytes = desc.dstPos.x;
  descDrv.dstY = desc.dstPos.y;
  descDrv.dstZ = desc.dstPos.z;
  descDrv.dstLOD = 0;

  if (desc.srcArray != nullptr) {
    descDrv.srcMemoryType = hipMemoryTypeArray;
    descDrv.srcArray = desc.srcArray;
    // When reffering to array memory, hipPos::x is in elements.
    descDrv.srcXInBytes *= getElementSize(desc.srcArray);
  }

  if (desc.srcPtr.ptr != nullptr) {
    descDrv.srcMemoryType = std::get<0>(hip::getMemoryType(desc.kind));
    descDrv.srcHost = desc.srcPtr.ptr;
    descDrv.srcDevice = desc.srcPtr.ptr;
    descDrv.srcPitch = desc.srcPtr.pitch;
    descDrv.srcHeight = desc.srcPtr.ysize;
  }

  if (desc.dstArray != nullptr) {
    descDrv.dstMemoryType = hipMemoryTypeArray;
    descDrv.dstArray = desc.dstArray;
    // When reffering to array memory, hipPos::x is in elements.
    descDrv.dstXInBytes *= getElementSize(desc.dstArray);
  }

  if (desc.dstPtr.ptr != nullptr) {
    descDrv.dstMemoryType = std::get<1>(getMemoryType(desc.kind));
    descDrv.dstHost = desc.dstPtr.ptr;
    descDrv.dstDevice = desc.dstPtr.ptr;
    descDrv.dstPitch = desc.dstPtr.pitch;
    descDrv.dstHeight = desc.dstPtr.ysize;
  }

  // If a HIP array is participating in the copy, the extent is defined in terms of that array's elements.
  if ((desc.srcArray != nullptr) && (desc.dstArray == nullptr)) {
    descDrv.WidthInBytes *= getElementSize(desc.srcArray);
  } else if ((desc.srcArray == nullptr) && (desc.dstArray != nullptr)) {
    descDrv.WidthInBytes *= getElementSize(desc.dstArray);
  } else if ((desc.srcArray != nullptr) && (desc.dstArray != nullptr)) {
    descDrv.WidthInBytes *= getElementSize(desc.dstArray);
  }

  return descDrv;
}

inline
hipResourceType getResourceType(const HIPresourcetype resType) {
  // These two enums should be isomorphic.
  return static_cast<hipResourceType>(resType);
}

inline
HIPresourcetype getResourceType(const hipResourceType resType) {
  // These two enums should be isomorphic.
  return static_cast<HIPresourcetype>(resType);
}

inline
hipResourceDesc getResourceDesc(const HIP_RESOURCE_DESC& resDesc) {
  hipResourceDesc desc;

  desc.resType = getResourceType(resDesc.resType);
  switch (desc.resType) {
  case hipResourceTypeArray:
    desc.res.array.array = resDesc.res.array.hArray;
    break;
  case hipResourceTypeMipmappedArray:
    desc.res.mipmap.mipmap = resDesc.res.mipmap.hMipmappedArray;
    break;
  case hipResourceTypeLinear:
    desc.res.linear.devPtr = resDesc.res.linear.devPtr;
    desc.res.linear.desc = getChannelFormatDesc(resDesc.res.linear.numChannels, resDesc.res.linear.format);
    desc.res.linear.sizeInBytes = resDesc.res.linear.sizeInBytes;
    break;
  case hipResourceTypePitch2D:
    desc.res.pitch2D.devPtr = resDesc.res.pitch2D.devPtr;
    desc.res.pitch2D.desc = getChannelFormatDesc(resDesc.res.pitch2D.numChannels, resDesc.res.pitch2D.format);
    desc.res.pitch2D.width = resDesc.res.pitch2D.width;
    desc.res.pitch2D.height = resDesc.res.pitch2D.height;
    desc.res.pitch2D.pitchInBytes = resDesc.res.pitch2D.pitchInBytes;
    break;
  default:
    break;
  }

  return desc;
}

inline
HIP_RESOURCE_DESC getResourceDesc(const hipResourceDesc& resDesc) {
  HIP_RESOURCE_DESC desc;

  desc.resType = getResourceType(resDesc.resType);
  switch (desc.resType) {
  case HIP_RESOURCE_TYPE_ARRAY:
    desc.res.array.hArray = resDesc.res.array.array;
    break;
  case HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY:
    desc.res.mipmap.hMipmappedArray = resDesc.res.mipmap.mipmap;
    break;
  case HIP_RESOURCE_TYPE_LINEAR:
    desc.res.linear.devPtr = resDesc.res.linear.devPtr;
    desc.res.linear.numChannels = getNumChannels(resDesc.res.linear.desc);
    desc.res.linear.format = getArrayFormat(resDesc.res.linear.desc);
    desc.res.linear.sizeInBytes = resDesc.res.linear.sizeInBytes;
    break;
  case HIP_RESOURCE_TYPE_PITCH2D:
    desc.res.pitch2D.devPtr = resDesc.res.pitch2D.devPtr;
    desc.res.pitch2D.numChannels = getNumChannels(resDesc.res.pitch2D.desc);
    desc.res.pitch2D.format = getArrayFormat(resDesc.res.pitch2D.desc);
    desc.res.pitch2D.width = resDesc.res.pitch2D.width;
    desc.res.pitch2D.height = resDesc.res.pitch2D.height;
    desc.res.pitch2D.pitchInBytes = resDesc.res.pitch2D.pitchInBytes;
    break;
  default:
    break;
  }

  return desc;
}

inline
hipTextureAddressMode getAddressMode(const HIPaddress_mode mode) {
  // These two enums should be isomorphic.
  return static_cast<hipTextureAddressMode>(mode);
}

inline
HIPaddress_mode getAddressMode(const hipTextureAddressMode mode) {
  // These two enums should be isomorphic.
  return static_cast<HIPaddress_mode>(mode);
}

inline
hipTextureFilterMode getFilterMode(const HIPfilter_mode mode) {
  // These two enums should be isomorphic.
  return static_cast<hipTextureFilterMode>(mode);
}

inline
HIPfilter_mode getFilterMode(const hipTextureFilterMode mode) {
  // These two enums should be isomorphic.
  return static_cast<HIPfilter_mode>(mode);
}

inline
hipTextureReadMode getReadMode(const unsigned int flags) {
  if (flags & HIP_TRSF_READ_AS_INTEGER) {
    return hipReadModeElementType;
  } else {
    return hipReadModeNormalizedFloat;
  }
}

inline
unsigned int getReadMode(const hipTextureReadMode mode) {
  if (mode ==  hipReadModeElementType) {
    return HIP_TRSF_READ_AS_INTEGER;
  } else {
    return 0;
  }
}

inline
int getsRGB(const unsigned int flags) {
  if (flags & HIP_TRSF_SRGB) {
    return 1;
  } else {
    return 0;
  }
}

inline
unsigned int getsRGB(const int sRGB) {
  if (sRGB == 1) {
    return HIP_TRSF_SRGB;
  } else {
    return 0;
  }
}

inline
int getNormalizedCoords(const unsigned int flags) {
  if (flags & HIP_TRSF_NORMALIZED_COORDINATES) {
    return 1;
  } else {
    return 0;
  }
}

inline
unsigned int getNormalizedCoords(const int normalizedCoords) {
  if (normalizedCoords == 1) {
    return HIP_TRSF_NORMALIZED_COORDINATES;
  } else {
    return 0;
  }
}

inline
hipTextureDesc getTextureDesc(const HIP_TEXTURE_DESC& texDesc) {
  hipTextureDesc desc;

  desc.addressMode[0] = getAddressMode(texDesc.addressMode[0]);
  desc.addressMode[1] = getAddressMode(texDesc.addressMode[1]);
  desc.addressMode[2] = getAddressMode(texDesc.addressMode[2]);
  desc.filterMode = getFilterMode(texDesc.filterMode);
  desc.readMode = getReadMode(texDesc.flags);
  desc.sRGB = getsRGB(texDesc.flags);
  std::memcpy(desc.borderColor, texDesc.borderColor, sizeof(desc.borderColor));
  desc.normalizedCoords = getNormalizedCoords(texDesc.flags);
  desc.maxAnisotropy = texDesc.maxAnisotropy;
  desc.mipmapFilterMode = getFilterMode(texDesc.mipmapFilterMode);
  desc.mipmapLevelBias = texDesc.mipmapLevelBias;
  desc.minMipmapLevelClamp = texDesc.minMipmapLevelClamp;
  desc.maxMipmapLevelClamp = texDesc.maxMipmapLevelClamp;

  return desc;
}

inline
HIP_TEXTURE_DESC getTextureDesc(const hipTextureDesc& texDesc) {
  HIP_TEXTURE_DESC desc;

  desc.addressMode[0] = getAddressMode(texDesc.addressMode[0]);
  desc.addressMode[1] = getAddressMode(texDesc.addressMode[1]);
  desc.addressMode[2] = getAddressMode(texDesc.addressMode[2]);
  desc.filterMode = getFilterMode(texDesc.filterMode);
  desc.flags = 0;
  desc.flags |= getReadMode(texDesc.readMode);
  desc.flags |= getsRGB(texDesc.sRGB);
  desc.flags |= getNormalizedCoords(texDesc.normalizedCoords);
  desc.maxAnisotropy = texDesc.maxAnisotropy;
  desc.mipmapFilterMode = getFilterMode(texDesc.mipmapFilterMode);
  desc.mipmapLevelBias = texDesc.mipmapLevelBias;
  desc.minMipmapLevelClamp = texDesc.minMipmapLevelClamp;
  desc.maxMipmapLevelClamp = texDesc.maxMipmapLevelClamp;
  std::memcpy(desc.borderColor, texDesc.borderColor, sizeof(desc.borderColor));

  return desc;
}

inline
hipResourceViewFormat getResourceViewFormat(const HIPresourceViewFormat format) {
  // These two enums should be isomorphic.
  return static_cast<hipResourceViewFormat>(format);
}

inline
HIPresourceViewFormat getResourceViewFormat(const hipResourceViewFormat format) {
  // These two enums should be isomorphic.
  return static_cast<HIPresourceViewFormat>(format);
}

inline
hipResourceViewDesc getResourceViewDesc(const HIP_RESOURCE_VIEW_DESC& resViewDesc) {
  hipResourceViewDesc desc;

  desc.format = getResourceViewFormat(resViewDesc.format);
  desc.width = resViewDesc.width;
  desc.height = resViewDesc.height;
  desc.depth = resViewDesc.depth;
  desc.firstMipmapLevel = resViewDesc.firstMipmapLevel;
  desc.lastMipmapLevel = resViewDesc.lastMipmapLevel;
  desc.firstLayer = resViewDesc.firstLayer;
  desc.lastLayer = resViewDesc.lastLayer;

  return desc;
}

inline
HIP_RESOURCE_VIEW_DESC getResourceViewDesc(const hipResourceViewDesc& resViewDesc) {
  HIP_RESOURCE_VIEW_DESC desc;

  desc.format = getResourceViewFormat(resViewDesc.format);
  desc.width = resViewDesc.width;
  desc.height = resViewDesc.height;
  desc.depth = resViewDesc.depth;
  desc.firstMipmapLevel = resViewDesc.firstMipmapLevel;
  desc.lastMipmapLevel = resViewDesc.lastMipmapLevel;
  desc.firstLayer = resViewDesc.firstLayer;
  desc.lastLayer = resViewDesc.lastLayer;

  return desc;
}

inline
size_t getElementSize(const hipChannelFormatDesc &desc) {
  return (desc.x / 8) * getNumChannels(desc);
}
};
