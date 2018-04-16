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

#include <hip/hip_runtime.h>
#include <hip/hcc_detail/texture_types.h>
#include "hip_internal.hpp"

void getChannelOrderAndType(const hipChannelFormatDesc& desc, enum hipTextureReadMode readMode,
                            cl_channel_order* channelOrder, cl_channel_type* channelType) {
    if (desc.x != 0 && desc.y != 0 && desc.z != 0 && desc.w != 0) {
        *channelOrder = CL_RGBA;
    } else if (desc.x != 0 && desc.y != 0 && desc.z != 0 && desc.w == 0) {
        *channelOrder = CL_RGB;
    } else if (desc.x != 0 && desc.y != 0 && desc.z == 0 && desc.w == 0) {
        *channelOrder = CL_RG;
    } else if (desc.x != 0 && desc.y == 0 && desc.z == 0 && desc.w == 0) {
        *channelOrder = CL_R;
    } else {
    }

    switch (desc.f) {
        case hipChannelFormatKindUnsigned:
            switch (desc.x) {
                case 32:
                    *channelType = CL_UNSIGNED_INT32;
                    break;
                case 16:
                    *channelType = readMode == hipReadModeNormalizedFloat
                                       ? CL_UNORM_INT16
                                       : CL_UNSIGNED_INT16;
                    break;
                case 8:
                    *channelType = readMode == hipReadModeNormalizedFloat
                                       ? CL_UNORM_INT8
                                       : CL_UNSIGNED_INT8;
                    break;
                default:
                    *channelType = CL_UNSIGNED_INT32;
            }
            break;
        case hipChannelFormatKindSigned:
            switch (desc.x) {
                case 32:
                    *channelType = CL_SIGNED_INT32;
                    break;
                case 16:
                    *channelType = readMode == hipReadModeNormalizedFloat
                                       ? CL_SNORM_INT16
                                       : CL_SIGNED_INT16;
                    break;
                case 8:
                    *channelType = readMode == hipReadModeNormalizedFloat
                                       ? CL_SNORM_INT8
                                       : CL_SIGNED_INT8;
                    break;
                default:
                    *channelType = CL_SIGNED_INT32;
            }
            break;
        case hipChannelFormatKindFloat:
            switch (desc.x) {
                case 32:
                    *channelType = CL_FLOAT;
                    break;
                case 16:
                    *channelType = CL_HALF_FLOAT;
                    break;
                case 8:
                    break;
                default:
                    *channelType = CL_FLOAT;
            }
            break;
        case hipChannelFormatKindNone:
        default:
            break;
    }
}

hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const hipResourceViewDesc* pResViewDesc) {
  HIP_INIT_API(pTexObject, pResDesc, pTexDesc, pResViewDesc);

  if (!g_context->devices()[0]->info().imageSupport_) {
    return hipErrorInvalidValue;
  }

  amd::Image* image = nullptr;

  cl_image_format image_format;
  getChannelOrderAndType(pResDesc->res.pitch2D.desc, pTexDesc->readMode,
    &image_format.image_channel_order, &image_format.image_channel_data_type);

  const amd::Image::Format imageFormat(image_format);

  amd::Memory* memory = nullptr;

  switch (pResDesc->resType) {
    case hipResourceTypeArray:
      assert(0);
      break;
    case hipResourceTypeMipmappedArray:
      assert(0);
      break;
    case hipResourceTypeLinear:
      assert(pResViewDesc == nullptr);

      memory = amd::SvmManager::FindSvmBuffer(pResDesc->res.linear.devPtr);
      image = new (*g_context) amd::Image(*memory->asBuffer(), CL_MEM_OBJECT_IMAGE1D, memory->getMemFlags(), imageFormat,
                                          pResDesc->res.linear.sizeInBytes / imageFormat.getElementSize(), 1, 1,
                                          pResDesc->res.linear.sizeInBytes, 0);
      break;
    case hipResourceTypePitch2D:
      assert(pResViewDesc == nullptr);

      memory = amd::SvmManager::FindSvmBuffer(pResDesc->res.pitch2D.devPtr);
      image = new (*g_context) amd::Image(*memory->asBuffer(), CL_MEM_OBJECT_IMAGE2D, memory->getMemFlags(), imageFormat,
                                          pResDesc->res.pitch2D.width, pResDesc->res.pitch2D.height, 1,
                                          pResDesc->res.pitch2D.pitchInBytes, 0);
      break;
    default: return hipErrorInvalidValue;
  }
  *pTexObject = reinterpret_cast<hipTextureObject_t>(as_cl(image));

  return hipErrorUnknown;
}

hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
  HIP_INIT_API(textureObject);

  as_amd(reinterpret_cast<cl_mem>(textureObject))->release();

  return hipSuccess;
}

hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject) {
  HIP_INIT_API(pResDesc, textureObject);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipGetTextureObjectResourceViewDesc(hipResourceViewDesc* pResViewDesc,
                                               hipTextureObject_t textureObject) {
  HIP_INIT_API(pResViewDesc, textureObject);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc,
                                          hipTextureObject_t textureObject) {
  HIP_INIT_API(pTexDesc, textureObject);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipBindTexture(size_t* offset, textureReference* tex, const void* devPtr,
                          const hipChannelFormatDesc* desc, size_t size) {
  HIP_INIT_API(offset, tex, devPtr, desc, size);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipBindTexture2D(size_t* offset, textureReference* tex, const void* devPtr,
                            const hipChannelFormatDesc* desc, size_t width, size_t height,
                            size_t pitch) {
  HIP_INIT_API(offset, tex, devPtr, desc, width, height, pitch);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipBindTextureToArray(textureReference* tex, hipArray_const_t array,
                                 const hipChannelFormatDesc* desc) {
  HIP_INIT_API(tex, array, desc);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t ihipBindTextureToArrayImpl(int dim, enum hipTextureReadMode readMode,
                                      hipArray_const_t array,
                                      const struct hipChannelFormatDesc& desc,
                                      textureReference* tex) {
  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipBindTextureToMipmappedArray(textureReference* tex,
                                          hipMipmappedArray_const_t mipmappedArray,
                                          const hipChannelFormatDesc* desc) {
  HIP_INIT_API(tex, mipmappedArray, desc);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipUnbindTexture(const textureReference* tex) {
  HIP_INIT_API(tex);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array) {
  HIP_INIT_API(desc, array);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipGetTextureAlignmentOffset(size_t* offset, const textureReference* tex) {
  HIP_INIT_API(offset, tex);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipGetTextureReference(const textureReference** tex, const void* symbol) {
  HIP_INIT_API(tex, symbol);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipTexRefSetFormat(textureReference* tex, hipArray_Format fmt, int NumPackedComponents) {
  HIP_INIT_API(tex, fmt, NumPackedComponents);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipTexRefSetFlags(textureReference* tex, unsigned int flags) {
  HIP_INIT_API(tex, flags);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipTexRefSetFilterMode(textureReference* tex, hipTextureFilterMode fm) {
  HIP_INIT_API(tex, fm);

  assert(0 && "Unimplemented");

  return hipErrorUnknown; 
}

hipError_t hipTexRefSetAddressMode(textureReference* tex, int dim, hipTextureAddressMode am) {
  HIP_INIT_API(tex, dim, am);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipTexRefSetArray(textureReference* tex, hipArray_const_t array, unsigned int flags) {
  HIP_INIT_API(tex, array, flags);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipTexRefSetAddress(size_t* offset, textureReference* tex, hipDeviceptr_t devPtr,
                               size_t size) {
  HIP_INIT_API(offset, tex, devPtr, size);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipTexRefSetAddress2D(textureReference* tex, const HIP_ARRAY_DESCRIPTOR* desc,
                                 hipDeviceptr_t devPtr, size_t pitch) {
  HIP_INIT_API(tex, desc, devPtr, pitch);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}
