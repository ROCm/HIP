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
#include "platform/sampler.hpp"

namespace hip {
  struct TextureObject {
    uint32_t imageSRD[HIP_IMAGE_OBJECT_SIZE_DWORD];
    uint32_t samplerSRD[HIP_SAMPLER_OBJECT_SIZE_DWORD];
    amd::Image* image;
    amd::Sampler* sampler;
    hipResourceDesc resDesc;
  };
};

void getDrvChannelOrderAndType(const enum hipArray_Format Format, unsigned int NumChannels,
                               cl_channel_order* channelOrder,
                               cl_channel_type* channelType) {
  switch (Format) {
    case HIP_AD_FORMAT_UNSIGNED_INT8:
      *channelType = CL_UNSIGNED_INT8;
      break;
    case HIP_AD_FORMAT_UNSIGNED_INT16:
      *channelType = CL_UNSIGNED_INT16;
      break;
    case HIP_AD_FORMAT_UNSIGNED_INT32:
      *channelType = CL_UNSIGNED_INT32;
      break;
    case HIP_AD_FORMAT_SIGNED_INT8:
      *channelType = CL_SIGNED_INT8;
      break;
    case HIP_AD_FORMAT_SIGNED_INT16:
      *channelType = CL_SIGNED_INT16;
      break;
    case HIP_AD_FORMAT_SIGNED_INT32:
      *channelType = CL_SIGNED_INT32;
      break;
    case HIP_AD_FORMAT_HALF:
      *channelType = CL_HALF_FLOAT;
      break;
    case HIP_AD_FORMAT_FLOAT:
      *channelType = CL_FLOAT;
      break;
    default:
      break;
  }

  if (NumChannels == 4) {
    *channelOrder = CL_RGBA;
  } else if (NumChannels == 2) {
    *channelOrder = CL_RG;
  } else if (NumChannels == 1) {
    *channelOrder = CL_R;
  }
}

void setDescFromChannelType(cl_channel_type channelType, hipChannelFormatDesc* desc) {

  memset(desc, 0x00, sizeof(hipChannelFormatDesc));

  switch (channelType) {
    case CL_SIGNED_INT8:
    case CL_SIGNED_INT16:
    case CL_SIGNED_INT32:
      desc->f = hipChannelFormatKindSigned;
      break;
    case CL_UNSIGNED_INT8:
    case CL_UNSIGNED_INT16:
    case CL_UNSIGNED_INT32:
      desc->f = hipChannelFormatKindUnsigned;
      break;
    case CL_HALF_FLOAT:
    case CL_FLOAT:
      desc->f = hipChannelFormatKindFloat;
      break;
    default:
      desc->f = hipChannelFormatKindNone;
      break;
  }

  switch (channelType) {
    case CL_SIGNED_INT8:
    case CL_UNSIGNED_INT8:
      desc->x = 8;
      break;
    case CL_SIGNED_INT16:
    case CL_UNSIGNED_INT16:
    case CL_HALF_FLOAT:
      desc->x = 16;
      break;
    case CL_SIGNED_INT32:
    case CL_UNSIGNED_INT32:
    case CL_FLOAT:
      desc->x = 32;
      break;
    default:
      desc->x = 0;
      break;
  }
}

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

void getByteSizeFromChannelFormatKind(enum hipChannelFormatKind channelFormatKind, size_t* byteSize) {
    switch (channelFormatKind)
    {
      case hipChannelFormatKindSigned:
        *byteSize = sizeof(int);
        break;
      case hipChannelFormatKindUnsigned:
        *byteSize = sizeof(unsigned int);
        break;
      case hipChannelFormatKindFloat:
        *byteSize = sizeof(float);
        break;
      case hipChannelFormatKindNone:
        *byteSize = sizeof(size_t);
        break;
      default:
        *byteSize = 1;
        break;
    }
}

amd::Sampler* fillSamplerDescriptor(enum hipTextureAddressMode addressMode,
                           enum hipTextureFilterMode filterMode, int normalizedCoords) {
#ifndef CL_FILTER_NONE
#define CL_FILTER_NONE 0x1142
#endif
  uint32_t filter_mode = CL_FILTER_NONE;
  switch (filterMode) {
    case hipFilterModePoint:
      filter_mode = CL_FILTER_NEAREST;
      break;
    case hipFilterModeLinear:
      filter_mode = CL_FILTER_LINEAR;
      break;
  }

  uint32_t address_mode = CL_ADDRESS_NONE;
  switch (addressMode) {
    case hipAddressModeWrap:
      address_mode = CL_ADDRESS_REPEAT;
      break;
    case hipAddressModeClamp:
      address_mode = CL_ADDRESS_CLAMP;
      break;
    case hipAddressModeMirror:
      address_mode = CL_ADDRESS_MIRRORED_REPEAT;
      break;
    case hipAddressModeBorder:
      address_mode = CL_ADDRESS_CLAMP_TO_EDGE;
      break;
  }
  amd::Sampler* sampler =  new amd::Sampler(*hip::getCurrentContext(),
                          normalizedCoords == CL_TRUE,
                          address_mode, filter_mode, CL_FILTER_NONE, 0.f, CL_MAXFLOAT);
  if (sampler == nullptr) {
    return nullptr;
  }
  if (!sampler->create()) {
    delete sampler;
    return nullptr;
  }
  return sampler;
}

hip::TextureObject* ihipCreateTextureObject(const hipResourceDesc& resDesc, amd::Image& image, amd::Sampler& sampler) {
  hip::TextureObject* texture;
  ihipMalloc(reinterpret_cast<void**>(&texture), sizeof(hip::TextureObject), CL_MEM_SVM_FINE_GRAIN_BUFFER);

  if (texture == nullptr) {
    return nullptr;
  }

  device::Memory* imageMem = image.getDeviceMemory(*hip::getCurrentContext()->devices()[0]);
  memcpy(texture->imageSRD, imageMem->cpuSrd(), sizeof(uint32_t)*HIP_IMAGE_OBJECT_SIZE_DWORD);
  texture->image = &image;

  device::Sampler* devSampler = sampler.getDeviceSampler(*hip::getCurrentContext()->devices()[0]);
  memcpy(texture->samplerSRD, devSampler->hwState(), sizeof(uint32_t)*HIP_SAMPLER_OBJECT_SIZE_DWORD);
  texture->sampler = &sampler;

  memcpy(&texture->resDesc, &resDesc, sizeof(hipResourceDesc));

  return texture;
}

hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const hipResourceViewDesc* pResViewDesc) {
  HIP_INIT_API(NONE, pTexObject, pResDesc, pTexDesc, pResViewDesc);

  amd::Device* device = hip::getCurrentContext()->devices()[0];

  if (!device->info().imageSupport_) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Image* image = nullptr;

  cl_image_format image_format;
  getChannelOrderAndType(pResDesc->res.pitch2D.desc, pTexDesc->readMode,
    &image_format.image_channel_order, &image_format.image_channel_data_type);

  const amd::Image::Format imageFormat(image_format);

  amd::Memory* memory = nullptr;
  size_t offset = 0;
  switch (pResDesc->resType) {
    case hipResourceTypeArray:
      {
        memory = getMemoryObject(pResDesc->res.array.array->data, offset);

        getChannelOrderAndType(pResDesc->res.array.array->desc, pTexDesc->readMode,
                             &image_format.image_channel_order, &image_format.image_channel_data_type);
        const amd::Image::Format imageFormat(image_format);
        switch (pResDesc->res.array.array->type) {
          case hipArrayLayered:
          case hipArrayCubemap:
            assert(0);
            break;
          case hipArraySurfaceLoadStore:
          case hipArrayTextureGather:
          case hipArrayDefault:
          default:
            image = new (*hip::getCurrentContext()) amd::Image(*memory->asBuffer(),
              CL_MEM_OBJECT_IMAGE2D, memory->getMemFlags(), imageFormat,
              pResDesc->res.array.array->width, pResDesc->res.array.array->height, 1, 0, 0);
            break;
        }
      }
      break;
    case hipResourceTypeMipmappedArray:
      assert(0);
      break;
    case hipResourceTypeLinear:
      {
        assert(pResViewDesc == nullptr);
        memory = getMemoryObject(pResDesc->res.linear.devPtr, offset);

        getChannelOrderAndType(pResDesc->res.linear.desc, pTexDesc->readMode,
                             &image_format.image_channel_order, &image_format.image_channel_data_type);
        const amd::Image::Format imageFormat(image_format);

        image = new (*hip::getCurrentContext()) amd::Image(*memory->asBuffer(),
          CL_MEM_OBJECT_IMAGE2D, memory->getMemFlags(), imageFormat,
          pResDesc->res.linear.sizeInBytes / imageFormat.getElementSize(), 1, 1,
          pResDesc->res.linear.sizeInBytes, 0);
      }
      break;
    case hipResourceTypePitch2D:
      assert(pResViewDesc == nullptr);
      memory = getMemoryObject(pResDesc->res.pitch2D.devPtr, offset);

      image = new (*hip::getCurrentContext()) amd::Image(*memory->asBuffer(),
        CL_MEM_OBJECT_IMAGE2D, memory->getMemFlags(), imageFormat,
        pResDesc->res.pitch2D.width, pResDesc->res.pitch2D.height, 1,
        pResDesc->res.pitch2D.pitchInBytes, 0);
      break;
    default: HIP_RETURN(hipErrorInvalidValue);
  }

  if (!image->create()) {
    delete image;
    HIP_RETURN(hipErrorMemoryAllocation);
  }

  amd::Sampler* sampler = fillSamplerDescriptor(pTexDesc->addressMode[0], pTexDesc->filterMode, pTexDesc->normalizedCoords);

  *pTexObject = reinterpret_cast<hipTextureObject_t>(ihipCreateTextureObject(*pResDesc, *image, *sampler));

  HIP_RETURN(hipSuccess);
}

void ihipDestroyTextureObject(hip::TextureObject* texture) {
  texture->image->release();
  texture->sampler->release();

  hipFree(texture);
}

hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
  HIP_INIT_API(NONE, textureObject);

  hip::TextureObject* texture = reinterpret_cast<hip::TextureObject*>(textureObject);

  ihipDestroyTextureObject(texture);

  HIP_RETURN(hipSuccess);
}

hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject) {
  HIP_INIT_API(NONE, pResDesc, textureObject);

  hip::TextureObject* texture = reinterpret_cast<hip::TextureObject*>(textureObject);

  if (pResDesc != nullptr && texture != nullptr) {
    memcpy(pResDesc, &(texture->resDesc), sizeof(hipResourceDesc));
  }

  HIP_RETURN(hipErrorInvalidValue);
}

hipError_t hipGetTextureObjectResourceViewDesc(hipResourceViewDesc* pResViewDesc,
                                               hipTextureObject_t textureObject) {
  HIP_INIT_API(NONE, pResViewDesc, textureObject);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc,
                                          hipTextureObject_t textureObject) {
  HIP_INIT_API(NONE, pTexDesc, textureObject);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t ihipBindTexture(cl_mem_object_type type,
                           size_t* offset, textureReference* tex, const void* devPtr,
                           const hipChannelFormatDesc* desc, size_t width, size_t height,
                           size_t pitch) {
  if (tex == nullptr) {
    return hipErrorInvalidImage;
  }
  if (hip::getCurrentContext()) {
    cl_image_format image_format;
    size_t byteSize;
    size_t rowPitch = 0;
    size_t depth = 0;
    size_t slicePitch = 0;

    getChannelOrderAndType(*desc, hipReadModeElementType,
      &image_format.image_channel_order, &image_format.image_channel_data_type);
    getByteSizeFromChannelFormatKind(desc->f, &byteSize);
    const amd::Image::Format imageFormat(image_format);
    amd::Memory* memory = getMemoryObject(devPtr, *offset);

    switch (type) {
       case CL_MEM_OBJECT_IMAGE3D:
         rowPitch = width * byteSize;
         depth = pitch;
         slicePitch = rowPitch * height;
         break;
       case CL_MEM_OBJECT_IMAGE2D:
       default:
         rowPitch = pitch;
         depth = 1;
         slicePitch = 0;
         break;
    }

    amd::Image* image = new (*hip::getCurrentContext()) amd::Image(*memory->asBuffer(),
                type, memory->getMemFlags(), imageFormat, width, height, depth, rowPitch, slicePitch);
    if (!image->create()) {
      delete image;
      return hipErrorMemoryAllocation;
    }

    *offset = 0;
    if (tex->textureObject) {
      ihipDestroyTextureObject(reinterpret_cast<hip::TextureObject*>(tex->textureObject));
    }
    amd::Sampler* sampler = fillSamplerDescriptor(tex->addressMode[0], tex->filterMode, tex->normalized);

    hipResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(hipResourceDesc));
    switch (type) {
      case CL_MEM_OBJECT_IMAGE1D:
        resDesc.resType = hipResourceTypeLinear;
        resDesc.res.linear.devPtr = const_cast<void*>(devPtr);
        resDesc.res.linear.desc = *desc;
        resDesc.res.linear.sizeInBytes = image->getSize();
        break;
      case CL_MEM_OBJECT_IMAGE2D:
        resDesc.resType = hipResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = const_cast<void*>(devPtr);
        resDesc.res.pitch2D.desc = *desc;
        resDesc.res.pitch2D.width = width;
        resDesc.res.pitch2D.height = height;
        resDesc.res.pitch2D.pitchInBytes = pitch;
        break;
      case CL_MEM_OBJECT_IMAGE3D:
        resDesc.resType = hipResourceTypeArray;
        resDesc.res.array.array = (hipArray*)malloc(sizeof(hipArray));
        resDesc.res.array.array->desc = *desc;
        resDesc.res.array.array->width = width;
        resDesc.res.array.array->height = height;
        resDesc.res.array.array->depth = depth;
        resDesc.res.array.array->Format = tex->format;
        resDesc.res.array.array->NumChannels = tex->numChannels;
        resDesc.res.array.array->isDrv = false;
        resDesc.res.array.array->textureType = hipTextureType3D;
        resDesc.res.array.array->data = const_cast<void*>(devPtr);
        break;
      default:
        resDesc.resType = hipResourceTypeArray;
        resDesc.res.array.array = nullptr;
        break;
    }

    tex->textureObject = reinterpret_cast<hipTextureObject_t>(ihipCreateTextureObject(resDesc, *image, *sampler));
    if(type == CL_MEM_OBJECT_IMAGE3D) {
      free(resDesc.res.array.array);
    }
    memset(&resDesc, 0, sizeof(hipResourceDesc));
    return hipSuccess;
  }
  return hipErrorInvalidValue;
}

hipError_t hipBindTexture(size_t* offset, textureReference* tex, const void* devPtr,
                          const hipChannelFormatDesc* desc, size_t size) {
  HIP_INIT_API(NONE, offset, tex, devPtr, desc, size);

  if (desc == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  cl_image_format image_format;
  getChannelOrderAndType(*desc, hipReadModeElementType,
    &image_format.image_channel_order, &image_format.image_channel_data_type);
  const amd::Image::Format imageFormat(image_format);

  HIP_RETURN(ihipBindTexture(CL_MEM_OBJECT_IMAGE1D, offset, tex, devPtr, desc, size / imageFormat.getElementSize(), 1, size));
}

hipError_t hipBindTexture2D(size_t* offset, textureReference* tex, const void* devPtr,
                            const hipChannelFormatDesc* desc, size_t width, size_t height,
                            size_t pitch) {
  HIP_INIT_API(NONE, offset, tex, devPtr, desc, width, height, pitch);

  HIP_RETURN(ihipBindTexture(CL_MEM_OBJECT_IMAGE2D, offset, tex, devPtr, desc, width, height, pitch));
}

hipError_t hipBindTextureToArray(textureReference* tex, hipArray_const_t array,
                                 const hipChannelFormatDesc* desc) {
  HIP_INIT_API(NONE, tex, array, desc);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t ihipBindTextureImpl(TlsData* tls, int dim, enum hipTextureReadMode readMode, size_t* offset,
                               const void* devPtr, const struct hipChannelFormatDesc* desc,
                               size_t size, textureReference* tex) {
  HIP_INIT_API(NONE, dim, readMode, offset, devPtr, size, tex);

  assert(1 == dim);

  HIP_RETURN(ihipBindTexture(CL_MEM_OBJECT_IMAGE1D, offset, tex, devPtr, desc, size, 1, 0));
}

hipError_t ihipBindTextureToArrayImpl(TlsData* tls, int dim, enum hipTextureReadMode readMode,
                                      hipArray_const_t array,
                                      const struct hipChannelFormatDesc& desc,
                                      textureReference* tex) {
  HIP_INIT_API(NONE, dim, readMode, &desc, array, tex);

  cl_mem_object_type clType;
  size_t offset = 0;

  switch (dim) {
    case 1:
      clType = CL_MEM_OBJECT_IMAGE1D;
      break;
    case 2:
      clType = CL_MEM_OBJECT_IMAGE2D;
      break;
    case 3:
      clType = CL_MEM_OBJECT_IMAGE3D;
      break;
    default:
      HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipBindTexture(clType, &offset, tex, array->data, &desc, array->width,
                             array->height, array->depth));
}

hipError_t hipBindTextureToMipmappedArray(textureReference* tex,
                                          hipMipmappedArray_const_t mipmappedArray,
                                          const hipChannelFormatDesc* desc) {
  HIP_INIT_API(NONE, tex, mipmappedArray, desc);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t ihipUnbindTextureImpl(const hipTextureObject_t& textureObject) {

  ihipDestroyTextureObject(reinterpret_cast<hip::TextureObject*>(textureObject));

  return hipSuccess;
}

hipError_t hipUnbindTexture(const textureReference* tex) {
  HIP_INIT_API(NONE, tex);

  ihipDestroyTextureObject(reinterpret_cast<hip::TextureObject*>(tex->textureObject));

  HIP_RETURN(hipSuccess);
}

hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array) {
  HIP_INIT_API(NONE, desc, array);

  if (desc != nullptr) {
    *desc = array->desc;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipGetTextureAlignmentOffset(size_t* offset, const textureReference* tex) {
  HIP_INIT_API(NONE, offset, tex);

  if ((offset == nullptr) || (tex == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *offset = 0;

  HIP_RETURN(hipSuccess);
}

hipError_t hipGetTextureReference(const textureReference** tex, const void* symbol) {
  HIP_INIT_API(NONE, tex, symbol);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipTexRefSetFormat(textureReference* tex, hipArray_Format fmt, int NumPackedComponents) {
  HIP_INIT_API(NONE, tex, fmt, NumPackedComponents);

  if (tex == nullptr) {
    HIP_RETURN(hipErrorInvalidImage);
  }

  tex->format = fmt;
  tex->numChannels = NumPackedComponents;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetFlags(textureReference* tex, unsigned int flags) {
  HIP_INIT_API(NONE, tex, flags);

  if (tex == nullptr) {
    HIP_RETURN(hipErrorInvalidImage);
  }

  tex->normalized = flags;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetFilterMode(textureReference* tex, hipTextureFilterMode fm) {
  HIP_INIT_API(NONE, tex, fm);

  if (tex == nullptr) {
    HIP_RETURN(hipErrorInvalidImage);
  }

  tex->filterMode = fm;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetAddressMode(hipTextureAddressMode* am, textureReference tex, int dim) {
  HIP_INIT_API(NONE, am, &tex, dim);

  if ((am == nullptr) || (dim >= 3)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *am = tex.addressMode[dim];

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetAddressMode(textureReference* tex, int dim, hipTextureAddressMode am) {
  HIP_INIT_API(NONE, tex, dim, am);

  if (tex == nullptr) {
    HIP_RETURN(hipErrorInvalidImage);
  }

  tex->addressMode[dim] = am;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetArray(hipArray_t* array, textureReference tex) {
  HIP_INIT_API(NONE, array, &tex);

  hip::TextureObject* texture = nullptr;

  if ((array == nullptr) || (*array == nullptr)) {
    HIP_RETURN(hipErrorInvalidImage);
  }

  texture = reinterpret_cast<hip::TextureObject *>(tex.textureObject);
  if(hipResourceTypeArray != texture->resDesc.resType){
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (texture->resDesc.res.array.array == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  **array = *(texture->resDesc.res.array.array);

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetArray(textureReference* tex, hipArray_const_t array, unsigned int flags) {
  HIP_INIT_API(NONE, tex, array, flags);

  size_t offset = 0;

  if ((tex == nullptr) || (array == nullptr)) {
    HIP_RETURN(hipErrorInvalidImage);
  }

  HIP_RETURN(ihipBindTexture(CL_MEM_OBJECT_IMAGE2D, &offset, tex, array->data, &array->desc, array->width,
                             array->height, 0));
}

hipError_t hipTexRefGetAddress(hipDeviceptr_t* dev_ptr, textureReference tex) {
  HIP_INIT_API(NONE, dev_ptr, &tex);

  hip::TextureObject* texture = nullptr;
  device::Memory* dev_mem = nullptr;

  texture = reinterpret_cast<hip::TextureObject *>(tex.textureObject);
  if ((texture == nullptr) || (texture->image == nullptr)) {
    HIP_RETURN(hipErrorInvalidImage);
  }

  dev_mem = texture->image->getDeviceMemory(*hip::getCurrentContext()->devices()[0]);
  if (dev_mem == nullptr) {
    HIP_RETURN(hipErrorInvalidImage);
  }

  *dev_ptr = reinterpret_cast<void*>(dev_mem->virtualAddress());

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetAddress(size_t* offset, textureReference* tex, hipDeviceptr_t devPtr,
                               size_t size) {
  HIP_INIT_API(NONE, offset, tex, devPtr, size);

  if (tex == nullptr) {
    HIP_RETURN(hipErrorInvalidImage);
  }

  cl_image_format image_format;
  getDrvChannelOrderAndType(tex->format, tex->numChannels,
    &image_format.image_channel_order, &image_format.image_channel_data_type);
  const amd::Image::Format imageFormat(image_format);

  HIP_RETURN(ihipBindTexture(CL_MEM_OBJECT_IMAGE1D, offset, tex, devPtr, &tex->channelDesc, size / imageFormat.getElementSize(), 1, size));
}

hipError_t hipTexRefSetAddress2D(textureReference* tex, const HIP_ARRAY_DESCRIPTOR* desc,
                                 hipDeviceptr_t devPtr, size_t pitch) {
  HIP_INIT_API(NONE, tex, desc, devPtr, pitch);

  if (desc == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  size_t offset;
  HIP_RETURN(ihipBindTexture(CL_MEM_OBJECT_IMAGE2D, &offset, tex, devPtr, &tex->channelDesc, desc->Width, desc->Height, pitch));
}
