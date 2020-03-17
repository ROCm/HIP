/* Copyright (c) 2015-present Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>
#include <hip/hcc_detail/texture_types.h>
#include "hip_internal.hpp"
#include "hip_conversions.hpp"
#include "platform/sampler.hpp"

namespace hip {
  struct TextureObject {
    uint32_t imageSRD[HIP_IMAGE_OBJECT_SIZE_DWORD];
    uint32_t samplerSRD[HIP_SAMPLER_OBJECT_SIZE_DWORD];
    amd::Image* image;
    amd::Sampler* sampler;
    hipResourceDesc resDesc;
    hipTextureDesc texDesc;
    hipResourceViewDesc resViewDesc;

    TextureObject(amd::Image* image_,
                  amd::Sampler* sampler_,
                  const hipResourceDesc& resDesc_,
                  const hipTextureDesc& texDesc_,
                  const hipResourceViewDesc& resViewDesc_) :
      image(image_),
      sampler(sampler_),
      resDesc(resDesc_),
      texDesc(texDesc_),
      resViewDesc(resViewDesc_) {
      amd::Context& context = *hip::getCurrentDevice()->asContext();
      amd::Device& device = *context.devices()[0];

      device::Memory* imageMem = image->getDeviceMemory(device);
      std::memcpy(imageSRD, imageMem->cpuSrd(), sizeof(imageSRD));

      device::Sampler* samplerMem = sampler->getDeviceSampler(device);
      std::memcpy(samplerSRD, samplerMem->hwState(), sizeof(samplerSRD));
    }
  };
};

amd::Image* ihipImageCreate(const cl_channel_order channelOrder,
                            const cl_channel_type channelType,
                            const cl_mem_object_type imageType,
                            const size_t imageWidth,
                            const size_t imageHeight,
                            const size_t imageDepth,
                            const size_t imageArraySize,
                            const size_t imageRowPitch,
                            const size_t imageSlicePitch,
                            const uint32_t numMipLevels,
                            amd::Memory* buffer);

hipError_t ihipCreateTextureObject(hipTextureObject_t* pTexObject,
                                   const hipResourceDesc* pResDesc,
                                   const hipTextureDesc* pTexDesc,
                                   const hipResourceViewDesc* pResViewDesc) {
  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  const device::Info& info = device->info();

  // pResViewDesc can only be specified if the type of resource is a HIP array or a HIP mipmapped array.
  if ((pResViewDesc != nullptr) &&
      ((pResDesc->resType != hipResourceTypeArray) && (pResDesc->resType != hipResourceTypeMipmappedArray))) {
    return hipErrorInvalidValue;
  }

  // If hipResourceDesc::resType is set to hipResourceTypeArray,
  // hipResourceDesc::res::array::array must be set to a valid HIP array handle.
  if ((pResDesc->resType == hipResourceTypeArray) &&
      (pResDesc->res.array.array == nullptr)) {
    return hipErrorInvalidValue;
  }

  // If hipResourceDesc::resType is set to hipResourceTypeMipmappedArray,
  // hipResourceDesc::res::mipmap::mipmap must be set to a valid HIP mipmapped array handle
  // and hipTextureDesc::normalizedCoords must be set to true.
  if ((pResDesc->resType == hipResourceTypeMipmappedArray) &&
      ((pResDesc->res.mipmap.mipmap == nullptr) || (pTexDesc->normalizedCoords == 0))) {
    return hipErrorInvalidValue;
  }

  // If hipResourceDesc::resType is set to hipResourceTypeLinear,
  // hipResourceDesc::res::linear::devPtr must be set to a valid device pointer, that is aligned to hipDeviceProp::textureAlignment.
  // The total number of elements in the linear address range cannot exceed hipDeviceProp::maxTexture1DLinear.
  if ((pResDesc->resType == hipResourceTypeLinear) &&
      ((pResDesc->res.linear.devPtr == nullptr) ||
       (!amd::isMultipleOf(pResDesc->res.linear.devPtr, info.imageBaseAddressAlignment_)) ||
       (pResDesc->res.linear.sizeInBytes >= info.imageMaxBufferSize_))) {
    return hipErrorInvalidValue;
  }

  // If hipResourceDesc::resType is set to hipResourceTypePitch2D,
  // hipResourceDesc::res::pitch2D::devPtr must be set to a valid device pointer, that is aligned to hipDeviceProp::textureAlignment.
  // hipResourceDesc::res::pitch2D::width and hipResourceDesc::res::pitch2D::height specify the width and height of the array in elements,
  // and cannot exceed hipDeviceProp::maxTexture2DLinear[0] and hipDeviceProp::maxTexture2DLinear[1] respectively.
  // hipResourceDesc::res::pitch2D::pitchInBytes specifies the pitch between two rows in bytes and has to be aligned to hipDeviceProp::texturePitchAlignment.
  // Pitch cannot exceed hipDeviceProp::maxTexture2DLinear[2].
  if ((pResDesc->resType == hipResourceTypePitch2D) &&
      ((pResDesc->res.pitch2D.devPtr == nullptr) ||
       (!amd::isMultipleOf(pResDesc->res.pitch2D.devPtr, info.imageBaseAddressAlignment_)) ||
       (pResDesc->res.pitch2D.width >= info.image2DMaxWidth_) ||
       (pResDesc->res.pitch2D.height >= info.image2DMaxHeight_) ||
       (!amd::isMultipleOf(pResDesc->res.pitch2D.pitchInBytes, info.imagePitchAlignment_)))) {
    // TODO check pitch limits.
    return hipErrorInvalidValue;
  }

  // Mipmaps are currently not supported.
  if (pResDesc->resType == hipResourceTypeMipmappedArray) {
    return hipErrorNotSupported;
  }
  // We don't program the border_color_ptr field in the HW sampler SRD.
  if (pTexDesc->addressMode[0] == hipAddressModeBorder) {
    return hipErrorNotSupported;
  }
  // We don't program the force_degamma/skip_degamma fields in the HW sampler SRD.
  if (pTexDesc->sRGB == 1) {
    return hipErrorNotSupported;
  }
  // We don't program the max_ansio_ratio field in the the HW sampler SRD.
  if (pTexDesc->maxAnisotropy != 0) {
    return hipErrorNotSupported;
  }
  // We don't program the lod_bias field in the HW sampler SRD.
  if (pTexDesc->mipmapLevelBias != 0) {
    return hipErrorNotSupported;
  }
  // We don't program the min_lod field in the HW sampler SRD.
  if (pTexDesc->minMipmapLevelClamp != 0) {
    return hipErrorNotSupported;
  }
  // We don't program the max_lod field in the HW sampler SRD.
  if (pTexDesc->maxMipmapLevelClamp != 0) {
    return hipErrorNotSupported;
  }

  // TODO VDI assumes all dimensions have the same addressing mode.
  cl_addressing_mode addressMode = CL_ADDRESS_NONE;
  // If hipTextureDesc::normalizedCoords is set to zero,
  // hipAddressModeWrap and hipAddressModeMirror won't be supported
  // and will be switched to hipAddressModeClamp.
  if ((pTexDesc->normalizedCoords == 0) &&
      ((pTexDesc->addressMode[0] == hipAddressModeWrap) || (pTexDesc->addressMode[0] == hipAddressModeMirror))) {
    addressMode = hip::getCLAddressingMode(hipAddressModeClamp);
  }
  // hipTextureDesc::addressMode is ignored if hipResourceDesc::resType is hipResourceTypeLinear
  else if (pResDesc->resType != hipResourceTypeLinear) {
    addressMode = hip::getCLAddressingMode(pTexDesc->addressMode[0]);
  }

#ifndef CL_FILTER_NONE
#define CL_FILTER_NONE 0x1142
#endif
  cl_filter_mode filterMode = CL_FILTER_NONE;
#undef CL_FILTER_NONE
  // hipTextureDesc::filterMode is ignored if hipResourceDesc::resType is hipResourceTypeLinear.
  if (pResDesc->resType != hipResourceTypeLinear) {
    filterMode = hip::getCLFilterMode(pTexDesc->filterMode);
  }

#ifndef CL_FILTER_NONE
#define CL_FILTER_NONE 0x1142
#endif
  cl_filter_mode mipFilterMode = CL_FILTER_NONE;
#undef CL_FILTER_NONE
  if (pResDesc->resType == hipResourceTypeMipmappedArray) {
    mipFilterMode = hip::getCLFilterMode(pTexDesc->mipmapFilterMode);
  }

  amd::Sampler* sampler = new amd::Sampler(*hip::getCurrentDevice()->asContext(),
                                           pTexDesc->normalizedCoords,
                                           addressMode,
                                           filterMode,
                                           mipFilterMode,
                                           pTexDesc->minMipmapLevelClamp,
                                           pTexDesc->maxMipmapLevelClamp);

  if (sampler == nullptr) {
    return hipErrorOutOfMemory;
  }

  if (!sampler->create()) {
    delete sampler;
    return hipErrorOutOfMemory;
  }

  amd::Image* image = nullptr;
  switch (pResDesc->resType) {
  case hipResourceTypeArray: {
    cl_mem memObj = reinterpret_cast<cl_mem>(pResDesc->res.array.array->data);
    if (!is_valid(memObj)) {
      return hipErrorInvalidValue;
    }
    image = as_amd(memObj)->asImage();

    hipTextureReadMode readMode = pTexDesc->readMode;
    // 32-bit integer format will not be promoted, regardless of whether or not
    // this hipTextureDesc::readMode is set hipReadModeNormalizedFloat is specified.
    if ((hip::getElementSize(pResDesc->res.array.array->Format) == 4) &&
        (pResDesc->res.array.array->Format != HIP_AD_FORMAT_FLOAT)) {
      readMode = hipReadModeElementType;
    }

    // We need to create an image view if the user requested to use normalized pixel values,
    // due to already having the image created with a different format.
    if ((pResViewDesc != nullptr) ||
        (readMode == hipReadModeNormalizedFloat)) {
      // TODO VDI currently right now can only change the format of the image.
      const cl_channel_order channelOrder = (pResViewDesc != nullptr) ? hip::getCLChannelOrder(hip::getNumChannels(pResViewDesc->format)) :
                                                                        hip::getCLChannelOrder(pResDesc->res.array.array->NumChannels);
      const cl_channel_type channelType = (pResViewDesc != nullptr) ? hip::getCLChannelType(hip::getArrayFormat(pResViewDesc->format), readMode) :
                                                                      hip::getCLChannelType(pResDesc->res.array.array->Format, readMode);
      const amd::Image::Format imageFormat(cl_image_format{channelOrder, channelType});
      if (!imageFormat.isValid()) {
        return hipErrorInvalidValue;
      }

      image = image->createView(*hip::getCurrentDevice()->asContext(), imageFormat, nullptr);
      if (image == nullptr) {
        return hipErrorInvalidValue;
      }
    }
    break;
  }
  case hipResourceTypeMipmappedArray: {
    ShouldNotReachHere();
    break;
  }
  case hipResourceTypeLinear: {
    const cl_channel_order channelOrder = hip::getCLChannelOrder(hip::getNumChannels(pResDesc->res.linear.desc));
    const cl_channel_type channelType = hip::getCLChannelType(hip::getArrayFormat(pResDesc->res.linear.desc), pTexDesc->readMode);
    const amd::Image::Format imageFormat({channelOrder, channelType});
    const cl_mem_object_type imageType = hip::getCLMemObjectType(pResDesc->resType);
    size_t offset = 0;
    image = ihipImageCreate(channelOrder,
                            channelType,
                            imageType,
                            (pResDesc->res.linear.sizeInBytes / imageFormat.getElementSize()), /* imageWidth */
                            0, /* imageHeight */
                            0, /* imageDepth */
                            0, /* imageArraySize */
                            0, /* imageRowPitch */
                            0, /* imageSlicePitch */
                            0, /* numMipLevels */
                            getMemoryObject(pResDesc->res.linear.devPtr, offset));
    // TODO take care of non-zero offset.
    assert(offset == 0);
    if (image == nullptr) {
      return hipErrorInvalidValue;
    }
    break;
  }
  case hipResourceTypePitch2D: {
    const cl_channel_order channelOrder = hip::getCLChannelOrder(hip::getNumChannels(pResDesc->res.pitch2D.desc));
    const cl_channel_type channelType = hip::getCLChannelType(hip::getArrayFormat(pResDesc->res.pitch2D.desc), pTexDesc->readMode);
    const cl_mem_object_type imageType = hip::getCLMemObjectType(pResDesc->resType);
    size_t offset = 0;
    image = ihipImageCreate(channelOrder,
                            channelType,
                            imageType,
                            pResDesc->res.pitch2D.width, /* imageWidth */
                            pResDesc->res.pitch2D.height, /* imageHeight */
                            0, /* imageDepth */
                            0, /* imageArraySize */
                            pResDesc->res.pitch2D.pitchInBytes, /* imageRowPitch */
                            0, /* imageSlicePitch */
                            0, /* numMipLevels */
                            getMemoryObject(pResDesc->res.pitch2D.devPtr, offset));
    // TODO take care of non-zero offset.
    assert(offset == 0);
    if (image == nullptr) {
      return hipErrorInvalidValue;
    }
    break;
  }
  }

  void *texObjectBuffer = nullptr;
  ihipMalloc(&texObjectBuffer, sizeof(hip::TextureObject), CL_MEM_SVM_FINE_GRAIN_BUFFER);
  if (texObjectBuffer == nullptr) {
    return hipErrorOutOfMemory;
  }
  *pTexObject = reinterpret_cast<hipTextureObject_t>(new (texObjectBuffer) hip::TextureObject{image, sampler, *pResDesc, *pTexDesc, (pResViewDesc != nullptr) ? *pResViewDesc : hipResourceViewDesc{}});

  return hipSuccess;
}

hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject,
                                  const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const hipResourceViewDesc* pResViewDesc) {
  HIP_INIT_API(hipCreateTextureObject, pTexObject, pResDesc, pTexDesc, pResViewDesc);

  HIP_RETURN(ihipCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc));
}


hipError_t ihipDestroyTextureObject(hipTextureObject_t texObject) {
  if (texObject == nullptr) {
    return hipErrorInvalidValue;
  }

  hip::TextureObject* hipTexObject = reinterpret_cast<hip::TextureObject*>(texObject);
  const hipResourceType type = hipTexObject->resDesc.resType;
  const bool isImageFromBuffer = (type == hipResourceTypeLinear) || (type == hipResourceTypePitch2D);
  const bool isImageView = ((type == hipResourceTypeArray) || (type == hipResourceTypeMipmappedArray)) &&
                           !hipTexObject->image->isParent();
  if (isImageFromBuffer || isImageView) {
    hipTexObject->image->release();
  }

  // TODO Should call ihipFree() to not polute the api trace.
  return hipFree(hipTexObject);
}

hipError_t hipDestroyTextureObject(hipTextureObject_t texObject) {
  HIP_INIT_API(hipDestroyTextureObject, texObject);

  HIP_RETURN(ihipDestroyTextureObject(texObject));
}


hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t texObject) {
  HIP_INIT_API(hipGetTextureObjectResourceDesc, pResDesc, texObject);

  if ((pResDesc == nullptr) ||
      (texObject == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::TextureObject* hipTexObject = reinterpret_cast<hip::TextureObject*>(texObject);
  *pResDesc = hipTexObject->resDesc;

  HIP_RETURN(hipSuccess);
}

hipError_t hipGetTextureObjectResourceViewDesc(hipResourceViewDesc* pResViewDesc,
                                               hipTextureObject_t texObject) {
  HIP_INIT_API(hipGetTextureObjectResourceViewDesc, pResViewDesc, texObject);

  if ((pResViewDesc == nullptr) ||
      (texObject == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::TextureObject* hipTexObject = reinterpret_cast<hip::TextureObject*>(texObject);
  *pResViewDesc = hipTexObject->resViewDesc;

  HIP_RETURN(hipSuccess);
}

hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc,
                                          hipTextureObject_t texObject) {
  HIP_INIT_API(hipGetTextureObjectTextureDesc, pTexDesc, texObject);

  if ((pTexDesc == nullptr) ||
      (texObject == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::TextureObject* hipTexObject = reinterpret_cast<hip::TextureObject*>(texObject);
  *pTexDesc = hipTexObject->texDesc;

  HIP_RETURN(hipSuccess);
}

hipError_t ihipBindTexture(size_t* offset,
                           const textureReference* texref,
                           const void* devPtr,
                           const hipChannelFormatDesc* desc,
                           size_t size,
                           hipTextureReadMode readMode) {
  if ((texref == nullptr) ||
      (devPtr == nullptr) ||
      (desc == nullptr)) {
    return hipErrorInvalidValue;
  }

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texref->textureObject);

  // If the device memory pointer was returned from hipMalloc(),
  // the offset is guaranteed to be 0 and NULL may be passed as the offset parameter.
  // TODO enforce alignment on devPtr.
  if (offset != nullptr) {
    *offset = 0;
  }

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypeLinear;
  resDesc.res.linear.devPtr = const_cast<void*>(devPtr);
  resDesc.res.linear.desc = *desc;
  resDesc.res.linear.sizeInBytes = size;

  hipTextureDesc texDesc = hip::getTextureDesc(texref, readMode);

  return ihipCreateTextureObject(const_cast<hipTextureObject_t*>(&texref->textureObject), &resDesc, &texDesc, nullptr);
}

hipError_t ihipBindTexture2D(size_t* offset,
                             const textureReference* texref,
                             const void* devPtr,
                             const hipChannelFormatDesc* desc,
                             size_t width,
                             size_t height,
                             size_t pitch,
                             hipTextureReadMode readMode) {
  if ((texref == nullptr) ||
      (devPtr == nullptr) ||
      (desc == nullptr)) {
    return hipErrorInvalidValue;
  }

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texref->textureObject);

  // If the device memory pointer was returned from hipMalloc(),
  // the offset is guaranteed to be 0 and NULL may be passed as the offset parameter.
  // TODO enforce alignment on devPtr.
  if (offset != nullptr) {
    *offset = 0;
  }

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = const_cast<void*>(devPtr);
  resDesc.res.pitch2D.desc = *desc;
  resDesc.res.pitch2D.width = width;
  resDesc.res.pitch2D.height = height;
  resDesc.res.pitch2D.pitchInBytes = pitch;

  hipTextureDesc texDesc = hip::getTextureDesc(texref, readMode);

  return ihipCreateTextureObject(const_cast<hipTextureObject_t*>(&texref->textureObject), &resDesc, &texDesc, nullptr);
}

hipError_t hipBindTexture2D(size_t* offset,
                            const textureReference* texref,
                            const void* devPtr,
                            const hipChannelFormatDesc* desc,
                            size_t width,
                            size_t height,
                            size_t pitch) {
  HIP_INIT_API(hipBindTexture2D, offset, texref, devPtr, desc, width, height, pitch);

  // TODO need compiler support to extract the read mode from textureReference.
  HIP_RETURN(ihipBindTexture2D(offset, texref, devPtr, desc, width, height, pitch, hipReadModeElementType));
}

hipError_t ihipBindTextureToArray(const textureReference* texref,
                                  hipArray_const_t array,
                                  const hipChannelFormatDesc* desc,
                                  hipTextureReadMode readMode) {
  if ((texref == nullptr) ||
      (array == nullptr) ||
      (desc == nullptr)) {
    return hipErrorInvalidValue;
  }

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texref->textureObject);

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = const_cast<hipArray_t>(array);

  hipTextureDesc texDesc = hip::getTextureDesc(texref, readMode);

  hipResourceViewFormat format = hip::getResourceViewFormat(*desc);
  hipResourceViewDesc resViewDesc = hip::getResourceViewDesc(array, format);

  return ihipCreateTextureObject(const_cast<hipTextureObject_t*>(&texref->textureObject), &resDesc, &texDesc, &resViewDesc);
}

hipError_t hipBindTextureToArray(const textureReference* texref,
                                 hipArray_const_t array,
                                 const hipChannelFormatDesc* desc) {
  HIP_INIT_API(hipBindTextureToArray, texref, array, desc);

  // TODO need compiler support to extract the read mode from textureReference.
  HIP_RETURN(ihipBindTextureToArray(texref, array, desc, hipReadModeElementType));
}

hipError_t ihipBindTextureImpl(TlsData *tls,
                               int dim,
                               hipTextureReadMode readMode,
                               size_t* offset,
                               const void* devPtr,
                               const hipChannelFormatDesc* desc,
                               size_t size,
                               textureReference* texref) {
  HIP_INIT_API(ihipBindTextureImpl, tls, dim, readMode, offset, devPtr, desc, size, texref);

  (void)dim; // Silence compiler warnings.

  HIP_RETURN(ihipBindTexture(offset, texref, devPtr, desc, size, readMode));
}

hipError_t ihipBindTextureToArrayImpl(TlsData *tls,
                                      int dim,
                                      hipTextureReadMode readMode,
                                      hipArray_const_t array,
                                      const hipChannelFormatDesc& desc,
                                      textureReference* texref) {
  // TODO overload operator<<(ostream&, hipChannelFormatDesc&).
  HIP_INIT_API(ihipBindTextureToArrayImpl, tls, dim, readMode, array, &desc, texref);

  (void)dim; // Silence compiler warnings.

  HIP_RETURN(ihipBindTextureToArray(texref, array, &desc, readMode));
}

hipError_t ihipBindTextureToMipmappedArray(textureReference* texref,
                                           hipMipmappedArray_const_t mipmappedArray,
                                           const hipChannelFormatDesc* desc,
                                           hipTextureReadMode readMode) {
  if ((texref == nullptr) ||
      (mipmappedArray == nullptr) ||
      (desc == nullptr)) {
    return hipErrorInvalidValue;
  }

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texref->textureObject);

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypeMipmappedArray;
  resDesc.res.mipmap.mipmap = const_cast<hipMipmappedArray_t>(mipmappedArray);

  hipTextureDesc texDesc = hip::getTextureDesc(texref, readMode);

  hipResourceViewFormat format = hip::getResourceViewFormat(*desc);
  hipResourceViewDesc resViewDesc = hip::getResourceViewDesc(mipmappedArray, format);

  return ihipCreateTextureObject(const_cast<hipTextureObject_t*>(&texref->textureObject), &resDesc, &texDesc, &resViewDesc);
}

hipError_t hipBindTextureToMipmappedArray(textureReference* texref,
                                          hipMipmappedArray_const_t mipmappedArray,
                                          const hipChannelFormatDesc* desc) {
  HIP_INIT_API(hipBindTextureToMipmappedArray, texref, mipmappedArray, desc);

  // TODO need compiler support to extract the read mode from textureReference.
  HIP_RETURN(ihipBindTextureToMipmappedArray(texref, mipmappedArray, desc, hipReadModeElementType));
}

hipError_t ihipBindTexture2DImpl(int dim,
                                 hipTextureReadMode readMode,
                                 size_t* offset,
                                 const void* devPtr,
                                 const hipChannelFormatDesc* desc,
                                 size_t width,
                                 size_t height,
                                 textureReference* texref,
                                 size_t pitch) {
  HIP_INIT_API(ihipBindTexture2DImpl, dim, readMode, offset, devPtr, desc, width, height, texref, pitch);

  (void)dim; // Silence compiler warnings.

  HIP_RETURN(ihipBindTexture2D(offset, texref, devPtr, desc, width, height, pitch, readMode));
}

hipError_t ihipUnbindTextureImpl(const hipTextureObject_t& textureObject) {
  // TODO overload operator<<(ostream&, hipTextureObject_t&).
  HIP_INIT_API(ihipUnbindTextureImpl, &textureObject);

  HIP_RETURN(ihipDestroyTextureObject(textureObject));
}

hipError_t hipUnbindTexture(const textureReference* texref) {
  HIP_INIT_API(hipUnbindTexture, texref);

  if (texref == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const hipTextureObject_t textureObject = texref->textureObject;
  const_cast<textureReference*>(texref)->textureObject = nullptr;

  HIP_RETURN(ihipDestroyTextureObject(textureObject));
}

hipError_t hipBindTexture(size_t* offset,
                          const textureReference* texref,
                          const void* devPtr,
                          const hipChannelFormatDesc* desc,
                          size_t size) {
  HIP_INIT_API(hipBindTexture, offset, texref, devPtr, desc, size);

  // TODO need compiler support to extract the read mode from textureReference.
  HIP_RETURN(ihipBindTexture(offset, texref, devPtr, desc, size, hipReadModeElementType));
}

hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc,
                             hipArray_const_t array) {
  HIP_INIT_API(hipGetChannelDesc, desc, array);

  if ((desc == nullptr) ||
      (array == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // It is UB to call hipGetChannelDesc() on an array created via hipArrayCreate()/hipArray3DCreate().
  // This is due to hip not differentiating between runtime and driver types.
  *desc = array->desc;

  HIP_RETURN(hipSuccess);
}

hipError_t hipGetTextureAlignmentOffset(size_t* offset,
                                        const textureReference* texref) {
  HIP_INIT_API(hipGetTextureAlignmentOffset, offset, texref);

  if ((offset == nullptr) ||
      (texref == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // TODO enforce alignment on devPtr.
  *offset = 0;

  HIP_RETURN(hipSuccess);
}

hipError_t hipGetTextureReference(const textureReference** texref, const void* symbol) {
  HIP_INIT_API(hipGetTextureReference, texref, symbol);

  assert(0 && "Unimplemented");

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipTexRefSetFormat(textureReference* texRef,
                              hipArray_Format fmt,
                              int NumPackedComponents) {
  HIP_INIT_API(hipTexRefSetFormat, texRef, fmt, NumPackedComponents);

  if (texRef == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  texRef->format = fmt;
  texRef->numChannels = NumPackedComponents;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetFlags(textureReference* texRef,
                             unsigned int Flags) {
  HIP_INIT_API(hipTexRefSetFlags, texRef, Flags);

  if (texRef == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // TODO add textureReference::flags.
  // Using textureReference::normalized for this purpose is OK for now,
  // because calling hipTexRefGetFlags() on a textureRefence after hipBindTexture() is UB
  // due to HIP not differentiating between runtime and driver api.
  texRef->normalized = Flags;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetFilterMode(textureReference* texRef,
                                  hipTextureFilterMode fm) {
  HIP_INIT_API(hipTexRefSetFilterMode, texRef, fm);

  if (texRef == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  texRef->filterMode = fm;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetAddressMode(hipTextureAddressMode* pam,
                                   textureReference texRef,
                                   int dim) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetAddressMode, pam, &texRef, dim);

  if (pam == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Currently, the only valid value for dim are 0 and 1.
  if ((dim != 0) || (dim != 1)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pam = texRef.addressMode[dim];

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetAddressMode(textureReference* texRef,
                                   int dim,
                                   hipTextureAddressMode am) {
  HIP_INIT_API(hipTexRefSetAddressMode, texRef, dim, am);

  if (texRef == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if ((dim < 0) || (dim > 2)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  texRef->addressMode[dim] = am;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetArray(hipArray_t* pArray,
                             textureReference texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetArray, pArray, &texRef);

  if (pArray == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipResourceDesc resDesc = {};
  // TODO use ihipGetTextureObjectResourceDesc() to not pollute the API trace.
  hipError_t error = hipGetTextureObjectResourceDesc(&resDesc, texRef.textureObject);
  if (error != hipSuccess) {
    return HIP_RETURN(error);
  }

  switch (resDesc.resType) {
  case hipResourceTypeLinear:
  case hipResourceTypePitch2D:
  case hipResourceTypeMipmappedArray:
    HIP_RETURN(hipErrorInvalidValue);
  case hipResourceTypeArray:
    *pArray = resDesc.res.array.array;
    break;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetArray(textureReference* texRef,
                             hipArray_const_t array,
                             unsigned int flags) {
  HIP_INIT_API(hipTexRefSetArray, texRef, array, flags);

  if ((texRef == nullptr) ||
      (array == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (flags != HIP_TRSA_OVERRIDE_FORMAT) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texRef->textureObject);

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = const_cast<hipArray_t>(array);

  // TODO need compiler support to extract the read mode from textureReference.
  hipTextureDesc texDesc = hip::getTextureDesc(texRef, hipReadModeElementType);

  hipResourceViewFormat format = hip::getResourceViewFormat(hip::getChannelFormatDesc(texRef->numChannels, texRef->format));
  hipResourceViewDesc resViewDesc = hip::getResourceViewDesc(array, format);

  HIP_RETURN(ihipCreateTextureObject(&texRef->textureObject, &resDesc, &texDesc, &resViewDesc));
}

hipError_t hipTexRefGetAddress(hipDeviceptr_t* dptr,
                               textureReference texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetAddress, dptr, &texRef);

  if (dptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipResourceDesc resDesc = {};
  // TODO use ihipGetTextureObjectResourceDesc() to not pollute the API trace.
  hipError_t error = hipGetTextureObjectResourceDesc(&resDesc, texRef.textureObject);
  if (error != hipSuccess) {
    return HIP_RETURN(error);
  }

  switch (resDesc.resType) {
  // Need to verify.
  // If the texture reference is not bound to any device memory range,
  // return hipErroInvalidValue.
  case hipResourceTypeArray:
  case hipResourceTypeMipmappedArray:
    HIP_RETURN(hipErrorInvalidValue);
  case hipResourceTypeLinear:
    *dptr = resDesc.res.linear.devPtr;
    break;
  case hipResourceTypePitch2D:
    *dptr = resDesc.res.pitch2D.devPtr;
    break;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetAddress(size_t* ByteOffset,
                               textureReference* texRef,
                               hipDeviceptr_t dptr,
                               size_t bytes) {
  HIP_INIT_API(hipTexRefSetAddress, ByteOffset, texRef, dptr, bytes);

  if (texRef == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texRef->textureObject);

  // If the device memory pointer was returned from hipMemAlloc(),
  // the offset is guaranteed to be 0 and NULL may be passed as the ByteOffset parameter.
  // TODO enforce alignment on devPtr.
  if (ByteOffset != nullptr) {
    *ByteOffset = 0;
  }

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypeLinear;
  resDesc.res.linear.devPtr = dptr;
  resDesc.res.linear.desc = hip::getChannelFormatDesc(texRef->numChannels, texRef->format);
  resDesc.res.linear.sizeInBytes = bytes;

  // TODO add textureReference::flags.
  // Using textureReference::normalized for this purpose is OK for now,
  // because calling hipTexRefGetFlags() on a textureRefence after hipBindTexture()
  // due to HIP not differentiating between runtime and driver api.
  hipTextureReadMode readMode = hip::getReadMode(texRef->normalized);
  texRef->sRGB = hip::getSRGB(texRef->normalized);
  texRef->normalized = hip::getNormalizedCoords(texRef->normalized);
  hipTextureDesc texDesc = hip::getTextureDesc(texRef, readMode);

  HIP_RETURN(ihipCreateTextureObject(&texRef->textureObject, &resDesc, &texDesc, nullptr));
}

hipError_t hipTexRefSetAddress2D(textureReference* texRef,
                                 const HIP_ARRAY_DESCRIPTOR* desc,
                                 hipDeviceptr_t dptr,
                                 size_t Pitch) {
  HIP_INIT_API(hipTexRefSetAddress2D, texRef, desc, dptr, Pitch);

  if ((texRef == nullptr) ||
      (desc == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texRef->textureObject);

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypePitch2D;
  resDesc.res.linear.devPtr = dptr;
  resDesc.res.linear.desc = hip::getChannelFormatDesc(desc->NumChannels, desc->Format); // Need to verify.
  resDesc.res.pitch2D.width = desc->Width;
  resDesc.res.pitch2D.height = desc->Height;
  resDesc.res.pitch2D.pitchInBytes = Pitch;

  // TODO add textureReference::flags.
  // Using textureReference::normalized for this purpose is OK for now,
  // because calling hipTexRefGetFlags() on a textureRefence after hipBindTexture()
  // due to HIP not differentiating between runtime and driver api.
  hipTextureReadMode readMode = hip::getReadMode(texRef->normalized);
  texRef->sRGB = hip::getSRGB(texRef->normalized);
  texRef->normalized = hip::getNormalizedCoords(texRef->normalized);
  hipTextureDesc texDesc = hip::getTextureDesc(texRef, readMode);

  HIP_RETURN(ihipCreateTextureObject(&texRef->textureObject, &resDesc, &texDesc, nullptr));
}

hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f) {
  return {x, y, z, w, f};
}

hipError_t ihipBindTextureToMipmappedArrayImpl(TlsData *tls,
                                               int dim,
                                               hipTextureReadMode readMode,
                                               hipMipmappedArray_const_t mipmappedArray,
                                               const struct hipChannelFormatDesc& desc,
                                               textureReference* texref) {
  // TODO overload operator<<(ostream&, hipChannelFormatDesc&).
  HIP_INIT_API(ihipBindTextureToMipmappedArrayImpl, tls, dim, readMode, mipmappedArray, &desc, texref);

  (void)dim; // Silence compiler warnings.

  HIP_RETURN(ihipBindTextureToMipmappedArray(texref, mipmappedArray, &desc, readMode));
}

hipError_t hipTexRefGetBorderColor(float* pBorderColor,
                                   textureReference texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetBorderColor, pBorderColor, &texRef);

  if (pBorderColor == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // TODO add textureReference::borderColor.
  assert(false && "textureReference::borderColor is missing in header");
  // std::memcpy(pBorderColor, texRef.borderColor, sizeof(texRef.borderColor));

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetFilterMode(hipTextureFilterMode* pfm,
                                  textureReference texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetFilterMode, pfm, &texRef);

  if (pfm == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pfm = texRef.filterMode;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetFlags(unsigned int* pFlags,
                             textureReference texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetFlags, pFlags, &texRef);

  if (pFlags == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // TODO add textureReference::flags.
  // Using textureReference::normalized for this purpose is OK for now,
  // because calling hipTexRefGetFlags() on a textureRefence after hipBindTexture()
  // due to HIP not differentiating between runtime and driver api.
  *pFlags = texRef.normalized;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetFormat(hipArray_Format* pFormat,
                              int* pNumChannels,
                              textureReference texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetFormat, pFormat, pNumChannels, &texRef);

  if ((pFormat == nullptr) ||
      (pNumChannels == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pFormat = texRef.format;
  *pNumChannels = texRef.numChannels;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetMaxAnisotropy(int* pmaxAnsio,
                                     textureReference texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetMaxAnisotropy, pmaxAnsio, &texRef);

  if (pmaxAnsio == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pmaxAnsio = texRef.maxAnisotropy;

  HIP_RETURN(hipErrorInvalidValue);
}

hipError_t hipTexRefGetMipmapFilterMode(hipTextureFilterMode* pfm,
                                        textureReference texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetMipmapFilterMode, pfm, &texRef);

  if (pfm == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pfm = texRef.mipmapFilterMode;

  HIP_RETURN(hipErrorInvalidValue);
}

hipError_t hipTexRefGetMipmapLevelBias(float* pbias,
                                       textureReference texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetMipmapLevelBias, pbias, &texRef);

  if (pbias == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pbias = texRef.mipmapLevelBias;

  HIP_RETURN(hipErrorInvalidValue);
}

hipError_t hipTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp,
                                        float* pmaxMipmapLevelClamp,
                                        textureReference texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetMipmapLevelClamp, pminMipmapLevelClamp, pmaxMipmapLevelClamp, &texRef);

  if ((pminMipmapLevelClamp == nullptr) ||
      (pmaxMipmapLevelClamp == nullptr)){
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pminMipmapLevelClamp = texRef.minMipmapLevelClamp;
  *pmaxMipmapLevelClamp = texRef.maxMipmapLevelClamp;

  HIP_RETURN(hipErrorInvalidValue);
}

hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t* pArray,
                                      textureReference texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetMipMappedArray, pArray, &texRef);

  if (pArray == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipResourceDesc resDesc = {};
  // TODO use ihipGetTextureObjectResourceDesc() to not pollute the API trace.
  hipError_t error = hipGetTextureObjectResourceDesc(&resDesc, texRef.textureObject);
  if (error != hipSuccess) {
    return HIP_RETURN(error);
  }

  switch (resDesc.resType) {
  case hipResourceTypeLinear:
  case hipResourceTypePitch2D:
  case hipResourceTypeArray:
    HIP_RETURN(hipErrorInvalidValue);
  case hipResourceTypeMipmappedArray:
    *pArray = resDesc.res.mipmap.mipmap;
    break;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetBorderColor(textureReference* texRef,
                                   float* pBorderColor) {
  HIP_INIT_API(hipTexRefSetBorderColor, texRef, pBorderColor);

  if ((texRef == nullptr) ||
      (pBorderColor == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // TODO add textureReference::borderColor.
  assert(false && "textureReference::borderColor is missing in header");
  // std::memcpy(texRef.borderColor, pBorderColor, sizeof(texRef.borderColor));

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetMaxAnisotropy(textureReference* texRef,
                                     unsigned int maxAniso) {
  HIP_INIT_API(hipTexRefSetMaxAnisotropy, texRef, maxAniso);

  if (texRef == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  texRef->maxAnisotropy = maxAniso;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetMipmapFilterMode(textureReference* texRef,
                                        hipTextureFilterMode fm) {
  HIP_INIT_API(hipTexRefSetMipmapFilterMode, texRef, fm);

  if (texRef == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  texRef->mipmapFilterMode = fm;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetMipmapLevelBias(textureReference* texRef,
                                       float bias) {
  HIP_INIT_API(hipTexRefSetMipmapLevelBias, texRef, bias);

  if (texRef == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  texRef->mipmapLevelBias = bias;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetMipmapLevelClamp(textureReference* texRef,
                                        float minMipMapLevelClamp,
                                        float maxMipMapLevelClamp) {
  HIP_INIT_API(hipTexRefSetMipmapLevelClamp, minMipMapLevelClamp, maxMipMapLevelClamp);

  if (texRef == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  texRef->minMipmapLevelClamp = minMipMapLevelClamp;
  texRef->maxMipmapLevelClamp = maxMipMapLevelClamp;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetMipmappedArray(textureReference* texRef,
                                      hipMipmappedArray* mipmappedArray,
                                      unsigned int Flags) {
  HIP_INIT_API(hipTexRefSetMipmappedArray, texRef, mipmappedArray, Flags);

  if ((texRef == nullptr) ||
      (mipmappedArray == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (Flags != HIP_TRSA_OVERRIDE_FORMAT) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texRef->textureObject);

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypeMipmappedArray;
  resDesc.res.mipmap.mipmap = mipmappedArray;

  // TODO need compiler support to extract the read mode from textureReference.
  hipTextureDesc texDesc = hip::getTextureDesc(texRef, hipReadModeElementType);

  hipResourceViewFormat format = hip::getResourceViewFormat(hip::getChannelFormatDesc(texRef->numChannels, texRef->format));
  hipResourceViewDesc resViewDesc = hip::getResourceViewDesc(mipmappedArray, format);

  HIP_RETURN(ihipCreateTextureObject(&texRef->textureObject, &resDesc, &texDesc, &resViewDesc));
}