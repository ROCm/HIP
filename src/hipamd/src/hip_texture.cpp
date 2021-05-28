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
#include <hip/amd_detail/amd_texture_types.h>
#include "hip_internal.hpp"
#include "hip_platform.hpp"
#include "hip_conversions.hpp"
#include "platform/sampler.hpp"

hipError_t ihipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                      amd::HostQueue& queue, bool isAsync = false);

struct __hip_texture {
  uint32_t imageSRD[HIP_IMAGE_OBJECT_SIZE_DWORD];
  uint32_t samplerSRD[HIP_SAMPLER_OBJECT_SIZE_DWORD];
  amd::Image* image;
  amd::Sampler* sampler;
  hipResourceDesc resDesc;
  hipTextureDesc texDesc;
  hipResourceViewDesc resViewDesc;

  __hip_texture(amd::Image* image_,
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

  // Validate input params
  if (pTexObject == nullptr || pResDesc == nullptr || pTexDesc == nullptr) {
    return hipErrorInvalidValue;
  }

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
       ((pResDesc->res.linear.sizeInBytes / hip::getElementSize(pResDesc->res.linear.desc)) >= info.imageMaxBufferSize_))) {
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

  // TODO ROCclr assumes all dimensions have the same addressing mode.
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
    if ((pResDesc->res.array.array->Format == HIP_AD_FORMAT_SIGNED_INT32) ||
        (pResDesc->res.array.array->Format == HIP_AD_FORMAT_UNSIGNED_INT32)) {
      readMode = hipReadModeElementType;
    }

    // We need to create an image view if the user requested to use normalized pixel values,
    // due to already having the image created with a different format.
    if ((pResViewDesc != nullptr) ||
        (readMode == hipReadModeNormalizedFloat) ||
        (pTexDesc->sRGB == 1)) {
      // TODO ROCclr currently right now can only change the format of the image.
      const cl_channel_order channelOrder = (pResViewDesc != nullptr) ? hip::getCLChannelOrder(hip::getNumChannels(pResViewDesc->format), pTexDesc->sRGB) :
                                                                        hip::getCLChannelOrder(pResDesc->res.array.array->NumChannels, pTexDesc->sRGB);
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
    const cl_channel_order channelOrder = hip::getCLChannelOrder(hip::getNumChannels(pResDesc->res.linear.desc), pTexDesc->sRGB);
    const cl_channel_type channelType = hip::getCLChannelType(hip::getArrayFormat(pResDesc->res.linear.desc), pTexDesc->readMode);
    const amd::Image::Format imageFormat({channelOrder, channelType});
    const cl_mem_object_type imageType = hip::getCLMemObjectType(pResDesc->resType);
    const size_t imageSizeInBytes = pResDesc->res.linear.sizeInBytes;
    amd::Memory* buffer = getMemoryObjectWithOffset(pResDesc->res.linear.devPtr, imageSizeInBytes);
    image = ihipImageCreate(channelOrder,
                            channelType,
                            imageType,
                            imageSizeInBytes / imageFormat.getElementSize(), /* imageWidth */
                            0, /* imageHeight */
                            0, /* imageDepth */
                            0, /* imageArraySize */
                            0, /* imageRowPitch */
                            0, /* imageSlicePitch */
                            0, /* numMipLevels */
                            buffer);
    buffer->release();
    if (image == nullptr) {
      return hipErrorInvalidValue;
    }
    break;
  }
  case hipResourceTypePitch2D: {
    const cl_channel_order channelOrder = hip::getCLChannelOrder(hip::getNumChannels(pResDesc->res.pitch2D.desc), pTexDesc->sRGB);
    const cl_channel_type channelType = hip::getCLChannelType(hip::getArrayFormat(pResDesc->res.pitch2D.desc), pTexDesc->readMode);
    const amd::Image::Format imageFormat({channelOrder, channelType});
    const cl_mem_object_type imageType = hip::getCLMemObjectType(pResDesc->resType);
    const size_t imageSizeInBytes = pResDesc->res.pitch2D.width * imageFormat.getElementSize() +
                                    pResDesc->res.pitch2D.pitchInBytes * (pResDesc->res.pitch2D.height - 1);
    amd::Memory* buffer = getMemoryObjectWithOffset(pResDesc->res.pitch2D.devPtr, imageSizeInBytes);
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
                            buffer);
    buffer->release();
    if (image == nullptr) {
      return hipErrorInvalidValue;
    }
    break;
  }
  }

  void *texObjectBuffer = nullptr;
  ihipMalloc(&texObjectBuffer, sizeof(__hip_texture), CL_MEM_SVM_FINE_GRAIN_BUFFER);
  if (texObjectBuffer == nullptr) {
    return hipErrorOutOfMemory;
  }
  *pTexObject = new (texObjectBuffer) __hip_texture{image, sampler, *pResDesc, *pTexDesc, (pResViewDesc != nullptr) ? *pResViewDesc : hipResourceViewDesc{}};

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
    return hipSuccess;
  }

  const hipResourceType type = texObject->resDesc.resType;
  const bool isImageFromBuffer = (type == hipResourceTypeLinear) || (type == hipResourceTypePitch2D);
  const bool isImageView = ((type == hipResourceTypeArray) || (type == hipResourceTypeMipmappedArray)) &&
                           !texObject->image->isParent();
  if (isImageFromBuffer || isImageView) {
    texObject->image->release();
  }

  // TODO Should call ihipFree() to not polute the api trace.
  return hipFree(texObject);
}

hipError_t hipDestroyTextureObject(hipTextureObject_t texObject) {
  HIP_INIT_API(hipDestroyTextureObject, texObject);

  HIP_RETURN(ihipDestroyTextureObject(texObject));
}


hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t texObject) {
  HIP_INIT_API(hipGetTextureObjectResourceDesc, pResDesc, texObject);

  if ((pResDesc == nullptr) || (texObject == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pResDesc = texObject->resDesc;

  HIP_RETURN(hipSuccess);
}

hipError_t hipGetTextureObjectResourceViewDesc(hipResourceViewDesc* pResViewDesc,
                                               hipTextureObject_t texObject) {
  HIP_INIT_API(hipGetTextureObjectResourceViewDesc, pResViewDesc, texObject);

  if ((pResViewDesc == nullptr) || (texObject == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pResViewDesc = texObject->resViewDesc;

  HIP_RETURN(hipSuccess);
}

hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc,
                                          hipTextureObject_t texObject) {
  HIP_INIT_API(hipGetTextureObjectTextureDesc, pTexDesc, texObject);

  if ((pTexDesc == nullptr) || (texObject == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pTexDesc = texObject->texDesc;

  HIP_RETURN(hipSuccess);
}

inline bool ihipGetTextureAlignmentOffset(size_t* offset,
                                          const void* devPtr) {
  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  const device::Info& info = device->info();

  const char* alignedDevPtr = amd::alignUp(static_cast<const char*>(devPtr), info.imageBaseAddressAlignment_);
  const size_t alignedOffset = alignedDevPtr - static_cast<const char*>(devPtr);

  // If the device memory pointer was returned from hipMalloc(),
  // the offset is guaranteed to be 0 and NULL may be passed as the offset parameter.
  if ((alignedOffset != 0) && (offset == nullptr)) {
    LogPrintfError("Texture object not aligned with offset %u \n", alignedOffset);
    return false;
  }

  if (offset != nullptr) {
    *offset = alignedOffset;
  }

  return true;
}

hipError_t ihipBindTexture(size_t* offset,
                           const textureReference* texref,
                           const void* devPtr,
                           const hipChannelFormatDesc* desc,
                           size_t size) {
  if ((texref == nullptr) ||
      (devPtr == nullptr) ||
      (desc == nullptr)) {
    return hipErrorInvalidValue;
  }

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texref->textureObject);

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypeLinear;
  resDesc.res.linear.devPtr = const_cast<void*>(devPtr);
  resDesc.res.linear.desc = *desc;
  resDesc.res.linear.sizeInBytes = size;

  if (ihipGetTextureAlignmentOffset(offset, devPtr)) {
    // Align the user ptr to HW requirments.
    resDesc.res.linear.devPtr = static_cast<char*>(const_cast<void*>(devPtr)) - *offset;
  } else {
    return hipErrorInvalidValue;
  }

  hipTextureDesc texDesc = hip::getTextureDesc(texref);

  return ihipCreateTextureObject(const_cast<hipTextureObject_t*>(&texref->textureObject), &resDesc, &texDesc, nullptr);
}

hipError_t ihipBindTexture2D(size_t* offset,
                             const textureReference* texref,
                             const void* devPtr,
                             const hipChannelFormatDesc* desc,
                             size_t width,
                             size_t height,
                             size_t pitch) {
  if ((texref == nullptr) ||
      (devPtr == nullptr) ||
      (desc == nullptr)) {
    return hipErrorInvalidValue;
  }

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texref->textureObject);

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = const_cast<void*>(devPtr);
  resDesc.res.pitch2D.desc = *desc;
  resDesc.res.pitch2D.width = width;
  resDesc.res.pitch2D.height = height;
  resDesc.res.pitch2D.pitchInBytes = pitch;

  if (ihipGetTextureAlignmentOffset(offset, devPtr)) {
    // Align the user ptr to HW requirments.
    resDesc.res.pitch2D.devPtr = static_cast<char*>(const_cast<void*>(devPtr)) - *offset;
  } else {
    return hipErrorInvalidValue;
  }

  hipTextureDesc texDesc = hip::getTextureDesc(texref);

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

  hipDeviceptr_t refDevPtr = nullptr;
  size_t refDevSize = 0;

  HIP_RETURN_ONFAIL(PlatformState::instance().getStatGlobalVar(texref, ihipGetDevice(), &refDevPtr,
                                                               &refDevSize));

  assert(refDevSize == sizeof(textureReference));
  hipError_t err = ihipBindTexture2D(offset, texref, devPtr, desc, width, height, pitch);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }
  // Copy to device.
  amd::HostQueue* queue = hip::getNullStream();
  HIP_RETURN(ihipMemcpy(refDevPtr, texref, refDevSize, hipMemcpyHostToDevice, *queue));
}

hipError_t ihipBindTextureToArray(const textureReference* texref,
                                  hipArray_const_t array,
                                  const hipChannelFormatDesc* desc) {
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

  hipTextureDesc texDesc = hip::getTextureDesc(texref);

  hipResourceViewFormat format = hip::getResourceViewFormat(*desc);
  hipResourceViewDesc resViewDesc = hip::getResourceViewDesc(array, format);

  return ihipCreateTextureObject(const_cast<hipTextureObject_t*>(&texref->textureObject), &resDesc, &texDesc, &resViewDesc);
}

hipError_t hipBindTextureToArray(const textureReference* texref,
                                 hipArray_const_t array,
                                 const hipChannelFormatDesc* desc) {
  HIP_INIT_API(hipBindTextureToArray, texref, array, desc);

  hipDeviceptr_t refDevPtr = nullptr;
  size_t refDevSize = 0;
  HIP_RETURN_ONFAIL(PlatformState::instance().getStatGlobalVar(texref, ihipGetDevice(), &refDevPtr,
                                                               &refDevSize));

  assert(refDevSize == sizeof(textureReference));
  hipError_t err = ihipBindTextureToArray(texref, array, desc);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }
  // Copy to device.
  amd::HostQueue* queue = hip::getNullStream();
  HIP_RETURN(ihipMemcpy(refDevPtr, texref, refDevSize, hipMemcpyHostToDevice, *queue));
}

hipError_t ihipBindTextureToMipmappedArray(const textureReference* texref,
                                           hipMipmappedArray_const_t mipmappedArray,
                                           const hipChannelFormatDesc* desc) {
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

  hipTextureDesc texDesc = hip::getTextureDesc(texref);

  hipResourceViewFormat format = hip::getResourceViewFormat(*desc);
  hipResourceViewDesc resViewDesc = hip::getResourceViewDesc(mipmappedArray, format);

  return ihipCreateTextureObject(const_cast<hipTextureObject_t*>(&texref->textureObject), &resDesc, &texDesc, &resViewDesc);
}

hipError_t hipBindTextureToMipmappedArray(const textureReference* texref,
                                          hipMipmappedArray_const_t mipmappedArray,
                                          const hipChannelFormatDesc* desc) {
  HIP_INIT_API(hipBindTextureToMipmappedArray, texref, mipmappedArray, desc);

  hipDeviceptr_t refDevPtr = nullptr;
  size_t refDevSize = 0;

  HIP_RETURN_ONFAIL(PlatformState::instance().getStatGlobalVar(texref, ihipGetDevice(), &refDevPtr,
                                                               &refDevSize));

  assert(refDevSize == sizeof(textureReference));
  hipError_t err = ihipBindTextureToMipmappedArray(texref, mipmappedArray, desc);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }
  // Copy to device.
  amd::HostQueue* queue = hip::getNullStream();
  HIP_RETURN(ihipMemcpy(refDevPtr, texref, refDevSize, hipMemcpyHostToDevice, *queue));
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

  hipDeviceptr_t refDevPtr = nullptr;
  size_t refDevSize = 0;
  HIP_RETURN_ONFAIL(PlatformState::instance().getStatGlobalVar(texref, ihipGetDevice(), &refDevPtr,
                                                               &refDevSize));
  assert(refDevSize == sizeof(textureReference));
  hipError_t err = ihipBindTexture(offset, texref, devPtr, desc, size);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }
  // Copy to device.
  amd::HostQueue* queue = hip::getNullStream();
  HIP_RETURN(ihipMemcpy(refDevPtr, texref, refDevSize, hipMemcpyHostToDevice, *queue));
}

hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc,
                             hipArray_const_t array) {
  HIP_INIT_API(hipGetChannelDesc, desc, array);

  if ((desc == nullptr) || (array == nullptr)) {
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

  if ((offset == nullptr) || (texref == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // TODO enforce alignment on devPtr.
  *offset = 0;

  HIP_RETURN(hipSuccess);
}

hipError_t hipGetTextureReference(const textureReference** texref, const void* symbol) {
  HIP_INIT_API(hipGetTextureReference, texref, symbol);

  if (texref == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *texref = reinterpret_cast<const textureReference *>(symbol);

  HIP_RETURN(hipSuccess);
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

  texRef->readMode = hipReadModeNormalizedFloat;
  texRef->normalized = 0;
  texRef->sRGB = 0;

  if (Flags & HIP_TRSF_READ_AS_INTEGER) {
    texRef->readMode = hipReadModeElementType;
  }

  if (Flags & HIP_TRSF_NORMALIZED_COORDINATES) {
    texRef->normalized = 1;
  }

  if (Flags & HIP_TRSF_SRGB) {
    texRef->sRGB = 1;
  }

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
                                   const textureReference* texRef,
                                   int dim) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetAddressMode, pam, texRef, dim);

  if ((pam == nullptr) || (texRef == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Currently, the only valid value for dim are 0 and 1.
  if ((dim != 0) && (dim != 1)) {
    LogPrintfError(
        "Currently only 2 dimensions (0,1) are valid,"
        "dim : %d \n",
        dim);
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pam = texRef->addressMode[dim];

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
    LogPrintfError(
        "Currently only 3 dimensions (0,1,2) are valid,"
        "dim : %d \n",
        dim);
    HIP_RETURN(hipErrorInvalidValue);
  }

  texRef->addressMode[dim] = am;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetArray(hipArray_t* pArray,
                             const textureReference* texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetArray, pArray, texRef);

  if ((pArray == nullptr) || (texRef == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipResourceDesc resDesc = {};
  // TODO use ihipGetTextureObjectResourceDesc() to not pollute the API trace.
  hipError_t error = hipGetTextureObjectResourceDesc(&resDesc, texRef->textureObject);
  if (error != hipSuccess) {
    HIP_RETURN(error);
  }

  switch (resDesc.resType) {
  case hipResourceTypeLinear:
  case hipResourceTypePitch2D:
  case hipResourceTypeMipmappedArray: {
    HIP_RETURN(hipErrorInvalidValue);
  }
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

  if ((texRef == nullptr) || (array == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (flags != HIP_TRSA_OVERRIDE_FORMAT) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipDeviceptr_t refDevPtr = nullptr;
  size_t refDevSize = 0;

  HIP_RETURN_ONFAIL(PlatformState::instance().getDynTexGlobalVar(texRef, &refDevPtr, &refDevSize));
  assert(refDevSize == sizeof(textureReference));

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texRef->textureObject);

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = const_cast<hipArray_t>(array);

  hipTextureDesc texDesc = hip::getTextureDesc(texRef);

  hipResourceViewFormat format = hip::getResourceViewFormat(hip::getChannelFormatDesc(texRef->numChannels, texRef->format));
  hipResourceViewDesc resViewDesc = hip::getResourceViewDesc(array, format);

  hipError_t err = ihipCreateTextureObject(&texRef->textureObject, &resDesc, &texDesc, &resViewDesc);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }
  // Copy to device.
  amd::HostQueue* queue = hip::getNullStream();
  HIP_RETURN(ihipMemcpy(refDevPtr, texRef, refDevSize, hipMemcpyHostToDevice, *queue));
}

hipError_t hipTexRefGetAddress(hipDeviceptr_t* dptr,
                               const textureReference* texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetAddress, dptr, texRef);

  if ((dptr == nullptr) || (texRef == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipResourceDesc resDesc = {};
  // TODO use ihipGetTextureObjectResourceDesc() to not pollute the API trace.
  hipError_t error = hipGetTextureObjectResourceDesc(&resDesc, texRef->textureObject);
  if (error != hipSuccess) {
    LogPrintfError("hipGetTextureObjectResourceDesc failed with error code: %s \n",
                   hipGetErrorName(error));
    HIP_RETURN(error);
  }

  switch (resDesc.resType) {
  // Need to verify.
  // If the texture reference is not bound to any device memory range,
  // return hipErroInvalidValue.
  case hipResourceTypeArray:
  case hipResourceTypeMipmappedArray: {
    HIP_RETURN(hipErrorInvalidValue);
  }
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

  hipDeviceptr_t refDevPtr = nullptr;
  size_t refDevSize = 0;
  HIP_RETURN_ONFAIL(PlatformState::instance().getDynTexGlobalVar(texRef, &refDevPtr, &refDevSize));
  assert(refDevSize == sizeof(textureReference));

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texRef->textureObject);

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypeLinear;
  resDesc.res.linear.devPtr = dptr;
  resDesc.res.linear.desc = hip::getChannelFormatDesc(texRef->numChannels, texRef->format);
  resDesc.res.linear.sizeInBytes = bytes;

  if (ihipGetTextureAlignmentOffset(ByteOffset, dptr)) {
    // Align the user ptr to HW requirments.
    resDesc.res.linear.devPtr = static_cast<char*>(dptr) - *ByteOffset;
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipTextureDesc texDesc = hip::getTextureDesc(texRef);

  hipError_t err = ihipCreateTextureObject(&texRef->textureObject, &resDesc, &texDesc, nullptr);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }
  // Copy to device.
  amd::HostQueue* queue = hip::getNullStream();
  HIP_RETURN(ihipMemcpy(refDevPtr, texRef, refDevSize, hipMemcpyHostToDevice, *queue));
}

hipError_t hipTexRefSetAddress2D(textureReference* texRef,
                                 const HIP_ARRAY_DESCRIPTOR* desc,
                                 hipDeviceptr_t dptr,
                                 size_t Pitch) {
  HIP_INIT_API(hipTexRefSetAddress2D, texRef, desc, dptr, Pitch);

  if ((texRef == nullptr) || (desc == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipDeviceptr_t refDevPtr = nullptr;
  size_t refDevSize = 0;
  HIP_RETURN_ONFAIL(PlatformState::instance().getDynTexGlobalVar(texRef, &refDevPtr, &refDevSize));
  assert(refDevSize == sizeof(textureReference));

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

  hipTextureDesc texDesc = hip::getTextureDesc(texRef);

  hipError_t err = ihipCreateTextureObject(&texRef->textureObject, &resDesc, &texDesc, nullptr);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }
  // Copy to device.
  amd::HostQueue* queue = hip::getNullStream();
  HIP_RETURN(ihipMemcpy(refDevPtr, texRef, refDevSize, hipMemcpyHostToDevice, *queue));
}

hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f) {
  return {x, y, z, w, f};
}

hipError_t hipTexRefGetBorderColor(float* pBorderColor,
                                   const textureReference* texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetBorderColor, pBorderColor, texRef);

  if ((pBorderColor == nullptr) || (texRef == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // TODO add textureReference::borderColor.
  assert(false && "textureReference::borderColor is missing in header");
  // std::memcpy(pBorderColor, texRef.borderColor, sizeof(texRef.borderColor));

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetFilterMode(hipTextureFilterMode* pfm,
                                  const textureReference* texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetFilterMode, pfm, texRef);

  if ((pfm == nullptr) || (texRef == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pfm = texRef->filterMode;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetFlags(unsigned int* pFlags,
                             const textureReference* texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetFlags, pFlags, texRef);

  if ((pFlags == nullptr) || (texRef == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pFlags = 0;

  if (texRef->readMode == hipReadModeElementType) {
    *pFlags |= HIP_TRSF_READ_AS_INTEGER;
  }

  if (texRef->normalized == 1) {
    *pFlags |= HIP_TRSF_NORMALIZED_COORDINATES;
  }

  if (texRef->sRGB == 1) {
    *pFlags |= HIP_TRSF_SRGB;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetFormat(hipArray_Format* pFormat,
                              int* pNumChannels,
                              const textureReference* texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetFormat, pFormat, pNumChannels, texRef);

  if ((pFormat == nullptr) || (pNumChannels == nullptr) ||
      (texRef == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pFormat = texRef->format;
  *pNumChannels = texRef->numChannels;

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefGetMaxAnisotropy(int* pmaxAnsio,
                                     const textureReference* texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetMaxAnisotropy, pmaxAnsio, texRef);

  if ((pmaxAnsio == nullptr) || (texRef == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pmaxAnsio = texRef->maxAnisotropy;

  HIP_RETURN(hipErrorInvalidValue);
}

hipError_t hipTexRefGetMipmapFilterMode(hipTextureFilterMode* pfm,
                                        const textureReference* texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetMipmapFilterMode, pfm, texRef);

  if ((pfm == nullptr) || (texRef == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pfm = texRef->mipmapFilterMode;

  HIP_RETURN(hipErrorInvalidValue);
}

hipError_t hipTexRefGetMipmapLevelBias(float* pbias,
                                       const textureReference* texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetMipmapLevelBias, pbias, texRef);

  if ((pbias == nullptr) || (texRef == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pbias = texRef->mipmapLevelBias;

  HIP_RETURN(hipErrorInvalidValue);
}

hipError_t hipTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp,
                                        float* pmaxMipmapLevelClamp,
                                        const textureReference* texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetMipmapLevelClamp, pminMipmapLevelClamp, pmaxMipmapLevelClamp, texRef);

  if ((pminMipmapLevelClamp == nullptr) || (pmaxMipmapLevelClamp == nullptr) ||
      (texRef == nullptr)){
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pminMipmapLevelClamp = texRef->minMipmapLevelClamp;
  *pmaxMipmapLevelClamp = texRef->maxMipmapLevelClamp;

  HIP_RETURN(hipErrorInvalidValue);
}

hipError_t hipTexRefGetMipmappedArray(hipMipmappedArray_t* pArray,
                                      const textureReference* texRef) {
  // TODO overload operator<<(ostream&, textureReference&).
  HIP_INIT_API(hipTexRefGetMipmappedArray, pArray, &texRef);

  if ((pArray == nullptr) || (texRef == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipResourceDesc resDesc = {};
  // TODO use ihipGetTextureObjectResourceDesc() to not pollute the API trace.
  hipError_t error = hipGetTextureObjectResourceDesc(&resDesc, texRef->textureObject);
  if (error != hipSuccess) {
    HIP_RETURN(error);
  }

  switch (resDesc.resType) {
  case hipResourceTypeLinear:
  case hipResourceTypePitch2D:
  case hipResourceTypeArray: {
    HIP_RETURN(hipErrorInvalidValue);
  }
  case hipResourceTypeMipmappedArray:
    *pArray = resDesc.res.mipmap.mipmap;
    break;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexRefSetBorderColor(textureReference* texRef,
                                   float* pBorderColor) {
  HIP_INIT_API(hipTexRefSetBorderColor, texRef, pBorderColor);

  if ((texRef == nullptr) || (pBorderColor == nullptr)) {
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

  if ((texRef == nullptr) || (mipmappedArray == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (Flags != HIP_TRSA_OVERRIDE_FORMAT) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipDeviceptr_t refDevPtr = nullptr;
  size_t refDevSize = 0;
  HIP_RETURN_ONFAIL(PlatformState::instance().getDynTexGlobalVar(texRef, &refDevPtr, &refDevSize));
  assert(refDevSize == sizeof(textureReference));

  // Any previous address or HIP array state associated with the texture reference is superseded by this function.
  // Any memory previously bound to hTexRef is unbound.
  // No need to check for errors.
  ihipDestroyTextureObject(texRef->textureObject);

  hipResourceDesc resDesc = {};
  resDesc.resType = hipResourceTypeMipmappedArray;
  resDesc.res.mipmap.mipmap = mipmappedArray;

  hipTextureDesc texDesc = hip::getTextureDesc(texRef);

  hipResourceViewFormat format = hip::getResourceViewFormat(hip::getChannelFormatDesc(texRef->numChannels, texRef->format));
  hipResourceViewDesc resViewDesc = hip::getResourceViewDesc(mipmappedArray, format);

  hipError_t err = ihipCreateTextureObject(&texRef->textureObject, &resDesc, &texDesc, &resViewDesc);
  if (err != hipSuccess) {
    HIP_RETURN(err);
  }
  // Copy to device.
  amd::HostQueue* queue = hip::getNullStream();
  HIP_RETURN(ihipMemcpy(refDevPtr, texRef, refDevSize, hipMemcpyHostToDevice, *queue));
}

hipError_t hipTexObjectCreate(hipTextureObject_t* pTexObject,
                              const HIP_RESOURCE_DESC* pResDesc,
                              const HIP_TEXTURE_DESC* pTexDesc,
                              const HIP_RESOURCE_VIEW_DESC* pResViewDesc) {
  HIP_INIT_API(hipTexObjectCreate, pTexObject, pResDesc, pTexDesc, pResViewDesc);

  if ((pTexObject == nullptr) || (pResDesc == nullptr) || (pTexDesc == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipResourceDesc resDesc = hip::getResourceDesc(*pResDesc);
  hipTextureDesc texDesc = hip::getTextureDesc(*pTexDesc);

  if (pResViewDesc != nullptr) {
    hipResourceViewDesc resViewDesc = hip::getResourceViewDesc(*pResViewDesc);
    HIP_RETURN(ihipCreateTextureObject(pTexObject, &resDesc, &texDesc, &resViewDesc));
  } else {
    HIP_RETURN(ihipCreateTextureObject(pTexObject, &resDesc, &texDesc, nullptr));
  }
}

hipError_t hipTexObjectDestroy(hipTextureObject_t texObject) {
  HIP_INIT_API(hipTexObjectDestroy, texObject);

  HIP_RETURN(ihipDestroyTextureObject(texObject));
}

hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC* pResDesc,
                                       hipTextureObject_t texObject) {
  HIP_INIT_API(hipTexObjectGetResourceDesc, pResDesc, texObject);

  if ((pResDesc == nullptr) || (texObject == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pResDesc = hip::getResourceDesc(texObject->resDesc);

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC* pResViewDesc,
                                           hipTextureObject_t texObject) {
  HIP_INIT_API(hipTexObjectGetResourceViewDesc, pResViewDesc, texObject);

  if ((pResViewDesc == nullptr) || (texObject == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pResViewDesc = hip::getResourceViewDesc(texObject->resViewDesc);

  HIP_RETURN(hipSuccess);
}

hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC* pTexDesc,
                                      hipTextureObject_t texObject) {
  HIP_INIT_API(hipTexObjectGetTextureDesc, pTexDesc, texObject);

  if ((pTexDesc == nullptr) || (texObject == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pTexDesc = hip::getTextureDesc(texObject->texDesc);

  HIP_RETURN(hipSuccess);
}
