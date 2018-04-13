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


hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const hipResourceViewDesc* pResViewDesc) {
  HIP_INIT_API(pTexObject, pResDesc, pTexDesc, pResViewDesc);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
  HIP_INIT_API(textureObject);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
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