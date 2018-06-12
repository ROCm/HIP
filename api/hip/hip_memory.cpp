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
#include "hip_internal.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/memory.hpp"

extern void getChannelOrderAndType(const hipChannelFormatDesc& desc,
                                   enum hipTextureReadMode readMode,
                                   cl_channel_order* channelOrder,
                                   cl_channel_type* channelType);

extern void getDrvChannelOrderAndType(const enum hipArray_Format Format,
                                      unsigned int NumChannels,
                                      cl_channel_order* channelOrder,
                                      cl_channel_type* channelType);

amd::Memory* getMemoryObject(const void* ptr, size_t& offset) {
  amd::Memory *memObj = amd::MemObjMap::FindMemObj(ptr);
  if (memObj != nullptr) {
    offset = reinterpret_cast<size_t>(ptr) - reinterpret_cast<size_t>(memObj->getSvmPtr());
  }
  return memObj;
}

hipError_t ihipMalloc(void** ptr, size_t sizeBytes, unsigned int flags)
{
  if (sizeBytes == 0) {
    *ptr = nullptr;
    return hipSuccess;
  }
  else if (ptr == nullptr) {
    return hipErrorInvalidValue;
  }

  if (hip::getCurrentContext()->devices()[0]->info().maxMemAllocSize_ < sizeBytes) {
    return hipErrorOutOfMemory;
  }

  *ptr = amd::SvmBuffer::malloc(*hip::getCurrentContext(), flags, sizeBytes, hip::getCurrentContext()->devices()[0]->info().memBaseAddrAlign_);
  if (*ptr == nullptr) {
    return hipErrorOutOfMemory;
  }

  return hipSuccess;
}

hipError_t ihipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                      amd::HostQueue& queue, bool isAsync = false) {
  amd::Command* command = nullptr;
  amd::Command::EventWaitList waitList;
  size_t sOffset = 0;
  amd::Memory *srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory *dstMemory = getMemoryObject(dst, dOffset);

  amd::Coord3D srcOffset(sOffset, 0, 0);
  amd::Coord3D dstOffset(dOffset, 0, 0);

  if (kind == hipMemcpyDefault) {
    // Determine kind on VA
    if (srcMemory == nullptr && dstMemory != nullptr) {
      kind = hipMemcpyHostToDevice;
    } else if (srcMemory != nullptr && dstMemory == nullptr) {
      kind = hipMemcpyDeviceToHost;
    } else if (srcMemory != nullptr && dstMemory != nullptr) {
      kind = hipMemcpyDeviceToDevice;
    } else {
      kind = hipMemcpyHostToHost;
    }
  }

  switch (kind) {
  case hipMemcpyDeviceToHost:
    command = new amd::ReadMemoryCommand(queue, CL_COMMAND_READ_BUFFER, waitList,
              *srcMemory->asBuffer(), srcOffset, sizeBytes, dst);
    break;
  case hipMemcpyHostToDevice:
    command = new amd::WriteMemoryCommand(queue, CL_COMMAND_WRITE_BUFFER, waitList,
              *dstMemory->asBuffer(), dstOffset, sizeBytes, src);
    break;
  case hipMemcpyDeviceToDevice:
    command = new amd::CopyMemoryCommand(queue, CL_COMMAND_COPY_BUFFER, waitList,
              *srcMemory->asBuffer(),*dstMemory->asBuffer(), srcOffset, dstOffset, sizeBytes);
    break;
  case hipMemcpyHostToHost:
    memcpy(dst, src, sizeBytes);
    return hipSuccess;
  default:
    assert(!"Shouldn't reach here");
    break;
  }
  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  if (!isAsync) {
    command->awaitCompletion();
  }
  command->release();

  return hipSuccess;
}

hipError_t ihipMemset(void* dst, int value, size_t sizeBytes, amd::HostQueue& queue,
                      bool isAsync = false) {

  if (dst == nullptr) {
    return hipErrorInvalidValue;
  }

  size_t offset = 0;
  amd::Memory* memory = getMemoryObject(dst, offset);

  if (memory != nullptr) {
    // Device memory
    amd::Command::EventWaitList waitList;
    amd::Coord3D fillOffset(offset, 0, 0);
    amd::Coord3D fillSize(sizeBytes, 1, 1);
    amd::FillMemoryCommand* command =
        new amd::FillMemoryCommand(queue, CL_COMMAND_FILL_BUFFER, waitList, *memory->asBuffer(),
                                   &value, sizeof(char), fillOffset, fillSize);

    if (command == nullptr) {
      return hipErrorOutOfMemory;
    }

    command->enqueue();
    if (!isAsync) {
      command->awaitCompletion();
    }
    command->release();
  } else {
    // Host alloced memory
    memset(dst, value, sizeBytes);
  }

  return hipSuccess;
}

hipError_t hipMalloc(void** ptr, size_t sizeBytes) {
  HIP_INIT_API(ptr, sizeBytes);

  return ihipMalloc(ptr, sizeBytes, 0);
}

hipError_t hipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags) {
  HIP_INIT_API(ptr, sizeBytes, flags);

  if (ptr == nullptr) {
    return hipErrorInvalidValue;
  }
  *ptr = nullptr;

  const unsigned int coherentFlags = hipHostMallocCoherent | hipHostMallocNonCoherent;

  // can't have both Coherent and NonCoherent flags set at the same time
  if ((flags & coherentFlags) == coherentFlags) {
    return hipErrorInvalidValue;
  }

  return ihipMalloc(ptr, sizeBytes, CL_MEM_SVM_FINE_GRAIN_BUFFER | (flags << 16));
}

hipError_t hipFree(void* ptr) {
  if (amd::SvmBuffer::malloced(ptr)) {
    hip::syncStreams();
    hip::getNullStream()->finish();
    amd::SvmBuffer::free(*hip::getCurrentContext(), ptr);
    return hipSuccess;
  }
  return hipErrorInvalidValue;
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  HIP_INIT_API(dst, src, sizeBytes, kind);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();
  return ihipMemcpy(dst, src, sizeBytes, kind, *queue);
}

hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  HIP_INIT_API(dst, value, sizeBytes, stream);

  amd::HostQueue* queue;

  if (stream == nullptr) {
    hip::syncStreams();
    queue = hip::getNullStream();
  } else {
    hip::getNullStream()->finish();
    queue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }

  return ihipMemset(dst, value, sizeBytes, *queue, true);
}

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  HIP_INIT_API(dst, value, sizeBytes);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();

  return ihipMemset(dst, value, sizeBytes, *queue);
}

hipError_t hipMemPtrGetInfo(void *ptr, size_t *size) {
  HIP_INIT_API(ptr, size);

  size_t offset = 0;
  amd::Memory* svmMem = getMemoryObject(ptr, offset);

  if (svmMem == nullptr) {
    return hipErrorInvalidValue;
  }

  *size = svmMem->getSize();

  return hipSuccess;
}

hipError_t hipHostFree(void* ptr) {
  HIP_INIT_API(ptr);

  if (amd::SvmBuffer::malloced(ptr)) {
    amd::SvmBuffer::free(*hip::getCurrentContext(), ptr);
    return hipSuccess;
  }
  return hipErrorInvalidValue;
}

hipError_t hipFreeArray(hipArray* array) {
  HIP_INIT_API(array);

  if (amd::SvmBuffer::malloced(array->data)) {
    amd::SvmBuffer::free(*hip::getCurrentContext(), array->data);
    return hipSuccess;
  }
  return hipErrorInvalidValue;
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
  HIP_INIT_API(pbase, psize, dptr);

  // Since we are using SVM buffer DevicePtr and HostPtr is the same
  void* ptr = dptr;
  size_t offset = 0;
  amd::Memory* svmMem = getMemoryObject(ptr, offset);

  if (svmMem == nullptr) {
    return hipErrorInvalidDevicePointer;
  }

  *pbase = svmMem->getSvmPtr();
  *psize = svmMem->getSize();

  return hipSuccess;
}

hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  HIP_INIT_API(free, total);

  size_t freeMemory[2];
  amd::Device* device = hip::getCurrentContext()->devices()[0];
  if(device == nullptr) {
    return hipErrorInvalidDevice;
  }

  if(!device->globalFreeMemory(freeMemory)) {
    return hipErrorInvalidValue;
  }

  *free = freeMemory[0];
  *total = device->info().globalMemSize_;

return hipSuccess;
}

hipError_t ihipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height, size_t depth,
                           cl_mem_object_type imageType, const cl_image_format* image_format) {

  amd::Device* device = hip::getCurrentContext()->devices()[0];

  if ((width == 0) || (height == 0)) {
    *ptr = nullptr;
    return hipSuccess;
  }
  else if (!(device->info().image2DMaxWidth_ >= width &&
           device->info().image2DMaxHeight_ >= height ) || (ptr == nullptr)) {
    return hipErrorInvalidValue;
  }

  if (device->info().maxMemAllocSize_ < (width * height)) {
    return hipErrorOutOfMemory;
  }

  const amd::Image::Format imageFormat(*image_format);

  *pitch = width * imageFormat.getElementSize();

  size_t sizeBytes = *pitch * height * depth;
  *ptr = amd::SvmBuffer::malloc(*hip::getCurrentContext(), 0, sizeBytes,
                                device->info().memBaseAddrAlign_);

  if (*ptr == nullptr) {
    return hipErrorMemoryAllocation;
  }

  return hipSuccess;
}


hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
  HIP_INIT_API(ptr, pitch, width, height);

  const cl_image_format image_format = { CL_R, CL_UNSIGNED_INT8 };
  return ihipMallocPitch(ptr, pitch, width, height, 1, CL_MEM_OBJECT_IMAGE2D, &image_format);
}

hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
  HIP_INIT_API(pitchedDevPtr, &extent);

  size_t pitch = 0;

  if (pitchedDevPtr == nullptr) {
    return hipErrorInvalidValue;
  }

  const cl_image_format image_format = { CL_R, CL_UNSIGNED_INT8 };
  hipError_t status = hipSuccess;
  status = ihipMallocPitch(&pitchedDevPtr->ptr, &pitch, extent.width, extent.height, extent.depth,
                           CL_MEM_OBJECT_IMAGE3D, &image_format);

  if (status == hipSuccess) {
        pitchedDevPtr->pitch = pitch;
        pitchedDevPtr->xsize = extent.width;
        pitchedDevPtr->ysize = extent.height;
  }

  return status;
}

hipError_t hipArrayCreate(hipArray** array, const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
  HIP_INIT_API(array, pAllocateArray);

  if (array[0]->width == 0) {
    return hipErrorInvalidValue;
  }

  *array = (hipArray*)malloc(sizeof(hipArray));
  array[0]->drvDesc = *pAllocateArray;
  array[0]->width = pAllocateArray->width;
  array[0]->height = pAllocateArray->height;
  array[0]->isDrv = true;
  array[0]->textureType = hipTextureType2D;
  void** ptr = &array[0]->data;

  cl_channel_order channelOrder;
  cl_channel_type channelType;
  getDrvChannelOrderAndType(pAllocateArray->format, pAllocateArray->numChannels, 
                            &channelOrder, &channelType);

  const cl_image_format image_format = { channelOrder, channelType };
  size_t size = pAllocateArray->width;
  if (pAllocateArray->height > 0) {
      size = size * pAllocateArray->height;
  }

  size_t pitch = 0;
  hipError_t status = ihipMallocPitch(ptr, &pitch, array[0]->width, array[0]->height, 1, CL_MEM_OBJECT_IMAGE2D,
                      &image_format);

  return status;
}

hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc,
                          size_t width, size_t height, unsigned int flags) {
  HIP_INIT_API(array, desc, width, height, flags);

  if (width == 0) {
    return hipErrorInvalidValue;
  }

  *array = (hipArray*)malloc(sizeof(hipArray));
  array[0]->type = flags;
  array[0]->width = width;
  array[0]->height = height;
  array[0]->depth = 1;
  array[0]->desc = *desc;
  array[0]->isDrv = false;
  array[0]->textureType = hipTextureType2D;
  void** ptr = &array[0]->data;

  cl_channel_order channelOrder;
  cl_channel_type channelType;
  getChannelOrderAndType(*desc, hipReadModeElementType, &channelOrder, &channelType);

  const cl_image_format image_format = { channelOrder, channelType };

 // Dummy flags check
  switch (flags) {
    case hipArrayLayered:
    case hipArrayCubemap:
    case hipArraySurfaceLoadStore:
    case hipArrayTextureGather:
        assert(0 && "Unspported");
        break;
    case hipArrayDefault:
    default:
        break;
  }
  size_t pitch = 0;
  hipError_t status = ihipMallocPitch(ptr, &pitch, width, height, 1, CL_MEM_OBJECT_IMAGE2D,
                      &image_format);

  return status;
}

hipError_t hipMalloc3DArray(hipArray_t* array, const struct hipChannelFormatDesc* desc,
                            struct hipExtent extent, unsigned int flags) {
  HIP_INIT_API(array, desc, &extent, flags);

  *array = (hipArray*)malloc(sizeof(hipArray));
  array[0]->type = flags;
  array[0]->width = extent.width;
  array[0]->height = extent.height;
  array[0]->depth = extent.depth;
  array[0]->desc = *desc;
  array[0]->isDrv = false;
  array[0]->textureType = hipTextureType3D;
  void** ptr = &array[0]->data;

  cl_channel_order channelOrder;
  cl_channel_type channelType;
  getChannelOrderAndType(*desc, hipReadModeElementType, &channelOrder, &channelType);

  const cl_image_format image_format = { channelOrder, channelType };

 // Dummy flags check
  switch (flags) {
    case hipArrayLayered:
    case hipArrayCubemap:
    case hipArraySurfaceLoadStore:
    case hipArrayTextureGather:
        assert(0 && "Unspported");
        break;
    case hipArrayDefault:
    default:
        break;
  }
  size_t pitch = 0;
  hipError_t status = ihipMallocPitch(ptr, &pitch, extent.width, extent.height, extent.depth,
                      CL_MEM_OBJECT_IMAGE3D, &image_format);

  return status;
}

hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
  HIP_INIT_API(flagsPtr, hostPtr);

  if (flagsPtr == nullptr ||
      hostPtr == nullptr) {
    return hipErrorInvalidValue;
  }

  size_t offset = 0;
  amd::Memory* svmMem = getMemoryObject(hostPtr, offset);

  if (svmMem == nullptr) {
    return hipErrorInvalidValue;
  }

  *flagsPtr = svmMem->getMemFlags() >> 16;

  return hipSuccess;
}

hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  HIP_INIT_API(hostPtr, sizeBytes, flags);
  if(hostPtr != nullptr) {
    amd::Context *amdContext = hip::getCurrentContext();
    amd::Memory* mem = new (*amdContext) amd::Buffer(*amdContext, CL_MEM_USE_HOST_PTR, sizeBytes);

    if (!mem->create(hostPtr)) {
      mem->release();
      return hipErrorMemoryAllocation;
    }
    amd::MemObjMap::AddMemObj(hostPtr, mem);
    return hipSuccess;
  } else {
    return ihipMalloc(&hostPtr, sizeBytes, flags);
  }
}

hipError_t hipHostUnregister(void* hostPtr) {
  HIP_INIT_API(hostPtr);

  if (amd::SvmBuffer::malloced(hostPtr)) {
    hip::syncStreams();
    hip::getNullStream()->finish();
    amd::SvmBuffer::free(*hip::getCurrentContext(), hostPtr);
    return hipSuccess;
  } else {
    size_t offset = 0;
    amd::Memory* mem = getMemoryObject(hostPtr, offset);

    if(mem) {
      mem->release();
      return hipSuccess;
    }
  }

  return hipErrorInvalidValue;
}

// Deprecated function:
hipError_t hipHostAlloc(void** ptr, size_t sizeBytes, unsigned int flags) {
  return ihipMalloc(ptr, sizeBytes, flags);
};


hipError_t hipMemcpyToSymbol(const void* symbolName, const void* src, size_t count,
                             size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(symbolName, src, count, offset, kind);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyFromSymbol(void* dst, const void* symbolName, size_t count,
                               size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(symbolName, dst, count, offset, kind);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyToSymbolAsync(const void* symbolName, const void* src, size_t count,
                                  size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(symbolName, src, count, offset, kind, stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbolName, size_t count,
                                    size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(symbolName, dst, count, offset, kind, stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes) {
  HIP_INIT_API(dst, src, sizeBytes);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();

  return ihipMemcpy(reinterpret_cast<void*>(dst), (const void*) src, sizeBytes, hipMemcpyHostToDevice, *queue);
}

hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes) {
  HIP_INIT_API(dst, src, sizeBytes);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();

  return ihipMemcpy(reinterpret_cast<void*>(dst), (const void*) src, sizeBytes, hipMemcpyDeviceToHost, *queue);
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
  HIP_INIT_API(dst, src, sizeBytes);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();

  return ihipMemcpy(reinterpret_cast<void*>(dst), (const void*) src, sizeBytes, hipMemcpyDeviceToDevice, *queue);
}

hipError_t hipMemcpyHtoH(void* dst, void* src, size_t sizeBytes) {
  HIP_INIT_API(dst, src, sizeBytes);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();

  return ihipMemcpy(reinterpret_cast<void*>(dst), (const void*) src, sizeBytes, hipMemcpyHostToHost, *queue);
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(dst, src, sizeBytes, kind, stream);

  amd::HostQueue* queue = nullptr;

  if (stream == nullptr) {
    hip::syncStreams();
    queue = hip::getNullStream();
  } else {
    hip::getNullStream()->finish();
    queue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }

  return ihipMemcpy(dst, src, sizeBytes, kind, *queue, true);
}


hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes,
                              hipStream_t stream) {
  HIP_INIT_API(dst, src, sizeBytes, stream);

  amd::HostQueue* queue = nullptr;

  if (stream == nullptr) {
    hip::syncStreams();
    queue = hip::getNullStream();
  } else {
    hip::getNullStream()->finish();
    queue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }

  return ihipMemcpy(reinterpret_cast<void*>(dst), (const void*) src, sizeBytes, hipMemcpyHostToDevice,
                    *queue, true);
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream) {
  HIP_INIT_API(dst, src, sizeBytes, stream);

  amd::HostQueue* queue = nullptr;

  if (stream == nullptr) {
    hip::syncStreams();
    queue = hip::getNullStream();
  } else {
    hip::getNullStream()->finish();
    queue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }

  return ihipMemcpy(reinterpret_cast<void*>(dst), (const void*) src, sizeBytes, hipMemcpyDeviceToDevice,
                   *queue, true);
}

hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream) {
  HIP_INIT_API(dst, src, sizeBytes, stream);

  amd::HostQueue* queue = nullptr;

  if (stream == nullptr) {
    hip::syncStreams();
    queue = hip::getNullStream();
  } else {
    hip::getNullStream()->finish();
    queue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }

  return ihipMemcpy(reinterpret_cast<void*>(dst), (const void*) src, sizeBytes, hipMemcpyDeviceToHost,
                   *queue, true);
}

hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
  HIP_INIT_API(pCopy);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t ihipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, amd::HostQueue& queue,
                            bool isAsync = false) {
  // Create buffer rectangle info structure
  amd::BufferRect srcRect;
  amd::BufferRect dstRect;

  size_t region[3] = {width, height, 1};
  size_t src_slice_pitch = spitch * height;
  size_t dst_slice_pitch = dpitch * height;
  size_t sOrigin[3] = { };
  size_t dOrigin[3] = { };
  amd::Memory* srcPtr = getMemoryObject(src, sOrigin[0]);
  amd::Memory* dstPtr = getMemoryObject(dst, dOrigin[0]);

  if (kind == hipMemcpyDefault) {
    // Determine kind on VA
    if (srcPtr == nullptr && dstPtr != nullptr) {
      kind = hipMemcpyHostToDevice;
    } else if (srcPtr != nullptr && dstPtr == nullptr) {
      kind = hipMemcpyDeviceToHost;
    } else if (srcPtr != nullptr && dstPtr != nullptr) {
      kind = hipMemcpyDeviceToDevice;
    } else {
      kind = hipMemcpyHostToHost;
    }
  }

  amd::Coord3D size(region[0], region[1], region[2]);

  if (!srcRect.create(sOrigin, region, spitch, src_slice_pitch) ||
      !dstRect.create(dOrigin, region, dpitch, dst_slice_pitch)) {
    return hipErrorInvalidValue;
  }

  amd::Coord3D srcStart(srcRect.start_, 0, 0);
  amd::Coord3D dstStart(dstRect.start_, 0, 0);
  amd::Coord3D srcEnd(srcRect.end_, 1, 1);
  amd::Coord3D dstEnd(dstRect.end_, 1, 1);

  amd::Command* command = nullptr;
  amd::Command::EventWaitList waitList;
  switch (kind) {
  case hipMemcpyDeviceToHost:
    command = new amd::ReadMemoryCommand(queue, CL_COMMAND_READ_BUFFER_RECT, waitList,
              *srcPtr->asBuffer(), srcStart, size, dst, srcRect, dstRect);
    break;
  case hipMemcpyHostToDevice:
    command = new amd::WriteMemoryCommand(queue, CL_COMMAND_WRITE_BUFFER_RECT, waitList,
              *dstPtr->asBuffer(), dstStart, size, src, dstRect, srcRect);
    break;
  case hipMemcpyDeviceToDevice:
    command = new amd::CopyMemoryCommand(queue, CL_COMMAND_COPY_BUFFER_RECT, waitList, *srcPtr->asBuffer(),
              *dstPtr->asBuffer(), srcStart, dstStart, size, srcRect, dstRect);
    break;
  case hipMemcpyHostToHost:
    for(unsigned int y = 0; y < height; y++) {
      void* pDst = reinterpret_cast<void*>(reinterpret_cast<size_t>(dst) + y * dpitch);
      void* pSrc = reinterpret_cast<void*>(reinterpret_cast<size_t>(src) + y * spitch);
      memcpy(pDst, pSrc, width);
    }
    return hipSuccess;
  default:
    assert(!"Shouldn't reach here");
    break;
  }

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();

  if (!isAsync) {
    command->awaitCompletion();
  }
  command->release();

  return hipSuccess;
}

hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(dst, dpitch, src, spitch, width, height, kind);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();

  return ihipMemcpy2D(dst, dpitch, src, spitch, width, height, kind, *queue);
}


hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(dst, dpitch, src, spitch, width, height, kind, stream);

  amd::HostQueue* queue;

  if (stream == nullptr) {
    hip::syncStreams();
    queue = hip::getNullStream();
  } else {
    hip::getNullStream()->finish();
    queue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }

  return ihipMemcpy2D(dst, dpitch, src, spitch, width, height, kind, *queue, true);
}

hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(dst, wOffset, hOffset, src, spitch, width, height, kind);

  if (dst->data == nullptr) {
    return hipErrorUnknown;
  }

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();

  size_t dpitch = dst->width;

  switch (dst[0].desc.f) {
    case hipChannelFormatKindSigned:
      dpitch *= sizeof(int);
      break;
    case hipChannelFormatKindUnsigned:
      dpitch *= sizeof(unsigned int);
      break;
    case hipChannelFormatKindFloat:
      dpitch *= sizeof(float);
      break;
    case hipChannelFormatKindNone:
      dpitch *= sizeof(size_t);
      break;
    default:
      dpitch *= 1;
      break;
  }

  if ((wOffset + width > (dpitch)) || width > spitch) {
      return hipErrorUnknown;
  }

  // Create buffer rectangle info structure
  amd::BufferRect srcRect;
  amd::BufferRect dstRect;

  size_t region[3] = {width, height, 1};
  size_t src_slice_pitch = spitch * height;
  size_t dst_slice_pitch = dpitch * height;
  size_t sOrigin[3] = { };
  size_t dOrigin[3] = {wOffset, hOffset, 0};
  size_t sz = 0;
  amd::Memory* srcPtr = getMemoryObject(src, sz);
  amd::Memory* dstPtr = getMemoryObject(dst->data, sz);

  if (kind == hipMemcpyDefault) {
    // Determine kind on VA
    if (srcPtr == nullptr && dstPtr != nullptr) {
      kind = hipMemcpyHostToDevice;
    } else if (srcPtr != nullptr && dstPtr == nullptr) {
      kind = hipMemcpyDeviceToHost;
    } else if (srcPtr != nullptr && dstPtr != nullptr) {
      kind = hipMemcpyDeviceToDevice;
    } else {
      kind = hipMemcpyHostToHost;
    }
  }

  amd::Coord3D size(region[0], region[1], region[2]);

  if (!srcRect.create(sOrigin, region, spitch, src_slice_pitch) ||
      !dstRect.create(dOrigin, region, dpitch, dst_slice_pitch)) {
    return hipErrorInvalidValue;
  }

  amd::Coord3D srcStart(srcRect.start_, 0, 0);
  amd::Coord3D dstStart(dstRect.start_, 0, 0);
  amd::Coord3D srcEnd(srcRect.end_, 1, 1);
  amd::Coord3D dstEnd(dstRect.end_, 1, 1);

  amd::Command* command = nullptr;
  amd::Command::EventWaitList waitList;

  void* newDst = nullptr;

  switch (kind) {
  case hipMemcpyDeviceToHost:
    command = new amd::ReadMemoryCommand(*queue, CL_COMMAND_READ_BUFFER_RECT, waitList,
              *srcPtr->asBuffer(), srcStart, size, dst->data, srcRect, dstRect);
    break;
  case hipMemcpyHostToDevice:
    command = new amd::WriteMemoryCommand(*queue, CL_COMMAND_WRITE_BUFFER_RECT, waitList,
              *dstPtr->asBuffer(), dstStart, size, src, dstRect, srcRect);
    break;
  case hipMemcpyDeviceToDevice:
    command = new amd::CopyMemoryCommand(*queue, CL_COMMAND_COPY_BUFFER_RECT, waitList, *srcPtr->asBuffer(),
              *dstPtr->asBuffer(), srcStart, dstStart, size, srcRect, dstRect);
    break;
  case hipMemcpyHostToHost:
    newDst = reinterpret_cast<void*>(reinterpret_cast<size_t>(dst->data)
                                          + dpitch * hOffset + wOffset);
    for(unsigned int y = 0; y < height; y++) {
      void* pDst = reinterpret_cast<void*>(reinterpret_cast<size_t>(newDst) + y * dpitch);
      void* pSrc = reinterpret_cast<void*>(reinterpret_cast<size_t>(src) + y * spitch);
      memcpy(pDst, pSrc, width);
    }
    return hipSuccess;
  default:
    assert(!"Shouldn't reach here");
    break;
  }

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();

  command->awaitCompletion();

  command->release();

  return hipSuccess;

}

hipError_t hipMemcpyToArray(hipArray* dstArray, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(dstArray, wOffset, hOffset, src, count, kind);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();

  amd::Command* command = nullptr;
  amd::Command::EventWaitList waitList;
  amd::Memory* memory;
  size_t offset = 0;
  amd::Coord3D dstOffset(wOffset, hOffset, 0);

  switch (kind) {
  case hipMemcpyDeviceToHost:
    assert(!"Invalid case");
  case hipMemcpyHostToDevice:
    memory = getMemoryObject(dstArray->data, offset);
    assert(offset == 0);
    command = new amd::WriteMemoryCommand(*queue, CL_COMMAND_WRITE_BUFFER, waitList,
      *memory->asBuffer(), dstOffset, count, src);
    break;
  case hipMemcpyDeviceToDevice:
  case hipMemcpyDefault:
  default:
    assert(!"Shouldn't reach here");
    break;
  }
  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->awaitCompletion();
  command->release();

  return hipSuccess;
}

hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(dst, srcArray, wOffset, hOffset, count, kind);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();

  amd::Command* command = nullptr;
  amd::Command::EventWaitList waitList;
  amd::Memory* memory;

  size_t offset = 0;
  amd::Coord3D srcOffset(wOffset, hOffset, 0);

  switch (kind) {
  case hipMemcpyHostToDevice:
    assert(!"Invalid case");
  case hipMemcpyDeviceToHost:
    memory = getMemoryObject(srcArray->data, offset);
    assert(offset == 0);
    command = new amd::ReadMemoryCommand(*queue, CL_COMMAND_READ_BUFFER, waitList,
      *memory->asBuffer(), srcOffset, count, dst);
    break;
  case hipMemcpyDeviceToDevice:
  case hipMemcpyDefault:
  default:
    assert(!"Shouldn't reach here");
    break;
  }
  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->awaitCompletion();
  command->release();

  return hipSuccess;
}

hipError_t hipMemcpyHtoA(hipArray* dstArray, size_t dstOffset, const void* srcHost, size_t count) {
  HIP_INIT_API(dstArray, dstOffset, srcHost, count);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();

  amd::Command::EventWaitList waitList;
  size_t offset = 0;
  amd::Memory* memory = getMemoryObject(dstArray->data, offset);
  assert(offset == 0);
  amd::Command* command = new amd::WriteMemoryCommand(*queue, CL_COMMAND_WRITE_BUFFER, waitList,
      *memory->asBuffer(), dstOffset, count, srcHost);

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->awaitCompletion();
  command->release();

  return hipSuccess;
}

hipError_t hipMemcpyAtoH(void* dst, hipArray* srcArray, size_t srcOffset, size_t count) {
  HIP_INIT_API(dst, srcArray, srcOffset, count);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();

  amd::Command::EventWaitList waitList;
  size_t offset = 0;
  amd::Memory* memory = getMemoryObject(srcArray->data, offset);
  assert(offset == 0);
  amd::Command* command = new amd::ReadMemoryCommand(*queue, CL_COMMAND_READ_BUFFER, waitList,
      *memory->asBuffer(), srcOffset, count, dst);

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->awaitCompletion();
  command->release();

  return hipSuccess;
}

hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p) {
  HIP_INIT_API(p);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();

  size_t byteSize;
  size_t srcPitchInBytes;
  size_t dstPitchInbytes;
  void* srcPtr;
  void* dstPtr;
  size_t srcOrigin[3];
  size_t dstOrigin[3];
  size_t region[3];
  if (p->dstArray != nullptr) {
    switch (p->dstArray->desc.f) {
      case hipChannelFormatKindSigned:
        byteSize = sizeof(int);
        break;
      case hipChannelFormatKindUnsigned:
        byteSize = sizeof(unsigned int);
        break;
      case hipChannelFormatKindFloat:
        byteSize = sizeof(float);
        break;
      case hipChannelFormatKindNone:
        byteSize = sizeof(size_t);
        break;
      default:
        byteSize = 1;
        break;
    }
    region[2] = p->Depth;
    region[1] = p->Height;
    region[0] = p->WidthInBytes * byteSize;
    srcOrigin[0] = p->srcXInBytes/byteSize;
    srcOrigin[1] = p->srcY;
    srcOrigin[2] = p->srcZ;
    dstPitchInbytes = p->dstArray->width * byteSize;
    srcPitchInBytes = p->srcPitch;
    srcPtr = (void*)p->srcHost;
    dstPtr = p->dstArray->data;
    dstOrigin[0] = p->dstXInBytes/byteSize;
    dstOrigin[1] = p->dstY;
    dstOrigin[2] = p->dstZ;
  } else {
    region[2] = p->extent.depth;
    region[1] = p->extent.height;
    region[0] = p->extent.width;
    srcOrigin[0] = p->srcXInBytes;
    srcOrigin[1] = p->srcY;
    srcOrigin[2] = p->srcZ;
    srcPitchInBytes = p->srcPtr.pitch;
    dstPitchInbytes = p->dstPtr.pitch;
    srcPtr = p->srcPtr.ptr;
    dstPtr = p->dstPtr.ptr;
    dstOrigin[0] = p->dstXInBytes;
    dstOrigin[1] = p->dstY;
    dstOrigin[2] = p->dstZ;
  }

  // Create buffer rectangle info structure
  amd::BufferRect srcRect;
  amd::BufferRect dstRect;
  size_t offset = 0;
  amd::Memory* src = getMemoryObject(srcPtr, offset);
  assert(offset == 0);
  amd::Memory* dst = getMemoryObject(dstPtr, offset);
  assert(offset == 0);

  size_t src_slice_pitch = srcPitchInBytes * p->srcHeight;
  size_t dst_slice_pitch = dstPitchInbytes * p->dstHeight;

  if (!srcRect.create(srcOrigin, region, srcPitchInBytes, src_slice_pitch) ||
      !dstRect.create(dstOrigin, region, dstPitchInbytes, dst_slice_pitch)) {
    return hipErrorInvalidValue;
  }

  amd::Coord3D srcStart(srcRect.start_, 0, 0);
  amd::Coord3D dstStart(dstRect.start_, 0, 0);
  amd::Coord3D srcEnd(srcRect.end_, 1, 1);
  amd::Coord3D dstEnd(dstRect.end_, 1, 1);

  if (!src->asBuffer()->validateRegion(srcStart, srcEnd) ||
      !dst->asBuffer()->validateRegion(dstStart, dstEnd)) {
    return hipErrorInvalidValue;
  }

  // Check if regions overlap each other
  if ((src->asBuffer() == dst->asBuffer()) &&
      (std::abs(static_cast<long>(srcOrigin[0]) - static_cast<long>(dstOrigin[0])) <
       static_cast<long>(region[0])) &&
      (std::abs(static_cast<long>(srcOrigin[1]) - static_cast<long>(dstOrigin[1])) <
       static_cast<long>(region[1])) &&
      (std::abs(static_cast<long>(srcOrigin[2]) - static_cast<long>(dstOrigin[2])) <
       static_cast<long>(region[2]))) {
    return hipErrorUnknown;
  }

  amd::Command::EventWaitList waitList;
  amd::Coord3D size(region[0], region[1], region[2]);

  amd::CopyMemoryCommand* command =
      new amd::CopyMemoryCommand(*queue, CL_COMMAND_COPY_BUFFER_RECT, waitList, *src->asBuffer(),
                                 *dst->asBuffer(), srcStart, dstStart, size, srcRect, dstRect);

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->awaitCompletion();
  command->release();

  return hipSuccess;
}

hipError_t ihipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height,
                        amd::HostQueue& queue, bool isAsync = false) {

  if (dst == nullptr) {
    return hipErrorInvalidValue;
  }

  size_t sizeBytes = pitch * height;
  size_t offset = 0;

  amd::Memory* memory = getMemoryObject(dst, offset);

  if (memory != nullptr) {
    // Device memory
    amd::Command::EventWaitList waitList;
    amd::Coord3D fillOffset(offset, 0, 0);
    amd::Coord3D fillSize(sizeBytes, 1, 1);

    // TODO: Byte copies are inefficient. Combine multiple writes inside runtime
    amd::FillMemoryCommand* command =
        new amd::FillMemoryCommand(queue, CL_COMMAND_FILL_BUFFER, waitList, *memory->asBuffer(),
                                   &value, sizeof(char), fillOffset, fillSize);

    if (command == nullptr) {
      return hipErrorOutOfMemory;
    }

    command->enqueue();
    if(!isAsync) {
      command->awaitCompletion();
    }
    command->release();
  } else {
    // Host alloced memory
    memset(dst, value, sizeBytes);
  }

  return hipSuccess;
}

hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
  HIP_INIT_API(dst, pitch, value, width, height);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();
  return ihipMemset2D(dst, pitch, value, width, height, *queue);
}

hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value,
                            size_t width, size_t height, hipStream_t stream) {
  HIP_INIT_API(dst, pitch, value, width, height, stream);

  amd::HostQueue* queue = nullptr;
  if (stream == nullptr) {
    hip::syncStreams();
    queue = hip::getNullStream();
  } else {
    hip::getNullStream()->finish();
    queue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }

  return ihipMemset2D(dst, pitch, value, width, height, *queue, true);
}

hipError_t hipMemsetD8(hipDeviceptr_t dst, unsigned char value, size_t sizeBytes) {
  HIP_INIT_API(dst, value, sizeBytes);

  if (dst == nullptr) {
    return hipErrorInvalidValue;
  }

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();
  size_t offset = 0;
  amd::Command::EventWaitList waitList;
  amd::Memory* memory = getMemoryObject(dst, offset);
  if (memory != nullptr) {
    // Device memory
    amd::Coord3D fillOffset(offset, 0, 0);
    amd::Coord3D fillSize(sizeBytes, 1, 1);
    amd::FillMemoryCommand* command =
        new amd::FillMemoryCommand(*queue, CL_COMMAND_FILL_BUFFER, waitList, *memory->asBuffer(),
                                   &value, sizeof(char), fillOffset, fillSize);

    if (command == nullptr) {
      return hipErrorOutOfMemory;
    }

    command->enqueue();
    command->awaitCompletion();
    command->release();
  } else {
    // Host alloced memory
    memset(dst, value, sizeBytes);
  }

  return hipSuccess;
}

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
  HIP_INIT_API(handle, devPtr);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags) {
  HIP_INIT_API(devPtr, &handle, flags);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipIpcCloseMemHandle(void* devPtr) {
  HIP_INIT_API(devPtr);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f) {
    hipChannelFormatDesc cd;
    cd.x = x;
    cd.y = y;
    cd.z = z;
    cd.w = w;
    cd.f = f;
    return cd;
}

hipError_t hipHostGetDevicePointer(void** devicePointer, void* hostPointer, unsigned flags) {
  HIP_INIT_API(devicePointer, hostPointer, flags);

  if (!amd::MemObjMap::FindMemObj(hostPointer)) {
    return hipErrorInvalidValue;
  }
  // right now we have SVM
  *devicePointer = hostPointer;

  return hipSuccess;
}

hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
  HIP_INIT_API(attributes, ptr);

  size_t offset = 0;
  amd::Memory* memObj = getMemoryObject(ptr, offset);
  amd::Context &memObjCtx = memObj->getContext();
  int device = 0;

  if (memObj != nullptr) {
    attributes->memoryType = hipMemoryTypeDevice;
    attributes->hostPointer = memObj->getSvmPtr();
    attributes->devicePointer = memObj->getSvmPtr();
    attributes->isManaged = 0;
    attributes->allocationFlags = memObj->getMemFlags();
    for (auto& ctx : g_devices) {
      ++device;
      if (*ctx == memObjCtx) {
        attributes->device = device;
        break;
      }
    }
  } else {
    attributes->memoryType = hipMemoryTypeHost;
    attributes->hostPointer = (void*)ptr;
    attributes->devicePointer = 0;
    attributes->device = -1;
    attributes->isManaged = 0;
    attributes->allocationFlags = 0;
  }

  return hipSuccess;
}
