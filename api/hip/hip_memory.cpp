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

hipError_t hipMalloc(void** ptr, size_t sizeBytes) {
  HIP_INIT_API(ptr, sizeBytes);

  return ihipMalloc(ptr, sizeBytes, 0);
}

hipError_t hipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags) {
  HIP_INIT_API(ptr, sizeBytes, flags);

  return ihipMalloc(ptr, sizeBytes, CL_MEM_SVM_FINE_GRAIN_BUFFER);
}

hipError_t hipFree(void* ptr) {
  if (amd::SvmBuffer::malloced(ptr)) {
    amd::SvmBuffer::free(*hip::getCurrentContext(), ptr);
    return hipSuccess;
  }
  return hipErrorInvalidValue;
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  HIP_INIT_API(dst, src, sizeBytes, kind);

  amd::HostQueue* queue = hip::getNullStream();

  if (queue == nullptr) {
    return hipErrorOutOfMemory;
  }

  amd::Command* command = nullptr;
  amd::Command::EventWaitList waitList;
  amd::Memory *srcMemory = nullptr;
  amd::Memory *dstMemory = nullptr;

  srcMemory = amd::SvmManager::FindSvmBuffer(src);
  dstMemory = amd::SvmManager::FindSvmBuffer(dst);

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
    command = new amd::ReadMemoryCommand(*queue, CL_COMMAND_READ_BUFFER, waitList,
              *srcMemory->asBuffer(), 0, sizeBytes, dst);
    break;
  case hipMemcpyHostToDevice:
    command = new amd::WriteMemoryCommand(*queue, CL_COMMAND_WRITE_BUFFER, waitList,
              *dstMemory->asBuffer(), 0, sizeBytes, src);
    break;
  case hipMemcpyDeviceToDevice:
    command = new amd::CopyMemoryCommand(*queue, CL_COMMAND_COPY_BUFFER, waitList,
              *srcMemory->asBuffer(),*dstMemory->asBuffer(), 0, 0, sizeBytes);
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

// FIXME: virtualize MemoryCommand::validateMemory()
#if 0
  // Make sure we have memory for the command execution
  if (CL_SUCCESS != command->validateMemory()) {
    delete command;
    return hipErrorMemoryAllocation;
  }
#endif

  command->enqueue();
  command->awaitCompletion();
  command->release();

  return hipSuccess;
}

hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  HIP_INIT_API(dst, value, sizeBytes, stream);

  amd::HostQueue* queue;

  if (stream == nullptr) {
    queue = hip::getNullStream();
    if (queue == nullptr) {
      return hipErrorOutOfMemory;
    }
  } else {
    queue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }

  amd::Command::EventWaitList waitList;
  amd::Memory* memory = amd::SvmManager::FindSvmBuffer(dst);

  amd::Coord3D fillOffset(0, 0, 0);
  amd::Coord3D fillSize(sizeBytes, 1, 1);
  amd::FillMemoryCommand* command =
      new amd::FillMemoryCommand(*queue, CL_COMMAND_FILL_BUFFER, waitList, *memory->asBuffer(),
                                 &value, sizeof(int), fillOffset, fillSize);

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->release();

  return hipSuccess;
}

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  HIP_INIT_API(dst, value, sizeBytes);

  amd::HostQueue* queue = hip::getNullStream();

  if (queue == nullptr) {
    return hipErrorOutOfMemory;
  }

  amd::Command::EventWaitList waitList;
  amd::Memory* memory = amd::SvmManager::FindSvmBuffer(dst);


  amd::Coord3D fillOffset(0, 0, 0);
  amd::Coord3D fillSize(sizeBytes, 1, 1);
  amd::FillMemoryCommand* command =
      new amd::FillMemoryCommand(*queue, CL_COMMAND_FILL_BUFFER, waitList, *memory->asBuffer(),
                                 &value, sizeof(int), fillOffset, fillSize);

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->awaitCompletion();
  command->release();

  return hipSuccess;
}

hipError_t hipMemPtrGetInfo(void *ptr, size_t *size) {
  HIP_INIT_API(ptr, size);

  amd::Memory* svmMem = amd::SvmManager::FindSvmBuffer(ptr);

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
  amd::Memory* svmMem = amd::SvmManager::FindSvmBuffer(ptr);

  if (svmMem == nullptr) {
    return hipErrorInvalidDevicePointer;
  }

  *pbase = ptr;
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
  *ptr = amd::SvmBuffer::malloc(*hip::getCurrentContext(), CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeBytes,
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

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  HIP_INIT_API(hostPtr, sizeBytes, flags);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipHostUnregister(void* hostPtr) {
  HIP_INIT_API(hostPtr);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

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

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes) {
  HIP_INIT_API(dst, src, sizeBytes);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes) {
  HIP_INIT_API(dst, src, sizeBytes);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyHtoH(void* dst, void* src, size_t sizeBytes) {
  HIP_INIT_API(dst, src, sizeBytes);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(dst, src, sizeBytes, kind, stream);

  amd::Command* command = nullptr;
  amd::Command::EventWaitList waitList;
  amd::HostQueue* queue;
  amd::Memory *srcMemory = nullptr;
  amd::Memory *dstMemory = nullptr;

  if (stream == nullptr) {
    queue = hip::getNullStream();
    if (queue == nullptr) {
      return hipErrorOutOfMemory;
    }
  } else {
    queue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }

  srcMemory = amd::SvmManager::FindSvmBuffer(src);
  dstMemory = amd::SvmManager::FindSvmBuffer(dst);

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
    command = new amd::ReadMemoryCommand(*queue, CL_COMMAND_READ_BUFFER, waitList,
      *srcMemory->asBuffer(), 0, sizeBytes, dst);
    break;
  case hipMemcpyHostToDevice:
    command = new amd::WriteMemoryCommand(*queue, CL_COMMAND_WRITE_BUFFER, waitList,
      *dstMemory->asBuffer(), 0, sizeBytes, src);
    break;
  case hipMemcpyDeviceToDevice:
    command = new amd::CopyMemoryCommand(*queue, CL_COMMAND_COPY_BUFFER, waitList,
              *srcMemory->asBuffer(),*dstMemory->asBuffer(), 0, 0, sizeBytes);
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
  command->release();

  return hipSuccess;
}


hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes,
                              hipStream_t stream) {
  HIP_INIT_API(dst, src, sizeBytes, stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream) {
  HIP_INIT_API(dst, src, sizeBytes, stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream) {
  HIP_INIT_API(dst, src, sizeBytes, stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(dst, dpitch, src, spitch, width, height, kind);

  amd::HostQueue* queue = hip::getNullStream();
  if (queue == nullptr) {
    return hipErrorOutOfMemory;
  }

  // Create buffer rectangle info structure
  amd::BufferRect srcRect;
  amd::BufferRect dstRect;
  amd::Memory* srcPtr = amd::SvmManager::FindSvmBuffer(src);
  amd::Memory* dstPtr = amd::SvmManager::FindSvmBuffer(dst);
  size_t region[3] = {width, height, 0};
  size_t src_slice_pitch = spitch * height;
  size_t dst_slice_pitch = dpitch * height;
  size_t origin[3] = { };

  if (!srcRect.create(origin, region, spitch, src_slice_pitch) ||
      !dstRect.create(origin, region, dpitch, dst_slice_pitch)) {
    return hipErrorInvalidValue;
  }

  amd::Coord3D srcStart(srcRect.start_, 0, 0);
  amd::Coord3D dstStart(dstRect.start_, 0, 0);
  amd::Coord3D srcEnd(srcRect.end_, 1, 1);
  amd::Coord3D dstEnd(dstRect.end_, 1, 1);

  if (!srcPtr->asBuffer()->validateRegion(srcStart, srcEnd) ||
      !dstPtr->asBuffer()->validateRegion(dstStart, dstEnd)) {
    return hipErrorInvalidValue;
  }

  amd::Command::EventWaitList waitList;
  amd::Coord3D size(region[0], region[1], region[2]);

  amd::CopyMemoryCommand* command =
      new amd::CopyMemoryCommand(*queue, CL_COMMAND_COPY_BUFFER_RECT, waitList, *srcPtr->asBuffer(),
                                 *dstPtr->asBuffer(), srcStart, dstStart, size, srcRect, dstRect);

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->awaitCompletion();
  command->release();

  return hipSuccess;
}

hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
  HIP_INIT_API(pCopy);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(dst, dpitch, src, spitch, width, height, kind, stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(dst, wOffset, hOffset, src, spitch, width, height, kind);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyToArray(hipArray* dstArray, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(dstArray, wOffset, hOffset, src, count, kind);

  amd::HostQueue* queue = hip::getNullStream();
  if (queue == nullptr) {
    return hipErrorOutOfMemory;
  }

  amd::Command* command = nullptr;
  amd::Command::EventWaitList waitList;
  amd::Memory* memory;

  amd::Coord3D dstOffset(wOffset, hOffset, 0);

  switch (kind) {
  case hipMemcpyDeviceToHost:
    assert(!"Invalid case");
    /* fall thru */
  case hipMemcpyHostToDevice:
    memory = amd::SvmManager::FindSvmBuffer(dstArray->data);
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

  amd::HostQueue* queue = hip::getNullStream();
  if (queue == nullptr) {
    return hipErrorOutOfMemory;
  }

  amd::Command* command = nullptr;
  amd::Command::EventWaitList waitList;
  amd::Memory* memory;

  amd::Coord3D srcOffset(wOffset, hOffset, 0);

  switch (kind) {
  case hipMemcpyHostToDevice:
    assert(!"Invalid case");
    /* fall thru */
  case hipMemcpyDeviceToHost:
    memory = amd::SvmManager::FindSvmBuffer(srcArray->data);
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

  amd::HostQueue* queue = hip::getNullStream();
  if (queue == nullptr) {
    return hipErrorOutOfMemory;
  }

  amd::Command::EventWaitList waitList;
  amd::Memory* memory = amd::SvmManager::FindSvmBuffer(dstArray->data);
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

  amd::HostQueue* queue = hip::getNullStream();
  if (queue == nullptr) {
    return hipErrorOutOfMemory;
  }

  amd::Command::EventWaitList waitList;
  amd::Memory* memory = amd::SvmManager::FindSvmBuffer(srcArray->data);
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

  amd::HostQueue* queue = hip::getNullStream();

  if (queue == nullptr) {
    return hipErrorOutOfMemory;
  }

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
  amd::Memory* src = amd::SvmManager::FindSvmBuffer(srcPtr);
  amd::Memory* dst = amd::SvmManager::FindSvmBuffer(dstPtr);

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

hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
  HIP_INIT_API(dst, pitch, value, width, height);

  amd::HostQueue* queue = hip::getNullStream();
  if (queue == nullptr) {
    return hipErrorOutOfMemory;
  }

  amd::Command::EventWaitList waitList;
  amd::Memory* memory = amd::SvmManager::FindSvmBuffer(dst);

  amd::Coord3D fillOffset(0, 0, 0);

  size_t sizeBytes = pitch * height;
  amd::Coord3D fillSize(sizeBytes, 1, 1);
  amd::FillMemoryCommand* command =
      new amd::FillMemoryCommand(*queue, CL_COMMAND_FILL_BUFFER, waitList, *memory->asBuffer(),
                                 &value, sizeof(int), fillOffset, fillSize);

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->awaitCompletion();
  command->release();

  return hipSuccess;
}

hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value,
                            size_t width, size_t height, hipStream_t stream) {
  HIP_INIT_API(dst, pitch, value, width, height, stream);

  amd::HostQueue* queue = nullptr;
  if (stream == nullptr) {
    queue = hip::getNullStream();
    if (queue == nullptr) {
      return hipErrorOutOfMemory;
    }
  } else {
    queue = as_amd(reinterpret_cast<cl_command_queue>(stream))->asHostQueue();
  }

  amd::Command::EventWaitList waitList;
  amd::Memory* memory = amd::SvmManager::FindSvmBuffer(dst);

  amd::Coord3D fillOffset(0, 0, 0);

  size_t sizeBytes = pitch * height;
  amd::Coord3D fillSize(sizeBytes, 1, 1);
  amd::FillMemoryCommand* command =
      new amd::FillMemoryCommand(*queue, CL_COMMAND_FILL_BUFFER, waitList, *memory->asBuffer(),
                                 &value, sizeof(int), fillOffset, fillSize);

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->release();

  return hipSuccess;
}

hipError_t hipMemsetD8(hipDeviceptr_t dst, unsigned char value, size_t sizeBytes) {
  HIP_INIT_API(dst, value, sizeBytes);

  amd::HostQueue* queue = hip::getNullStream();
  if (queue == nullptr) {
    return hipErrorOutOfMemory;
  }

  amd::Command::EventWaitList waitList;
  amd::Memory* memory = amd::SvmManager::FindSvmBuffer(dst);

  amd::Coord3D fillOffset(0, 0, 0);
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

  if (!amd::SvmBuffer::malloced(hostPointer)) {
    return hipErrorInvalidValue;
  }
  // right now we have SVM
  *devicePointer = hostPointer;

  return hipSuccess;
}

hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
  HIP_INIT_API(attributes, ptr);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}
