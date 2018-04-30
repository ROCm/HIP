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

extern void getChannelOrderAndType(const hipChannelFormatDesc& desc, enum hipTextureReadMode readMode,
                                    cl_channel_order* channelOrder, cl_channel_type* channelType);

hipError_t ihipMalloc(void** ptr, size_t sizeBytes, unsigned int flags)
{
  if (sizeBytes == 0) {
    *ptr = nullptr;
    return hipSuccess;
  }
  else if (!ptr) {
    return hipErrorInvalidValue;
  }

  if (g_context->devices()[0]->info().maxMemAllocSize_ < sizeBytes) {
    return hipErrorOutOfMemory;
  }

  *ptr = amd::SvmBuffer::malloc(*g_context, flags, sizeBytes, g_context->devices()[0]->info().memBaseAddrAlign_);
  if (!*ptr) {
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
    amd::SvmBuffer::free(*g_context, ptr);
    return hipSuccess;
  }
  return hipErrorInvalidValue;
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  HIP_INIT_API(dst, src, sizeBytes, kind);

  amd::Device* device = g_context->devices()[0];

  amd::HostQueue* queue = new amd::HostQueue(*g_context, *device, 0,
                                             amd::CommandQueue::RealTimeDisabled,
                                             amd::CommandQueue::Priority::Normal);
  if (!queue) {
    return hipErrorOutOfMemory;
  }

  amd::Command* command = nullptr;
  amd::Command::EventWaitList waitList;
  amd::Memory* memory;

  switch (kind) {
  case hipMemcpyDeviceToHost:
    memory = amd::SvmManager::FindSvmBuffer(src);
    command = new amd::ReadMemoryCommand(*queue, CL_COMMAND_READ_BUFFER, waitList,
      *memory->asBuffer(), 0, sizeBytes, dst);
    break;
  case hipMemcpyHostToDevice:
    memory = amd::SvmManager::FindSvmBuffer(dst);
    command = new amd::WriteMemoryCommand(*queue, CL_COMMAND_WRITE_BUFFER, waitList,
      *memory->asBuffer(), 0, sizeBytes, src);
    break;
  default:
    assert(!"Shouldn't reach here");
    break;
  }
  if (!command) {
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

  queue->release();

  return hipSuccess;
}

hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  HIP_INIT_API(dst, value, sizeBytes, stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  HIP_INIT_API(dst, value, sizeBytes);

  amd::Device* device = g_context->devices()[0];

  amd::HostQueue* queue = new amd::HostQueue(*g_context, *device, 0,
                                             amd::CommandQueue::RealTimeDisabled,
                                             amd::CommandQueue::Priority::Normal);
  if (!queue) {
    return hipErrorOutOfMemory;
  }

  amd::Command::EventWaitList waitList;
  amd::Memory* memory = amd::SvmManager::FindSvmBuffer(dst);


  amd::Coord3D fillOffset(0, 0, 0);
  amd::Coord3D fillSize(sizeBytes, 1, 1);
  amd::FillMemoryCommand* command =
      new amd::FillMemoryCommand(*queue, CL_COMMAND_FILL_BUFFER, waitList, *memory->asBuffer(),
                                 &value, sizeof(int), fillOffset, fillSize);

  if (!command) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->awaitCompletion();
  command->release();

  queue->release();

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
    amd::SvmBuffer::free(*g_context, ptr);
    return hipSuccess;
  }
  return hipErrorInvalidValue;
}

hipError_t hipFreeArray(hipArray* array) {
  HIP_INIT_API(array);

  if (amd::SvmBuffer::malloced(array->data)) {
    amd::SvmBuffer::free(*g_context, array->data);
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
  amd::Device* device = g_context->devices()[0];
  if(!device) {
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

  amd::Device* device = g_context->devices()[0];

  if ((width == 0) || (height == 0)) {
    *ptr = nullptr;
    return hipSuccess;
  }
  else if (!(device->info().image2DMaxWidth_ >= width &&
           device->info().image2DMaxHeight_ >= height ) || (ptr == nullptr)) {
    return hipErrorInvalidValue;
  }

  if (g_context->devices()[0]->info().maxMemAllocSize_ < (width * height)) {
    return hipErrorOutOfMemory;
  }

  const amd::Image::Format imageFormat(*image_format);

  *pitch = width * imageFormat.getElementSize();

  size_t sizeBytes = *pitch * height * depth;
  *ptr = amd::SvmBuffer::malloc(*g_context, CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeBytes,
                                g_context->devices()[0]->info().memBaseAddrAlign_);

  if (!*ptr) {
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

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
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

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
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

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
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

  amd::Device* device = g_context->devices()[0];

  amd::HostQueue* queue = new amd::HostQueue(*g_context, *device, 0,
                                             amd::CommandQueue::RealTimeDisabled,
                                             amd::CommandQueue::Priority::Normal);
  if (!queue) {
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
  default:
    assert(!"Shouldn't reach here");
    break;
  }
  if (!command) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->awaitCompletion();
  command->release();

  queue->release();

  return hipSuccess;
}

hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(dst, srcArray, wOffset, hOffset, count, kind);

  amd::Device* device = g_context->devices()[0];

  amd::HostQueue* queue = new amd::HostQueue(*g_context, *device, 0,
                                             amd::CommandQueue::RealTimeDisabled,
                                             amd::CommandQueue::Priority::Normal);
  if (!queue) {
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
  default:
    assert(!"Shouldn't reach here");
    break;
  }
  if (!command) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  command->awaitCompletion();
  command->release();

  queue->release();

  return hipSuccess;
}

hipError_t hipMemcpyHtoA(hipArray* dstArray, size_t dstOffset, const void* srcHost, size_t count) {
  HIP_INIT_API(dstArray, dstOffset, srcHost, count);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyAtoH(void* dst, hipArray* srcArray, size_t srcOffset, size_t count) {
  HIP_INIT_API(dst, srcArray, srcOffset, count);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p) {
  HIP_INIT_API(p);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
  HIP_INIT_API(dst, pitch, value, width, height);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemsetD8(hipDeviceptr_t dst, unsigned char value, size_t sizeBytes) {
  HIP_INIT_API(dst, value, sizeBytes);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
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

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
  HIP_INIT_API(attributes, ptr);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}
