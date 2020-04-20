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
#include "hip_internal.hpp"
#include "hip_conversions.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/memory.hpp"

amd::Memory* getMemoryObject(const void* ptr, size_t& offset) {
  amd::Memory *memObj = amd::MemObjMap::FindMemObj(ptr);
  if (memObj != nullptr) {
    if (memObj->getSvmPtr() != nullptr) {
      // SVM pointer
      offset = reinterpret_cast<size_t>(ptr) - reinterpret_cast<size_t>(memObj->getSvmPtr());
    } else if (memObj->getHostMem() != nullptr) {
      // Prepinned memory
      offset = reinterpret_cast<size_t>(ptr) - reinterpret_cast<size_t>(memObj->getHostMem());
    } else {
      ShouldNotReachHere();
    }
  }
  return memObj;
}

hipError_t ihipFree(void *ptr)
{
  if (ptr == nullptr) {
    return hipSuccess;
  }
  if (amd::SvmBuffer::malloced(ptr)) {
    for (auto& dev : g_devices) {
      amd::HostQueue* queue = hip::getNullStream(*dev->asContext());
      if (queue != nullptr) {
        queue->finish();
      }
      hip::syncStreams(dev->deviceId());
    }
    amd::SvmBuffer::free(*hip::getCurrentDevice()->asContext(), ptr);
    return hipSuccess;
  }
  return hipErrorInvalidValue;
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

  amd::Context* amdContext = ((flags & CL_MEM_SVM_FINE_GRAIN_BUFFER) != 0)?
    hip::host_device->asContext() : hip::getCurrentDevice()->asContext();

  if (amdContext == nullptr) {
    return hipErrorOutOfMemory;
  }

  if (amdContext->devices()[0]->info().maxMemAllocSize_ < sizeBytes) {
    return hipErrorOutOfMemory;
  }

  *ptr = amd::SvmBuffer::malloc(*amdContext, flags, sizeBytes, amdContext->devices()[0]->info().memBaseAddrAlign_);
  if (*ptr == nullptr) {
    return hipErrorOutOfMemory;
  }
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%-5d: [%zx] ihipMalloc ptr=0x%zx",  getpid(),std::this_thread::get_id(), *ptr);
  return hipSuccess;
}

hipError_t ihipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                      amd::HostQueue& queue, bool isAsync = false) {
  if (sizeBytes == 0) {
    // Skip if nothing needs writing.
    return hipSuccess;
  }

  amd::Command* command = nullptr;
  amd::Command::EventWaitList waitList;

  size_t sOffset = 0;
  amd::Memory *srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory *dstMemory = getMemoryObject(dst, dOffset);
  amd::Device* queueDevice = &queue.device();

  if (((srcMemory == nullptr) && (dstMemory == nullptr)) ||
      (kind == hipMemcpyHostToHost)) {
    queue.finish();
    memcpy(dst, src, sizeBytes);
    return hipSuccess;
  } else if ((srcMemory == nullptr) && (dstMemory != nullptr)) {
    amd::HostQueue* pQueue = &queue;
    if (queueDevice != dstMemory->getContext().devices()[0]) {
      pQueue = hip::getNullStream(dstMemory->getContext());
      waitList.push_back(queue.getLastQueuedCommand(true));
    }
    command = new amd::WriteMemoryCommand(*pQueue, CL_COMMAND_WRITE_BUFFER, waitList,
              *dstMemory->asBuffer(), dOffset, sizeBytes, src);
    isAsync = false;
  } else if ((srcMemory != nullptr) && (dstMemory == nullptr)) {
    amd::HostQueue* pQueue = &queue;
    if (queueDevice != srcMemory->getContext().devices()[0]) {
      pQueue = hip::getNullStream(srcMemory->getContext());
      waitList.push_back(queue.getLastQueuedCommand(true));
    }
    command = new amd::ReadMemoryCommand(*pQueue, CL_COMMAND_READ_BUFFER, waitList,
              *srcMemory->asBuffer(), sOffset, sizeBytes, dst);
    isAsync = false;
  } else if ((srcMemory != nullptr) && (dstMemory != nullptr)) {
    if (queueDevice != srcMemory->getContext().devices()[0]) {
      amd::Coord3D srcOffset(sOffset, 0, 0);
      amd::Coord3D dstOffset(dOffset, 0, 0);
      amd::Coord3D copySize(sizeBytes, 1, 1);
      command = new amd::CopyMemoryP2PCommand(queue, CL_COMMAND_COPY_BUFFER, waitList,
                *srcMemory->asBuffer(),*dstMemory->asBuffer(), srcOffset, dstOffset, copySize);
      command->enqueue();
      if (!isAsync) {
        command->awaitCompletion();
      }
      command->release();
      return hipSuccess;
    }
    if (queueDevice != dstMemory->getContext().devices()[0]) {
      amd::Coord3D srcOffset(sOffset, 0, 0);
      amd::Coord3D dstOffset(dOffset, 0, 0);
      amd::Coord3D copySize(sizeBytes, 1, 1);
      command = new amd::CopyMemoryP2PCommand(queue, CL_COMMAND_COPY_BUFFER, waitList,
                *srcMemory->asBuffer(),*dstMemory->asBuffer(), srcOffset, dstOffset, copySize);
      command->enqueue();
      if (!isAsync) {
        command->awaitCompletion();
      }
      command->release();
      return hipSuccess;
    }
    command = new amd::CopyMemoryCommand(queue, CL_COMMAND_COPY_BUFFER, waitList,
              *srcMemory->asBuffer(),*dstMemory->asBuffer(), sOffset, dOffset, sizeBytes);
  }

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  command->enqueue();
  if (!isAsync) {
    command->awaitCompletion();
  }
  command->release();

  if (waitList.size() > 0) {
    waitList[0]->release();
  }

  return hipSuccess;
}

hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags) {
  HIP_INIT_API(hipExtMallocWithFlags, ptr, sizeBytes, flags);

  if (flags != hipDeviceMallocDefault &&
      flags != hipDeviceMallocFinegrained) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipMalloc(ptr, sizeBytes, (flags & hipDeviceMallocFinegrained)? CL_MEM_SVM_ATOMICS: 0));
}

hipError_t hipMalloc(void** ptr, size_t sizeBytes) {
  HIP_INIT_API(hipMalloc, ptr, sizeBytes);

  HIP_RETURN(ihipMalloc(ptr, sizeBytes, 0));
}

hipError_t hipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags) {
  HIP_INIT_API(hipHostMalloc, ptr, sizeBytes, flags);

  if (ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *ptr = nullptr;

  const unsigned int coherentFlags = hipHostMallocCoherent | hipHostMallocNonCoherent;

  // can't have both Coherent and NonCoherent flags set at the same time
  if ((flags & coherentFlags) == coherentFlags) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  unsigned int ihipFlags = CL_MEM_SVM_FINE_GRAIN_BUFFER | (flags << 16);
  if (flags == 0 ||
      flags & (hipHostMallocCoherent | hipHostMallocMapped) ||
     (!(flags & hipHostMallocNonCoherent) && HIP_HOST_COHERENT)) {
    ihipFlags |= CL_MEM_SVM_ATOMICS;
  }

  HIP_RETURN(ihipMalloc(ptr, sizeBytes, ihipFlags));
}

hipError_t hipMallocManaged(void** devPtr, size_t size,
                            unsigned int flags) {
  HIP_INIT_API(hipMallocManaged, devPtr, size, flags);

  if (flags != hipMemAttachGlobal) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipMalloc(devPtr, size, CL_MEM_SVM_FINE_GRAIN_BUFFER));
}

hipError_t hipFree(void* ptr) {
  HIP_INIT_API(hipFree, ptr);

  HIP_RETURN(ihipFree(ptr));
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy, dst, src, sizeBytes, kind);

  hip::syncStreams();
  amd::HostQueue* queue = hip::getNullStream();
  HIP_RETURN(ihipMemcpy(dst, src, sizeBytes, kind, *queue));
}

hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes,
                               hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyWithStream, dst, src, sizeBytes, kind, stream);

  amd::HostQueue* queue = hip::getQueue(stream);

  HIP_RETURN(ihipMemcpy(dst, src, sizeBytes, kind, *queue, false));
}

hipError_t hipMemPtrGetInfo(void *ptr, size_t *size) {
  HIP_INIT_API(hipMemPtrGetInfo, ptr, size);

  size_t offset = 0;
  amd::Memory* svmMem = getMemoryObject(ptr, offset);

  if (svmMem == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *size = svmMem->getSize();

  HIP_RETURN(hipSuccess);
}

hipError_t hipHostFree(void* ptr) {
  HIP_INIT_API(hipHostFree, ptr);

  HIP_RETURN(ihipFree(ptr));
}

hipError_t ihipArrayDestroy(hipArray* array) {
  if (array == nullptr) {
    return hipErrorInvalidValue;
  }

  cl_mem memObj = reinterpret_cast<cl_mem>(array->data);
  if (is_valid(memObj) == false) {
    return hipErrorInvalidValue;
  }
  for (auto& dev : g_devices) {
    amd::HostQueue* queue = hip::getNullStream(*dev->asContext());
    if (queue != nullptr) {
      queue->finish();
    }
    hip::syncStreams(dev->deviceId());
  }
  as_amd(memObj)->release();

  delete array;

  return hipSuccess;
}

hipError_t hipFreeArray(hipArray* array) {
  HIP_INIT_API(hipFreeArray, array);

  HIP_RETURN(ihipArrayDestroy(array));
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
  HIP_INIT_API(hipMemGetAddressRange, pbase, psize, dptr);

  // Since we are using SVM buffer DevicePtr and HostPtr is the same
  void* ptr = dptr;
  size_t offset = 0;
  amd::Memory* svmMem = getMemoryObject(ptr, offset);

  if (svmMem == nullptr) {
    HIP_RETURN(hipErrorInvalidDevicePointer);
  }

  *pbase = svmMem->getSvmPtr();
  *psize = svmMem->getSize();

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  HIP_INIT_API(hipMemGetInfo, free, total);

  size_t freeMemory[2];
  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  if(device == nullptr) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if(!device->globalFreeMemory(freeMemory)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *free = freeMemory[0] * Ki;
  *total = device->info().globalMemSize_;

  HIP_RETURN(hipSuccess);
}

hipError_t ihipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height, size_t depth,
                           cl_mem_object_type imageType, const cl_image_format* image_format) {

  amd::Device* device = hip::getCurrentDevice()->devices()[0];

  if (ptr == nullptr) {
    return hipErrorInvalidValue;
  }

  if ((width == 0) || (height == 0)) {
    *ptr = nullptr;
    return hipSuccess;
  }

  const amd::Image::Format imageFormat(*image_format);

  *pitch = amd::alignUp(width * imageFormat.getElementSize(), device->info().imagePitchAlignment_);

  size_t sizeBytes = *pitch * height * depth;

  if (device->info().maxMemAllocSize_ < sizeBytes) {
    return hipErrorOutOfMemory;
  }

  *ptr = amd::SvmBuffer::malloc(*hip::getCurrentDevice()->asContext(), 0, sizeBytes,
                                device->info().memBaseAddrAlign_);

  if (*ptr == nullptr) {
    return hipErrorOutOfMemory;
  }

  return hipSuccess;
}


hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
  HIP_INIT_API(hipMallocPitch, ptr, pitch, width, height);

  const cl_image_format image_format = { CL_R, CL_UNSIGNED_INT8 };
  HIP_RETURN(ihipMallocPitch(ptr, pitch, width, height, 1, CL_MEM_OBJECT_IMAGE2D, &image_format));
}

hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
  HIP_INIT_API(hipMalloc3D, pitchedDevPtr, extent);

  size_t pitch = 0;

  if (pitchedDevPtr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
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

  HIP_RETURN(status);
}

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
                            amd::Memory* buffer) {
  const amd::Image::Format imageFormat({channelOrder, channelType});
  if (!imageFormat.isValid()) {
    return nullptr;
  }

  amd::Context& context = *hip::getCurrentDevice()->asContext();
  if (!imageFormat.isSupported(context, imageType)) {
    return nullptr;
  }

  const std::vector<amd::Device*>& devices = context.devices();
  if (!devices[0]->info().imageSupport_) {
    return nullptr;
  }

  if (!amd::Image::validateDimensions(devices,
                                      imageType,
                                      imageWidth,
                                      imageHeight,
                                      imageDepth,
                                      imageArraySize)) {
    return nullptr;
  }

  // TODO validate the image descriptor.

  amd::Image* image = nullptr;
  if (buffer != nullptr) {
    switch (imageType) {
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
    case CL_MEM_OBJECT_IMAGE2D:
      image = new (context) amd::Image(*buffer->asBuffer(),
                                       imageType,
                                       CL_MEM_READ_WRITE,
                                       imageFormat,
                                       imageWidth,
                                       (imageHeight == 0) ? 1 : imageHeight,
                                       (imageDepth == 0) ? 1 : imageDepth,
                                       imageRowPitch,
                                       imageSlicePitch);
      break;
    default:
      ShouldNotReachHere();
    }
  } else {
    switch (imageType) {
    case CL_MEM_OBJECT_IMAGE1D:
    case CL_MEM_OBJECT_IMAGE2D:
    case CL_MEM_OBJECT_IMAGE3D:
      image = new (context) amd::Image(context,
                                      imageType,
                                      CL_MEM_READ_WRITE,
                                      imageFormat,
                                      imageWidth,
                                      (imageHeight == 0) ? 1 : imageHeight,
                                      (imageDepth == 0) ? 1 : imageDepth,
                                      imageWidth * imageFormat.getElementSize(), /* row pitch */
                                      imageWidth * imageHeight * imageFormat.getElementSize(), /* slice pitch */
                                      numMipLevels);
      break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      image = new (context) amd::Image(context,
                                       imageType,
                                       CL_MEM_READ_WRITE,
                                       imageFormat,
                                       imageWidth,
                                       imageArraySize,
                                       1, /* image depth */
                                       imageWidth * imageFormat.getElementSize(),
                                       imageWidth * imageHeight * imageFormat.getElementSize(),
                                       numMipLevels);
      break;
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      image = new (context) amd::Image(context,
                                       imageType,
                                       CL_MEM_READ_WRITE,
                                       imageFormat,
                                       imageWidth,
                                       imageHeight,
                                       imageArraySize,
                                       imageWidth * imageFormat.getElementSize(),
                                       imageWidth * imageHeight * imageFormat.getElementSize(),
                                       numMipLevels);
      break;
    default:
      ShouldNotReachHere();
    }
  }

  if (image == nullptr) {
    return nullptr;
  }

  if (!image->create(nullptr)) {
    delete image;
    return nullptr;
  }

  return image;
}

hipError_t ihipArrayCreate(hipArray** array,
                           const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray,
                           unsigned int numMipmapLevels) {
  // NumChannels specifies the number of packed components per HIP array element; it may be 1, 2, or 4;
  if ((pAllocateArray->NumChannels != 1) &&
      (pAllocateArray->NumChannels != 2) &&
      (pAllocateArray->NumChannels != 4)) {
    return hipErrorInvalidValue;
  }

  if ((pAllocateArray->Flags & hipArraySurfaceLoadStore) ||
      (pAllocateArray->Flags & hipArrayCubemap) ||
      (pAllocateArray->Flags & hipArrayTextureGather)) {
    return hipErrorNotSupported;
  }

  const cl_channel_order channelOrder = hip::getCLChannelOrder(pAllocateArray->NumChannels, 0);
  const cl_channel_type channelType = hip::getCLChannelType(pAllocateArray->Format, hipReadModeElementType);
  const cl_mem_object_type imageType = hip::getCLMemObjectType(pAllocateArray->Width,
                                                               pAllocateArray->Height,
                                                               pAllocateArray->Depth,
                                                               pAllocateArray->Flags);

  amd::Image* image = ihipImageCreate(channelOrder,
                                      channelType,
                                      imageType,
                                      pAllocateArray->Width,
                                      pAllocateArray->Height,
                                      pAllocateArray->Depth,
                                      // The number of layers is determined by the depth extent.
                                      pAllocateArray->Depth, /* array size */
                                      0, /* row pitch */
                                      0, /* slice pitch */
                                      numMipmapLevels,
                                      nullptr /* buffer */);

  if (image == nullptr) {
    return hipErrorInvalidValue;
  }

  cl_mem memObj = as_cl<amd::Memory>(image);
  *array = new hipArray{reinterpret_cast<void*>(memObj)};

  // It is UB to call hipGet*() on an array created via hipArrayCreate()/hipArray3DCreate().
  // This is due to hip not differentiating between runtime and driver types.
  // TODO change the hipArray struct in driver_types.h.
  (*array)->desc = hip::getChannelFormatDesc(pAllocateArray->NumChannels, pAllocateArray->Format);
  (*array)->width = pAllocateArray->Width;
  (*array)->height = pAllocateArray->Height;
  (*array)->depth = pAllocateArray->Depth;
  (*array)->Format = pAllocateArray->Format;
  (*array)->NumChannels = pAllocateArray->NumChannels;

  return hipSuccess;
}

hipError_t hipArrayCreate(hipArray** array,
                          const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
  HIP_INIT_API(hipArrayCreate, array, pAllocateArray);

  HIP_ARRAY3D_DESCRIPTOR desc = {pAllocateArray->Width,
                                 pAllocateArray->Height,
                                 0, /* Depth */
                                 pAllocateArray->Format,
                                 pAllocateArray->NumChannels,
                                 hipArrayDefault /* Flags */};

  HIP_RETURN(ihipArrayCreate(array, &desc, 0));
}


hipError_t hipMallocArray(hipArray** array,
                          const hipChannelFormatDesc* desc,
                          size_t width,
                          size_t height,
                          unsigned int flags) {
  HIP_INIT_API(hipMallocArray, array, desc, width, height, flags);

  HIP_ARRAY3D_DESCRIPTOR allocateArray = {width,
                                          height,
                                          0, /* Depth */
                                          hip::getArrayFormat(*desc),
                                          hip::getNumChannels(*desc),
                                          flags};

  HIP_RETURN(ihipArrayCreate(array, &allocateArray, 0 /* numMipLevels */));
}

hipError_t hipArray3DCreate(hipArray** array,
                            const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray) {
  HIP_INIT_API(hipArray3DCreate, array, pAllocateArray);

  HIP_RETURN(ihipArrayCreate(array, pAllocateArray, 0 /* numMipLevels */));
}

hipError_t hipMalloc3DArray(hipArray_t* array,
                            const hipChannelFormatDesc* desc,
                            hipExtent extent,
                            unsigned int flags) {
  HIP_INIT_API(hipMalloc3DArray, array, desc, extent, flags);

  HIP_ARRAY3D_DESCRIPTOR allocateArray = {extent.width,
                                          extent.height,
                                          extent.depth,
                                          hip::getArrayFormat(*desc),
                                          hip::getNumChannels(*desc),
                                          flags};

  HIP_RETURN(ihipArrayCreate(array, &allocateArray, 0));
}

hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
  HIP_INIT_API(hipHostGetFlags, flagsPtr, hostPtr);

  if (flagsPtr == nullptr ||
      hostPtr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  size_t offset = 0;
  amd::Memory* svmMem = getMemoryObject(hostPtr, offset);

  if (svmMem == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *flagsPtr = svmMem->getMemFlags() >> 16;

  HIP_RETURN(hipSuccess);
}

hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  HIP_INIT_API(hipHostRegister, hostPtr, sizeBytes, flags);
  if(hostPtr != nullptr) {
    amd::Memory* mem = new (*hip::host_device->asContext()) amd::Buffer(*hip::host_device->asContext(), CL_MEM_USE_HOST_PTR | CL_MEM_SVM_ATOMICS, sizeBytes);

    constexpr bool sysMemAlloc = false;
    constexpr bool skipAlloc = false;
    constexpr bool forceAlloc = true;
    if (!mem->create(hostPtr, sysMemAlloc, skipAlloc, forceAlloc)) {
      mem->release();
      HIP_RETURN(hipErrorOutOfMemory);
    }

    for (const auto& device: hip::getCurrentDevice()->devices()) {
      // Since the amd::Memory object is shared between all devices
      // it's fine to have multiple addresses mapped to it
      const device::Memory* devMem = mem->getDeviceMemory(*device);
      amd::MemObjMap::AddMemObj(reinterpret_cast<void*>(devMem->virtualAddress()), mem);
    }

    amd::MemObjMap::AddMemObj(hostPtr, mem);
    HIP_RETURN(hipSuccess);
  } else {
    HIP_RETURN(ihipMalloc(&hostPtr, sizeBytes, flags));
  }
}

hipError_t hipHostUnregister(void* hostPtr) {
  HIP_INIT_API(hipHostUnregister, hostPtr);

  for (auto& dev : g_devices) {
    amd::HostQueue* queue = hip::getNullStream(*dev->asContext());
    if (queue != nullptr) {
      queue->finish();
    }
    hip::syncStreams(dev->deviceId());
  }

  if (amd::SvmBuffer::malloced(hostPtr)) {
    amd::SvmBuffer::free(*hip::host_device->asContext(), hostPtr);
    HIP_RETURN(hipSuccess);
  } else {
    size_t offset = 0;
    amd::Memory* mem = getMemoryObject(hostPtr, offset);

    if(mem) {
      for (const auto& device: hip::getCurrentDevice()->devices()) {
        const device::Memory* devMem = mem->getDeviceMemory(*device);
        amd::MemObjMap::RemoveMemObj(reinterpret_cast<void*>(devMem->virtualAddress()));
      }
      amd::MemObjMap::RemoveMemObj(hostPtr);
      mem->release();
      HIP_RETURN(hipSuccess);
    }
  }

  HIP_RETURN(hipErrorInvalidValue);
}

// Deprecated function:
hipError_t hipHostAlloc(void** ptr, size_t sizeBytes, unsigned int flags) {
  HIP_RETURN(ihipMalloc(ptr, sizeBytes, flags));
};


hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t count,
                             size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpyToSymbol, symbol, src, count, offset, kind);

  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  std::string symbolName;
  if (!PlatformState::instance().findSymbol(symbol, symbolName)) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  /* Get address and size for the global symbol */
  if (!PlatformState::instance().getGlobalVar(symbolName.c_str(), ihipGetDevice(), nullptr,
                                              &device_ptr, &sym_size)) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }

  /* Size Check to make sure offset is correct */
  if ((offset + count) != sym_size) {
    return HIP_RETURN(hipErrorInvalidDevicePointer);
  }

  device_ptr = reinterpret_cast<address>(device_ptr) + offset;

  /* Copy memory from source to destination address */
  HIP_RETURN(hipMemcpy(device_ptr, src, count, kind));
}

hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t count,
                               size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpyFromSymbol, symbol, dst, count, offset, kind);

  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  std::string symbolName;
  if (!PlatformState::instance().findSymbol(symbol, symbolName)) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  /* Get address and size for the global symbol */
  if (!PlatformState::instance().getGlobalVar(symbolName.c_str(), ihipGetDevice(), nullptr,
                                              &device_ptr, &sym_size)) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }

  /* Size Check to make sure offset is correct */
  if ((offset + count) != sym_size) {
    return HIP_RETURN(hipErrorInvalidDevicePointer);
  }

  device_ptr = reinterpret_cast<address>(device_ptr) + offset;

  /* Copy memory from source to destination address */
  HIP_RETURN(hipMemcpy(dst, device_ptr, count, kind));
}

hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count,
                                  size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyToSymbolAsync, symbol, src, count, offset, kind, stream);

  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  std::string symbolName;
  if (!PlatformState::instance().findSymbol(symbol, symbolName)) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  /* Get address and size for the global symbol */
  if (!PlatformState::instance().getGlobalVar(symbolName.c_str(), ihipGetDevice(), nullptr,
                                              &device_ptr, &sym_size)) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }

  /* Size Check to make sure offset is correct */
  if ((offset + count) != sym_size) {
    return HIP_RETURN(hipErrorInvalidDevicePointer);
  }

  device_ptr = reinterpret_cast<address>(device_ptr) + offset;

  /* Copy memory from source to destination address */
  HIP_RETURN(hipMemcpyAsync(device_ptr, src, count, kind, stream));
}

hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count,
                                    size_t offset, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyFromSymbolAsync, symbol, dst, count, offset, kind, stream);

  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  std::string symbolName;
  if (!PlatformState::instance().findSymbol(symbol, symbolName)) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  /* Get address and size for the global symbol */
  if (!PlatformState::instance().getGlobalVar(symbolName.c_str(), ihipGetDevice(), nullptr,
                                              &device_ptr, &sym_size)) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }

  /* Size Check to make sure offset is correct */
  if ((offset + count) != sym_size) {
    return HIP_RETURN(hipErrorInvalidDevicePointer);
  }

  device_ptr = reinterpret_cast<address>(device_ptr) + offset;

  /* Copy memory from source to destination address */
  HIP_RETURN(hipMemcpyAsync(dst, device_ptr, count, kind, stream));
}

hipError_t hipMemcpyHtoD(hipDeviceptr_t dstDevice,
                         void* srcHost,
                         size_t ByteCount) {
  HIP_INIT_API(hipMemcpyHtoD, dstDevice, srcHost, ByteCount);

  HIP_RETURN(ihipMemcpy(dstDevice, srcHost, ByteCount, hipMemcpyHostToDevice, *hip::getQueue(nullptr)));
}

hipError_t hipMemcpyDtoH(void* dstHost,
                         hipDeviceptr_t srcDevice,
                         size_t ByteCount) {
  HIP_INIT_API(hipMemcpyDtoH, dstHost, srcDevice, ByteCount);

  HIP_RETURN(ihipMemcpy(dstHost, srcDevice, ByteCount, hipMemcpyDeviceToHost, *hip::getQueue(nullptr)));
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t dstDevice,
                         hipDeviceptr_t srcDevice,
                         size_t ByteCount) {
  HIP_INIT_API(hipMemcpyDtoD, dstDevice, srcDevice, ByteCount);

  HIP_RETURN(ihipMemcpy(dstDevice, srcDevice, ByteCount, hipMemcpyDeviceToDevice, *hip::getQueue(nullptr)));
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyAsync, dst, src, sizeBytes, kind, stream);

  amd::HostQueue* queue = hip::getQueue(stream);

  HIP_RETURN(ihipMemcpy(dst, src, sizeBytes, kind, *queue, true));
}

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dstDevice,
                              void* srcHost,
                              size_t ByteCount,
                              hipStream_t stream) {
  HIP_INIT_API(hipMemcpyHtoDAsync, dstDevice, srcHost, ByteCount, stream);

  HIP_RETURN(ihipMemcpy(dstDevice, srcHost, ByteCount, hipMemcpyHostToDevice, *hip::getQueue(stream), true));
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dstDevice,
                              hipDeviceptr_t srcDevice,
                              size_t ByteCount,
                              hipStream_t stream) {
  HIP_INIT_API(hipMemcpyDtoDAsync, dstDevice, srcDevice, ByteCount, stream);

  HIP_RETURN(ihipMemcpy(dstDevice, srcDevice, ByteCount, hipMemcpyDeviceToDevice, *hip::getQueue(stream), true));
}

hipError_t hipMemcpyDtoHAsync(void* dstHost,
                              hipDeviceptr_t srcDevice,
                              size_t ByteCount,
                              hipStream_t stream) {
  HIP_INIT_API(hipMemcpyDtoHAsync, dstHost, srcDevice, ByteCount, stream);

  HIP_RETURN(ihipMemcpy(dstHost, srcDevice, ByteCount, hipMemcpyDeviceToHost, *hip::getQueue(stream), true));
}

hipError_t ihipMemcpyAtoD(hipArray* srcArray,
                          void* dstDevice,
                          amd::Coord3D srcOrigin,
                          amd::Coord3D dstOrigin,
                          amd::Coord3D copyRegion,
                          size_t dstRowPitch,
                          size_t dstSlicePitch,
                          hipStream_t stream,
                          bool isAsync = false) {
  cl_mem srcMemObj = reinterpret_cast<cl_mem>(srcArray->data);
  if (is_valid(srcMemObj) == false) {
    return hipErrorInvalidValue;
  }

  amd::Image* srcImage = as_amd(srcMemObj)->asImage();
  size_t dstOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dstDevice, dstOffset);

  amd::BufferRect srcRect;
  if (!srcRect.create(static_cast<size_t*>(srcOrigin), static_cast<size_t*>(copyRegion), srcImage->getRowPitch(), srcImage->getSlicePitch())) {
    return hipErrorInvalidValue;
  }

  amd::BufferRect dstRect;
  if (!dstRect.create(static_cast<size_t*>(dstOrigin), static_cast<size_t*>(copyRegion), dstRowPitch, dstSlicePitch)) {
    return hipErrorInvalidValue;
  }
  dstRect.start_ += dstOffset;
  dstRect.end_ += dstOffset;

  const size_t copySizeInBytes = copyRegion[0] * copyRegion[1] * copyRegion[2] * srcImage->getImageFormat().getElementSize();
  if (!srcImage->validateRegion(srcOrigin, copyRegion) ||
      !dstMemory->validateRegion(dstOrigin, {copySizeInBytes, 0, 0})) {
    return hipErrorInvalidValue;
  }

  amd::CopyMemoryCommand* command = new amd::CopyMemoryCommand(*hip::getQueue(stream),
                                                               CL_COMMAND_COPY_IMAGE_TO_BUFFER,
                                                               amd::Command::EventWaitList{},
                                                               *srcImage,
                                                               *dstMemory,
                                                               srcOrigin,
                                                               dstOrigin,
                                                               copyRegion,
                                                               srcRect,
                                                               dstRect);

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

hipError_t ihipMemcpyDtoA(void* srcDevice,
                          hipArray* dstArray,
                          amd::Coord3D srcOrigin,
                          amd::Coord3D dstOrigin,
                          amd::Coord3D copyRegion,
                          size_t srcRowPitch,
                          size_t srcSlicePitch,
                          hipStream_t stream,
                          bool isAsync = false) {
  cl_mem dstMemObj = reinterpret_cast<cl_mem>(dstArray->data);
  if (is_valid(dstMemObj) == false) {
    return hipErrorInvalidValue;
  }

  size_t srcOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(srcDevice, srcOffset);
  amd::Image* dstImage = as_amd(dstMemObj)->asImage();

  amd::BufferRect srcRect;
  if (!srcRect.create(static_cast<size_t*>(srcOrigin), static_cast<size_t*>(copyRegion), srcRowPitch, srcSlicePitch)) {
    return hipErrorInvalidValue;
  }
  srcRect.start_ += srcOffset;
  srcRect.end_ += srcOffset;

  amd::BufferRect dstRect;
  if (!dstRect.create(static_cast<size_t*>(dstOrigin), static_cast<size_t*>(copyRegion), dstImage->getRowPitch(), dstImage->getSlicePitch())) {
    return hipErrorInvalidValue;
  }

  const size_t copySizeInBytes = copyRegion[0] * copyRegion[1] * copyRegion[2] * dstImage->getImageFormat().getElementSize();
  if (!srcMemory->validateRegion(srcOrigin, {copySizeInBytes, 0, 0}) ||
      !dstImage->validateRegion(dstOrigin, copyRegion)) {
    return hipErrorInvalidValue;
  }

  amd::CopyMemoryCommand* command = new amd::CopyMemoryCommand(*hip::getQueue(stream),
                                                               CL_COMMAND_COPY_BUFFER_TO_IMAGE,
                                                               amd::Command::EventWaitList{},
                                                               *srcMemory,
                                                               *dstImage,
                                                               srcOrigin,
                                                               dstOrigin,
                                                               copyRegion,
                                                               srcRect,
                                                               dstRect);

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

hipError_t ihipMemcpyDtoD(void* srcDevice,
                          void* dstDevice,
                          amd::Coord3D srcOrigin,
                          amd::Coord3D dstOrigin,
                          amd::Coord3D copyRegion,
                          size_t srcRowPitch,
                          size_t srcSlicePitch,
                          size_t dstRowPitch,
                          size_t dstSlicePitch,
                          hipStream_t stream,
                          bool isAsync = false) {
  size_t srcOffset = 0;
  amd::Memory *srcMemory = getMemoryObject(srcDevice, srcOffset);
  size_t dstOffset = 0;
  amd::Memory *dstMemory = getMemoryObject(dstDevice, dstOffset);

  amd::BufferRect srcRect;
  if (!srcRect.create(static_cast<size_t*>(srcOrigin), static_cast<size_t*>(copyRegion), srcRowPitch, srcSlicePitch)) {
    return hipErrorInvalidValue;
  }
  srcRect.start_ += srcOffset;
  srcRect.end_ += srcOffset;

  amd::Coord3D srcStart(srcRect.start_, 0, 0);
  amd::Coord3D srcSize(srcRect.end_ - srcRect.start_, 1, 1);
  if (!srcMemory->validateRegion(srcStart, srcSize)) {
    return hipErrorInvalidValue;
  }

  amd::BufferRect dstRect;
  if (!dstRect.create(static_cast<size_t*>(dstOrigin), static_cast<size_t*>(copyRegion), dstRowPitch, dstSlicePitch)) {
    return hipErrorInvalidValue;
  }
  dstRect.start_ += dstOffset;
  dstRect.end_ += dstOffset;

  amd::Coord3D dstStart(dstRect.start_, 0, 0);
  amd::Coord3D dstSize(dstRect.end_ - dstRect.start_, 1, 1);
  if (!dstMemory->validateRegion(dstStart, dstSize)) {
    return hipErrorInvalidValue;
  }

  amd::CopyMemoryCommand* command = new amd::CopyMemoryCommand(*hip::getQueue(stream),
                                                               CL_COMMAND_COPY_BUFFER_RECT,
                                                               amd::Command::EventWaitList{},
                                                               *srcMemory,
                                                               *dstMemory,
                                                               srcStart,
                                                               dstStart,
                                                               copyRegion,
                                                               srcRect,
                                                               dstRect);

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

hipError_t ihipMemcpyDtoH(void* srcDevice,
                          void* dstHost,
                          amd::Coord3D srcOrigin,
                          amd::Coord3D dstOrigin,
                          amd::Coord3D copyRegion,
                          size_t srcRowPitch,
                          size_t srcSlicePitch,
                          size_t dstRowPitch,
                          size_t dstSlicePitch,
                          hipStream_t stream,
                          bool isAsync = false) {
  size_t srcOffset = 0;
  amd::Memory *srcMemory = getMemoryObject(srcDevice, srcOffset);

  amd::BufferRect srcRect;
  if (!srcRect.create(static_cast<size_t*>(srcOrigin), static_cast<size_t*>(copyRegion), srcRowPitch, srcSlicePitch)) {
    return hipErrorInvalidValue;
  }
  srcRect.start_ += srcOffset;
  srcRect.end_ += srcOffset;

  amd::Coord3D srcStart(srcRect.start_, 0, 0);
  amd::Coord3D srcSize(srcRect.end_ - srcRect.start_, 1, 1);
  if (!srcMemory->validateRegion(srcStart, srcSize)) {
    return hipErrorInvalidValue;
  }

  amd::BufferRect dstRect;
  if (!dstRect.create(static_cast<size_t*>(dstOrigin), static_cast<size_t*>(copyRegion), dstRowPitch, dstSlicePitch)) {
    return hipErrorInvalidValue;
  }

  amd::ReadMemoryCommand* command = new amd::ReadMemoryCommand(*hip::getQueue(stream),
                                                               CL_COMMAND_READ_BUFFER_RECT,
                                                               amd::Command::EventWaitList{},
                                                               *srcMemory,
                                                               srcStart,
                                                               copyRegion,
                                                               dstHost,
                                                               srcRect,
                                                               dstRect);

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

hipError_t ihipMemcpyHtoD(const void* srcHost,
                          void* dstDevice,
                          amd::Coord3D srcOrigin,
                          amd::Coord3D dstOrigin,
                          amd::Coord3D copyRegion,
                          size_t srcRowPitch,
                          size_t srcSlicePitch,
                          size_t dstRowPitch,
                          size_t dstSlicePitch,
                          hipStream_t stream,
                          bool isAsync = false) {
  size_t dstOffset = 0;
  amd::Memory *dstMemory = getMemoryObject(dstDevice, dstOffset);

  amd::BufferRect srcRect;
  if (!srcRect.create(static_cast<size_t*>(srcOrigin), static_cast<size_t*>(copyRegion), srcRowPitch, srcSlicePitch)) {
    return hipErrorInvalidValue;
  }

  amd::BufferRect dstRect;
  if (!dstRect.create(static_cast<size_t*>(dstOrigin), static_cast<size_t*>(copyRegion), dstRowPitch, dstSlicePitch)) {
    return hipErrorInvalidValue;
  }
  dstRect.start_ += dstOffset;
  dstRect.end_ += dstOffset;

  amd::Coord3D dstStart(dstRect.start_, 0, 0);
  amd::Coord3D dstSize(dstRect.end_ - dstRect.start_, 1, 1);
  if (!dstMemory->validateRegion(dstStart, dstSize)) {
    return hipErrorInvalidValue;
  }

  amd::WriteMemoryCommand* command = new amd::WriteMemoryCommand(*hip::getQueue(stream),
                                                                 CL_COMMAND_WRITE_BUFFER_RECT,
                                                                 amd::Command::EventWaitList{},
                                                                 *dstMemory,
                                                                 dstStart,
                                                                 copyRegion,
                                                                 srcHost,
                                                                 dstRect,
                                                                 srcRect);

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

hipError_t ihipMemcpyHtoH(const void* srcHost,
                          void* dstHost,
                          amd::Coord3D srcOrigin,
                          amd::Coord3D dstOrigin,
                          amd::Coord3D copyRegion,
                          size_t srcRowPitch,
                          size_t srcSlicePitch,
                          size_t dstRowPitch,
                          size_t dstSlicePitch) {
  amd::BufferRect srcRect;
  if (!srcRect.create(static_cast<size_t*>(srcOrigin), static_cast<size_t*>(copyRegion), srcRowPitch, srcSlicePitch)) {
    return hipErrorInvalidValue;
  }

  amd::BufferRect dstRect;
  if (!dstRect.create(static_cast<size_t*>(dstOrigin), static_cast<size_t*>(copyRegion), dstRowPitch, dstSlicePitch)) {
    return hipErrorInvalidValue;
  }

  for (size_t slice = 0; slice < copyRegion[2]; slice++) {
    for (size_t row = 0; row < copyRegion[1]; row++) {
      const void* srcRow = static_cast<const char*>(srcHost) + srcRect.start_ + row * srcRect.rowPitch_ + slice * srcRect.slicePitch_;
      void* dstRow = static_cast<char*>(dstHost) + dstRect.start_ + row * dstRect.rowPitch_ + slice * dstRect.slicePitch_;
      std::memcpy(dstRow, srcRow, copyRegion[0]);
    }
  }

  return hipSuccess;
}

hipError_t ihipMemcpyAtoA(hipArray* srcArray,
                          hipArray* dstArray,
                          amd::Coord3D srcOrigin,
                          amd::Coord3D dstOrigin,
                          amd::Coord3D copyRegion,
                          hipStream_t stream,
                          bool isAsync = false) {
  cl_mem srcMemObj = reinterpret_cast<cl_mem>(srcArray->data);
  cl_mem dstMemObj = reinterpret_cast<cl_mem>(dstArray->data);
  if (!is_valid(srcMemObj) || !is_valid(dstMemObj)) {
    return hipErrorInvalidValue;
  }

  amd::Image* srcImage = as_amd(srcMemObj)->asImage();
  amd::Image* dstImage = as_amd(dstMemObj)->asImage();

  // HIP assumes the width is in bytes, but OCL assumes it's in pixels.
  // Note that src and dst should have the same element size.
  assert(srcImage->getImageFormat().getElementSize() == dstImage->getImageFormat().getElementSize());
  const size_t elementSize = srcImage->getImageFormat().getElementSize();
  static_cast<size_t*>(srcOrigin)[0] /= elementSize;
  static_cast<size_t*>(dstOrigin)[0] /= elementSize;
  static_cast<size_t*>(copyRegion)[0] /= elementSize;

  if (!srcImage->validateRegion(srcOrigin, copyRegion) ||
      !dstImage->validateRegion(dstOrigin, copyRegion)) {
    return hipErrorInvalidValue;
  }

  amd::CopyMemoryCommand* command = new amd::CopyMemoryCommand(*hip::getQueue(stream),
                                                               CL_COMMAND_COPY_IMAGE,
                                                               amd::Command::EventWaitList{},
                                                               *srcImage,
                                                               *dstImage,
                                                               srcOrigin,
                                                               dstOrigin,
                                                               copyRegion);

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

hipError_t ihipMemcpyHtoA(const void* srcHost,
                          hipArray* dstArray,
                          amd::Coord3D srcOrigin,
                          amd::Coord3D dstOrigin,
                          amd::Coord3D copyRegion,
                          size_t srcRowPitch,
                          size_t srcSlicePitch,
                          hipStream_t stream,
                          bool isAsync = false) {
  if (srcHost == nullptr) {
    return hipErrorInvalidValue;
  }

  cl_mem dstMemObj = reinterpret_cast<cl_mem>(dstArray->data);
  if (is_valid(dstMemObj) == false) {
    return hipErrorInvalidValue;
  }

  amd::BufferRect srcRect;
  if (!srcRect.create(static_cast<size_t*>(srcOrigin), static_cast<size_t*>(copyRegion), srcRowPitch, srcSlicePitch)) {
    return hipErrorInvalidValue;
  }

  amd::Image* dstImage = as_amd(dstMemObj)->asImage();
  // HIP assumes the width is in bytes, but OCL assumes it's in pixels.
  const size_t elementSize = dstImage->getImageFormat().getElementSize();
  static_cast<size_t*>(dstOrigin)[0] /= elementSize;
  static_cast<size_t*>(copyRegion)[0] /= elementSize;

  if (!dstImage->validateRegion(dstOrigin, copyRegion)) {
    return hipErrorInvalidValue;
  }

  amd::WriteMemoryCommand* command = new amd::WriteMemoryCommand(*hip::getQueue(stream),
                                                                 CL_COMMAND_WRITE_IMAGE,
                                                                 amd::Command::EventWaitList{},
                                                                 *dstImage,
                                                                 dstOrigin,
                                                                 copyRegion,
                                                                 static_cast<const char*>(srcHost) + srcRect.start_,
                                                                 srcRowPitch,
                                                                 srcSlicePitch);

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

hipError_t ihipMemcpyAtoH(hipArray* srcArray,
                          void* dstHost,
                          amd::Coord3D srcOrigin,
                          amd::Coord3D dstOrigin,
                          amd::Coord3D copyRegion,
                          size_t dstRowPitch,
                          size_t dstSlicePitch,
                          hipStream_t stream,
                          bool isAsync = false) {
  cl_mem srcMemObj = reinterpret_cast<cl_mem>(srcArray->data);
  if (!is_valid(srcMemObj)) {
    return hipErrorInvalidValue;
  }

  if (dstHost == nullptr) {
    return hipErrorInvalidValue;
  }

  amd::BufferRect dstRect;
  if (!dstRect.create(static_cast<size_t*>(dstOrigin), static_cast<size_t*>(copyRegion), dstRowPitch, dstSlicePitch)) {
    return hipErrorInvalidValue;
  }


  amd::Image* srcImage = as_amd(srcMemObj)->asImage();
  // HIP assumes the width is in bytes, but OCL assumes it's in pixels.
  const size_t elementSize = srcImage->getImageFormat().getElementSize();
  static_cast<size_t*>(srcOrigin)[0] /= elementSize;
  static_cast<size_t*>(copyRegion)[0] /= elementSize;

  if (!srcImage->validateRegion(srcOrigin, copyRegion) ||
      !srcImage->isRowSliceValid(dstRowPitch, dstSlicePitch, copyRegion[0], copyRegion[1])) {
    return hipErrorInvalidValue;
  }

  amd::ReadMemoryCommand* command = new amd::ReadMemoryCommand(*hip::getQueue(stream),
                                                               CL_COMMAND_READ_IMAGE,
                                                               amd::Command::EventWaitList{},
                                                               *srcImage,
                                                               srcOrigin,
                                                               copyRegion,
                                                               static_cast<char*>(dstHost) + dstRect.start_,
                                                               dstRowPitch,
                                                               dstSlicePitch);

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

hipError_t ihipMemcpyParam3D(const HIP_MEMCPY3D* pCopy,
                             hipStream_t stream,
                             bool isAsync = false) {
  // If {src/dst}MemoryType is hipMemoryTypeUnified, {src/dst}Device and {src/dst}Pitch specify the (unified virtual address space)
  // base address of the source data and the bytes per row to apply. {src/dst}Array is ignored.
  hipMemoryType srcMemoryType = pCopy->srcMemoryType;
  if (srcMemoryType == hipMemoryTypeUnified) {
    srcMemoryType = amd::MemObjMap::FindMemObj(pCopy->srcDevice) ? hipMemoryTypeDevice : hipMemoryTypeHost;
  }
  hipMemoryType dstMemoryType = pCopy->dstMemoryType;
  if (dstMemoryType == hipMemoryTypeUnified) {
    dstMemoryType = amd::MemObjMap::FindMemObj(pCopy->dstDevice) ? hipMemoryTypeDevice : hipMemoryTypeHost;
  }

  amd::Coord3D srcOrigin = {pCopy->srcXInBytes, pCopy->srcY, pCopy->srcZ};
  amd::Coord3D dstOrigin = {pCopy->dstXInBytes, pCopy->dstY, pCopy->dstZ};
  amd::Coord3D copyRegion = {pCopy->WidthInBytes, (pCopy->Height != 0) ? pCopy->Height : 1, (pCopy->Depth != 0) ? pCopy->Depth :  1};

  if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeHost)) {
    // Host to Host.
    return ihipMemcpyHtoH(pCopy->srcHost, pCopy->dstHost, srcOrigin, dstOrigin, copyRegion, pCopy->srcPitch, pCopy->srcPitch * pCopy->srcHeight, pCopy->dstPitch, pCopy->dstPitch * pCopy->dstHeight);
  } else if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeDevice)) {
    // Host to Device.
    return ihipMemcpyHtoD(pCopy->srcHost, pCopy->dstDevice, srcOrigin, dstOrigin, copyRegion, pCopy->srcPitch, pCopy->srcPitch * pCopy->srcHeight, pCopy->dstPitch, pCopy->dstPitch * pCopy->dstHeight, stream, isAsync);
  } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeHost)) {
    // Device to Host.
    return ihipMemcpyDtoH(pCopy->srcDevice, pCopy->dstHost, srcOrigin, dstOrigin, copyRegion, pCopy->srcPitch, pCopy->srcPitch * pCopy->srcHeight, pCopy->dstPitch, pCopy->dstPitch * pCopy->dstHeight, stream, isAsync);
  } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeDevice)) {
    // Device to Device.
    return ihipMemcpyDtoD(pCopy->srcDevice, pCopy->dstDevice, srcOrigin, dstOrigin, copyRegion, pCopy->srcPitch, pCopy->srcPitch * pCopy->srcHeight, pCopy->dstPitch, pCopy->dstPitch * pCopy->dstHeight, stream, isAsync);
  } else if ((srcMemoryType == hipMemoryTypeHost) && (dstMemoryType == hipMemoryTypeArray)) {
    // Host to Image.
    return ihipMemcpyHtoA(pCopy->srcHost, pCopy->dstArray, srcOrigin, dstOrigin, copyRegion, pCopy->srcPitch, pCopy->srcPitch * pCopy->srcHeight, stream, isAsync);
  } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeHost)) {
    // Image to Host.
    return ihipMemcpyAtoH(pCopy->srcArray, pCopy->dstHost, srcOrigin, dstOrigin, copyRegion, pCopy->dstPitch, pCopy->dstPitch * pCopy->dstHeight, stream, isAsync);
  } else if ((srcMemoryType == hipMemoryTypeDevice) && (dstMemoryType == hipMemoryTypeArray)) {
    // Device to Image.
    return ihipMemcpyDtoA(pCopy->srcDevice, pCopy->dstArray, srcOrigin, dstOrigin, copyRegion, pCopy->srcPitch, pCopy->srcPitch * pCopy->srcHeight, stream, isAsync);
  } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeDevice)) {
    // Image to Device.
    return ihipMemcpyAtoD(pCopy->srcArray, pCopy->dstDevice, srcOrigin, dstOrigin, copyRegion, pCopy->dstPitch, pCopy->dstPitch * pCopy->dstHeight, stream, isAsync);
  } else if ((srcMemoryType == hipMemoryTypeArray) && (dstMemoryType == hipMemoryTypeArray)) {
    // Image to Image.
    return ihipMemcpyAtoA(pCopy->srcArray, pCopy->dstArray, srcOrigin, dstOrigin, copyRegion, stream, isAsync);
  } else {
    ShouldNotReachHere();
  }

  return hipSuccess;
}

hipError_t ihipMemcpyParam2D(const hip_Memcpy2D* pCopy,
                             hipStream_t stream,
                             bool isAsync = false) {
  HIP_MEMCPY3D desc = hip::getDrvMemcpy3DDesc(*pCopy);

  return ihipMemcpyParam3D(&desc, stream, isAsync);
}

hipError_t ihipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                        size_t height, hipMemcpyKind kind, hipStream_t stream, bool isAsync = false) {
  hip_Memcpy2D desc = {};

  desc.srcXInBytes = 0;
  desc.srcY = 0;
  desc.srcMemoryType = std::get<0>(hip::getMemoryType(kind));
  desc.srcHost = src;
  desc.srcDevice = const_cast<void*>(src);
  desc.srcArray = nullptr; // Ignored.
  desc.srcPitch = spitch;

  desc.dstXInBytes = 0;
  desc.dstY = 0;
  desc.dstMemoryType = std::get<1>(hip::getMemoryType(kind));
  desc.dstHost = dst;
  desc.dstDevice = dst;
  desc.dstArray = nullptr; // Ignored.
  desc.dstPitch = dpitch;

  desc.WidthInBytes = width;
  desc.Height = height;

  return ihipMemcpyParam2D(&desc, stream, isAsync);
}

hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
  HIP_INIT_API(hipMemcpyParam2D, pCopy);

  HIP_RETURN(ihipMemcpyParam2D(pCopy, nullptr));
}

hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy2D, dst, dpitch, src, spitch, width, height, kind);

  HIP_RETURN(ihipMemcpy2D(dst, dpitch, src, spitch, width, height, kind, nullptr));
}

hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpy2DAsync, dst, dpitch, src, spitch, width, height, kind, stream);

  HIP_RETURN(ihipMemcpy2D(dst, dpitch, src, spitch, width, height, kind, stream, true));
}

hipError_t ihipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream, bool isAsync = false) {
  hip_Memcpy2D desc = {};

  desc.srcXInBytes = 0;
  desc.srcY = 0;
  desc.srcMemoryType = std::get<0>(hip::getMemoryType(kind));
  desc.srcHost = const_cast<void*>(src);
  desc.srcDevice = const_cast<void*>(src);
  desc.srcArray = nullptr;
  desc.srcPitch = spitch;

  desc.dstXInBytes = wOffset;
  desc.dstY = hOffset;
  desc.dstMemoryType = hipMemoryTypeArray;
  desc.dstHost = nullptr;
  desc.dstDevice = nullptr;
  desc.dstArray = dst;
  desc.dstPitch = 0; // Ignored.

  desc.WidthInBytes = width;
  desc.Height = height;

  return ihipMemcpyParam2D(&desc, stream, isAsync);
}

hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy2DToArray, dst, wOffset, hOffset, src, spitch, width, height, kind);

  HIP_RETURN(ihipMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind, nullptr));
}

hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src, size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpyToArray, dst, wOffset, hOffset, src, count, kind);

  if (dst == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const size_t arrayHeight = (dst->height != 0) ? dst->height : 1;
  const size_t witdthInBytes = count / arrayHeight;

  const size_t height = (count / dst->width) / hip::getElementSize(dst);

  HIP_RETURN(ihipMemcpy2DToArray(dst, wOffset, hOffset, src, 0 /* spitch */, witdthInBytes, height, kind, nullptr));
}

hipError_t ihipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream, bool isAsync = false) {
  hip_Memcpy2D desc = {};

  desc.srcXInBytes = wOffsetSrc;
  desc.srcY = hOffsetSrc;
  desc.srcMemoryType = hipMemoryTypeArray;
  desc.srcHost = nullptr;
  desc.srcDevice = nullptr;
  desc.srcArray = const_cast<hipArray_t>(src);
  desc.srcPitch = 0; // Ignored.

  desc.dstXInBytes = 0;
  desc.dstY = 0;
  desc.dstMemoryType = std::get<1>(hip::getMemoryType(kind));
  desc.dstHost = dst;
  desc.dstDevice = dst;
  desc.dstArray = nullptr;
  desc.dstPitch = dpitch;

  desc.WidthInBytes = width;
  desc.Height = height;

  return ihipMemcpyParam2D(&desc, stream, isAsync);
}

hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t src, size_t wOffsetSrc, size_t hOffset, size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpyFromArray, dst, src, wOffsetSrc, hOffset, count, kind);

  if (src == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const size_t arrayHeight = (src->height != 0) ? src->height : 1;
  const size_t witdthInBytes = count / arrayHeight;

  const size_t height = (count / src->width) / hip::getElementSize(src);

  HIP_RETURN(ihipMemcpy2DFromArray(dst, 0 /* dpitch */, src, wOffsetSrc, hOffset, witdthInBytes, height, kind, nullptr));
}

hipError_t hipMemcpyHtoA(hipArray* dstArray,
                         size_t dstOffset,
                         const void* srcHost,
                         size_t ByteCount) {
  HIP_INIT_API(hipMemcpyHtoA, dstArray, dstOffset, srcHost, ByteCount);

  HIP_RETURN(ihipMemcpyHtoA(srcHost, dstArray, {0, 0, 0}, {dstOffset, 0, 0}, {ByteCount, 1, 1}, 0, 0, nullptr));
}

hipError_t hipMemcpyAtoH(void* dstHost,
                         hipArray* srcArray,
                         size_t srcOffset,
                         size_t ByteCount) {
  HIP_INIT_API(hipMemcpyAtoH, dstHost, srcArray, srcOffset, ByteCount);

  HIP_RETURN(ihipMemcpyAtoH(srcArray, dstHost, {srcOffset, 0, 0}, {0, 0, 0}, {ByteCount, 1, 1}, 0, 0, nullptr));
}

hipError_t ihipMemcpy3D(const hipMemcpy3DParms* p,
                        hipStream_t stream,
                        bool isAsync = false) {
  // The struct passed to hipMemcpy3D() must specify one of srcArray or srcPtr and one of dstArray or dstPtr.
  // Passing more than one non-zero source or destination will cause hipMemcpy3D() to return an error.
  if (((p->srcArray != nullptr) && (p->srcPtr.ptr != nullptr)) ||
      ((p->dstArray != nullptr) && (p->dstPtr.ptr != nullptr))) {
    return hipErrorInvalidValue;
  }

  // If the source and destination are both arrays, hipMemcpy3D() will return an error if they do not have the same element size.
  if (((p->srcArray != nullptr) && (p->dstArray != nullptr)) &&
      (hip::getElementSize(p->dstArray) != hip::getElementSize(p->dstArray))) {
    return hipErrorInvalidValue;
  }

  const HIP_MEMCPY3D desc = hip::getDrvMemcpy3DDesc(*p);

  return ihipMemcpyParam3D(&desc, stream, isAsync);
}

hipError_t hipMemcpy3D(const hipMemcpy3DParms* p) {
  HIP_INIT_API(hipMemcpy3D, p);

  HIP_RETURN(ihipMemcpy3D(p, nullptr));
}

hipError_t hipMemcpy3DAsync(const hipMemcpy3DParms* p, hipStream_t stream) {
  HIP_INIT_API(hipMemcpy3DAsync, p, stream);

  HIP_RETURN(ihipMemcpy3D(p, stream, true));
}

hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy) {
  HIP_INIT_API(hipDrvMemcpy3D, pCopy);

  HIP_RETURN(ihipMemcpyParam3D(pCopy, nullptr));
}

hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream) {
  HIP_INIT_API(hipDrvMemcpy3DAsync, pCopy, stream);

  HIP_RETURN(ihipMemcpyParam3D(pCopy, stream, true));
}

hipError_t ihipMemset(void* dst, int value, size_t valueSize, size_t sizeBytes,
                      hipStream_t stream, bool isAsync = false) {
  if (sizeBytes == 0) {
    // Skip if nothing needs filling.
    return hipSuccess;
  }

  if (dst == nullptr) {
    return hipErrorInvalidValue;
  }

  size_t offset = 0;
  amd::HostQueue* queue = hip::getQueue(stream);
  amd::Memory* memory = getMemoryObject(dst, offset);

  if (memory != nullptr) {
    // Device memory
    amd::Command::EventWaitList waitList;
    amd::Coord3D fillOffset(offset, 0, 0);
    amd::Coord3D fillSize(sizeBytes, 1, 1);
    amd::FillMemoryCommand* command =
        new amd::FillMemoryCommand(*queue, CL_COMMAND_FILL_BUFFER, waitList, *memory->asBuffer(),
                                   &value, valueSize, fillOffset, fillSize);

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

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  HIP_INIT_API(hipMemset, dst, value, sizeBytes);

  HIP_RETURN(ihipMemset(dst, value, sizeof(int8_t), sizeBytes, nullptr));
}

hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
  HIP_INIT_API(hipMemsetAsync, dst, value, sizeBytes, stream);

  HIP_RETURN(ihipMemset(dst, value, sizeof(int8_t), sizeBytes, stream, true));
}

hipError_t hipMemsetD8(hipDeviceptr_t dst, unsigned char value, size_t count) {
  HIP_INIT_API(hipMemsetD8, dst, value, count);

  HIP_RETURN(ihipMemset(dst, value, sizeof(int8_t), count * sizeof(int8_t), nullptr));
}

hipError_t hipMemsetD8Async(hipDeviceptr_t dst, unsigned char value, size_t count,
                            hipStream_t stream) {
  HIP_INIT_API(hipMemsetD8Async, dst, value, count, stream);

  HIP_RETURN(ihipMemset(dst, value, sizeof(int8_t), count * sizeof(int8_t), stream, true));
}

hipError_t hipMemsetD16(hipDeviceptr_t dst, unsigned short value, size_t count) {
  HIP_INIT_API(hipMemsetD16, dst, value, count);

  HIP_RETURN(ihipMemset(dst, value, sizeof(int16_t), count * sizeof(int16_t), nullptr));
}

hipError_t hipMemsetD16Async(hipDeviceptr_t dst, unsigned short value, size_t count,
                             hipStream_t stream) {
  HIP_INIT_API(hipMemsetD16Async, dst, value, count, stream);

  HIP_RETURN(ihipMemset(dst, value, sizeof(int16_t), count * sizeof(int16_t), stream, true));
}

hipError_t hipMemsetD32(hipDeviceptr_t dst, int value, size_t count) {
  HIP_INIT_API(hipMemsetD32, dst, value, count);

  HIP_RETURN(ihipMemset(dst, value, sizeof(int32_t), count * sizeof(int32_t), nullptr));
}

hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count,
                             hipStream_t stream) {
  HIP_INIT_API(hipMemsetD32Async, dst, value, count, stream);

  HIP_RETURN(ihipMemset(dst, value, sizeof(int32_t), count * sizeof(int32_t), stream, true));
}

hipError_t ihipMemset3D(hipPitchedPtr pitchedDevPtr,
                        int value,
                        hipExtent extent,
                        hipStream_t stream,
                        bool isAsync = false) {
  if (pitchedDevPtr.pitch == extent.width) {
    return ihipMemset(pitchedDevPtr.ptr, value, sizeof(int8_t), extent.width * extent.height * extent.depth, stream, isAsync);
  }

  // Workaround for cases when pitch > row untill fill kernel will be updated to support pitch.
  // Fallback to filling one row at a time.

  amd::HostQueue* queue = hip::getQueue(stream);

  size_t offset = 0;
  amd::Memory* memory = getMemoryObject(pitchedDevPtr.ptr, offset);

  amd::Coord3D origin(offset);
  amd::Coord3D region(pitchedDevPtr.xsize, pitchedDevPtr.ysize, extent.depth);
  amd::BufferRect rect;
  if (!rect.create(static_cast<size_t*>(origin), static_cast<size_t*>(region), pitchedDevPtr.pitch, 0)) {
    return hipErrorInvalidValue;
  }

  if (memory != nullptr) {
    std::vector<amd::FillMemoryCommand*> commands;

    for (size_t slice = 0; slice < extent.depth; slice++) {
      for (size_t row = 0; row < extent.height; row++) {
        const size_t rowOffset = rect.offset(0, row, slice);
        amd::FillMemoryCommand* command = new amd::FillMemoryCommand(*queue,
                                                                     CL_COMMAND_FILL_BUFFER,
                                                                     amd::Command::EventWaitList{},
                                                                     *memory->asBuffer(),
                                                                     &value,
                                                                     sizeof(int8_t),
                                                                     amd::Coord3D{rowOffset, 0, 0},
                                                                     amd::Coord3D{extent.width, 1, 1});

        command->enqueue();
        commands.push_back(command);
      }
    }

    for (auto &command: commands) {
      if (!isAsync) {
        command->awaitCompletion();
      }
      command->release();
    }
  } else {
    for (size_t slice = 0; slice < extent.depth; slice++) {
      for (size_t row = 0; row < extent.height; row++) {
        const size_t rowOffset = rect.offset(0, row, slice);
        std::memset(pitchedDevPtr.ptr, value, extent.width);
      }
    }
  }

  return hipSuccess;
}

hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height) {
  HIP_INIT_API(hipMemset2D, dst, pitch, value, width, height);

  HIP_RETURN(ihipMemset3D({dst, pitch, width, height}, value, {width, height, 1}, nullptr));
}

hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value,
                            size_t width, size_t height, hipStream_t stream) {
  HIP_INIT_API(hipMemset2DAsync, dst, pitch, value, width, height, stream);

  HIP_RETURN(ihipMemset3D({dst, pitch, width, height}, value, {width, height, 1}, stream, true));
}

hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent) {
  HIP_INIT_API(hipMemset3D, pitchedDevPtr, value, extent);

  HIP_RETURN(ihipMemset3D(pitchedDevPtr, value, extent, nullptr));
}

hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent, hipStream_t stream) {
  HIP_INIT_API(hipMemset3DAsync, pitchedDevPtr, value, extent, stream);

  HIP_RETURN(ihipMemset3D(pitchedDevPtr, value, extent, stream, false));
}

hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes,
                            size_t height, unsigned int elementSizeBytes) {
  HIP_INIT_API(hipMemAllocPitch, dptr, pitch, widthInBytes, height, elementSizeBytes);

  HIP_RETURN(hipMallocPitch(dptr, pitch, widthInBytes, height));
}

hipError_t hipMemAllocHost(void** ptr, size_t size) {
  HIP_INIT_API(hipMemAllocHost, ptr, size);

  HIP_RETURN(hipHostMalloc(ptr, size, 0));
}

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* dev_ptr) {
  HIP_INIT_API(hipIpcGetMemHandle, handle, dev_ptr);

  size_t offset = 0;
  amd::Memory* amd_mem_obj = nullptr;
  device::Memory* dev_mem_obj = nullptr;
  ihipIpcMemHandle_t* ihandle = nullptr;

  if ((handle == nullptr) || (dev_ptr == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  /* Get AMD::Memory object corresponding to this pointer */
  amd_mem_obj = getMemoryObject(dev_ptr, offset);
  if (amd_mem_obj == nullptr) {
    HIP_RETURN(hipErrorInvalidDevicePointer);
  }

  /* Get Device::Memory object pointer */
  dev_mem_obj = amd_mem_obj->getDeviceMemory(*hip::getCurrentDevice()->devices()[0],false);
  if (dev_mem_obj == nullptr) {
    HIP_RETURN(hipErrorInvalidDevicePointer);
  }

  /* Create an handle for IPC. Store the memory size inside the handle */
  ihandle = reinterpret_cast<ihipIpcMemHandle_t *>(handle);
  dev_mem_obj->IpcCreate(offset, &(ihandle->psize), &(ihandle->ipc_handle));

  HIP_RETURN(hipSuccess);
}

hipError_t hipIpcOpenMemHandle(void** dev_ptr, hipIpcMemHandle_t handle, unsigned int flags) {
  HIP_INIT_API(hipIpcOpenMemHandle, dev_ptr, &handle, flags);

  amd::Memory* amd_mem_obj = nullptr;
  amd::Device* device = nullptr;
  ihipIpcMemHandle_t* ihandle = nullptr;

  if (dev_ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  /* Call the IPC Attach from Device class */
  device = hip::getCurrentDevice()->devices()[0];
  ihandle = reinterpret_cast<ihipIpcMemHandle_t *>(&handle);

  amd_mem_obj = device->IpcAttach(&(ihandle->ipc_handle), ihandle->psize, flags, dev_ptr);
  if (amd_mem_obj == nullptr) {
    HIP_RETURN(hipErrorInvalidDevicePointer);
  }

  /* Add the memory to the MemObjMap */
  amd::MemObjMap::AddMemObj(*dev_ptr, amd_mem_obj);

  HIP_RETURN(hipSuccess);
}

hipError_t hipIpcCloseMemHandle(void* dev_ptr) {
  HIP_INIT_API(hipIpcCloseMemHandle, dev_ptr);

  size_t offset = 0;
  amd::Device* device = nullptr;
  amd::Memory* amd_mem_obj = nullptr;

  hip::syncStreams();
  hip::getNullStream()->finish();

  if (dev_ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  /* Get the amd::Memory object */
  amd_mem_obj = getMemoryObject(dev_ptr, offset);
  if (amd_mem_obj == nullptr) {
    HIP_RETURN(hipErrorInvalidDevicePointer);
  }

  /* Call IPC Detach from Device class */
  device = hip::getCurrentDevice()->devices()[0];
  if (device == nullptr) {
    HIP_RETURN(hipErrorNoDevice);
  }

  /* Remove the memory from MemObjMap */
  amd::MemObjMap::RemoveMemObj(amd_mem_obj);

  /* detach the memory */
  device->IpcDetach(*amd_mem_obj);

  HIP_RETURN(hipSuccess);
}

hipError_t hipHostGetDevicePointer(void** devicePointer, void* hostPointer, unsigned flags) {
  HIP_INIT_API(hipHostGetDevicePointer, devicePointer, hostPointer, flags);

  size_t offset = 0;

  amd::Memory* memObj = getMemoryObject(hostPointer, offset);
  if (!memObj) {
    HIP_RETURN(hipErrorInvalidValue);
  }
*devicePointer = reinterpret_cast<void*>(memObj->getDeviceMemory(*hip::getCurrentDevice()->devices()[0])->virtualAddress() + offset);

  HIP_RETURN(hipSuccess);
}

hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
  HIP_INIT_API(hipPointerGetAttributes, attributes, ptr);

  size_t offset = 0;
  amd::Memory* memObj = getMemoryObject(ptr, offset);
  int device = 0;

  if (memObj != nullptr) {
    attributes->memoryType = (CL_MEM_SVM_FINE_GRAIN_BUFFER & memObj->getMemFlags())? hipMemoryTypeHost : hipMemoryTypeDevice;
    attributes->hostPointer = memObj->getSvmPtr();
    attributes->devicePointer = memObj->getSvmPtr();
    attributes->isManaged = 0;
    attributes->allocationFlags = memObj->getMemFlags() >> 16;

    amd::Context* memObjCtx = &memObj->getContext();
    if (hip::host_device->asContext() == memObjCtx) {
        attributes->device = ihipGetDevice();
        HIP_RETURN(hipSuccess);
    }
    for (auto& ctx : g_devices) {
      if (ctx->asContext() == memObjCtx) {
        attributes->device = device;
        HIP_RETURN(hipSuccess);
      }
      ++device;
    }
    HIP_RETURN(hipErrorInvalidDevice);
  }

  HIP_RETURN(hipErrorInvalidValue);
}

hipError_t hipArrayDestroy(hipArray* array) {
  HIP_INIT_API(hipArrayDestroy, array);

  HIP_RETURN(ihipArrayDestroy(array));
}

hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor,
                                   hipArray* array) {
  HIP_INIT_API(hipArray3DGetDescriptor, pArrayDescriptor, array);

  assert(false && "Unimplemented");

  HIP_RETURN(hipSuccess);
}

hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor,
                                 hipArray* array) {
  HIP_INIT_API(hipArrayGetDescriptor, pArrayDescriptor, array);

  assert(false && "Unimplemented");

  HIP_RETURN(hipSuccess);
}

hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy,
                                 hipStream_t stream) {
  HIP_INIT_API(hipMemcpyParam2D, pCopy);

  HIP_RETURN(ihipMemcpyParam2D(pCopy, stream, true));
}

hipError_t ihipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst, size_t hOffsetDst, hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream, bool isAsync = false) {
  hip_Memcpy2D desc = {};

  desc.srcXInBytes = wOffsetSrc;
  desc.srcY = hOffsetSrc;
  desc.srcMemoryType = hipMemoryTypeArray;
  desc.srcHost = nullptr;
  desc.srcDevice = nullptr;
  desc.srcArray = const_cast<hipArray_t>(src);
  desc.srcPitch = 0; // Ignored.

  desc.dstXInBytes = wOffsetDst;
  desc.dstY = hOffsetDst;
  desc.dstMemoryType = hipMemoryTypeArray;
  desc.dstHost = nullptr;
  desc.dstDevice = nullptr;
  desc.dstArray = dst;
  desc.dstPitch = 0; // Ignored.

  desc.WidthInBytes = width;
  desc.Height = height;

  return ihipMemcpyParam2D(&desc, stream, isAsync);
}

hipError_t hipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst, size_t hOffsetDst, hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy2DArrayToArray, dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);

  HIP_RETURN(ihipMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind, nullptr));
}

hipError_t hipMemcpyArrayToArray(hipArray_t dst, size_t wOffsetDst, size_t hOffsetDst, hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpyArrayToArray, dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);

  HIP_RETURN(ihipMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind, nullptr));
}

hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffsetSrc, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy2DFromArray, dst, dpitch, src, wOffsetSrc, hOffset, width, height, kind);

  HIP_RETURN(ihipMemcpy2DFromArray(dst, dpitch, src, wOffsetSrc, hOffset, width, height, kind, nullptr));
}

hipError_t hipMemcpy2DFromArrayAsync(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpy2DFromArrayAsync, dst, dpitch, src, wOffsetSrc, hOffsetSrc, width, height, kind, stream);

  HIP_RETURN(ihipMemcpy2DFromArray(dst, dpitch, src, wOffsetSrc, hOffsetSrc, width, height, kind, stream, true));
}

hipError_t hipMemcpyFromArrayAsync(void* dst, hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyFromArrayAsync, dst, src, wOffsetSrc, hOffsetSrc, count, kind, stream);

  if (src == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const size_t arrayHeight = (src->height != 0) ? src->height : 1;
  const size_t widthInBytes = count / arrayHeight;

  const size_t height = (count / src->width) / hip::getElementSize(src);

  HIP_RETURN(ihipMemcpy2DFromArray(dst, 0 /* dpitch */, src, wOffsetSrc, hOffsetSrc, widthInBytes, height, kind, stream, true));
}

hipError_t hipMemcpy2DToArrayAsync(hipArray* dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpy2DToArrayAsync, dst, wOffset, hOffset, src, spitch, width, height, kind);

  HIP_RETURN(ihipMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind, stream, true));
}

hipError_t hipMemcpyToArrayAsync(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, hipMemcpyKind kind, hipStream_t stream) {
  HIP_INIT_API(hipMemcpyToArrayAsync, dst, wOffset, hOffset, src, count, kind);

  if (dst == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  const size_t arrayHeight = (dst->height != 0) ? dst->height : 1;
  const size_t widthInBytes = count / arrayHeight;

  const size_t height = (count / dst->width) / hip::getElementSize(dst);

  HIP_RETURN(ihipMemcpy2DToArray(dst, wOffset, hOffset, src, 0 /* spitch */, widthInBytes, height, kind, stream, true));
}

hipError_t hipMemcpyAtoA(hipArray* dstArray,
                         size_t dstOffset,
                         hipArray* srcArray,
                         size_t srcOffset,
                         size_t ByteCount) {
  HIP_INIT_API(hipMemcpyAtoA, dstArray, dstOffset, srcArray, srcOffset, ByteCount);

  HIP_RETURN(ihipMemcpyAtoA(srcArray, dstArray, {srcOffset, 0, 0}, {dstOffset, 0, 0}, {ByteCount, 1, 1}, nullptr));
}

hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice,
                         hipArray* srcArray,
                         size_t srcOffset,
                         size_t ByteCount) {
  HIP_INIT_API(hipMemcpyAtoD, dstDevice, srcArray, srcOffset, ByteCount);

  HIP_RETURN(ihipMemcpyAtoD(srcArray, dstDevice, {srcOffset, 0, 0}, {0, 0, 0}, {ByteCount, 1, 1}, 0, 0, nullptr));
}

hipError_t hipMemcpyAtoHAsync(void* dstHost,
                              hipArray* srcArray,
                              size_t srcOffset,
                              size_t ByteCount,
                              hipStream_t stream) {
  HIP_INIT_API(hipMemcpyAtoHAsync, dstHost, srcArray, srcOffset, ByteCount, stream);

  HIP_RETURN(ihipMemcpyAtoH(srcArray, dstHost, {srcOffset, 0, 0}, {0, 0, 0}, {ByteCount, 1, 1}, 0, 0, stream, true));
}

hipError_t hipMemcpyDtoA(hipArray* dstArray,
                        size_t dstOffset,
                        hipDeviceptr_t srcDevice,
                        size_t ByteCount) {
  HIP_INIT_API(hipMemcpyDtoA, dstArray, dstOffset, srcDevice, ByteCount);

  HIP_RETURN(ihipMemcpyDtoA(srcDevice, dstArray, {0, 0, 0}, {dstOffset, 0, 0}, {ByteCount, 1, 1}, 0, 0, nullptr));
}

hipError_t hipMemcpyHtoAAsync(hipArray* dstArray,
                              size_t dstOffset,
                              const void* srcHost,
                              size_t ByteCount,
                              hipStream_t stream) {
  HIP_INIT_API(hipMemcpyHtoAAsync, dstArray, dstOffset, srcHost, ByteCount, stream);

  HIP_RETURN(ihipMemcpyHtoA(srcHost, dstArray, {0, 0, 0}, {dstOffset, 0, 0}, {ByteCount, 1, 1}, 0, 0, stream, true));
}

hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* pHandle,
                                   HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                                   unsigned int numMipmapLevels) {
  HIP_INIT_API(hipMipmappedArrayCreate, pHandle, pMipmappedArrayDesc, numMipmapLevels);

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipMallocMipmappedArray(hipMipmappedArray_t *mipmappedArray,
                                   const hipChannelFormatDesc* desc,
                                   hipExtent extent,
                                   unsigned int numLevels,
                                   unsigned int flags) {
  HIP_INIT_API(hipMallocMipmappedArray, mipmappedArray, desc, extent, numLevels, flags);

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray) {
  HIP_INIT_API(hipMipmappedArrayDestroy, hMipmappedArray);

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray) {
  HIP_INIT_API(hipFreeMipmappedArray, mipmappedArray);

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray,
                                     hipMipmappedArray_t hMipMappedArray,
                                     unsigned int level) {
  HIP_INIT_API(hipMipmappedArrayGetLevel, pLevelArray, hMipMappedArray, level);

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipGetMipmappedArrayLevel(hipArray_t *levelArray,
                                     hipMipmappedArray_const_t mipmappedArray,
                                     unsigned int level) {
  HIP_INIT_API(hipGetMipmappedArrayLevel, levelArray, mipmappedArray, level);

  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipMallocHost(void** ptr,
                         size_t size) {
  HIP_INIT_API(hipMallocHost, ptr, size);

  if (ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipMalloc(ptr, size, CL_MEM_SVM_FINE_GRAIN_BUFFER));
}

hipError_t hipFreeHost(void *ptr) {
  HIP_INIT_API(hipFreeHost, ptr);

  HIP_RETURN(ihipFree(ptr));
}
