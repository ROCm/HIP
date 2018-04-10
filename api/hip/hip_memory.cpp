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

  amd::Command* command;
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

hipError_t hipMemsetAsync(void* dst, int  value, size_t sizeBytes, hipStream_t stream) {
  HIP_INIT_API(dst, value, sizeBytes, stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  HIP_INIT_API(dst, value, sizeBytes);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemPtrGetInfo(void *ptr, size_t *size) {
  HIP_INIT_API(ptr, size);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipHostFree(void* ptr) {
  HIP_INIT_API(ptr);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipFreeArray(hipArray* array) {
  HIP_INIT_API(array);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
  HIP_INIT_API(pbase, psize, dptr);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  HIP_INIT_API(free, total);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
  HIP_INIT_API(ptr, pitch, width, height);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent) {
  HIP_INIT_API(pitchedDevPtr, &extent);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipArrayCreate(hipArray** array, const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
  HIP_INIT_API(array, pAllocateArray);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc,
                          size_t width, size_t height, unsigned int flags) {
  HIP_INIT_API(array, desc, width, height, flags);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMalloc3DArray(hipArray_t* array, const struct hipChannelFormatDesc* desc,
                            struct hipExtent extent, unsigned int flags) {
  HIP_INIT_API(array, desc, &extent, flags);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
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

hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(dst, wOffset, hOffset, src, count, kind);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(dst, srcArray, wOffset, hOffset, count, kind);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
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
