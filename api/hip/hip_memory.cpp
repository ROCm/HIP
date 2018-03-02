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

hipError_t hipMalloc(void** ptr, size_t sizeBytes)
{
  HIP_INIT_API(ptr, sizeBytes);

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

  amd::Memory* mem = new (*g_context) amd::Buffer(*g_context, 0, sizeBytes);
  if (!mem) {
    return hipErrorOutOfMemory;
  }

  if (!mem->create(nullptr)) {
    return hipErrorMemoryAllocation;
  }

  *ptr = reinterpret_cast<void*>(as_cl(mem));

  return hipSuccess;
}

hipError_t hipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags)
{
  HIP_INIT_API(ptr, sizeBytes, flags);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipFree(void* ptr)
{
  if (!is_valid(reinterpret_cast<cl_mem>(ptr))) {
    return hipErrorInvalidValue;
  }
  as_amd(reinterpret_cast<cl_mem>(ptr))->release();
  return hipSuccess;
}

hipError_t hipMemcpyAsync(void* dst,
                          const void* src,
                          size_t sizeBytes,
                          hipMemcpyKind kind,
                          hipStream_t stream)
{
  HIP_INIT_API(dst, src, sizeBytes, kind, stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}


hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
{
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

  switch (kind) {
  case hipMemcpyDeviceToHost:
    command = new amd::ReadMemoryCommand(*queue, CL_COMMAND_READ_BUFFER, waitList,
      *as_amd(reinterpret_cast<cl_mem>(const_cast<void*>(src)))->asBuffer(), 0, sizeBytes, dst);
    break;
  case hipMemcpyHostToDevice:
    command = new amd::WriteMemoryCommand(*queue, CL_COMMAND_WRITE_BUFFER, waitList,
      *as_amd(reinterpret_cast<cl_mem>(dst))->asBuffer(), 0, sizeBytes, src);
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

hipError_t hipMemsetAsync(void* dst, int  value, size_t sizeBytes, hipStream_t stream )
{
  HIP_INIT_API(dst, value, sizeBytes, stream);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemset(void* dst, int value, size_t sizeBytes)
{
  HIP_INIT_API(dst, value, sizeBytes);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}

hipError_t hipMemPtrGetInfo(void *ptr, size_t *size)
{
  HIP_INIT_API(ptr, size);

  assert(0 && "Unimplemented");

  return hipErrorUnknown;
}
