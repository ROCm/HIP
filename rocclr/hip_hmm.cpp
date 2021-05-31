/* Copyright (c) 2020-present Advanced Micro Devices, Inc.

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

// Forward declaraiton of a function
hipError_t ihipMallocManaged(void** ptr, size_t size, unsigned int align = 0);

// Make sure HIP defines match ROCclr to avoid double conversion
static_assert(hipCpuDeviceId == amd::CpuDeviceId, "CPU device ID mismatch with ROCclr!");
static_assert(hipInvalidDeviceId == amd::InvalidDeviceId,
              "Invalid device ID mismatch with ROCclr!");

static_assert(static_cast<uint32_t>(hipMemAdviseSetReadMostly) ==
              amd::MemoryAdvice::SetReadMostly, "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseUnsetReadMostly) ==
              amd::MemoryAdvice::UnsetReadMostly, "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseSetPreferredLocation) ==
              amd::MemoryAdvice::SetPreferredLocation, "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseUnsetPreferredLocation) ==
              amd::MemoryAdvice::UnsetPreferredLocation, "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseSetAccessedBy) ==
              amd::MemoryAdvice::SetAccessedBy, "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseUnsetAccessedBy) ==
              amd::MemoryAdvice::UnsetAccessedBy, "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseSetCoarseGrain) ==
              amd::MemoryAdvice::SetCoarseGrain, "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemAdviseUnsetCoarseGrain) ==
              amd::MemoryAdvice::UnsetCoarseGrain, "Enum mismatch with ROCclr!");

static_assert(static_cast<uint32_t>(hipMemRangeAttributeReadMostly) ==
              amd::MemRangeAttribute::ReadMostly, "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemRangeAttributePreferredLocation) ==
              amd::MemRangeAttribute::PreferredLocation, "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemRangeAttributeAccessedBy) ==
              amd::MemRangeAttribute::AccessedBy, "Enum mismatch with ROCclr!");
static_assert(static_cast<uint32_t>(hipMemRangeAttributeLastPrefetchLocation) ==
              amd::MemRangeAttribute::LastPrefetchLocation, "Enum mismatch with ROCclr!");

// ================================================================================================
hipError_t hipMallocManaged(void** dev_ptr, size_t size, unsigned int flags) {
  HIP_INIT_API(hipMallocManaged, dev_ptr, size, flags);

  if ((dev_ptr == nullptr) || (size == 0) ||
      ((flags != hipMemAttachGlobal) && (flags != hipMemAttachHost))) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(ihipMallocManaged(dev_ptr, size), *dev_ptr);
}

// ================================================================================================
hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device,
                               hipStream_t stream) {
  HIP_INIT_API(hipMemPrefetchAsync, dev_ptr, count, device, stream);

  if ((dev_ptr == nullptr) || (count == 0)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  amd::HostQueue* queue = nullptr;
  bool cpu_access = (device == hipCpuDeviceId) ? true : false;

  // Pick the specified stream or Null one from the provided device
  if (stream != nullptr) {
    queue = hip::getQueue(stream);
  } else {
    if (!cpu_access) {
      queue = g_devices[device]->NullStream();
    } else {
      queue = hip::getCurrentDevice()->NullStream();
    }
  }
  if (queue == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Command::EventWaitList waitList;
  amd::SvmPrefetchAsyncCommand* command =
      new amd::SvmPrefetchAsyncCommand(*queue, waitList, dev_ptr, count, cpu_access);
  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }

  if (!command->validateMemory()) {
    delete command;
    HIP_RETURN(hipErrorInvalidValue);
  }
  command->enqueue();
  command->release();

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device) {
  HIP_INIT_API(hipMemAdvise, dev_ptr, count, advice, device);

  if ((dev_ptr == nullptr) || (count == 0) ||
      ((device != hipCpuDeviceId) && (static_cast<size_t>(device) >= g_devices.size()))) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  amd::Device* dev = (device == hipCpuDeviceId) ?
    g_devices[0]->devices()[0] : g_devices[device]->devices()[0];
  bool use_cpu = (device == hipCpuDeviceId) ? true : false;

  // Set the allocation attributes in AMD HMM
  if (!dev->SetSvmAttributes(dev_ptr, count, static_cast<amd::MemoryAdvice>(advice), use_cpu)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemRangeGetAttribute(void* data, size_t data_size, hipMemRangeAttribute attribute,
                                   const void* dev_ptr, size_t count) {
  HIP_INIT_API(hipMemRangeGetAttribute, data, data_size, attribute, dev_ptr, count);

  if ((data == nullptr) || (data_size == 0) || (dev_ptr == nullptr) || (count == 0)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Shouldn't matter for which device the interface is called
  amd::Device* dev = g_devices[0]->devices()[0];

  // Get the allocation attribute from AMD HMM
  if (!dev->GetSvmAttributes(&data, &data_size, reinterpret_cast<int*>(&attribute), 1,
                             dev_ptr, count)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes,
                                    hipMemRangeAttribute* attributes, size_t num_attributes,
                                    const void* dev_ptr, size_t count) {
  HIP_INIT_API(hipMemRangeGetAttributes, data, data_sizes,
               attributes, num_attributes, dev_ptr, count);

  if ((data == nullptr) || (data_sizes == nullptr) || (attributes == nullptr) ||
      (num_attributes == 0) || (dev_ptr == nullptr) || (count == 0)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Shouldn't matter for which device the interface is called
  amd::Device* dev = g_devices[0]->devices()[0];
  // Get the allocation attributes from AMD HMM
  if (!dev->GetSvmAttributes(data, data_sizes, reinterpret_cast<int*>(attributes),
      num_attributes, dev_ptr, count)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipStreamAttachMemAsync(hipStream_t stream, hipDeviceptr_t* dev_ptr,
                                   size_t length, unsigned int flags) {
  HIP_INIT_API(hipStreamAttachMemAsync, stream, dev_ptr, length, flags);

  if ((stream == nullptr) || (dev_ptr == nullptr) || (length == 0)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Unclear what should be done for this interface in AMD HMM, since it's generic SVM alloc
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t ihipMallocManaged(void** ptr, size_t size, unsigned int align) {
  if (ptr == nullptr) {
    return hipErrorInvalidValue;
  } else if (size == 0) {
    *ptr = nullptr;
    return hipSuccess;
  }

  assert((hip::host_device->asContext()!= nullptr) && "Current host context must be valid");
  amd::Context& ctx = *hip::host_device->asContext();

  const amd::Device& dev = *ctx.devices()[0];

  // For now limit to the max allocation size on the device.
  // The apps should be able to go over the limit in the future
  if (dev.info().maxMemAllocSize_ < size) {
    return hipErrorMemoryAllocation;
  }

  // Allocate SVM fine grain buffer with the forced host pointer, avoiding explicit memory
  // allocation in the device driver
  *ptr = amd::SvmBuffer::malloc(ctx, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_ALLOC_HOST_PTR,
                                size, (align == 0) ? dev.info().memBaseAddrAlign_ : align);
  if (*ptr == nullptr) {
    return hipErrorMemoryAllocation;
  }

  ClPrint(amd::LOG_INFO, amd::LOG_API, "%-5d: [%zx] ihipMallocManaged ptr=0x%zx",  getpid(),
    std::this_thread::get_id(), *ptr);
  return hipSuccess;
}
