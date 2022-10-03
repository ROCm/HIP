/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

enum class LinearAllocs {
  malloc,
  mallocAndRegister,
  hipHostMalloc,
  hipMalloc,
  hipMallocManaged,
};

template <typename T> class LinearAllocGuard {
 public:
  LinearAllocGuard(const LinearAllocs allocation_type, const size_t size,
                   const unsigned int flags = 0u)
      : allocation_type_{allocation_type} {
    switch (allocation_type_) {
      case LinearAllocs::malloc:
        ptr_ = host_ptr_ = reinterpret_cast<T*>(malloc(size));
        break;
      case LinearAllocs::mallocAndRegister:
        host_ptr_ = reinterpret_cast<T*>(malloc(size));
        HIP_CHECK(hipHostRegister(host_ptr_, size, flags));
        HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&ptr_), host_ptr_, 0u));
        break;
      case LinearAllocs::hipHostMalloc:
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&ptr_), size, flags));
        host_ptr_ = ptr_;
        break;
      case LinearAllocs::hipMalloc:
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr_), size));
        break;
      case LinearAllocs::hipMallocManaged:
        HIP_CHECK(hipMallocManaged(reinterpret_cast<void**>(&ptr_), size, flags ? flags : 1u));
        host_ptr_ = ptr_;
    }
  }

  LinearAllocGuard(const LinearAllocGuard&) = delete;
  LinearAllocGuard(LinearAllocGuard&&) = delete;

  ~LinearAllocGuard() {
    // No Catch macros, don't want to possibly throw in the destructor
    switch (allocation_type_) {
      case LinearAllocs::malloc:
        free(ptr_);
        break;
      case LinearAllocs::mallocAndRegister:
        hipHostUnregister(host_ptr_);
        free(host_ptr_);
        break;
      case LinearAllocs::hipHostMalloc:
        hipHostFree(ptr_);
        break;
      case LinearAllocs::hipMalloc:
      case LinearAllocs::hipMallocManaged:
        hipFree(ptr_);
    }
  }

  T* ptr() { return ptr_; };
  T* const ptr() const { return ptr_; };
  T* host_ptr() { return host_ptr_; }
  T* const host_ptr() const { return host_ptr(); }

 private:
  const LinearAllocs allocation_type_;
  T* ptr_ = nullptr;
  T* host_ptr_ = nullptr;
};

enum class Streams { nullstream, perThread, created };

class StreamGuard {
 public:
  StreamGuard(const Streams stream_type) : stream_type_{stream_type} {
    switch (stream_type_) {
      case Streams::nullstream:
        stream_ = nullptr;
        break;
      case Streams::perThread:
        stream_ = hipStreamPerThread;
        break;
      case Streams::created:
        HIP_CHECK(hipStreamCreate(&stream_));
    }
  }

  StreamGuard(const StreamGuard&) = delete;
  StreamGuard(StreamGuard&&) = delete;

  ~StreamGuard() {
    if (stream_type_ == Streams::created) {
      hipStreamDestroy(stream_);
    }
  }

  hipStream_t stream() const { return stream_; }

 private:
  const Streams stream_type_;
  hipStream_t stream_;
};

inline unsigned int GenerateLinearAllocationFlagCombinations(const LinearAllocs allocation_type) {
  switch (allocation_type) {
    case LinearAllocs::mallocAndRegister:
      // TODO
      return 0;
    case LinearAllocs::hipHostMalloc:
      return GENERATE(hipHostMallocDefault, hipHostMallocPortable, hipHostMallocMapped,
                      hipHostMallocWriteCombined);
    case LinearAllocs::hipMallocManaged:
      // TODO
      return 1u;
    case LinearAllocs::malloc:
    case LinearAllocs::hipMalloc:
      return 0u;
    default:
      assert("Invalid LinearAllocs enumerator");
  }
}