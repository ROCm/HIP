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

#include <hip_array_common.hh>
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
        // Cast to void to suppress nodiscard warnings
        static_cast<void>(hipHostUnregister(host_ptr_));
        free(host_ptr_);
        break;
      case LinearAllocs::hipHostMalloc:
        static_cast<void>(hipHostFree(ptr_));
        break;
      case LinearAllocs::hipMalloc:
      case LinearAllocs::hipMallocManaged:
        static_cast<void>(hipFree(ptr_));
    }
  }

  T* ptr() const { return ptr_; };
  T* host_ptr() const { return host_ptr_; }

 private:
  const LinearAllocs allocation_type_;
  T* ptr_ = nullptr;
  T* host_ptr_ = nullptr;
};

template <typename T> class LinearAllocGuardMultiDim {
 protected:
  LinearAllocGuardMultiDim(hipExtent extent) : extent_{extent} {}

  ~LinearAllocGuardMultiDim() { static_cast<void>(hipFree(pitched_ptr_.ptr)); }

 public:
  T* ptr() const { return reinterpret_cast<T*>(pitched_ptr_.ptr); };

  size_t pitch() const { return pitched_ptr_.pitch; }

  hipExtent extent() const { return extent_; }

  hipPitchedPtr pitched_ptr() const { return pitched_ptr_; }

  size_t width() const { return extent_.width; }

  size_t width_logical() const { return extent_.width / sizeof(T); }

  size_t height() const { return extent_.height; }

 public:
  hipPitchedPtr pitched_ptr_;
  const hipExtent extent_;
};

template <typename T> class LinearAllocGuard2D : public LinearAllocGuardMultiDim<T> {
 public:
  LinearAllocGuard2D(const size_t width_logical, const size_t height)
      : LinearAllocGuardMultiDim<T>{make_hipExtent(width_logical * sizeof(T), height, 1)} {
    HIP_CHECK(hipMallocPitch(&this->pitched_ptr_.ptr, &this->pitched_ptr_.pitch,
                             this->extent_.width, this->extent_.height));
  }

  LinearAllocGuard2D(const LinearAllocGuard2D&) = delete;
  LinearAllocGuard2D(LinearAllocGuard2D&&) = delete;
};

template <typename T> class LinearAllocGuard3D : public LinearAllocGuardMultiDim<T> {
 public:
  LinearAllocGuard3D(const size_t width_logical, const size_t height, const size_t depth)
      : LinearAllocGuardMultiDim<T>{make_hipExtent(width_logical * sizeof(T), height, depth)} {
    HIP_CHECK(hipMalloc3D(&this->pitched_ptr_, this->extent_));
  }

  LinearAllocGuard3D(const hipExtent extent) : LinearAllocGuardMultiDim<T>(extent) {
    HIP_CHECK(hipMalloc3D(&this->pitched_ptr_, this->extent_));
  }

  LinearAllocGuard3D(const LinearAllocGuard3D&) = delete;
  LinearAllocGuard3D(LinearAllocGuard3D&&) = delete;

  size_t depth() const { return this->extent_.depth; }
};

template <typename T> class ArrayAllocGuard {
 public:
  // extent should contain logical width
  ArrayAllocGuard(const hipExtent extent, const unsigned int flags = 0u) : extent_{extent} {
    hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
    HIP_CHECK(hipMalloc3DArray(&ptr_, &desc, extent_, flags));
  }

  ~ArrayAllocGuard() { static_cast<void>(hipFreeArray(ptr_)); }

  ArrayAllocGuard(const ArrayAllocGuard&) = delete;
  ArrayAllocGuard(ArrayAllocGuard&&) = delete;

  hipArray_t ptr() const { return ptr_; }

  hipExtent extent() const { return extent_; }

 private:
  hipArray_t ptr_ = nullptr;
  const hipExtent extent_;
};

template <typename T> class DrvArrayAllocGuard {
 public:
  // extent should contain width in bytes
  DrvArrayAllocGuard(const hipExtent extent, const unsigned int flags = 0u) : extent_{extent} {
    HIP_ARRAY3D_DESCRIPTOR desc{};
    using vec_info = vector_info<T>;
    desc.Format = vec_info::format;
    desc.NumChannels = vec_info::size;
    desc.Width = extent_.width / sizeof(T);
    desc.Height = extent_.height;
    desc.Depth = extent_.depth;
    desc.Flags = flags;
    HIP_CHECK(hipArray3DCreate(&ptr_, &desc));
  }

  ~DrvArrayAllocGuard() { static_cast<void>(hipArrayDestroy(ptr_)); }

  DrvArrayAllocGuard(const DrvArrayAllocGuard&) = delete;
  DrvArrayAllocGuard(DrvArrayAllocGuard&&) = delete;

  hiparray ptr() const { return ptr_; }

  hipExtent extent() const { return extent_; }

 private:
  hiparray ptr_ = nullptr;
  const hipExtent extent_;
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
      static_cast<void>(hipStreamDestroy(stream_));
    }
  }

  hipStream_t stream() const { return stream_; }

 private:
  const Streams stream_type_;
  hipStream_t stream_;
};