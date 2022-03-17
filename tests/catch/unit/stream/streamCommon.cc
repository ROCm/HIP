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

#include "streamCommon.hh"

namespace hip {

inline namespace internal {

bool checkStreamPriority_(hipStream_t stream, bool checkPriority = false, int priority_ = 0) {
  int priority{0};
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  if (checkPriority) {
    if (priority_ != priority) {
      UNSCOPED_INFO("Priority Mismatch, Expected Priority: " << priority_
                                                             << " Actual Priority: " << priority);
      return false;
    }
  } else {
    int priority_low{0}, priority_high{0};
    HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    if (priority_low < priority || priority_high > priority) {
      UNSCOPED_INFO("Priority Mismatch, Expected Priority Range: "
                    << priority_low << " - " << priority_high << " Actual Priority: " << priority);
      return false;
    }
  }
  return true;
}

bool checkStreamFlags_(hipStream_t stream, bool checkFlags = false, unsigned flags_ = 0) {
  unsigned flags{0};
  HIP_CHECK(hipStreamGetFlags(stream, &flags));
  if (checkFlags) {
    if (flags_ != flags) {
      UNSCOPED_INFO("Flags Mismatch, Expected Flag: " << flags_ << " Actual Flag: " << flags);
      return false;
    }
  } else {
    if (flags != hipStreamDefault && flags != hipStreamNonBlocking) {
      UNSCOPED_INFO("Flags Mismatch, Expected Flag: " << hipStreamDefault << " or "
                                                      << hipStreamNonBlocking
                                                      << " Actual Flag: " << flags);
      return false;
    }
  }
  return true;
}
}  // namespace internal

inline namespace stream {

__device__ int defaultSemaphore = 0;

__global__ void signaling_kernel(int* semaphore) {
  size_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid == 0) {
    if (semaphore == nullptr) {
      atomicAdd(&defaultSemaphore, 1);
    } else {
      atomicAdd(semaphore, 1);
    }
  }
}

__global__ void waiting_kernel(int* semaphore) {
  size_t tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (tid == 0) {
    if (semaphore == nullptr) {
      while (atomicCAS(&defaultSemaphore, 1, 2) == 0) {
      }
    } else {
      while (atomicCAS(semaphore, 1, 2) == 0) {
      }
    }
  }
}

std::thread startSignalingThread(int* semaphore) {
  std::thread signalingThread([semaphore]() {
    hipStream_t signalingStream;
    HIP_CHECK(hipStreamCreateWithFlags(&signalingStream, hipStreamNonBlocking));

    signaling_kernel<<<1, 1, 0, signalingStream>>>(semaphore);
    HIP_CHECK(hipStreamSynchronize(signalingStream));
    HIP_CHECK(hipStreamDestroy(signalingStream));
  });

  return signalingThread;
}

bool checkStream(hipStream_t stream) {
  {  // Check default flags
    auto res = checkStreamFlags_(stream, true, hipStreamDefault);
    if (!res) return false;
  }

  {  // Check default Priority
    auto res = checkStreamPriority_(stream);
    if (!res) return false;
  }

  return true;
}

bool checkStreamPriorityAndFlags(hipStream_t stream, int priority, unsigned int flags) {
  {  // Check flags
    auto res = checkStreamFlags_(stream, true, flags);
    if (!res) return false;
  }

  {  // Check priority
    auto res = checkStreamPriority_(stream, true, priority);
    if (!res) return false;
  }

  return true;
}

}  // namespace stream
}  // namespace hip
