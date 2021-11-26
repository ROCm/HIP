/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
#include "hip_test_common.hh"

#ifdef __linux__
  #include <sys/sysinfo.h>
#else
  #include <windows.h>
  #include <sysinfoapi.h>
#endif

namespace HipTest {
static inline int getGeviceCount() {
  int dev = 0;
  HIP_CHECK(hipGetDeviceCount(&dev));
  return dev;
}

// Get Free Memory from the system
static size_t getMemoryAmount() {
#ifdef __linux__
  struct sysinfo info{};
  sysinfo(&info);
  return info.freeram / (1024 * 1024);  // MB
#elif defined(_WIN32)
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  GlobalMemoryStatusEx(&statex);
  return (statex.ullAvailPhys / (1024 * 1024));  // MB
#endif
}

static size_t getHostThreadCount(const size_t memPerThread,
                                           const size_t maxThreads) {
  if (memPerThread == 0) return 0;
  auto memAmount = getMemoryAmount();
  const auto processor_count = std::thread::hardware_concurrency();
  if (processor_count == 0 || memAmount == 0) return 0;
  size_t thread_count = 0;
  if ((processor_count * memPerThread) < memAmount)
    thread_count = processor_count;
  else
    thread_count = reinterpret_cast<size_t>(memAmount / memPerThread);
  if (maxThreads > 0) {
    return (thread_count > maxThreads) ? maxThreads : thread_count;
  }
  return thread_count;
}

}  // namespace HipTest
