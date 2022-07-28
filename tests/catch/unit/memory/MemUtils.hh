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
#include <memory>
#include <hip_test_common.hh>


static __global__ void waitKernel(clock_t offset) {
  auto time = clock();
  while (clock() - time < offset) {
  }
}

static __global__ void clock_kernel(clock_t clock_count, size_t* co) {
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count) {
    clock_offset = clock() - start_clock;
  }
  *co = clock_offset;
}

// number of clocks the device is running at (device frequency)
static size_t ticksPerMillisecond = 0;
// Var used in multithreaded test cases
constexpr size_t numAllocs = 10;


// helper function used to set the device frequency variable

static size_t findTicks() {
  hipDeviceProp_t prop;
  int device;
  size_t* clockOffset;
  HIP_CHECK(hipMalloc(&clockOffset, sizeof(size_t)));
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  constexpr float milliseconds = 1000;
  constexpr float tolerance = 0.02 * milliseconds;

  clock_t devFreq = static_cast<clock_t>(prop.clockRate);  // in kHz
  clock_t time = devFreq * milliseconds;
  // Warmup
  hipLaunchKernelGGL(clock_kernel, dim3(1), dim3(1), 0, 0, time, clockOffset);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // try 10 times to find device frequency
  // after 10 attempts the result is likely good enough so just accept it
  size_t co = 0;

  for (int attempts = 10; attempts > 0; attempts--) {
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(clock_kernel, dim3(1), dim3(1), 0, 0, time, clockOffset);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventSynchronize(stop));

    float executionTime = 0;
    HIP_CHECK(hipEventElapsedTime(&executionTime, start, stop));

    HIP_CHECK(hipMemcpy(&co, clockOffset, sizeof(size_t), hipMemcpyDeviceToHost));
    if (executionTime >= (milliseconds - tolerance) &&
        executionTime <= (milliseconds + tolerance)) {
      // Timing is within accepted tolerance, break here
      break;
    } else {
      auto off = fabs(milliseconds - executionTime) / milliseconds;
      if (executionTime >= milliseconds) {
        time -= (time * off);
        --attempts;
      } else {
        time += (time * off);
        --attempts;
      }
    }
  }
  HIP_CHECK(hipFree(clockOffset));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return co;
}

// Launches a kernel which runs for specified amount of time
// Note: The current implementation uses HIP_CHECK which is not thread safe!
// Note the function assumes execution on a single device, if changing devices between calls of
// runKernelForMs set ticksPerMillisecond back to 0 (should be avoided as it slows down testing
// significantly)
static void runKernelForMs(size_t millis, hipStream_t stream = nullptr) {
  if (ticksPerMillisecond == 0) {
    ticksPerMillisecond = findTicks();
  }
  hipLaunchKernelGGL(waitKernel, dim3(1), dim3(1), 0, stream, ticksPerMillisecond * millis / 1000);
  HIP_CHECK(hipGetLastError());
}
