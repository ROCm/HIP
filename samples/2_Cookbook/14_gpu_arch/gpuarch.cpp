/*
Copyright (c) 2020-Present Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip/hip_runtime.h"
#include <iostream>

#define THREADS_PER_BLOCK   64
#define BLOCKS_PER_GRID     4
#define SIZE                (BLOCKS_PER_GRID * THREADS_PER_BLOCK)
#define NOT_SUPPORTED       -99  // dummy number indicates unsupported operation

#define HIP_STATUS_CHECK(status)                                                                   \
  if (status != hipSuccess) {                                                                      \
    std::cout << "Got Status: " << status << " at Line: " << __LINE__ << std::endl;                \
    exit(0);                                                                                       \
  }

// Using __gfx*__ macro one can have GPU architecture specific code flow
// For example: If below kernel runs on gfx908 it will increment 'in' by 'value' and store into
// 'out'
//              but it will update with "NOT_SUPPORTED" for any other gfx archs.
__global__ void incrementKernel(int32_t* in, int32_t* out, int32_t value, size_t buffSize) {
  int index = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  if (index < buffSize) {
#if defined(__gfx908__)
    out[index] = in[index] + value;
#else
    out[index] = NOT_SUPPORTED;
#endif
  }
}

int main() {
  int32_t incrementValue = 10;
  // Host pointers
  int32_t* hInput = nullptr;
  int32_t* hOutput = nullptr;
  // Device pointers
  int32_t* dInput = nullptr;
  int32_t* dOutput = nullptr;

  size_t NBytes = SIZE * sizeof(int32_t);

  hInput = static_cast<int32_t*>(malloc(NBytes));
  hOutput = static_cast<int32_t*>(malloc(NBytes));

  HIP_STATUS_CHECK(hipMalloc(&dInput, NBytes));
  HIP_STATUS_CHECK(hipMalloc(&dOutput, NBytes));

  // Initialize host input/output buffers
  for (int i = 0; i < SIZE; ++i) {
    hInput[i] = i;
    hOutput[i] = 0;
  }

  // Initialize device input buffer
  HIP_STATUS_CHECK(hipMemcpy(dInput, hInput, NBytes, hipMemcpyHostToDevice));

  // Launch kernel
  hipLaunchKernelGGL(incrementKernel, dim3(BLOCKS_PER_GRID), dim3(THREADS_PER_BLOCK), 0, 0, dInput,
                     dOutput, incrementValue, SIZE);

  // Copy result back to host buffer
  HIP_STATUS_CHECK(hipMemcpy(hOutput, dOutput, NBytes, hipMemcpyDeviceToHost));

  bool flag = true;
  // verify data
  for (int i = 0; i < SIZE; ++i) {
    if (hOutput[i] != NOT_SUPPORTED && hOutput[i] != (hInput[i] + incrementValue)) {
      std::cout << "Error : Data mismatch found";
      exit(0);
    } else if (hOutput[i] == NOT_SUPPORTED) {
      flag &= false;
    }
  }
  if (flag == false) {
    std::cout << "Error: Kernel is supported for gfx908 architecture\n";
  } else {
    std::cout << "success\n";
  }
  return 0;
}