/*
Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.
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

#include "hip/hip_runtime.h"

extern "C" __global__ void bit_extract_kernel(uint32_t* C_d, const uint32_t*
                                              A_d, size_t N) {
  size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
  size_t stride = hipBlockDim_x * hipGridDim_x;

  for (size_t i = offset; i < N; i += stride) {
#ifdef __HIP_PLATFORM_AMD__
    C_d[i] = __bitextract_u32(A_d[i], 8, 4);
#else /* defined __HIP_PLATFORM_NVIDIA__ or other path */
    C_d[i] = ((A_d[i] & 0xf00) >> 8);
#endif
  }
}
