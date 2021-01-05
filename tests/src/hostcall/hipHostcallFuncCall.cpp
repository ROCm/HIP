/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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

/* HIT_START
 * BUILD: %t %s EXCLUDE_HIP_PLATFORM nvidia
 * HIT_END
 */

#include <test_common.h>

extern "C" __device__ HIP_vector_base<long long, 2>::Native_vec_ __ockl_call_host_function(
    uint64_t fptr, uint64_t arg0, uint64_t arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6);

static void callee(uint64_t* output, uint64_t* input) {
  output[0] = input[0] + 1;
  output[1] = input[1] + input[2];
}

__global__ void kernel(uint64_t fptr, uint64_t* retval0, uint64_t* retval1) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  uint64_t arg0 = (uint64_t)fptr;
  uint64_t arg1 = tid;
  uint64_t arg2 = 42;
  uint64_t arg3 = tid % 23;
  uint64_t arg4 = 0;
  uint64_t arg5 = 0;
  uint64_t arg6 = 0;
  uint64_t arg7 = 0;

  longlong2 result = {0, 0};
  if (tid % 71 != 1) {
    result.data = __ockl_call_host_function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    retval0[tid] = result.x;
    retval1[tid] = result.y;
  }
}

static bool test() {
  uint num_blocks = 5;
  uint threads_per_block = 1000;
  uint num_threads = num_blocks * threads_per_block;

  void* retval0_void;
  HIPCHECK(hipHostMalloc(&retval0_void, 8 * num_threads));
  auto retval0 = reinterpret_cast<uint64_t*>(retval0_void);
  for (uint i = 0; i != num_threads; ++i) {
    retval0[i] = 0x23232323;
  }

  void* retval1_void;
  HIPCHECK(hipHostMalloc(&retval1_void, 8 * num_threads));
  auto retval1 = reinterpret_cast<uint64_t*>(retval1_void);
  for (uint i = 0; i != num_threads; ++i) {
    retval1[i] = 0x23232323;
  }

  hipLaunchKernelGGL(kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0, (uint64_t)callee,
                     retval0, retval1);

  hipStreamSynchronize(0);

  for (uint i = 0; i != num_threads; ++i) {
    uint64_t value = retval0[i];
    if (i % 71 == 1) {
      if (value != 0x23232323) {
        printf("failed\n");
        return false;
      }
    } else {
      if (value != i + 1) {
        printf("failed\n");
        return false;
      }
    }
  }

  return true;
}

int main(int argc, char** argv) {
  if (!test()) {
    return 1;
  }

  printf("passed\n");
  return 0;
}
