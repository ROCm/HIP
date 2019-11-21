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
 * BUILD: %t %s EXCLUDE_HIP_PLATFORM all
 * HIT_END
 */

#include <test_common.h>

// This is NOT a real printf test. It is a test for calling a host function
// which happens to be a wrapper around system printf.

extern "C" __device__ HIP_vector_base<long, 2>::Native_vec_ __ockl_call_host_function(
    ulong fptr, ulong arg0, ulong arg1, ulong arg2, ulong arg3, ulong arg4, ulong arg5, ulong arg6);

// FuncCall service function that expects three arguments bundled in the
// request: the format string, and two uint64_t arguments.
void print_things_0(ulong* output, ulong* input) {
  auto fmt = reinterpret_cast<const char*>(input);
  auto arg0 = input[2];
  auto arg1 = input[3];
  output[0] = fprintf(stdout, fmt, arg0, arg1);
}

__global__ void kernel0(ulong fptr, ulong* retval) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  ulong arg0 = fptr;

  const char* str = "(%lu -> %lu)\n";
  ulong arg1 = 0;
  for (int ii = 0; ii != 8; ++ii) {
    arg1 |= (ulong)str[ii] << (8 * ii);
  }
  ulong arg2 = 0;
  for (int ii = 0; ii != 7; ++ii) {
    arg2 |= (ulong)str[ii + 8] << (8 * ii);
  }

  ulong arg3 = 42;
  ulong arg4 = tid;
  ulong arg5 = 0;
  ulong arg6 = 0;
  ulong arg7 = 0;

  long2 result = {0, 0};
  result.data = __ockl_call_host_function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
  *retval = result.x;
}

// FuncCall service function that expects two arguments bundled in the request:
// a kernel "name" and a uint64_t thread ID. The format string is built into the
// service function itself.
void print_things_1(ulong* output, const ulong* input) {
  auto name = reinterpret_cast<const char*>(input[0]);
  auto tid = input[1];
  output[0] = fprintf(stdout, "kernel: %s; tid: %lu\n", name, tid);
}

__global__ void kernel1(ulong fptr, ulong name, ulong* retval) {
  uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  ulong arg0 = fptr;
  ulong arg1 = name;
  ulong arg2 = tid;
  ulong arg3 = 0;
  ulong arg4 = 0;
  ulong arg5 = 0;
  ulong arg6 = 0;
  ulong arg7 = 0;

  long2 result = {0, 0};
  result.data = __ockl_call_host_function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
  *retval = result.x;
}

static bool test() {
  void* retval_void;
  HIPCHECK(hipHostMalloc(&retval_void, 8));
  auto retval = reinterpret_cast<uint64_t*>(retval_void);
  *retval = 0x23232323;

  hipLaunchKernelGGL(kernel0, dim3(1), dim3(1), 0, 0, (ulong)print_things_0, retval);
  hipStreamSynchronize(0);
  if (*retval != strlen("(42 -> 0)\n")) {
    return false;
  }

  *retval = 0x23232323;
  const char* name = "kernel1";
  hipLaunchKernelGGL(kernel1, dim3(1), dim3(1), 0, 0, (ulong)print_things_1, (ulong)name, retval);
  hipStreamSynchronize(0);
  if (*retval != strlen("kernel: kernel1; tid: 0\n")) {
    return false;
  }

  return true;
}

int main(int argc, char** argv) {
  if (!test()) {
    printf("failed\n");
    return 1;
  }
  printf("passed\n");
  return 0;
}
