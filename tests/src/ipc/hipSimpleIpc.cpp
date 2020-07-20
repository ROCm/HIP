/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"

#define N 1024
#define OFFSET 128

void single_process() {
  int* ipc_dptr = nullptr;
  int* ipc_hptr = nullptr;
  int* ipc_out_dptr = nullptr;
  int* ipc_out_hptr = nullptr;

  int* ipc_offset_dptr = nullptr;

  hipIpcMemHandle_t ipc_handle;
  hipIpcMemHandle_t ipc_offset_handle;

  HIPCHECK(hipMalloc((void**)&ipc_dptr, N * sizeof(int)));

  // Negative, Make sure we return error when an offset of original ptr is passed
  ipc_offset_dptr = ipc_dptr + (OFFSET * sizeof(int));
  assert(hipErrorInvalidDevicePointer == hipIpcGetMemHandle(&ipc_offset_handle, ipc_offset_dptr));

  // Get handle for the device_ptr
  HIPCHECK(hipIpcGetMemHandle(&ipc_handle, ipc_dptr));

  // Set Values @ Host Ptr
  ipc_hptr = new int[N];
  for (size_t idx = 0; idx < N; ++idx) {
     ipc_hptr[idx] = idx;
  }

  // Copy values to Device ptr
  HIPCHECK(hipMemset(ipc_dptr, 0x00, (N * sizeof(int))));
  HIPCHECK(hipMemcpy(ipc_dptr, ipc_hptr, (N * sizeof(int)), hipMemcpyHostToDevice));

  // Open handle to get dev_ptr
  ipc_out_hptr = new int[N];
  memset(ipc_out_hptr, 0x00, (N * sizeof(int)));
  HIPCHECK(hipIpcOpenMemHandle((void**)&ipc_out_dptr, ipc_handle, 0));

  // Copy Values from Device to Host and Check for correctness
  HIPCHECK(hipMemcpy(ipc_out_hptr, ipc_out_dptr, (N * sizeof(int)), hipMemcpyDeviceToHost));
  for (size_t idx = 0; idx < N; ++idx) {
    if(ipc_out_hptr[idx] != idx) {
      std::cout<<"Failing @ idx: "<<idx<<std::endl;
    }
  }

  //Close All Mem Handle
  HIPCHECK(hipIpcCloseMemHandle(ipc_out_dptr));
  HIPCHECK(hipFree(ipc_dptr));

  delete[] ipc_hptr;
  delete[] ipc_out_hptr;
}

void multi_process() {
  //To create and open IPC handle via multiple process
}

int main() {
  single_process();
  multi_process();
  passed();
}
