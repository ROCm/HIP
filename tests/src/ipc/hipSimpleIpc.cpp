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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"

#define NUM 1024

hipError_t single_process(int32_t offset) {
  int* ipc_dptr = nullptr;
  int* ipc_hptr = nullptr;
  int* ipc_out_dptr = nullptr;
  int* ipc_out_hptr = nullptr;

  int* ipc_offset_dptr = nullptr;
  hipIpcMemHandle_t ipc_offset_handle;

  HIPCHECK_RETURN_ONFAIL(hipMalloc(reinterpret_cast<void**>(&ipc_dptr), NUM * sizeof(int)));

  // Add offset to the dev_ptr
  ipc_offset_dptr = ipc_dptr + offset;
  // Get handle for the offsetted device_ptr
  HIPCHECK_RETURN_ONFAIL(hipIpcGetMemHandle(&ipc_offset_handle, ipc_offset_dptr));

  // Set Values @ Host Ptr
  ipc_hptr = new int[NUM];
  for (size_t idx = 0; idx < NUM; ++idx) {
     ipc_hptr[idx] = idx;
  }

  // Copy values to Device ptr
  HIPCHECK_RETURN_ONFAIL(hipMemset(ipc_dptr, 0x00, (NUM * sizeof(int))));
  HIPCHECK_RETURN_ONFAIL(hipMemcpy(ipc_dptr, ipc_hptr, (NUM * sizeof(int)), hipMemcpyHostToDevice));

  // Open handle to get dev_ptr
  ipc_out_hptr = new int[NUM];
  memset(ipc_out_hptr, 0x00, (NUM * sizeof(int)));
  HIPCHECK_RETURN_ONFAIL(hipIpcOpenMemHandle(reinterpret_cast<void**>(&ipc_out_dptr),
                                             ipc_offset_handle, 0));

  // Copy Values from Device to Host and Check for correctness
  HIPCHECK_RETURN_ONFAIL(hipMemcpy(ipc_out_hptr, ipc_out_dptr, (NUM * sizeof(int)), hipMemcpyDeviceToHost));
  for (size_t idx = offset; idx < NUM; ++idx) {
    if (ipc_out_hptr[idx-offset] != ipc_dptr[idx]) {
      std::cout<<"Failing @ idx: "<<idx<<std::endl;
    }
  }

  //Close All Mem Handle
  HIPCHECK_RETURN_ONFAIL(hipIpcCloseMemHandle(ipc_out_dptr));
  HIPCHECK_RETURN_ONFAIL(hipFree(ipc_dptr));

  delete[] ipc_hptr;
  delete[] ipc_out_hptr;

  return hipSuccess;
}

void positive_cases() {
  HIPCHECK(single_process(0));
  HIPCHECK(single_process(32));
  HIPCHECK(single_process(128));
  HIPCHECK(single_process(256));
  HIPCHECK(single_process(512));

  HIPCHECK(single_process(1023));
  HIPCHECK(single_process(47));
  HIPCHECK(single_process(191));
  HIPCHECK(single_process(1022));
}

void negative_cases() {
  HIPCHECK_API(single_process(-1), hipErrorInvalidDevicePointer);
  HIPCHECK_API(single_process(1024), hipErrorInvalidDevicePointer);
}

int main() {
  positive_cases();
  negative_cases();
  passed();
}
