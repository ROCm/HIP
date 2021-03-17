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
#include "MultiProcess.h"

#define NUM_ELEMS 1024
#define OFFSET 128

void multi_process(int num_process, bool debug_process) {

#ifdef __unix__

  int* ipc_dptr = nullptr;
  int* ipc_hptr = nullptr;
  int* ipc_out_dptr = nullptr;
  int* ipc_out_hptr = nullptr;

  MultiProcess<hipIpcMemHandle_t>* mProcess = new MultiProcess<hipIpcMemHandle_t>(num_process);
  mProcess->CreateShmem();
  pid_t pid = mProcess->SpawnProcess(debug_process);

  // Parent Process
  if (pid != 0) {
    hipIpcMemHandle_t ipc_handle;
    memset(&ipc_handle, 0x00, sizeof(hipIpcMemHandle_t));

    HIPCHECK(hipMalloc((void**)&ipc_dptr, NUM_ELEMS * sizeof(int)));
    HIPCHECK(hipIpcGetMemHandle(&ipc_handle, ipc_dptr));

    ipc_hptr = new int[NUM_ELEMS];
    for (size_t idx = 0; idx < NUM_ELEMS; ++idx) {
      ipc_hptr[idx] = idx;
    }

    HIPCHECK(hipMemset(ipc_dptr, 0x00, (NUM_ELEMS * sizeof(int))));
    HIPCHECK(hipMemcpy(ipc_dptr, ipc_hptr, (NUM_ELEMS * sizeof(int)), hipMemcpyHostToDevice));

    mProcess->WriteHandleToShmem(ipc_handle);

    mProcess->WaitTillAllChildReads();

  } else {
    ipc_out_hptr = new int[NUM_ELEMS];
    memset(ipc_out_hptr, 0x00, (NUM_ELEMS * sizeof(int)));

    hipIpcMemHandle_t ipc_handle;
    mProcess->ReadHandleFromShmem(ipc_handle);
    HIPCHECK(hipIpcOpenMemHandle((void**)&ipc_out_dptr, ipc_handle, 0));

    HIPCHECK(hipMemcpy(ipc_out_hptr, ipc_out_dptr, (NUM_ELEMS * sizeof(int)),
                       hipMemcpyDeviceToHost));
    for (size_t idx = 0; idx < NUM_ELEMS; ++idx) {
      if (ipc_out_hptr[idx] != idx) {
        std::cout<<"Failing @ idx: "<< idx << std::endl;
      }
    }
    mProcess->NotifyParentDone();
    HIPCHECK(hipIpcCloseMemHandle(ipc_out_dptr));
    delete[] ipc_out_hptr;
  }

  if (pid != 0) {
    delete mProcess;
  }

#endif /* __unix__ */

}


int main(int argc, char* argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
  multi_process((N < 64) ? N : 64, debug_test);
  passed();
}
