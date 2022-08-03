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

#include <hip/hip_runtime.h>
#include "hip/hip_runtime_api.h"
#include <hip_test_context.hh>


void inline hideDevices(const char* devices) {
#if HT_NVIDIA
  setenv("CUDA_VISIBLE_DEVICES", devices, 1);
#else
  setenv("HIP_VISIBLE_DEVICES", devices, 1);
  setenv("ROCR_VISIBLE_DEVICES", devices, 1);
#endif
}

void inline unhideAllDevices() {
#if HT_NVIDIA
  unsetenv("CUDA_VISIBLE_DEVICES");
#else
  unsetenv("HIP_VISIBLE_DEVICES");
  unsetenv("ROCR_VISIBLE_DEVICES");
#endif
}

/**
 * @brief Runs hipDeviceGetP2PAttribute with srcDevice = 0 and dstDevice = 1
 *        Expects 1 command line arg, which is the Device Visible String
 *
 * @return the error code returned by hipDeviceGetP2PAttribute
 */
int main(int argc, char** argv) {
  int value;
  const int srcDevice = 0;
  const int dstDevice = 1;
  const hipDeviceP2PAttr validAttr = hipDevP2PAttrAccessSupported;

  if (argc == 2) {
    hideDevices(argv[1]);
  }

  hipError_t error = hipDeviceGetP2PAttribute(&value, validAttr, srcDevice, dstDevice);

  if (argc == 2) {
    unhideAllDevices();
  }

  return error;
}