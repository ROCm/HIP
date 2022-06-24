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
#include <iostream>

// Expects 1 command line arg, which is the Device Visible String
int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Invalid number of args passed.\n"
              << "argc : " << argc << std::endl;
    for (int i = 0; i < argc; i++) {
      std::cerr << "  argv[" << i << "] : " << argv[0] << std::endl;
    }
    std::cerr << "The program expects device visibility string i.e. 0,1,2" << std::endl;
    return -1;
  }

  // disable visible_devices env from shell
#ifdef __HIP_PLATFORM_NVCC__
  unsetenv("CUDA_VISIBLE_DEVICES");
  setenv("CUDA_VISIBLE_DEVICES", argv[1], 1);
  HIP_CHECK(hipInit(0));
#else
  unsetenv("ROCR_VISIBLE_DEVICES");
  unsetenv("HIP_VISIBLE_DEVICES");
  setenv("ROCR_VISIBLE_DEVICES", argv[1], 1);
  setenv("HIP_VISIBLE_DEVICES", argv[1], 1);
#endif

  int count = 0;
  auto res = hipGetDeviceCount(&count);
  if (hipSuccess != res) {
    std::cerr << "HIP API returned : " << hipGetErrorString(res) << std::endl;
    return -1;
  }

#ifdef __HIP_PLATFORM_NVCC__
  unsetenv("CUDA_VISIBLE_DEVICES");
#else
  unsetenv("ROCR_VISIBLE_DEVICES");
  unsetenv("HIP_VISIBLE_DEVICES");
#endif
  return count;
}