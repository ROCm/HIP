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
#include <stdlib.h>

bool UNSETENV(std::string var) {
  int result = -1;
  #ifdef __unix__
    result = unsetenv(var.c_str());
  #else
    result = _putenv((var + '=').c_str());
  #endif
  return (result == 0) ? true: false;
}

bool SETENV(std::string var, std::string value, int overwrite) {
  int result = -1;
  #ifdef __unix__
    result = setenv(var.c_str(), value.c_str(), overwrite);
  #else
    result = _putenv((var + '=' + value).c_str());
  #endif
  return (result == 0) ? true: false;
}

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
  UNSETENV("CUDA_VISIBLE_DEVICES");
  SETENV("CUDA_VISIBLE_DEVICES", argv[1], 1);
  auto init_res = hipInit(0);
  if (hipSuccess != init_res) {
    std::cerr << "CUDA INIT API returned : " << hipGetErrorString(init_res) << std::endl;
    return -1;
  }
#else
  UNSETENV("ROCR_VISIBLE_DEVICES");
  UNSETENV("HIP_VISIBLE_DEVICES");
  SETENV("ROCR_VISIBLE_DEVICES", argv[1], 1);
  SETENV("HIP_VISIBLE_DEVICES", argv[1], 1);
#endif

  int count = 0;
  auto res = hipGetDeviceCount(&count);
  if (hipSuccess != res) {
    std::cerr << "HIP API returned : " << hipGetErrorString(res) << std::endl;
    return -1;
  }

#ifdef __HIP_PLATFORM_NVCC__
  UNSETENV("CUDA_VISIBLE_DEVICES");
#else
  UNSETENV("ROCR_VISIBLE_DEVICES");
  UNSETENV("HIP_VISIBLE_DEVICES");
#endif
  return count;
}