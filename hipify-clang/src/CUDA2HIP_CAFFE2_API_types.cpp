/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "CUDA2HIP.h"

// Map of all types
const std::map<llvm::StringRef, hipCounter> CUDA_CAFFE2_TYPE_NAME_MAP{

  // 5. Defines
  {"REGISTER_CUDA_OPERATOR",                                  {"REGISTER_HIP_OPERATOR",                                        "", CONV_DEFINE, API_CAFFE2}},
  {"REGISTER_CUDA_OPERATOR_CREATOR",                          {"REGISTER_HIP_OPERATOR_CREATOR",                                "", CONV_DEFINE, API_CAFFE2}},

  // 6. Classes
  {"CUDAContext",                                             {"HIPContext",                                                   "", CONV_TYPE, API_CAFFE2}},
};
