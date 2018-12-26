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

// Map of all functions
const std::map<llvm::StringRef, hipCounter> CUDA_FFT_FUNCTION_MAP{
  {"cufftPlan1d",                                         {"hipfftPlan1d",                                         "", CONV_LIB_FUNC, API_FFT}},
  {"cufftPlan2d",                                         {"hipfftPlan2d",                                         "", CONV_LIB_FUNC, API_FFT}},
  {"cufftPlan3d",                                         {"hipfftPlan3d",                                         "", CONV_LIB_FUNC, API_FFT}},
  {"cufftPlanMany",                                       {"hipfftPlanMany",                                       "", CONV_LIB_FUNC, API_FFT}},
  {"cufftMakePlan1d",                                     {"hipfftMakePlan1d",                                     "", CONV_LIB_FUNC, API_FFT}},
  {"cufftMakePlan2d",                                     {"hipfftMakePlan2d",                                     "", CONV_LIB_FUNC, API_FFT}},
  {"cufftMakePlan3d",                                     {"hipfftMakePlan3d",                                     "", CONV_LIB_FUNC, API_FFT}},
  {"cufftMakePlanMany",                                   {"hipfftMakePlanMany",                                   "", CONV_LIB_FUNC, API_FFT}},
  {"cufftMakePlanMany64",                                 {"hipfftMakePlanMany64",                                 "", CONV_LIB_FUNC, API_FFT}},
  {"cufftGetSizeMany64",                                  {"hipfftGetSizeMany64",                                  "", CONV_LIB_FUNC, API_FFT}},
  {"cufftEstimate1d",                                     {"hipfftEstimate1d",                                     "", CONV_LIB_FUNC, API_FFT}},
  {"cufftEstimate2d",                                     {"hipfftEstimate2d",                                     "", CONV_LIB_FUNC, API_FFT}},
  {"cufftEstimate3d",                                     {"hipfftEstimate3d",                                     "", CONV_LIB_FUNC, API_FFT}},
  {"cufftEstimateMany",                                   {"hipfftEstimateMany",                                   "", CONV_LIB_FUNC, API_FFT}},
  {"cufftCreate",                                         {"hipfftCreate",                                         "", CONV_LIB_FUNC, API_FFT}},
  {"cufftGetSize1d",                                      {"hipfftGetSize1d",                                      "", CONV_LIB_FUNC, API_FFT}},
  {"cufftGetSize2d",                                      {"hipfftGetSize2d",                                      "", CONV_LIB_FUNC, API_FFT}},
  {"cufftGetSize3d",                                      {"hipfftGetSize3d",                                      "", CONV_LIB_FUNC, API_FFT}},
  {"cufftGetSizeMany",                                    {"hipfftGetSizeMany",                                    "", CONV_LIB_FUNC, API_FFT}},
  {"cufftGetSize",                                        {"hipfftGetSize",                                        "", CONV_LIB_FUNC, API_FFT}},
  {"cufftSetWorkArea",                                    {"hipfftSetWorkArea",                                    "", CONV_LIB_FUNC, API_FFT}},
  {"cufftSetAutoAllocation",                              {"hipfftSetAutoAllocation",                              "", CONV_LIB_FUNC, API_FFT}},
  {"cufftExecC2C",                                        {"hipfftExecC2C",                                        "", CONV_LIB_FUNC, API_FFT}},
  {"cufftExecR2C",                                        {"hipfftExecR2C",                                        "", CONV_LIB_FUNC, API_FFT}},
  {"cufftExecC2R",                                        {"hipfftExecC2R",                                        "", CONV_LIB_FUNC, API_FFT}},
  {"cufftExecZ2Z",                                        {"hipfftExecZ2Z",                                        "", CONV_LIB_FUNC, API_FFT}},
  {"cufftExecD2Z",                                        {"hipfftExecD2Z",                                        "", CONV_LIB_FUNC, API_FFT}},
  {"cufftExecZ2D",                                        {"hipfftExecZ2D",                                        "", CONV_LIB_FUNC, API_FFT}},
  {"cufftSetStream",                                      {"hipfftSetStream",                                      "", CONV_LIB_FUNC, API_FFT}},
  {"cufftDestroy",                                        {"hipfftDestroy",                                        "", CONV_LIB_FUNC, API_FFT}},
  {"cufftGetVersion",                                     {"hipfftGetVersion",                                     "", CONV_LIB_FUNC, API_FFT}},
  {"cufftGetProperty",                                    {"hipfftGetProperty",                                    "", CONV_LIB_FUNC, API_FFT, HIP_UNSUPPORTED}},
};
