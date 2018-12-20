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
const std::map<llvm::StringRef, hipCounter> CUDA_FFT_TYPE_NAME_MAP{

  // cuFFT defines
  {"CUFFT_FORWARD",                                       {"HIPFFT_FORWARD",                                       "", CONV_NUMERIC_LITERAL, API_DNN}},    // -1
  {"CUFFT_INVERSE",                                       {"HIPFFT_BACKWARD",                                      "", CONV_NUMERIC_LITERAL, API_DNN}},    //  1
  {"CUFFT_COMPATIBILITY_DEFAULT",                         {"HIPFFT_COMPATIBILITY_DEFAULT",                         "", CONV_NUMERIC_LITERAL, API_DNN, HIP_UNSUPPORTED}},    //  CUFFT_COMPATIBILITY_FFTW_PADDING

  // cuFFT enums
  {"cufftResult_t",                                       {"hipfftResult_t",                                       "", CONV_TYPE, API_FFT}},
  {"cufftResult",                                         {"hipfftResult",                                         "", CONV_TYPE, API_FFT}},
  {"CUFFT_SUCCESS",                                       {"HIPFFT_SUCCESS",                                       "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x0  0
  {"CUFFT_INVALID_PLAN",                                  {"HIPFFT_INVALID_PLAN",                                  "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x1  1
  {"CUFFT_ALLOC_FAILED",                                  {"HIPFFT_ALLOC_FAILED",                                  "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x2  2
  {"CUFFT_INVALID_TYPE",                                  {"HIPFFT_INVALID_TYPE",                                  "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x3  3
  {"CUFFT_INVALID_VALUE",                                 {"HIPFFT_INVALID_VALUE",                                 "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x4  4
  {"CUFFT_INTERNAL_ERROR",                                {"HIPFFT_INTERNAL_ERROR",                                "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x5  5
  {"CUFFT_EXEC_FAILED",                                   {"HIPFFT_EXEC_FAILED",                                   "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x6  6
  {"CUFFT_SETUP_FAILED",                                  {"HIPFFT_SETUP_FAILED",                                  "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x7  7
  {"CUFFT_INVALID_SIZE",                                  {"HIPFFT_INVALID_SIZE",                                  "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x8  8
  {"CUFFT_UNALIGNED_DATA",                                {"HIPFFT_UNALIGNED_DATA",                                "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x9  9
  {"CUFFT_INCOMPLETE_PARAMETER_LIST",                     {"HIPFFT_INCOMPLETE_PARAMETER_LIST",                     "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0xA  10
  {"CUFFT_INVALID_DEVICE",                                {"HIPFFT_INVALID_DEVICE",                                "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0xB  11
  {"CUFFT_PARSE_ERROR",                                   {"HIPFFT_PARSE_ERROR",                                   "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0xC  12
  {"CUFFT_NO_WORKSPACE",                                  {"HIPFFT_NO_WORKSPACE",                                  "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0xD  13
  {"CUFFT_NOT_IMPLEMENTED",                               {"HIPFFT_NOT_IMPLEMENTED",                               "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0xE  14
  {"CUFFT_LICENSE_ERROR",                                 {"HIPFFT_LICENSE_ERROR",                                 "", CONV_NUMERIC_LITERAL, API_FFT, HIP_UNSUPPORTED}},
  {"CUFFT_NOT_SUPPORTED",                                 {"HIPFFT_NOT_SUPPORTED",                                 "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x10 16
  {"cufftType_t",                                         {"hipfftType_t",                                         "", CONV_TYPE, API_FFT}},
  {"cufftType",                                           {"hipfftType",                                           "", CONV_TYPE, API_FFT}},
  {"CUFFT_R2C",                                           {"HIPFFT_R2C",                                           "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x2a
  {"CUFFT_C2R",                                           {"HIPFFT_C2R",                                           "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x2c
  {"CUFFT_C2C",                                           {"HIPFFT_C2C",                                           "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x29
  {"CUFFT_D2Z",                                           {"HIPFFT_D2Z",                                           "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x6a
  {"CUFFT_Z2D",                                           {"HIPFFT_Z2D",                                           "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x6c
  {"CUFFT_Z2Z",                                           {"HIPFFT_Z2Z",                                           "", CONV_NUMERIC_LITERAL, API_FFT}},    //  0x69
  {"cufftCompatibility_t",                                {"hipfftCompatibility_t",                                "", CONV_TYPE, API_FFT, HIP_UNSUPPORTED}},
  {"cufftCompatibility",                                  {"hipfftCompatibility",                                  "", CONV_TYPE, API_FFT, HIP_UNSUPPORTED}},
  {"CUFFT_COMPATIBILITY_FFTW_PADDING",                    {"HIPFFT_COMPATIBILITY_FFTW_PADDING",                    "", CONV_NUMERIC_LITERAL, API_FFT, HIP_UNSUPPORTED}},    //  0x01

  // cuFFT types
  {"cufftReal",                                           {"hipfftReal",                                           "", CONV_TYPE, API_FFT}},
  {"cufftDoubleReal",                                     {"hipfftDoubleReal",                                     "", CONV_TYPE, API_FFT}},
  {"cufftComplex",                                        {"hipfftComplex",                                        "", CONV_TYPE, API_FFT}},
  {"cufftDoubleComplex",                                  {"hipfftDoubleComplex",                                  "", CONV_TYPE, API_FFT}},
  {"cufftHandle",                                         {"hipfftHandle",                                         "", CONV_TYPE, API_FFT}},
};
