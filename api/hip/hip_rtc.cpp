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

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <amd_comgr.h>

const char* hiprtcGetErrorString(hiprtcResult x) {
  switch (x) {
    case HIPRTC_SUCCESS:
      return "HIPRTC_SUCCESS";
    case HIPRTC_ERROR_OUT_OF_MEMORY:
      return "HIPRTC_ERROR_OUT_OF_MEMORY";
    case HIPRTC_ERROR_PROGRAM_CREATION_FAILURE:
      return "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case HIPRTC_ERROR_INVALID_INPUT:
      return "HIPRTC_ERROR_INVALID_INPUT";
    case HIPRTC_ERROR_INVALID_PROGRAM:
      return "HIPRTC_ERROR_INVALID_PROGRAM";
    case HIPRTC_ERROR_INVALID_OPTION:
      return "HIPRTC_ERROR_INVALID_OPTION";
    case HIPRTC_ERROR_COMPILATION:
      return "HIPRTC_ERROR_COMPILATION";
    case HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE:
      return "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
      return "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
      return "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
      return "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case HIPRTC_ERROR_INTERNAL_ERROR:
      return "HIPRTC_ERROR_INTERNAL_ERROR";
    default: 
      throw std::logic_error{"Invalid HIPRTC result."};
    };
}

hiprtcResult hiprtcAddNameExpression(hiprtcProgram p, const char* n) {
  return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcCompileProgram(hiprtcProgram p, int n, const char** o)
{
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcCreateProgram(hiprtcProgram* p, const char* src,
                                 const char* name, int n, const char** hdrs,
                                 const char** incs) {
  if (p == nullptr) {
    return HIPRTC_ERROR_INVALID_PROGRAM;
  }
  if (n < 0) {
    return HIPRTC_ERROR_INVALID_INPUT;
  }
  if (n && (hdrs == nullptr || incs == nullptr)) {
    return HIPRTC_ERROR_INVALID_INPUT;
  }

  return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcDestroyProgram(hiprtcProgram* p)
{
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetLoweredName(hiprtcProgram p, const char* n,
                                  const char** ln)
{
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetProgramLog(hiprtcProgram p, char* l)
{
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram p, std::size_t* sz)
{
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetCode(hiprtcProgram p, char* c)
{
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetCodeSize(hiprtcProgram p, std::size_t* sz)
{
    return HIPRTC_SUCCESS;
}