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

// Maps CUDA header names to HIP header names
const std::map<llvm::StringRef, hipCounter> CUDA_DEVICE_FUNC_MAP{
  {"umin",        {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"llmin",       {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"ullmin",      {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"umax",        {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"llmax",       {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"ullmax",      {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"__isinff",    {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"__isnanf",    {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"__finite",    {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"__finitef",   {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"__signbit",   {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"__isnan",     {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"__isinf",     {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"__signbitf",  {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"__signbitl",  {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"__finitel",   {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"__isinfl",    {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"__isnanl",    {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"_ldsign",     {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"_fdsign",     {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
  {"_Pow_int",    {"", "", CONV_DEVICE_FUNC, API_RUNTIME, UNSUPPORTED}},
};
