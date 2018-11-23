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

#include "hip/hip_runtime.h"
#include "hip/hcc_detail/hip_prof_api.h"

// HIP API callback/activity

api_callbacks_table_t callbacks_table;

extern std::string& FunctionSymbol(hipFunction_t f);
const char* hipKernelNameRef(const hipFunction_t f) { return FunctionSymbol(f).c_str(); }

hipError_t hipRegisterApiCallback(uint32_t id, void* fun, void* arg) {
  return callbacks_table.set_callback(id, reinterpret_cast<api_callbacks_table_t::fun_t>(fun), arg) ?
    hipSuccess : hipErrorInvalidValue;
}

hipError_t hipRemoveApiCallback(uint32_t id) {
  return callbacks_table.set_callback(id, NULL, NULL) ? hipSuccess : hipErrorInvalidValue;
}

hipError_t hipRegisterActivityCallback(uint32_t id, void* fun, void* arg) {
  return callbacks_table.set_activity(id, reinterpret_cast<api_callbacks_table_t::act_t>(fun), arg) ?
    hipSuccess : hipErrorInvalidValue;
}

hipError_t hipRemoveActivityCallback(uint32_t id) {
  return callbacks_table.set_activity(id, NULL, NULL) ? hipSuccess : hipErrorInvalidValue;
}
