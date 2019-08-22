/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include "platform/activity.hpp"

extern "C" void InitActivityCallback(void* id_callback, void* op_callback, void* arg) {
  activity_prof::CallbacksTable::init(reinterpret_cast<activity_prof::id_callback_fun_t>(id_callback),
                                      reinterpret_cast<activity_prof::callback_fun_t>(op_callback),
                                      arg);
}

extern "C" bool EnableActivityCallback(unsigned op, bool enable) {
  return activity_prof::CallbacksTable::SetEnabled(op, enable);
}

extern "C" const char* GetCmdName(unsigned op) {
  return getOclCommandKindString(static_cast<uint32_t>(op));
}
