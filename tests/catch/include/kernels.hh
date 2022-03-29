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

#pragma once

#include <hip_test_common.hh>
#include <map>

#ifndef RTC_ENABLED

__global__ void Set(int* Ad, int val);

/* Kernel Templates */
#include "vectorADD.inl"

#else

/*
 * Wrapper Macros that create a string representation of the kernel name.
 * In the case of kernel templates, a variadic template is used to ensure compatibility with
 * the launchKernel template when RTC is not enabled. If the kernel is inside a namespace, use the
 * "_NS" version of the Macro.
 */
#define FUNCTION_WRAPPER(param)                                                                    \
  std::string param() { return #param; }
#define TEMPLATE_WRAPPER(param)                                                                    \
  template <typename...> std::string param() { return #param; }
#define FUNCTION_WRAPPER_NS(param, namespace)                                                      \
  std::string param() { return #namespace "::" #param; }
#define TEMPLATE_WRAPPER_NS(param, namespace)                                                      \
  template <typename...> std::string param() { return #namespace "::" #param; }

FUNCTION_WRAPPER(Set);

namespace HipTest {
TEMPLATE_WRAPPER_NS(vectorADD, HipTest);
}

#endif
