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

/*
AtomicAdd with unsafe flag should trigger HW atomics
*/

#ifdef __AMDGCN_UNSAFE_FP_ATOMICS__
#error "Compiler change defining __AMDGCN_UNSAFE_FP_ATOMICS__ landed, remove this define"
#else
#define __AMDGCN_UNSAFE_FP_ATOMICS__
#endif

#include "AtomicAddTester.hh"

TEMPLATE_TEST_CASE("Unit_AtomicAdd_WithUnsafeFlag", "",
                   float, double) {
  // test FLAT unsafe add

  // testtype, coherent, unsafe, flag, global
  run<TestType, true, atomicSafety::none, true, false>(__FILE__);
}
