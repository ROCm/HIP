/*
Copyright (c) 2015-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_common.h"

class AtomicsTest {
public:
  AtomicsTest() {}
  ~AtomicsTest() {}

  template <typename T>
  bool get_system_atomics_ptr(T** host_pptr, T** device_pptr, size_t num_elems) {
    HIPCHECK(hipHostMalloc(host_pptr, num_elems * sizeof(T),
                           (hipHostMallocCoherent | hipHostMallocMapped)));

    if (*host_pptr == nullptr) {
      std::cout<<"Cannot get host ptr from hipHostMaloc"
               " with coherent & malloc mapped flags "<<std::endl;
      return false;
    }

    HIPCHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(device_pptr), *host_pptr, 0));
    if (*device_pptr == nullptr) {
      std::cout<<"Cannot get device_ptr for host_ptr: "<<host_pptr<<std::endl;
      return false;
    }

    HIPCHECK(hipMemset(*device_pptr, 0x00, num_elems * sizeof(T)));
    return true;
  }

  template <typename T>
  bool free_system_atomics_ptr(T* host_ptr) {
    HIPCHECK(hipFree(host_ptr));
    return true;
  }
};
