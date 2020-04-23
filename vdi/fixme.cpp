/* Copyright (c) 2015-present Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "vdi_common.hpp"
#include <icd/loader/icd_dispatch.h>

cl_icd_dispatch amd::ICDDispatchedObject::icdVendorDispatch_[] = {0};
amd::PlatformIDS amd::PlatformID::Platform = {amd::ICDDispatchedObject::icdVendorDispatch_};

RUNTIME_ENTRY(cl_int, clGetDeviceIDs,
              (cl_platform_id platform, cl_device_type device_type, cl_uint num_entries,
               cl_device_id* devices, cl_uint* num_devices)) {
  return CL_SUCCESS;
}
RUNTIME_EXIT
