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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// Test the HCC-specific API extensions for HIP:

/* HIT_START
 * BUILD: %t %s
 * RUN: %t EXCLUDE_HIP_PLATFORM all
 * HIT_END
 */

#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "hip/hip_hcc.h"
#include "test_common.h"

#define CHECK(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
    }


int main(int argc, char *argv[])
{
    int deviceId;
    CHECK (hipGetDevice(&deviceId));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, deviceId));
    printf ("info: running on device #%d %s\n", deviceId, props.name);

#ifdef __HCC__
    hc::accelerator acc;
    CHECK(hipHccGetAccelerator(deviceId, &acc));
    std::wcout << "device_path=" << acc.get_device_path() << "\n";

    hc::accelerator_view *av;
    CHECK(hipHccGetAcceleratorView(0/*nullStream*/, &av));
#endif


    passed();

};
