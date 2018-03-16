/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * Conformance test for checking functionality of
 * hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device);
 */

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "test_common.h"

int main() {
    int numDevices = 0;
    int major, minor;
    hipDevice_t device;
    HIPCHECK(hipGetDeviceCount(&numDevices));
    for (int i = 0; i < numDevices; i++) {
        HIPCHECK(hipDeviceGet(&device, i));
        HIPCHECK(hipDeviceComputeCapability(&major, &minor, device));
        HIPASSERT(major >= 0);
        HIPASSERT(minor >= 0);
    }
    passed();
}
