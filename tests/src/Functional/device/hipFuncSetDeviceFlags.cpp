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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "test_common.h"

int main() {
    unsigned flag = 0;
    HIPCHECK(hipDeviceReset());

    int deviceCount = 0;
    HIPCHECK(hipGetDeviceCount(&deviceCount));

    for (int j = 0; j < deviceCount; j++) {
        HIPCHECK(hipSetDevice(j));

        for (int i = 0; i < 4; i++) {
            flag = 1 << i;
            printf("Flag=%x\n", flag);
            HIPCHECK(hipSetDeviceFlags(flag));
            // HIPCHECK_API(hipSetDeviceFlags(flag), hipErrorInvalidValue);
        }

        flag = 0;
    }

    passed();
}
