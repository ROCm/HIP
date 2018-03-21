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

#include <stdio.h>
#include "hip/hip_runtime.h"
#include "test_common.h"

int main(void) {
    char pciBusId[13];
    int deviceCount = 0;
    HIPCHECK(hipGetDeviceCount(&deviceCount));
    HIPASSERT(deviceCount != 0);
    for (int i = 0; i < deviceCount; i++) {
        int pciBusID = -1;
        int pciDeviceID = -1;
        int pciDomainID = -1;
        int tempPciBusId = -1;
        int tempDeviceId = -1;
        HIPCHECK(hipDeviceGetPCIBusId(&pciBusId[0], 13, i));
        sscanf(pciBusId, "%04x:%02x:%02x", &pciDomainID, &pciBusID, &pciDeviceID);
        HIPCHECK(hipDeviceGetAttribute(&tempPciBusId, hipDeviceAttributePciBusId, i));
        if (pciBusID != tempPciBusId) {
            exit(EXIT_FAILURE);
        }
        HIPCHECK(hipDeviceGetByPCIBusId(&tempDeviceId, pciBusId));
        if (tempDeviceId != i) {
            exit(EXIT_FAILURE);
        }
    }
    passed();
}
