/*
Copyright (c) 2020-Present Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11 EXCLUDE_HIP_RUNTIME rocclr
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"

void NegativeTests(){

    // Null pointers
    {
        hipEvent_t start,end;
        float tms = 1.0f;
        HIPASSERT(hipEventElapsedTime(nullptr,start,end) == hipErrorInvalidValue);
#ifndef __HIP_PLATFORM_NVIDIA__
        // On NVCC platform API throws seg fault hence skipping
        HIPASSERT(hipEventElapsedTime(&tms,nullptr,end) == hipErrorInvalidHandle);
        HIPASSERT(hipEventElapsedTime(&tms,start,nullptr) == hipErrorInvalidHandle);
#endif
    }

    // Event created using disabled timing
    {
        float timeElapsed = 1.0f;
        hipEvent_t start, stop;
        HIPCHECK(hipEventCreateWithFlags(&start,hipEventDisableTiming));
        HIPCHECK(hipEventCreateWithFlags(&stop,hipEventDisableTiming));
        HIPASSERT(hipEventElapsedTime(&timeElapsed, start, stop) == hipErrorInvalidHandle);
    }

    // events created different devices
    {
        int devCount = 0;
        HIPCHECK(hipGetDeviceCount(&devCount));
        if (devCount > 1){
            // create event on dev=0
            HIPCHECK(hipSetDevice(0));
            hipEvent_t start;
            HIPCHECK(hipEventCreate(&start));

            HIPCHECK(hipEventRecord(start, nullptr));
            HIPCHECK(hipEventSynchronize(start));

            // create event on dev=1
            HIPCHECK(hipSetDevice(1));
            hipEvent_t stop;
            HIPCHECK(hipEventCreate(&stop));

            HIPCHECK(hipEventRecord(stop, nullptr));
            HIPCHECK(hipEventSynchronize(stop));

            float tElapsed = 1.0f;
            HIPASSERT(hipEventElapsedTime(&tElapsed,start,stop) == hipErrorInvalidHandle);
        }
    }
}

void PositiveTest(){
    hipEvent_t start;
    HIPCHECK(hipEventCreate(&start));

    hipEvent_t stop;
    HIPCHECK(hipEventCreate(&stop));

    HIPCHECK(hipEventRecord(start, nullptr));
    HIPCHECK(hipEventSynchronize(start));

    HIPCHECK(hipEventRecord(stop, nullptr));
    HIPCHECK(hipEventSynchronize(stop));

    float tElapsed = 1.0f;
    HIPCHECK(hipEventElapsedTime(&tElapsed,start,stop));
}

int main(){

    NegativeTests();
    PositiveTest();
    passed();
}
