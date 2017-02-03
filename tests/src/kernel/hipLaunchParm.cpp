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
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include"test_common.h"
#include "hip/hip_runtime_api.h"
#include<iostream>

__global__ void vAdd(hipLaunchParm lp, float *a){}


//---
//Some wrapper macro for testing:
#define WRAP(...) __VA_ARGS__

#include <sys/time.h>
#define GPU_PRINT_TIME(cmd, elapsed, quiet) do {\
        struct timeval start, stop;\
        float elapsed;\
        gettimeofday(&start, NULL);\
        hipDeviceSynchronize();\
        cmd;\
        hipDeviceSynchronize();\
        gettimeofday(&stop, NULL);\
    } while(0);



#define MY_LAUNCH(command, doTrace, msg) \
{\
    if (doTrace) printf ("TRACE: %s %s\n", msg, #command); \
    command;\
}


#define MY_LAUNCH_WITH_PAREN(command, doTrace, msg) \
{\
    if (doTrace) printf ("TRACE: %s %s\n", msg, #command); \
    (command);\
}



int main()
{
    float *Ad;
    hipMalloc((void**)&Ad, 1024);

    // Test the different hipLaunchParm options:
    hipLaunchKernel(vAdd, size_t(1024), 1, 0, 0, Ad);
    hipLaunchKernel(vAdd, 1024, dim3(1), 0, 0, Ad);
    hipLaunchKernel(vAdd, dim3(1024), 1, 0, 0, Ad);
    hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad);

    // Test case with hipLaunchKernel inside another macro:
    float e0;
    GPU_PRINT_TIME (hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad), e0, j);
    GPU_PRINT_TIME (WRAP(hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad)), e0, j);

#ifdef EXTRA_PARENS_1
    // Don't wrap hipLaunchKernel in extra set of parens:
    GPU_PRINT_TIME ((hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad)), e0, j);
#endif

    MY_LAUNCH (hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad), true, "firstCall");

    float *A;
    float e1;
    MY_LAUNCH_WITH_PAREN (hipMalloc(&A, 100), true, "launch2");

#ifdef EXTRA_PARENS_2
    //MY_LAUNCH_WITH_PAREN wraps cmd in () which can cause issues.
    MY_LAUNCH_WITH_PAREN (hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad), true, "firstCall");
#endif

    passed();
}
