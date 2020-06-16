/*
Copyright (c) 2015-2020 Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t 1.2 2.3
 * HIT_END
 */

#include <cstdlib>
#include <hip/hip_runtime.h>
#include "test_common.h"

int main(int argc, char* argv[]) {
    // Testing that the compiler supports host _Float16 conversions
    float init_value = atof(argv[1]);
    _Float16 value_float16 = static_cast<_Float16>(init_value);
    float result_value = static_cast<float>(value_float16);

    if(std::abs(result_value - init_value) >= 0.01){
        printf("init: %f\n", init_value);
        printf("result: %f\n", result_value);
        printf("diff: %f\n", std::abs(result_value - init_value));
        failed("Failed host _Float16 test.");
    }

    // Testing that the compiler supports host __fp16 conversions
    init_value = atof(argv[2]);
    __fp16 value_fp16 = static_cast<__fp16>(init_value);
    result_value = static_cast<float>(value_fp16);

    if(std::abs(result_value - init_value) >= 0.01){
        printf("init: %f\n", init_value);
        printf("result: %f\n", result_value);
        printf("diff: %f\n", std::abs(result_value - init_value));
        failed("Failed host __fp16 test.");
    }

    passed();
}
