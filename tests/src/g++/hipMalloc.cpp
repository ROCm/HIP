/*
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * */

/* HIT_START
 * BUILD_CMD: hipMalloc %cxx -D__HIP_PLATFORM_AMD__ -I%hip-path/include -I%rocm-path/include %S/%s -Wl,--rpath=%roc-path/lib %hip-path/lib/libamdhip64.so -o %T/%t -std=c++11 EXCLUDE_HIP_PLATFORM nvidia EXCLUDE_HIP_LIB_TYPE static
 * TEST: %t EXCLUDE_HIP_PLATFORM nvidia EXCLUDE_HIP_LIB_TYPE static
 * HIT_END
 */


#include <hip/hip_runtime_api.h>
#include <iostream>

int main() {
    int* Ad;
    hipMalloc((void**)&Ad, 1024);
    std::cout<<"PASSED!"<<std::endl;   
}
