/* Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * RUN: %t
 * HIT_END
 */

#include <iostream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include "hip/hip_runtime.h"
using namespace std;

string getRes() {
    FILE* in;
    char buff[512], buff_2[512];
    string str = "./hipMemoryAllocateCoherent";
    if (!(in = popen(str.c_str(), "r"))) {
        exit(1);
    }
    fgets(buff, sizeof(buff), in);
    fgets(buff_2, sizeof(buff_2), in);
    string str_buff = buff;
    str_buff += buff_2;
    pclose(in);
    return str_buff;
}

int main() {
    setenv("HIP_COHERENT_HOST_ALLOC", "1000,0,1", 1);
    string output = getRes();
    istringstream buffer(output);
    double res1, res2;
    buffer >> res1;
    buffer >> res2;
    if ((res2 - res1 * 2) > 0.000001) exit(1);
    std::cout << "PASSED" << std::endl;
    return 0;
}
