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

/* HIT_START
 * BUILD: %t %s test_common.cpp
 * HIT_END
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "clara/clara.hpp"
#include <string>
#include "hip/hip_runtime.h"
#include "test_common.h"

using namespace std;
using namespace clara;
inline clara::Parser cmdline_parser(bool& help, std::string& env, int &device, bool &retDevCnt) {
    return clara::Opt{retDevCnt}
        ["-c"]
        ("total number of GPUs available") |

        clara::Help{help} |

        clara::Opt{device,"device"}
        ["-d"]["--device"]
        ("select one GPU and return its pciBusID")  |

        clara::Opt{env,"Set Env Value"}
        ["-v"]["--EnvValue"]
        ("send the list to HIP_VISIBLE_DEVICES env var, syntax -v=<value>");
}

int main(int argc, char** argv) {
    bool help = false;
    bool retDevCnt = false;
    int c = 0;
    int device = INT_MAX;
    string env;

    auto cmd = cmdline_parser(help, env, device, retDevCnt);
    const auto r = cmd.parse(Args{argc, argv});
    if (!r) { std::cout<<"Valid device must be >= 0"<<std::endl; return -1;}

    if (help)
        cout << cmd << endl;

    if (!env.empty()) {
        setenv("HIP_VISIBLE_DEVICES", env.c_str(), 1);
        setenv("CUDA_VISIBLE_DEVICES", env.c_str(), 1);
        cout << "set env HIP_VISIBLE_DEVICES = " << env.c_str() << endl;

        // verify if the environment variable is set
        char* pPath;
        pPath = getenv("HIP_VISIBLE_DEVICES");
        if (pPath != NULL)
            printf("HIP_VISIBLE_DEVICES is %s\n", pPath);
        else
            printf("HIP_VISIBLE_DEVICES is not set\n");
    }

    // device init
    int devCount = 0;
    hipGetDeviceCount(&devCount);

    if (devCount == 0) {
        printf("No HIP enabled device\n");
        return -1;
    }
    if (device != INT_MAX && (device < 0 || device > devCount - 1)) {
        printf("Selected device %d is out of bound. Devices on your system are in range %d - %d\n",
               device, 0, devCount - 1);
        return -1;
    }

    if (retDevCnt) {
        std::cout << devCount << std::endl;
    }
    if (device != INT_MAX) {
        hipDevice_t deviceT;
        hipDeviceGet(&deviceT, device);

        char pciBusId[100];
        memset(pciBusId, 0, 100);
        hipDeviceGetPCIBusId(pciBusId, 100, deviceT);

        cout << pciBusId << endl;
    }
    exit(0);
}
