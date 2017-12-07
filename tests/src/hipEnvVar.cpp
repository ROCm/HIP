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
#include <getopt.h>
#include <iostream>
#include <string>
#include "hip/hip_runtime.h"

using namespace std;

void usage() {
    printf(
        "hipEnvVar [otpions]\n\
    -c,\t\ttotal number of available GPUs and their pciBusID\n\
    -d,\t\tselect one GPU and return its pciBusID\n\
    -v,\t\tsend the list to HIP_VISIBLE_DEVICES env var\n\
    -h,\t\tshow this help message\n\
    ");
}
int main(int argc, char** argv) {
    // string str = getenv("HIP_VISIBLE_DEVICES");
    // std::cout << "The current env HIP_VISIBLE_DEVICES is"<<str << std::endl;
    extern char* optarg;
    extern int optind;
    int c = 0;
    int retDevCnt = 0, retDevInfo = 0, setEnvVar = 0;
    int device = 0;
    string env;
    while ((c = getopt(argc, argv, "cd:v:h")) != -1) switch (c) {
            case 'c':
                retDevCnt = true;
                break;
            case 'd':
                retDevInfo = true;
                device = atoi(optarg);
                break;
            case 'v':
                setEnvVar = true;
                env = optarg;
                break;
            case 'h':
                usage();
                return 0;
            default:
                // usage();
                return -1;
        }

    if (setEnvVar) {
        // env = "export HIP_VISIBLE_DEVICES=" + env;
        // cout<<"The received env var is: "<<env<<endl;
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

    // printf("\nTotal number of GPU devices in the system is %d\n",devCount);

    if (devCount == 0) {
        printf("No HIP enabled device\n");
        return -1;
    }
    if (device < 0 || device > devCount - 1) {
        printf("Selected device %d is out of bound. Devices on your system are in range %d - %d\n",
               device, 0, devCount - 1);
        return -1;
    }

    if (retDevCnt) {
        // std::cout << "Total number of devices visible in system is "<< devCount  << std::endl;
        std::cout << devCount << std::endl;
    }
    if (retDevInfo) {
        hipDevice_t deviceT;
        hipDeviceGet(&deviceT, device);

        char pciBusId[100];
        memset(pciBusId, 0, 100);
        hipDeviceGetPCIBusId(pciBusId, 100, deviceT);

        cout << pciBusId << endl;
    }
    exit(0);
}
