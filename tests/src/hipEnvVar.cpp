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


#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <iostream>
#include <string>
#include <hip_runtime.h>

int debug = 0;

void usage() {
    printf("hipEnvVar [otpions]\n\
    -a,\t\ttotal number ofavailable GPUs and their pciBusID\n\
    -s,\t\tselect one GPU and return its pciBusID\n\
    -h,\t\tshow this help message\n\
    ");
}
int main(int argc, char **argv)
{
    extern char *optarg;
    extern int optind;
    int c, err = 0;
    int retDevCnt=0, retDevInfo=0;
    int device=0;
    //std::cout << "reach here!!" << std::endl;
    while ((c = getopt(argc, argv, "cd:h")) != -1)
        switch (c) {
        case 'c':
            retDevCnt = true;
            break;
        case 'd':
            retDevInfo = true;
            device = atoi(optarg);
            break;
        case 'h':
            usage();
            return 0;
            break;
        default :
            //usage();
            return -1;
            break;
        case '?':
            err = 1;
            break;
        }
    // device init
    int devCount=0;
    hipGetDeviceCount(&devCount);

    //printf("\nTotal number of GPU devices in the system is %d\n",devCount);

    if (devCount == 0) {
        printf("No HIP enabled device\n");
        return -1;
    }
    if (device < 0 || device > devCount -1) {
        printf("Selected device %d is out of bound. Devices on your system are in range %d - %d\n",
               device, 0, devCount -1);
        return -1;
    }
    if (retDevCnt) {
        std::cout << "Total number of devices visible in system is "<< devCount  << std::endl;
    }
    if (retDevInfo) {
        hipSetDevice(device);
        hipDeviceProp_t devProp;

        hipDeviceGetProperties(&devProp, device);
        if (devProp.major < 1) {
            printf("Device %d does not support HIP\n", device);
            return -1;
        }

        std::cout << "The selected device pciBusID is " << devProp.pciBusID << std::endl;
    }

    exit(0);
}

