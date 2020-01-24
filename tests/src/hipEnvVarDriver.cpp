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
 * BUILD: %t %s test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t
 * HIT_END
 */

#include <io.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include "hip/hip_runtime.h"
#include <chrono>
#include <thread>
#include "test_common.h"

using namespace std;

<<<<<<< HEAD
const string directed_dir = "." + string(PATH_SEPERATOR_STR) + "directed_tests" + string(PATH_SEPERATOR_STR) + "hipEnvVar";
const string dir = "." + string(PATH_SEPERATOR_STR) + "hipEnvVar";

int getDeviceNumber(bool print_cout=true) {
    char buff[512];
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

	//Don't print error if missing directed_dir file
    int fd = dup(fileno(stderr));
    freopen(NULL_DEVICE, "w", stderr);
    FILE* in = popen((directed_dir + " -c").c_str(), "r");
    if(fgets(buff, 512, in) == NULL){
        dup2(fd, fileno(stderr));
        close(fd);
        pclose(in);
        //Check at same level, and print error if missing both files
=======
const string directed_dir = "directed_tests" + string(PATH_SEPERATOR_STR) + "hipEnvVar";
const string dir = "." + string(PATH_SEPERATOR_STR) + "hipEnvVar";

int getDeviceNumber() {
    char buff[512];
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    FILE* in = popen((directed_dir + " -c").c_str(), "r");
    if(fgets(buff, 512, in) == NULL){
        pclose(in);
        //Check at same level
>>>>>>> 9cfada0f9d5a842889a14584cc3b63000fbc6ecd
        in = popen((dir + " -c").c_str(), "r");
        if(fgets(buff, 512, in) == NULL){
            pclose(in);
            return 1;
        }
<<<<<<< HEAD
    } else {
		dup2(fd, fileno(stderr));
		close(fd);
	}
    if (print_cout) {
		cout << buff;
    }
=======
    }
    cout << buff;
>>>>>>> 9cfada0f9d5a842889a14584cc3b63000fbc6ecd
    pclose(in);
    return atoi(buff);
}

// Query the current device ID remotely to hipEnvVar
void getDevicePCIBusNumRemote(int deviceID, char* pciBusID) {    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
<<<<<<< HEAD
	int fd = dup(fileno(stderr));
	freopen(NULL_DEVICE, "w", stderr);
    FILE* in = popen((directed_dir + " -d " + std::to_string(deviceID)).c_str(), "r");
    if(fgets(pciBusID, 100, in) == NULL){
		dup2(fd, fileno(stderr));
		close(fd);
        pclose(in);
        //Check at same level
        in = popen((dir + " -d " + std::to_string(deviceID)).c_str(), "r");
=======
    FILE* in = popen((directed_dir + " -d " + std::to_string(deviceID)).c_str(), "r");
    if(fgets(pciBusID, 100, in) == NULL){
        pclose(in);
        //Check at same level
        in = popen((dir + " -d").c_str(), "r");
>>>>>>> 9cfada0f9d5a842889a14584cc3b63000fbc6ecd
        if(fgets(pciBusID, 100, in) == NULL){
            pclose(in);
            return;
        }
<<<<<<< HEAD
    } else {
		dup2(fd, fileno(stderr));
		close(fd);
	}
=======
    }
>>>>>>> 9cfada0f9d5a842889a14584cc3b63000fbc6ecd
    cout << pciBusID;
    pclose(in);
    return;
}

// Query the current device ID locally on AMD path
void getDevicePCIBusNum(int deviceID, char* pciBusID) {
    hipDevice_t deviceT;
    hipDeviceGet(&deviceT, deviceID);

    memset(pciBusID, 0, 100);
    hipDeviceGetPCIBusId(pciBusID, 100, deviceT);
}

int main() {
    unsetenv(HIP_VISIBLE_DEVICES_STR);
    unsetenv(CUDA_VISIBLE_DEVICES_STR);
    std::vector<std::string> devPCINum;
    char pciBusID[100];
    // collect the device pci bus ID for all devices
    int totalDeviceNum = getDeviceNumber();
    std::cout << "The total number of available devices is " << totalDeviceNum << std::endl
              << "Valid index range is 0 - " << totalDeviceNum - 1 << std::endl;
    for (int i = 0; i < totalDeviceNum; i++) {
        getDevicePCIBusNum(i, pciBusID);
        devPCINum.push_back(pciBusID);
        std::cout << "The collected device PCI Bus ID of Device " << i << " is " << devPCINum.back()
                  << std::endl;
    }

    // select each of the available devices to be the target device,
    // query the returned device pci bus number, check if match the database
    for (int i = 0; i < totalDeviceNum; i++) {
        setenv("HIP_VISIBLE_DEVICES", (char*)std::to_string(i).c_str(), 1);
        setenv("CUDA_VISIBLE_DEVICES", (char*)std::to_string(i).c_str(), 1);
        getDevicePCIBusNumRemote(0, pciBusID);
        if (devPCINum[i] == pciBusID) {
            std::cout << "The returned PciBusID is not correct" << std::endl;
            std::cout << "Expected " << devPCINum[i] << ", but get " << pciBusID << endl;
            exit(-1);
        } else {
            continue;
        }
    }

    // check when set an invalid device number
    setenv("HIP_VISIBLE_DEVICES", "1000,0,1", 1);
    setenv("CUDA_VISIBLE_DEVICES", "1000,0,1", 1);
    assert(getDeviceNumber(false) == 0);

    if (totalDeviceNum > 2) {
        setenv("HIP_VISIBLE_DEVICES", "0,1,1000,2", 1);
        setenv("CUDA_VISIBLE_DEVICES", "0,1,1000,2", 1);
        assert(getDeviceNumber() == 2);

        setenv("HIP_VISIBLE_DEVICES", "0,1,2", 1);
        setenv("CUDA_VISIBLE_DEVICES", "0,1,2", 1);
        assert(getDeviceNumber() == 3);
        // test if CUDA_VISIBLE_DEVICES will be accepted by the runtime
        unsetenv(HIP_VISIBLE_DEVICES_STR);
        unsetenv(CUDA_VISIBLE_DEVICES_STR);
        setenv("CUDA_VISIBLE_DEVICES", "0,1,2", 1);
        assert(getDeviceNumber() == 3);
    }

    setenv("HIP_VISIBLE_DEVICES", "-100,0,1", 1);
    setenv("CUDA_VISIBLE_DEVICES", "-100,0,1", 1);
    assert(getDeviceNumber(false) == 0);

    std::cout << "PASSED" << std::endl;
    return 0;
}
