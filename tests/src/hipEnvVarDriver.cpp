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
#ifdef _WIN32
    #define PLATFORM_NAME "windows"
#elif __linux__
    #define PLATFORM_NAME "linux"
#else
    #define PLATFORM_NAME "NULL"
#endif

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

int getDeviceNumber() {
    FILE* directed_in;
    FILE* in;
    string directed_dir;
    string dir;
    char buff[512];

    if (PLATFORM_NAME == "windows"){
        directed_dir = "directed_tests\\hipEnvVar -c";
        dir = "hipEnvVar -c";
    } else {
        directed_dir = "./directed_tests/hipEnvVar -c";
        dir = "./hipEnvVar -c";
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    directed_in = popen(directed_dir.c_str(), "r");
    if(fgets(buff, 512, directed_in) != NULL){
        cout << buff;
        pclose(directed_in);
        return atoi(buff);
    }
    //Check at same level
    in = popen(dir.c_str(), "r");
    if(fgets(buff, 512, in) != NULL){
        cout << buff;
        pclose(in);
        return atoi(buff);
    }
    
    pclose(directed_in);
    pclose(in);
    return 1;
}

// Query the current device ID remotely to hipEnvVar
void getDevicePCIBusNumRemote(int deviceID, char* pciBusID) {
    FILE* directed_in;
    FILE* in;
    
    string directed_dir;
    string dir;
    if (PLATFORM_NAME == "windows"){
        directed_dir = "directed_tests\\hipEnvVar -d ";
        dir = "hipEnvVar.exe -d ";
    } else {
        directed_dir = "./directed_tests/hipEnvVar -d ";
        dir = "./hipEnvVar -d ";
    }

    directed_dir += std::to_string(deviceID);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    directed_in = popen(directed_dir.c_str(), "r");
    if(fgets(pciBusID, 100, directed_in) != NULL){
        cout << pciBusID;
        pclose(directed_in);
        return;
    }
    //Check at same level
    in = popen(dir.c_str(), "r");
    if(fgets(pciBusID, 100, in) != NULL){
        cout << pciBusID;
        pclose(in);
        return;
    }
    pclose(directed_in);
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
    if (PLATFORM_NAME == "windows"){
        _putenv("HIP_VISIBLE_DEVICES=");
        _putenv("CUDA_VISIBLE_DEVICES=");
    } else {
        unsetenv("HIP_VISIBLE_DEVICES");
        unsetenv("CUDA_VISIBLE_DEVICES");
    }

    std::vector<std::string> devPCINum;
    char pciBusID[100];
    // collect the device pci bus ID for all device
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
    assert(getDeviceNumber() == 0);

    if (totalDeviceNum > 2) {
        setenv("HIP_VISIBLE_DEVICES", "0,1,1000,2", 1);
        setenv("CUDA_VISIBLE_DEVICES", "0,1,1000,2", 1);
        assert(getDeviceNumber() == 2);

        setenv("HIP_VISIBLE_DEVICES", "0,1,2", 1);
        setenv("CUDA_VISIBLE_DEVICES", "0,1,2", 1);
        assert(getDeviceNumber() == 3);
        // test if CUDA_VISIBLE_DEVICES will be accepted by the runtime
        if (PLATFORM_NAME == "windows"){
            _putenv("HIP_VISIBLE_DEVICES=");
            _putenv("CUDA_VISIBLE_DEVICES=");
        } else {
            unsetenv("HIP_VISIBLE_DEVICES");
            unsetenv("CUDA_VISIBLE_DEVICES");
        }
        setenv("CUDA_VISIBLE_DEVICES", "0,1,2", 1);
        assert(getDeviceNumber() == 3);
    }

    setenv("HIP_VISIBLE_DEVICES", "-100,0,1", 1);
    setenv("CUDA_VISIBLE_DEVICES", "-100,0,1", 1);
    assert(getDeviceNumber() == 0);

    std::cout << "PASSED" << std::endl;
    return 0;
}
