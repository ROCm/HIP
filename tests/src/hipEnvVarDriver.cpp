/* Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>

using namespace std;

//./hipEnvVar -c -d 0 -h
  //putenv("SomeVariable=SomeValue");
 //putenv("export HIP_VISIBLE_DEVICES=0,1,2,3");

int getDeviceNumber(){
   FILE *in;
   char buff[512];
   string str;
   if(!(in = popen("./hipEnvVar -c", "r"))){
        return 1;
    }
    fgets(buff, sizeof(buff), in);
    pclose(in);
    return atoi(buff);
}

int getDevicePCIBusNum(int deviceID){
    FILE *in;
    char buff[512];
    string str = "./hipEnvVar -d ";
    str += std::to_string(deviceID);
    if(!(in = popen(str.c_str(), "r"))){
        return 1;
    }
    fgets(buff, sizeof(buff), in);
    pclose(in);
    return atoi(buff);
}

int main() {
    unsetenv("HIP_VISIBLE_DEVICES");
    unsetenv("CUDA_VISIBLE_DEVICES");
    //collect the device pci bus ID for all devices
    int totalDeviceNum = getDeviceNumber();
    std::cout << "The total number of available devices is " << totalDeviceNum<< std::endl
        <<"Valid index range is 0 - "<<totalDeviceNum-1<<std::endl;
    std::vector<int> devPCINum;
    for (int i = 0; i < totalDeviceNum ; i++) {
        devPCINum.push_back(getDevicePCIBusNum(i));
        std::cout <<"The collected device PCI Bus ID of Device "<<i<<" is "
            << getDevicePCIBusNum(i) << std::endl;
    }

    //select each of the available devices to be the target device,
    //query the returned device pci bus number, check if match the database
    for (int i = 0; i < totalDeviceNum ; i++) {
        setenv("HIP_VISIBLE_DEVICES",(char*)std::to_string(i).c_str(),1);
        setenv("CUDA_VISIBLE_DEVICES",(char*)std::to_string(i).c_str(),1);
        //cout<<"HIP_VISIBLE_DEVICES is "<<i<<" data in vector is "<<devPCINum[i]<<endl;
        //std::cout <<"Returned pci number is"<< getDevicePCIBusNum(0) << std::endl;
        if (devPCINum[i] != getDevicePCIBusNum(0)) {
            std::cout << "The returned PciBusID is not correct"
                << std::endl;
            exit(-1);
        } else {
            continue;
        }
    }

    //check when set an invalid device number
    setenv("HIP_VISIBLE_DEVICES","1000,0,1",1);
    setenv("CUDA_VISIBLE_DEVICES","1000,0,1",1);
    assert(getDeviceNumber() == 0);

    if(totalDeviceNum > 2){
        setenv("HIP_VISIBLE_DEVICES","0,1,1000,2",1);
        setenv("CUDA_VISIBLE_DEVICES","0,1,1000,2",1);
        assert(getDeviceNumber() == 2);

        setenv("HIP_VISIBLE_DEVICES","0,1,2",1);
        setenv("CUDA_VISIBLE_DEVICES","0,1,2",1);
        assert(getDeviceNumber() == 3);
        // test if CUDA_VISIBLE_DEVICES will be accepted by the runtime
        unsetenv("HIP_VISIBLE_DEVICES");
        unsetenv("CUDA_VISIBLE_DEVICES");
        setenv("CUDA_VISIBLE_DEVICES","0,1,2",1);
        assert(getDeviceNumber() == 3);
    }

    setenv("HIP_VISIBLE_DEVICES","-100,0,1",1);
    setenv("CUDA_VISIBLE_DEVICES","-100,0,1",1);
    assert(getDeviceNumber() == 0);

    std::cout << "PASSED" << std::endl;
    return 0;
}
