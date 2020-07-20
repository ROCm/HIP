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
// Test the device info API extensions for HIP:

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"

#include "test_common.h"

#define CHECK(error)                                                                               \
    if (error != hipSuccess) {                                                                     \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__,   \
                __LINE__);                                                                         \
        exit(EXIT_FAILURE);                                                                        \
    }

hipError_t test_hipDeviceGetAttribute(int deviceId, hipDeviceAttribute_t attr,
                                      int expectedValue = -1) {
    int value = 0;
    std::cout << "Test hipDeviceGetAttribute attribute " << attr;
    if (expectedValue != -1) {
        std::cout << " expected value " << expectedValue;
    }
    hipError_t e = hipDeviceGetAttribute(&value, attr, deviceId);
    std::cout << " actual value " << value << std::endl;
    if ((expectedValue != -1) && value != expectedValue) {
        std::cout << "fail" << std::endl;
        return hipErrorInvalidValue;
    }
    return hipSuccess;
}

hipError_t test_hipDeviceGetHdpAddress(int deviceId, hipDeviceAttribute_t attr,
                                       uint32_t* expectedValue = (uint32_t*)0xdeadbeef) {
    uint32_t* value = 0;
    std::cout << "Test hipDeviceGetHdpAddress attribute " << attr;
    if (expectedValue != (uint32_t*)0xdeadbeef) {
        std::cout << " expected value " << expectedValue;
    }
    hipError_t e = hipDeviceGetAttribute((int*) &value, attr, deviceId);
    std::cout << " actual value " << value << std::endl;
    if ((expectedValue != (uint32_t*)0xdeadbeef) && value != expectedValue) {
        std::cout << "fail" << std::endl;
        return hipErrorInvalidValue;
    }
    return hipSuccess;
}

int main(int argc, char* argv[]) {
    int deviceId;
    CHECK(hipGetDevice(&deviceId));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, deviceId));
    printf("info: running on device #%d %s\n", deviceId, props.name);

    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxThreadsPerBlock,
                                     props.maxThreadsPerBlock));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxBlockDimX,
                                     props.maxThreadsDim[0]));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxBlockDimY,
                                     props.maxThreadsDim[1]));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxBlockDimZ,
                                     props.maxThreadsDim[2]));
    CHECK(
        test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxGridDimX, props.maxGridSize[0]));
    CHECK(
        test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxGridDimY, props.maxGridSize[1]));
    CHECK(
        test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxGridDimZ, props.maxGridSize[2]));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxSharedMemoryPerBlock,
                                     props.sharedMemPerBlock));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeTotalConstantMemory,
                                     props.totalConstMem));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeWarpSize, props.warpSize));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxRegistersPerBlock,
                                     props.regsPerBlock));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeClockRate, props.clockRate));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMemoryClockRate,
                                     props.memoryClockRate));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMemoryBusWidth,
                                     props.memoryBusWidth));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMultiprocessorCount,
                                     props.multiProcessorCount));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeIsMultiGpuBoard,
                                     props.isMultiGpuBoard));  //
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeComputeMode, props.computeMode));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeL2CacheSize, props.l2CacheSize));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxThreadsPerMultiProcessor,
                                     props.maxThreadsPerMultiProcessor));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeComputeCapabilityMajor,
                                     props.major));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeComputeCapabilityMinor,
                                     props.minor));  //
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeConcurrentKernels,
                                     props.concurrentKernels));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributePciBusId, props.pciBusID));
    CHECK(
        test_hipDeviceGetAttribute(deviceId, hipDeviceAttributePciDeviceId, props.pciDeviceID));  //
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
                                     props.maxSharedMemoryPerMultiProcessor));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeIntegrated, props.integrated));

    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture1DWidth, props.maxTexture1D));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture2DWidth, props.maxTexture2D[0]));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture2DHeight, props.maxTexture2D[1]));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture3DWidth, props.maxTexture3D[0]));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture3DHeight, props.maxTexture3D[1]));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture3DDepth, props.maxTexture3D[2]));

    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeCooperativeLaunch, props.cooperativeLaunch));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeCooperativeMultiDeviceLaunch, props.cooperativeMultiDeviceLaunch));

#ifndef __HIP_PLATFORM_NVCC__
    CHECK(test_hipDeviceGetHdpAddress(deviceId, hipDeviceAttributeHdpMemFlushCntl, props.hdpMemFlushCntl));
    CHECK(test_hipDeviceGetHdpAddress(deviceId, hipDeviceAttributeHdpRegFlushCntl, props.hdpRegFlushCntl));
#endif
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxPitch, props.memPitch));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeTextureAlignment, props.textureAlignment));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeKernelExecTimeout, props.kernelExecTimeoutEnabled));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeCanMapHostMemory, props.canMapHostMemory));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeEccEnabled, props.ECCEnabled));
    CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeAsicRevision, props.asicRevision));
    passed();
};
