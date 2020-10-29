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
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t --tests 0x1
 * TEST: %t --tests 0x2
 * HIT_END
 */

#include <stdio.h>
#include <iostream>
#include "test_common.h"

hipError_t test_hipDeviceGetAttribute(int deviceId,
                                      hipDeviceAttribute_t attr,
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

hipError_t test_hipDeviceGetHdpAddress(int deviceId,
                hipDeviceAttribute_t attr,
                uint32_t* expectedValue) {
  uint32_t* value = 0;
  std::cout << "Test hipDeviceGetHdpAddress attribute " << attr;
  if (expectedValue != reinterpret_cast<uint32_t*>(0xdeadbeef)) {
    std::cout << " expected value " << expectedValue;
  }
  hipError_t e = hipDeviceGetAttribute(reinterpret_cast<int*>(&value),
                                       attr, deviceId);
  std::cout << " actual value " << value << std::endl;
  if ((expectedValue != reinterpret_cast<uint32_t*>(0xdeadbeef)) &&
       value != expectedValue) {
    std::cout << "fail" << std::endl;
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

bool testAttributeValues() {
  int deviceId;
  HIPCHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, deviceId));
  printf("info: running on device #%d %s\n", deviceId, props.name);

  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxThreadsPerBlock,
                                  props.maxThreadsPerBlock));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxBlockDimX,
                                  props.maxThreadsDim[0]));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxBlockDimY,
                                  props.maxThreadsDim[1]));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxBlockDimZ,
                                  props.maxThreadsDim[2]));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxGridDimX,
                                  props.maxGridSize[0]));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxGridDimY,
                                  props.maxGridSize[1]));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxGridDimZ,
                                  props.maxGridSize[2]));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                              hipDeviceAttributeMaxSharedMemoryPerBlock,
                              props.sharedMemPerBlock));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeTotalConstantMemory,
                                  props.totalConstMem));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeWarpSize,
                                      props.warpSize));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxRegistersPerBlock,
                                  props.regsPerBlock));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeClockRate,
                                      props.clockRate));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMemoryClockRate,
                                      props.memoryClockRate));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMemoryBusWidth,
                                      props.memoryBusWidth));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMultiprocessorCount,
                                  props.multiProcessorCount));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeIsMultiGpuBoard,
                                      props.isMultiGpuBoard));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeComputeMode,
                                      props.computeMode));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeL2CacheSize,
                                      props.l2CacheSize));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                          hipDeviceAttributeMaxThreadsPerMultiProcessor,
                          props.maxThreadsPerMultiProcessor));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeComputeCapabilityMajor,
                                  props.major));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeComputeCapabilityMinor,
                                  props.minor));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeConcurrentKernels,
                                      props.concurrentKernels));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributePciBusId,
                                      props.pciBusID));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributePciDeviceId,
                                      props.pciDeviceID));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                      hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
                      props.maxSharedMemoryPerMultiProcessor));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeIntegrated,
                                      props.integrated));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMaxTexture1DWidth,
                                      props.maxTexture1D));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                     hipDeviceAttributeMaxTexture2DWidth,
                                     props.maxTexture2D[0]));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMaxTexture2DHeight,
                                      props.maxTexture2D[1]));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMaxTexture3DWidth,
                                      props.maxTexture3D[0]));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMaxTexture3DHeight,
                                      props.maxTexture3D[1]));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMaxTexture3DDepth,
                                      props.maxTexture3D[2]));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeCooperativeLaunch,
                                      props.cooperativeLaunch));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                          hipDeviceAttributeCooperativeMultiDeviceLaunch,
                          props.cooperativeMultiDeviceLaunch));

#ifndef __HIP_PLATFORM_NVCC__
  HIPCHECK(test_hipDeviceGetHdpAddress(deviceId,
                                     hipDeviceAttributeHdpMemFlushCntl,
                                     props.hdpMemFlushCntl));
  HIPCHECK(test_hipDeviceGetHdpAddress(deviceId,
                                     hipDeviceAttributeHdpRegFlushCntl,
                                     props.hdpRegFlushCntl));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                            hipDeviceAttributeDirectManagedMemAccessFromHost,
                            props.directManagedMemAccessFromHost));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                            hipDeviceAttributeConcurrentManagedAccess,
                            props.concurrentManagedAccess));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                            hipDeviceAttributePageableMemoryAccess,
                            props.pageableMemoryAccess));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                    hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
                    props.pageableMemoryAccessUsesHostPageTables));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
                    props.cooperativeMultiDeviceUnmatchedFunc));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                  hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
                  props.cooperativeMultiDeviceUnmatchedGridDim));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                  hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
                  props.cooperativeMultiDeviceUnmatchedBlockDim));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                  hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
                  props.cooperativeMultiDeviceUnmatchedSharedMem));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeAsicRevision,
                                      props.asicRevision));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeManagedMemory,
                                      props.managedMemory));
#endif
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                     hipDeviceAttributeMaxPitch,
                                     props.memPitch));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeTextureAlignment,
                                      props.textureAlignment));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeKernelExecTimeout,
                                      props.kernelExecTimeoutEnabled));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeCanMapHostMemory,
                                      props.canMapHostMemory));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeEccEnabled,
                                      props.ECCEnabled));
  HIPCHECK(test_hipDeviceGetAttribute(deviceId,
                                    hipDeviceAttributeTexturePitchAlignment,
                                    props.texturePitchAlignment));
  return true;
}
/**
 * Validates negative scenarios for hipDeviceGetAttribute
 * scenario1: pi = nullptr
 * scenario2: device = -1 (Invalid Device)
 * scenario3: device = Non Existing Device
 * scenario4: attr = Invalid Attribute
 */
bool testInvalidParameters() {
  bool TestPassed = true;
  hipError_t ret;
  int deviceCount = 0;
  HIPCHECK(hipGetDeviceCount(&deviceCount));
  HIPASSERT(deviceCount != 0);
  printf("No.of gpus in the system: %d\n", deviceCount);
  // pi = nullptr
  int device;
  HIPCHECK(hipGetDevice(&device));
  ret = hipDeviceGetAttribute(nullptr, hipDeviceAttributePciBusId, device);
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {pi = nullptr} Failed \n");
  }
  // device = -1
  int pi = -1;
  ret = hipDeviceGetAttribute(&pi, hipDeviceAttributePciBusId, -1);
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {device = -1} Failed \n");
  }
  // device = Non Existing Device
  pi = -1;
  ret = hipDeviceGetAttribute(&pi, hipDeviceAttributePciBusId, deviceCount);
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {device = Non Existing Device} Failed \n");
  }
  // attr = Invalid Attribute
  pi = -1;
  ret = hipDeviceGetAttribute(&pi, static_cast<hipDeviceAttribute_t>(-1),
                              device);
  if (ret == hipSuccess) {
    TestPassed &= false;
    printf("Test {attr = Invalid Attribute} Failed \n");
  }
  return TestPassed;
}

int main(int argc, char* argv[]) {
  bool TestPassed = true;
  HipTest::parseStandardArguments(argc, argv, true);

  if (p_tests == 0x1) {
    TestPassed = testAttributeValues();
  } else if (p_tests == 0x2) {
    TestPassed = testInvalidParameters();
  } else {
    printf("Invalid Test Case \n");
    exit(1);
  }
  if (TestPassed) {
    passed();
  } else {
    failed("Test Case %x Failed!", p_tests);
  }
}
