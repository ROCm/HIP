/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

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
// Test the device info API extensions for HIP

#include <hip_test_common.hh>
#include <iostream>



static hipError_t test_hipDeviceGetAttribute(int deviceId,
                                      hipDeviceAttribute_t attr,
                                      int expectedValue = -1) {
  int value = 0;
  std::cout << "Test hipDeviceGetAttribute attribute " << attr;
  if (expectedValue != -1) {
    std::cout << " expected value " << expectedValue;
  }
  HIP_CHECK(hipDeviceGetAttribute(&value, attr, deviceId));
  std::cout << " actual value " << value << std::endl;
  if ((expectedValue != -1) && value != expectedValue) {
    std::cout << "fail" << std::endl;
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t test_hipDeviceGetHdpAddress(int deviceId,
                hipDeviceAttribute_t attr,
                uint32_t* expectedValue) {
  uint32_t* value = 0;
  std::cout << "Test hipDeviceGetHdpAddress attribute " << attr;
  if (expectedValue != reinterpret_cast<uint32_t*>(0xdeadbeef)) {
    std::cout << " expected value " << expectedValue;
  }
  HIP_CHECK(hipDeviceGetAttribute(reinterpret_cast<int*>(&value),
                                       attr, deviceId));
  std::cout << " actual value " << value << std::endl;
  if ((expectedValue != reinterpret_cast<uint32_t*>(0xdeadbeef)) &&
       value != expectedValue) {
    std::cout << "fail" << std::endl;
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

TEST_CASE("Unit_hipGetDeviceAttribute_CheckAttrValues") {
  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  printf("info: running on device #%d %s\n", deviceId, props.name);

  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxThreadsPerBlock,
                                  props.maxThreadsPerBlock));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxBlockDimX,
                                  props.maxThreadsDim[0]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxBlockDimY,
                                  props.maxThreadsDim[1]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxBlockDimZ,
                                  props.maxThreadsDim[2]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxGridDimX,
                                  props.maxGridSize[0]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxGridDimY,
                                  props.maxGridSize[1]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxGridDimZ,
                                  props.maxGridSize[2]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                              hipDeviceAttributeMaxSharedMemoryPerBlock,
                              props.sharedMemPerBlock));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeTotalConstantMemory,
                                  props.totalConstMem));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeWarpSize,
                                      props.warpSize));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMaxRegistersPerBlock,
                                  props.regsPerBlock));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeClockRate,
                                      props.clockRate));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMemoryClockRate,
                                      props.memoryClockRate));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMemoryBusWidth,
                                      props.memoryBusWidth));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeMultiprocessorCount,
                                  props.multiProcessorCount));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeIsMultiGpuBoard,
                                      props.isMultiGpuBoard));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeComputeMode,
                                      props.computeMode));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeL2CacheSize,
                                      props.l2CacheSize));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                          hipDeviceAttributeMaxThreadsPerMultiProcessor,
                          props.maxThreadsPerMultiProcessor));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeComputeCapabilityMajor,
                                  props.major));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                  hipDeviceAttributeComputeCapabilityMinor,
                                  props.minor));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeConcurrentKernels,
                                      props.concurrentKernels));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributePciBusId,
                                      props.pciBusID));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributePciDeviceId,
                                      props.pciDeviceID));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                      hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
                      props.maxSharedMemoryPerMultiProcessor));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeIntegrated,
                                      props.integrated));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMaxTexture1DWidth,
                                      props.maxTexture1D));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                     hipDeviceAttributeMaxTexture2DWidth,
                                     props.maxTexture2D[0]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMaxTexture2DHeight,
                                      props.maxTexture2D[1]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMaxTexture3DWidth,
                                      props.maxTexture3D[0]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMaxTexture3DHeight,
                                      props.maxTexture3D[1]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeMaxTexture3DDepth,
                                      props.maxTexture3D[2]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeCooperativeLaunch,
                                      props.cooperativeLaunch));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                          hipDeviceAttributeCooperativeMultiDeviceLaunch,
                          props.cooperativeMultiDeviceLaunch));

#if HT_AMD
  HIP_CHECK(test_hipDeviceGetHdpAddress(deviceId,
                                     hipDeviceAttributeHdpMemFlushCntl,
                                     props.hdpMemFlushCntl));
  HIP_CHECK(test_hipDeviceGetHdpAddress(deviceId,
                                     hipDeviceAttributeHdpRegFlushCntl,
                                     props.hdpRegFlushCntl));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                            hipDeviceAttributeDirectManagedMemAccessFromHost,
                            props.directManagedMemAccessFromHost));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                            hipDeviceAttributeConcurrentManagedAccess,
                            props.concurrentManagedAccess));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                            hipDeviceAttributePageableMemoryAccess,
                            props.pageableMemoryAccess));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                    hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
                    props.pageableMemoryAccessUsesHostPageTables));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
                    props.cooperativeMultiDeviceUnmatchedFunc));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                  hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
                  props.cooperativeMultiDeviceUnmatchedGridDim));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                  hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
                  props.cooperativeMultiDeviceUnmatchedBlockDim));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                  hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
                  props.cooperativeMultiDeviceUnmatchedSharedMem));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeAsicRevision,
                                      props.asicRevision));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeManagedMemory,
                                      props.managedMemory));
#endif

  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                     hipDeviceAttributeMaxPitch,
                                     props.memPitch));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeTextureAlignment,
                                      props.textureAlignment));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeKernelExecTimeout,
                                      props.kernelExecTimeoutEnabled));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeCanMapHostMemory,
                                      props.canMapHostMemory));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                      hipDeviceAttributeEccEnabled,
                                      props.ECCEnabled));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                    hipDeviceAttributeTexturePitchAlignment,
                                    props.texturePitchAlignment));
}

/**
 * Validates negative scenarios for hipDeviceGetAttribute
 * scenario1: pi = nullptr
 * scenario2: device = -1 (Invalid Device)
 * scenario3: device = Non Existing Device
 * scenario4: attr = Invalid Attribute
 */
TEST_CASE("Unit_hipDeviceGetAttribute_NegTst") {
  int deviceCount = 0;
  int pi = -1;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE(deviceCount != 0);
  printf("No.of gpus in the system: %d\n", deviceCount);

  int device;
  HIP_CHECK(hipGetDevice(&device));

  // pi is nullptr
  SECTION("pi is nullptr") {
    REQUIRE_FALSE(hipSuccess == hipDeviceGetAttribute(nullptr,
                                hipDeviceAttributePciBusId, device));
  }

  // device is -1
  SECTION("device is -1") {
    REQUIRE_FALSE(hipSuccess == hipDeviceGetAttribute(&pi,
                                hipDeviceAttributePciBusId, -1));
  }

  // device is Non Existing Device
  SECTION("device is Non Existing Device") {
    REQUIRE_FALSE(hipSuccess == hipDeviceGetAttribute(&pi,
                                hipDeviceAttributePciBusId, deviceCount));
  }

  // attr is Invalid Attribute
  SECTION("attr is invalid") {
    REQUIRE_FALSE(hipSuccess == hipDeviceGetAttribute(&pi,
                                static_cast<hipDeviceAttribute_t>(-1),
                                device));
  }
}
