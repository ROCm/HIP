/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <fstream>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>

// Common Definitions and Macros
#if HT_AMD
  static const char* visibility_env_var = "HIP_VISIBLE_DEVICES";
#else
  static const char* visibility_env_var = "CUDA_VISIBLE_DEVICES";
#endif

#define UNUSED(expr) do { (void)(expr); } while (0)

void Callback(hipStream_t stream, hipError_t status,
                       void* userData) {
  UNUSED(stream);
  HIP_CHECK(status);
  REQUIRE(userData == NULL);
}

// Dummy RTC function
static constexpr auto rtcfn {
    R"(
extern "C"
__global__ void fn()
{
}
)"};

// Dummy Kernel Function
static __global__ void fn() {
}

// No GPU test case for hipGetDeviceCount
static bool NoGpuTst_hipGetDeviceCount() {
  bool passed = false;
  hipError_t err;
  int nGpus = 0;
  err = hipGetDeviceCount(&nGpus);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipGetDeviceCount: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipSetDevice
static bool NoGpuTst_hipSetDevice() {
  bool passed = false;
  hipError_t err;
  err = hipSetDevice(0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipSetDevice: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipDeviceGetAttribute
static bool NoGpuTst_hipDeviceGetAttribute() {
  bool passed = false;
  hipError_t err;
  int pi = 0;
  hipDeviceAttribute_t attr = hipDeviceAttributeCudaCompatibleBegin;
  err = hipDeviceGetAttribute(&pi, attr, 0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceGetAttribute: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipDeviceGetP2PAttribute
static bool NoGpuTst_hipDeviceGetP2PAttribute() {
  bool passed = false;
  hipError_t err;
  int srcDevice = 0, dstDevice = 1;
  int value{-1};
  err = hipDeviceGetP2PAttribute(&value, hipDevP2PAttrPerformanceRank,
                                 srcDevice, dstDevice);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceGetP2PAttribute: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipDeviceGetPCIBusId
static bool NoGpuTst_hipDeviceGetPCIBusId() {
  bool passed = false;
  hipError_t err;
  int device = 0;
  char pciBusId[128];
  err = hipDeviceGetPCIBusId(pciBusId, 128, device);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceGetPCIBusId: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipDeviceGetByPCIBusId
static bool NoGpuTst_hipDeviceGetByPCIBusId() {
  bool passed = false;
  hipError_t err;
  int device = 0;
  char pciBusId[128];
  err = hipDeviceGetByPCIBusId(&device, pciBusId);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceGetByPCIBusId: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipDeviceTotalMem
static bool NoGpuTst_hipDeviceTotalMem() {
  bool passed = false;
  hipError_t err, err_exp;
#if HT_AMD
  err_exp = hipErrorNoDevice;
#else
  err_exp = hipErrorNotInitialized;
#endif
  int device = 0;
  size_t bytes = 0;
  err = hipDeviceTotalMem(&bytes, device);
  if (err == err_exp) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceTotalMem: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipFuncSetSharedMemConfig
static bool NoGpuTst_hipFuncSetSharedMemConfig() {
  bool passed = false;
  hipError_t err;
  err = hipFuncSetSharedMemConfig(reinterpret_cast<const void*>(&fn),
                                  hipSharedMemBankSizeDefault);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipFuncSetSharedMemConfig: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipStreamCreate
static bool NoGpuTst_hipStreamCreate() {
  bool passed = false;
  hipError_t err;
  hipStream_t stream = nullptr;
  err = hipStreamCreate(&stream);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipStreamCreate: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipStreamCreateWithPriority
static bool NoGpuTst_hipStreamCreateWithPriority() {
  bool passed = false;
  hipError_t err;
  hipStream_t mystream = nullptr;
  err = hipStreamCreateWithPriority(&mystream, 0, 0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipStreamCreateWithPriority: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipStreamSynchronize
static bool NoGpuTst_hipStreamSynchronize() {
  bool passed = false;
  hipError_t err;
  err = hipStreamSynchronize(0);  // sync with null stream
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipStreamSynchronize: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipStreamGetFlags
static bool NoGpuTst_hipStreamGetFlags() {
  bool passed = false;
  hipError_t err;
  unsigned int flags = 0;
  err = hipStreamGetFlags(0, &flags);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipStreamGetFlags: " <<
        hipGetErrorName(err));
  }
  return passed;
}
#if HT_AMD
// No GPU test case for hipExtStreamCreateWithCUMask
static bool NoGpuTst_hipExtStreamCreateWithCUMask() {
  bool passed = false;
  hipError_t err;
  hipStream_t mystream = nullptr;
  uint32_t cuMaskSize = 0;
  const uint32_t cuMask = 0;
  err = hipExtStreamCreateWithCUMask(&mystream, cuMaskSize, &cuMask);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipExtStreamCreateWithCUMask: " <<
        hipGetErrorName(err));
  }
  return passed;
}
#endif
// No GPU test case for hipStreamAddCallback
static bool NoGpuTst_hipStreamAddCallback() {
  bool passed = false;
  hipError_t err;
  void* userData = NULL;
  unsigned int flags = 0;
  err = hipStreamAddCallback(0, Callback, userData, flags);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipStreamAddCallback: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipEventCreateWithFlags
static bool NoGpuTst_hipEventCreateWithFlags() {
  bool passed = false;
  hipError_t err;
  hipEvent_t event = 0;
  err = hipEventCreateWithFlags(&event, hipEventDefault);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipEventCreateWithFlags: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipDeviceSynchronize
static bool NoGpuTst_hipDeviceSynchronize() {
  bool passed = false;
  hipError_t err;
  err = hipDeviceSynchronize();
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceSynchronize: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipGetDevice
static bool NoGpuTst_hipGetDevice() {
  bool passed = false;
  hipError_t err;
  int deviceId = 0;
  err = hipGetDevice(&deviceId);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipGetDevice: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipDeviceGetCacheConfig
static bool NoGpuTst_hipDeviceGetCacheConfig() {
  bool passed = false;
  hipError_t err;
  hipFuncCache_t cacheConfig = hipFuncCachePreferNone;
  err = hipDeviceGetCacheConfig(&cacheConfig);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceGetCacheConfig: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipDeviceGetSharedMemConfig
static bool NoGpuTst_hipDeviceGetSharedMemConfig() {
  bool passed = false;
  hipError_t err;
  hipSharedMemConfig pConfig = hipSharedMemBankSizeDefault;
  err = hipDeviceGetSharedMemConfig(&pConfig);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceGetSharedMemConfig: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipDeviceSetSharedMemConfig
static bool NoGpuTst_hipDeviceSetSharedMemConfig() {
  bool passed = false;
  hipError_t err;
  hipSharedMemConfig pConfig = hipSharedMemBankSizeDefault;
  err = hipDeviceSetSharedMemConfig(pConfig);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceSetSharedMemConfig: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipPointerGetAttributes
static bool NoGpuTst_hipPointerGetAttributes() {
  bool passed = false;
  hipError_t err;
  hipPointerAttribute_t attributes = {};
  char* A_d;
  err = hipPointerGetAttributes(&attributes, A_d);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipPointerGetAttributes: " <<
        hipGetErrorName(err));
  }
  return passed;
}
#if HT_AMD
// No GPU test case for hipExtMallocWithFlags
static bool NoGpuTst_hipExtMallocWithFlags() {
  bool passed = false;
  hipError_t err;
  int *Ptr = nullptr;
  unsigned int flags = 0;
  err = hipExtMallocWithFlags(reinterpret_cast<void**>(Ptr), 128, flags);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipExtMallocWithFlags: " <<
        hipGetErrorName(err));
  }
  return passed;
}
#endif
// No GPU test case for hipMallocManaged
static bool NoGpuTst_hipMallocManaged() {
  bool passed = false;
  hipError_t err;
  float *dev_ptr;
  err = hipMallocManaged(&dev_ptr, 128, hipMemAttachGlobal);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipMallocManaged: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipHostFree
static bool NoGpuTst_hipHostFree() {
  bool passed = false;
  hipError_t err;
  void* ptr1 = nullptr;
  err = hipHostFree(ptr1);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipHostFree: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test case for hipMemcpyWithStream
static bool NoGpuTst_hipMemcpyWithStream() {
  bool passed = false;
  hipError_t err;
  int *A_h, *B_h;
  A_h = reinterpret_cast<int*>(malloc(sizeof(int)));
  REQUIRE(A_h != nullptr);
  B_h = reinterpret_cast<int*>(malloc(sizeof(int)));
  REQUIRE(B_h != nullptr);
  err = hipMemcpyWithStream(A_h, B_h, 128, hipMemcpyHostToHost, 0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipMemcpyWithStream: " <<
        hipGetErrorName(err));
  }
  return passed;
}

#if HT_AMD
// No GPU test case for hipMallocMipmappedArray
static bool NoGpuTst_hipMallocMipmappedArray() {
  bool passed = false;
  hipError_t err;
  hipMipmappedArray_t mipmappedArray{};
  hipExtent extent;
  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
  err = hipMallocMipmappedArray(&mipmappedArray, &desc, extent, 0, 0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipMallocMipmappedArray: " <<
        hipGetErrorName(err));
  }
  return passed;
}
#endif
// No GPU test case for hipDeviceEnablePeerAccess
static bool NoGpuTst_hipDeviceEnablePeerAccess() {
  bool passed = false;
  hipError_t err;
  int peerDeviceId = 0;
  err = hipDeviceEnablePeerAccess(peerDeviceId, 0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceEnablePeerAccess: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipHostMalloc() API
static bool NoGpuTst_hipHostMalloc() {
  bool passed = false;
  hipError_t err;
  int *ptr;
  size_t size = 1024;
  unsigned int flag = hipHostMallocDefault;
  err = hipHostMalloc(&ptr, size, flag);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipHostMalloc: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipMalloc() API
static bool NoGpuTst_hipMalloc() {
  bool passed = false;
  hipError_t err;
  int *ptr;
  size_t size = 1024;
  err = hipMalloc(&ptr, size);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipMalloc: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipHostRegister() API
static bool NoGpuTst_hipHostRegister() {
  bool passed = false;
  hipError_t err;
  size_t size = 1024;
  unsigned int reg_flag = hipHostRegisterDefault;
  void * hostPtr = reinterpret_cast<void*>(malloc(size));
  err = hipHostRegister(hostPtr, size, reg_flag);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipHostRegister: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipMemcpy() API
static bool NoGpuTst_hipMemcpy() {
  bool passed = false;
  hipError_t err;
  size_t size = 1024;
  int *dst = reinterpret_cast<int*>(malloc(size));
  int *src = reinterpret_cast<int*>(malloc(size));
  err = hipMemcpy(dst, src, size, hipMemcpyHostToHost);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipMemcpy: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipMemAllocPitch() API
static bool NoGpuTst_hipMemAllocPitch() {
  bool passed = false;
  hipError_t err, err_exp;
#if HT_AMD
  err_exp = hipErrorNoDevice;
#else
  err_exp = hipErrorNotInitialized;
#endif
  size_t pitch;
  hipDeviceptr_t dptr;
  size_t widthInBytes = 128;
  size_t height = 128;
  unsigned int elementSizeBytes = 4;
  err = hipMemAllocPitch(&dptr, &pitch, widthInBytes, height, elementSizeBytes);
  if (err == err_exp) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipMemAllocPitch: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipMemGetInfo() API
static bool NoGpuTst_hipMemGetInfo() {
  bool passed = false;
  hipError_t err;
  size_t free, total;
  err = hipMemGetInfo(&free, &total);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipMemGetInfo: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipMalloc3DArray() API
static bool NoGpuTst_hipMalloc3DArray() {
  bool passed = false;
  hipError_t err;
  hipArray_t array;
  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
  hipExtent extent = make_hipExtent(64, 64, 64);
  err = hipMalloc3DArray(&array, &desc, extent, hipArrayDefault);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipMalloc3DArray: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipDeviceCanAccessPeer() API
static bool NoGpuTst_hipDeviceCanAccessPeer() {
  bool passed = false;
  hipError_t err;
  int canAccessPeer; int deviceId1 = 0; int peerDeviceId = 1;
  err =  hipDeviceCanAccessPeer(&canAccessPeer, deviceId1, peerDeviceId);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceCanAccessPeer: "
          <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipDeviceDisablePeerAccess() API
static bool NoGpuTst_hipDeviceDisablePeerAccess() {
  bool passed = false;
  hipError_t err;
  err =  hipDeviceDisablePeerAccess(1);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceDisablePeerAccess: "
          <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipDeviceGet() API
static bool NoGpuTst_hipDeviceGet() {
  bool passed = false;
  hipError_t err, err_exp;
#if HT_AMD
  err_exp = hipErrorNoDevice;
#else
  err_exp = hipErrorNotInitialized;
#endif
  hipDevice_t device;
  int ordinal = 0;
  err =  hipDeviceGet(&device, ordinal);
  if (err == err_exp) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceGet: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipDeviceComputeCapability() API
static bool NoGpuTst_hipDeviceComputeCapability() {
  bool passed = false;
  hipError_t err, err_exp;
#if HT_AMD
  err_exp = hipErrorNoDevice;
#else
  err_exp = hipErrorNotInitialized;
#endif
  int major;
  int minor;
  err = hipDeviceComputeCapability(&major, &minor, 0);
  if (err == err_exp) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceComputeCapability: "
          <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipDeviceGetName() API
static bool NoGpuTst_hipDeviceGetName() {
  bool passed = false;
  hipError_t err, err_exp;
#if HT_AMD
  err_exp = hipErrorNoDevice;
#else
  err_exp = hipErrorNotInitialized;
#endif
  char name[256];
  int len = 256;
  err = hipDeviceGetName(name, len, 0);
  if (err == err_exp) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceGetName: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipGetDeviceProperties() API
static bool NoGpuTst_hipGetDeviceProperties() {
  bool passed = false;
  hipError_t err;
  int deviceId = 0;
  hipDeviceProp_t prop;
  err =  hipGetDeviceProperties(&prop, deviceId);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipGetDeviceProperties: "
          <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipChooseDevice() API
static bool NoGpuTst_hipChooseDevice() {
  bool passed = false;
  hipError_t err;
  int device;
  hipDeviceProp_t prop;
  prop.major = 0;
  prop.minor = 0;
  err = hipChooseDevice(&device, &prop);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipChooseDevice: " <<hipGetErrorName(err));
  }
  return passed;
}
#if HT_AMD
// No GPU test for hipExtGetLinkTypeAndHopCount() API
static bool NoGpuTst_hipExtGetLinkTypeAndHopCount() {
  bool passed = false;
  hipError_t err;
  uint32_t linktype;
  uint32_t hopcount;
  err = hipExtGetLinkTypeAndHopCount(0, 1, &linktype, &hopcount);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipExtGetLinkTypeAndHopCount: "
          <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipExtStreamGetCUMask() API
static bool NoGpuTst_hipExtStreamGetCUMask() {
  bool passed = false;
  hipError_t err;
  uint32_t cuMaskSize = 1024;
  uint32_t cuMask;
  err = hipExtStreamGetCUMask(0, cuMaskSize, &cuMask);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipExtStreamGetCUMask: "
          <<hipGetErrorName(err));
  }
  return passed;
}
#endif

// No GPU test for hipFuncSetAttribute() API
static bool NoGpuTst_hipFuncSetAttribute() {
  bool passed = false;
  hipError_t err;
  hipFuncAttribute attr = hipFuncAttributeMaxDynamicSharedMemorySize;
  err = hipFuncSetAttribute(reinterpret_cast<const void*>(&fn), attr, 0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipFuncSetAttribute: "
          <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipFuncSetCacheConfig() API
static bool NoGpuTst_hipFuncSetCacheConfig() {
  bool passed = false;
  hipError_t err;
  hipFuncCache_t config = hipFuncCachePreferShared;
  err = hipFuncSetCacheConfig(reinterpret_cast<const void*>(&fn), config);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipFuncSetCacheConfig: "
          <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipStreamCreateWithFlags() API
static bool NoGpuTst_hipStreamCreateWithFlags() {
  bool passed = false;
  hipError_t err;
  hipStream_t stream;
  unsigned int flags = 0;
  err =  hipStreamCreateWithFlags(&stream, flags);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipStreamCreateWithFlags: "
          <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipDeviceGetStreamPriorityRange() API
static bool NoGpuTst_hipDeviceGetStreamPriorityRange() {
  bool passed = false;
  hipError_t err;
  int leastPriority;
  int greatestPriority;
  err =  hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceGetStreamPriorityRange: "
          <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipEventCreate() API
static bool NoGpuTst_hipEventCreate() {
  bool passed = false;
  hipError_t err;
  hipEvent_t event;
  err = hipEventCreate(&event);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipEventCreate: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipStreamGetPriority() API
static bool NoGpuTst_hipStreamGetPriority() {
  bool passed = false;
  hipError_t err;
  int priority;
  err =  hipStreamGetPriority(0, &priority);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipStreamGetPriority: "
          <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipDeviceReset() API
static bool NoGpuTst_hipDeviceReset() {
  bool passed = false;
  hipError_t err;
  err = hipDeviceReset();
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceReset: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipDeviceSetCacheConfig() API
static bool NoGpuTst_hipDeviceSetCacheConfig() {
  bool passed = false;
  hipError_t err;
  hipFuncCache_t cacheConfig = hipFuncCachePreferShared;
  err = hipDeviceSetCacheConfig(cacheConfig);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceSetCacheConfig: "
          <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipDeviceGetLimit() API
static bool NoGpuTst_hipDeviceGetLimit() {
  bool passed = false;
  hipError_t err;
  size_t pValue;
  hipLimit_t limit = hipLimitStackSize;
  err =  hipDeviceGetLimit(&pValue, limit);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipDeviceGetLimit: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipGetDeviceFlags() API
static bool NoGpuTst_hipGetDeviceFlags() {
  bool passed = false;
  hipError_t err;
  unsigned int flags;
  err =  hipGetDeviceFlags(&flags);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipGetDeviceFlags: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipSetDeviceFlags() API
static bool NoGpuTst_hipSetDeviceFlags() {
  bool passed = false;
  hipError_t err;
  err = hipSetDeviceFlags(0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipSetDeviceFlags: " <<hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipModuleLoad() API
static bool NoGpuTst_hipModuleLoad() {
  bool passed = false;
  hipError_t err;
  hipModule_t module;
  err = hipModuleLoad(&module, "dummy_kernel.code");
#if HT_AMD
  if (err == hipErrorNoDevice) {
    passed = true;
  }
#else
  if (err == hipErrorNotInitialized) {
    passed = true;
  }
#endif
  if (passed == false) {
    WARN("Error Code Returned by hipModuleLoad: " << hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipFuncGetAttributes() API
static bool NoGpuTst_hipFuncGetAttributes() {
  bool passed = false;
  hipError_t err;
  hipFuncAttributes attr{};
  err = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(&fn));
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipFuncGetAttributes: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipModuleLoadData() API
static bool NoGpuTst_hipModuleLoadData() {
  bool passed = false;
  hipError_t err;
  hipModule_t module;
  std::ifstream file("dummy_kernel.code", std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fsize);
  if (file.read(buffer.data(), fsize)) {
    err = hipModuleLoadData(&module, &buffer[0]);
#if HT_AMD
    if (err == hipErrorNoDevice) {
      passed = true;
    }
#else
    if (err == hipErrorNotInitialized) {
      passed = true;
    }
#endif
    if (passed == false) {
      WARN("Error Code Returned by hipModuleLoadData: " <<
            hipGetErrorName(err));
    }
  } else {
    WARN("File Read Failed");
  }
  return passed;
}

// No GPU test for hipModuleLoadDataEx() API
static bool NoGpuTst_hipModuleLoadDataEx() {
  bool passed = false;
  hipError_t err;
  hipModule_t module;
  std::ifstream file("dummy_kernel.code", std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fsize);
  if (file.read(buffer.data(), fsize)) {
    err = hipModuleLoadDataEx(&module, &buffer[0], 0, nullptr, nullptr);
#if HT_AMD
    if (err == hipErrorNoDevice) {
      passed = true;
    }
#else
    if (err == hipErrorNotInitialized) {
      passed = true;
    }
#endif
    if (passed == false) {
      WARN("Error Code Returned by hipModuleLoadDataEx: " <<
            hipGetErrorName(err));
    }
  } else {
    WARN("File Read Failed");
  }
  return true;
}

// No GPU test for hipLaunchCooperativeKernel() API
static bool NoGpuTst_hipLaunchCooperativeKernel() {
  bool passed = false;
  hipError_t err;
  dim3 dimBlock = dim3(1);
  dim3 dimGrid  = dim3(1);
  err = hipLaunchCooperativeKernel(reinterpret_cast<void*>(fn),
        dimGrid, dimBlock, nullptr, 0, 0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipLaunchCooperativeKernel: " <<
          hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipLaunchCooperativeKernelMultiDevice() API
static bool NoGpuTst_hipLaunchCooperativeKernelMultiDevice() {
  bool passed = false;
  hipError_t err;
  dim3 dimBlock;
  dim3 dimGrid;
  dimGrid.x = 1;
  dimGrid.y = 1;
  dimGrid.z = 1;
  dimBlock.x = 64;
  dimBlock.y = 1;
  dimBlock.z = 1;
  hipLaunchParams md_params[1];
  void *dev_params[1][1];
  dev_params[0][0] = nullptr;
  md_params[0].func = reinterpret_cast<void*>(fn);
  md_params[0].gridDim = dimGrid;
  md_params[0].blockDim = dimBlock;
  md_params[0].sharedMem = 0;
  md_params[0].stream = hipStreamPerThread;
  md_params[0].args = dev_params[0];

  err = hipLaunchCooperativeKernelMultiDevice(md_params, 1, 0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipLaunchCooperativeKernelMultiDevice: " <<
          hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipExtLaunchMultiKernelMultiDevice() API
#if HT_AMD
static bool NoGpuTst_hipExtLaunchMultiKernelMultiDevice() {
  bool passed = false;
  hipError_t err;
  dim3 dimBlock;
  dim3 dimGrid;
  dimGrid.x = 1;
  dimGrid.y = 1;
  dimGrid.z = 1;
  dimBlock.x = 64;
  dimBlock.y = 1;
  dimBlock.z = 1;
  hipLaunchParams md_params[1];
  md_params[0].func = reinterpret_cast<void*>(fn);
  md_params[0].gridDim = dimGrid;
  md_params[0].blockDim = dimBlock;
  md_params[0].sharedMem = 0;
  md_params[0].stream = hipStreamPerThread;
  md_params[0].args = nullptr;

  err = hipExtLaunchMultiKernelMultiDevice(md_params, 1, 0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipExtLaunchMultiKernelMultiDevice: " <<
          hipGetErrorName(err));
  }
  return passed;
}
#endif

// No GPU test for hipExtLaunchMultiKernelMultiDevice() API
static bool NoGpuTst_hipOccupancyMaxActiveBlocksPerMultiprocessor() {
  bool passed = false;
  hipError_t err;
  int numBlock = 0, blockSize = 256;
  err = hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock,
        fn, blockSize, 0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code hipOccupancyMaxActiveBlocksPerMultiprocessor: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags()
static bool NoGpuTst_hipOccupancyMaxActiveBlocksPerMultiprocessorFlags() {
  bool passed = false;
  hipError_t err;
  int numBlock = 0, blockSize = 256;
  err = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlock,
        fn, blockSize, 0, 0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags: "
         << hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipOccupancyMaxPotentialBlockSize()
static bool NoGpuTst_hipOccupancyMaxPotentialBlockSize() {
  bool passed = false;
  hipError_t err;
  int blocksizelimit = 256, blockSize = 0, gridsize = 0;
  err = hipOccupancyMaxPotentialBlockSize(&gridsize, &blockSize, fn,
        0, blocksizelimit);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipOccupancyMaxPotentialBlockSize: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hiprtcVersion()
static bool NoGpuTst_hiprtcVersion() {
  bool passed = false;
  int major = 0, minor = 0;
  hiprtcResult err = hiprtcVersion(&major, &minor);
  if (err == HIPRTC_SUCCESS) {
    passed = true;
  } else {
    WARN("Error Code Returned by hiprtcVersion: " <<
        hiprtcGetErrorString(err));
  }
  return passed;
}

// No GPU test for hiprtcCreateProgram() and hiprtcDestroyProgram()
static bool NoGpuTst_hiprtcCreateProgram_DestroyProg() {
  bool passed = false;
  hiprtcProgram prog;
  hiprtcResult err = hiprtcCreateProgram(&prog, rtcfn, "rtcfn.cu",
                                        0, nullptr, nullptr);
  if (err == HIPRTC_SUCCESS) {
    passed = true;
    err = hiprtcDestroyProgram(&prog);
    if (err == HIPRTC_SUCCESS) {
      passed = true;
    } else {
      passed = false;
      WARN("Error Code Returned by hiprtcDestroyProgram: " <<
            hiprtcGetErrorString(err));
    }
  } else {
    WARN("Error Code Returned by hiprtcCreateProgram: " <<
        hiprtcGetErrorString(err));
  }
  return passed;
}

// No GPU test for hiprtcAddNameExpression()
static bool NoGpuTst_hiprtcAddNameExpression() {
  bool passed = false;
  hiprtcProgram prog;
  hiprtcResult err = hiprtcCreateProgram(&prog, rtcfn, "rtcfn.cu",
                                        0, nullptr, nullptr);
  if (err == HIPRTC_SUCCESS) {
    err = hiprtcAddNameExpression(prog, "dummy_func");
    if (err == HIPRTC_SUCCESS) {
      passed = true;
    } else {
      passed = false;
      WARN("Error Code Returned by hiprtcAddNameExpression: " <<
            hiprtcGetErrorString(err));
    }
  }
  if (err == HIPRTC_SUCCESS) {
    err = hiprtcDestroyProgram(&prog);
  }
  return passed;
}

// No GPU test for hipMallocPitch()
#define SIZE_H 20
#define SIZE_W 179
static bool NoGpuTst_hipMallocPitch() {
  bool passed = false;
  hipError_t err;
  int* devPtr;
  size_t devPitch;
  err = hipMallocPitch(reinterpret_cast<void**>(&devPtr), &devPitch,
                                       SIZE_W*sizeof(int), SIZE_H);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipMallocPitch: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipMallocArray()
static bool NoGpuTst_hipMallocArray() {
  bool passed = false;
  hipError_t err;
  hipChannelFormatDesc channelDesc =
  hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
  hipArray* hipArray;
  err = hipMallocArray(&hipArray, &channelDesc, 256, 256);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipMallocArray: " <<
        hipGetErrorName(err));
  }
  return passed;
}
#if HT_AMD
// No GPU test for hipMipmappedArrayCreate()
static bool NoGpuTst_hipMipmappedArrayCreate() {
  bool passed = false;
  hipError_t err;
  unsigned int width = 256, height = 256, mipmap_level = 2;
  unsigned int orig_width = width;
  unsigned int orig_height = height;
  width /= pow(2, mipmap_level);
  height /= pow(2, mipmap_level);

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(32, 0, 0,
                                    0, hipChannelFormatKindFloat);
  HIP_ARRAY3D_DESCRIPTOR mipmapped_array_desc;
  memset(&mipmapped_array_desc, 0x00, sizeof(HIP_ARRAY3D_DESCRIPTOR));
  mipmapped_array_desc.Width = orig_width;
  mipmapped_array_desc.Height = orig_height;
  mipmapped_array_desc.Depth = 0;
  mipmapped_array_desc.Format = HIP_AD_FORMAT_FLOAT;
  mipmapped_array_desc.NumChannels =
  ((channelDesc.x != 0) + (channelDesc.y != 0) +\
  (channelDesc.z != 0) + (channelDesc.w != 0));
  mipmapped_array_desc.Flags = 0;

  hipMipmappedArray* mip_array_ptr;
  err = hipMipmappedArrayCreate(&mip_array_ptr, &mipmapped_array_desc,
        2*mipmap_level);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipMipmappedArrayCreate: " <<
        hipGetErrorName(err));
  }
  return passed;
}
#endif
#define SMALL_SIZE 4
static bool NoGpuTst_hipMalloc3D() {
  bool passed = false;
  hipError_t err;
  size_t width = SMALL_SIZE * sizeof(char);
  size_t height{SMALL_SIZE}, depth{SMALL_SIZE};
  hipPitchedPtr devPitchedPtr;
  hipExtent extent = make_hipExtent(width, height, depth);
  err = hipMalloc3D(&devPitchedPtr, extent);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipMalloc3D: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipGraphCreate() API
static bool NoGpuTst_hipGraphCreate() {
  bool passed = false;
  hipError_t err;
  hipGraph_t graph;

  err = hipGraphCreate(&graph, 0);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipGraphCreate: " <<
        hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipStreamBeginCapture() API
static bool NoGpuTst_hipStreamBeginCapture() {
  bool passed = false;
  hipError_t err;

  err = hipStreamBeginCapture(0, hipStreamCaptureModeGlobal);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipStreamBeginCapture with null stream: " <<
        hipGetErrorName(err));
  }

  err = hipStreamBeginCapture(hipStreamPerThread, hipStreamCaptureModeGlobal);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipStreamBeginCapture with "
         "hipStreamPerThread stream: " << hipGetErrorName(err));
  }
  return passed;
}

// No GPU test for hipStreamIsCapturing() API
static bool NoGpuTst_hipStreamIsCapturing() {
  bool passed = false;
  hipError_t err;
  hipStreamCaptureStatus cStatus;

  err = hipStreamIsCapturing(0, &cStatus);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipStreamIsCapturing: " <<
        hipGetErrorName(err));
  }
  err = hipStreamIsCapturing(hipStreamPerThread, &cStatus);
  if (err == hipErrorNoDevice) {
    passed = true;
  } else {
    WARN("Error Code Returned by hipStreamIsCapturing with "
         "hipStreamPerThread stream: " << hipGetErrorName(err));
  }
  return passed;
}
// Common function invoking individual tests
static bool NoGpuTst_Common(bool(*test_fn)()) {
  bool passed = false;
  int *Ptr = reinterpret_cast<int*>(mmap(NULL, sizeof(int),
                                        PROT_READ | PROT_WRITE,
                                        MAP_SHARED|MAP_ANONYMOUS, 0, 0));
  if (Ptr == MAP_FAILED) {
    WARN("Mapping Failed\n");
    return false;
  }
  // forking to get number of gpus on this system
  if (fork() == 0) {  //  child process
    int TotlGpus = 0;
    static_cast<void>(hipGetDeviceCount(&TotlGpus));
    Ptr[0] = TotlGpus;
    exit(0);
  } else {  // parent process
    wait(NULL);
    if (Ptr[0] > 0) {  // If there are gpus active on the system
      if ((setenv(visibility_env_var, "-1", 1)) != 0) {
        WARN("Unable to turn on HIP_VISIBLE_DEVICES, hence exiting!");
        return false;
      }
      passed = test_fn();
    } else if (Ptr[0] == 0) {
      WARN("GPUs not available!");
      passed = test_fn();
    }
  }
  return passed;
}

// Tests
TEST_CASE("Unit_NoGpuTst_hipGetDeviceCount") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipGetDeviceCount));
}

TEST_CASE("Unit_NoGpuTst_hipSetDevice") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipSetDevice));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceGetAttribute") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceGetAttribute));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceGetP2PAttribute") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceGetP2PAttribute));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceGetPCIBusId") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceGetPCIBusId));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceGetByPCIBusId") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceGetByPCIBusId));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceTotalMem") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceTotalMem));
}

TEST_CASE("Unit_NoGpuTst_hipFuncSetSharedMemConfig") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipFuncSetSharedMemConfig));
}

TEST_CASE("Unit_NoGpuTst_hipStreamCreate") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipStreamCreate));
}

TEST_CASE("Unit_NoGpuTst_hipStreamCreateWithPriority") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipStreamCreateWithPriority));
}

TEST_CASE("Unit_NoGpuTst_hipStreamSynchronize") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipStreamSynchronize));
}

TEST_CASE("Unit_NoGpuTst_hipStreamGetFlags") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipStreamGetFlags));
}
#if HT_AMD
TEST_CASE("Unit_NoGpuTst_hipExtStreamCreateWithCUMask") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipExtStreamCreateWithCUMask));
}
#endif
TEST_CASE("Unit_NoGpuTst_hipStreamAddCallback") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipStreamAddCallback));
}

TEST_CASE("Unit_NoGpuTst_hipEventCreateWithFlags") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipEventCreateWithFlags));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceSynchronize") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceSynchronize));
}

TEST_CASE("Unit_NoGpuTst_hipGetDevice") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipGetDevice));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceGetCacheConfig") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceGetCacheConfig));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceGetSharedMemConfig") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceGetSharedMemConfig));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceSetSharedMemConfig") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceSetSharedMemConfig));
}

TEST_CASE("Unit_NoGpuTst_hipPointerGetAttributes") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipPointerGetAttributes));
}
#if HT_AMD
TEST_CASE("Unit_NoGpuTst_hipExtMallocWithFlags") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipExtMallocWithFlags));
}
#endif
TEST_CASE("Unit_NoGpuTst_hipMallocManaged") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipMallocManaged));
}

TEST_CASE("Unit_NoGpuTst_hipHostFree") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipHostFree));
}

TEST_CASE("Unit_NoGpuTst_hipMemcpyWithStream") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipMemcpyWithStream));
}

TEST_CASE("Unit_NoGpuTst_hipMallocArray") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipMallocArray));
}
#if HT_AMD
TEST_CASE("Unit_NoGpuTst_hipMallocMipmappedArray") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipMallocMipmappedArray));
}
#endif
TEST_CASE("Unit_NoGpuTst_hipDeviceEnablePeerAccess") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceEnablePeerAccess));
}

TEST_CASE("Unit_NoGpuTst_hipHostMalloc") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipHostMalloc));
}

TEST_CASE("Unit_NoGpuTst_hipMalloc") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipMalloc));
}

TEST_CASE("Unit_NoGpuTst_hipHostRegister") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipHostRegister));
}

TEST_CASE("Unit_NoGpuTst_hipMemcpy") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipMemcpy));
}

TEST_CASE("Unit_NoGpuTst_hipMemAllocPitch") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipMemAllocPitch));
}

TEST_CASE("Unit_NoGpuTst_hipMemGetInfo") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipMemGetInfo));
}

TEST_CASE("Unit_NoGpuTst_hipMalloc3DArray") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipMalloc3DArray));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceCanAccessPeer") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceCanAccessPeer));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceDisablePeerAccess") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceDisablePeerAccess));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceGet") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceGet));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceComputeCapability") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceComputeCapability));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceGetName") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceGetName));
}

TEST_CASE("Unit_NoGpuTst_hipGetDeviceProperties") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipGetDeviceProperties));
}

TEST_CASE("Unit_NoGpuTst_hipChooseDevice") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipChooseDevice));
}

#if HT_AMD
TEST_CASE("Unit_NoGpuTst_hipExtGetLinkTypeAndHopCount") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipExtGetLinkTypeAndHopCount));
}

TEST_CASE("Unit_NoGpuTst_hipExtStreamGetCUMask") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipExtStreamGetCUMask));
}
#endif

TEST_CASE("Unit_NoGpuTst_hipFuncSetAttribute") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipFuncSetAttribute));
}

TEST_CASE("Unit_NoGpuTst_hipFuncSetCacheConfig") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipFuncSetCacheConfig));
}

TEST_CASE("Unit_NoGpuTst_hipStreamCreateWithFlags") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipStreamCreateWithFlags));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceGetStreamPriorityRange") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceGetStreamPriorityRange));
}

TEST_CASE("Unit_NoGpuTst_hipEventCreate") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipEventCreate));
}

TEST_CASE("Unit_NoGpuTst_hipStreamGetPriority") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipStreamGetPriority));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceReset") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceReset));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceSetCacheConfig") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceSetCacheConfig));
}

TEST_CASE("Unit_NoGpuTst_hipDeviceGetLimit") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipDeviceGetLimit));
}

TEST_CASE("Unit_NoGpuTst_hipGetDeviceFlags") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipGetDeviceFlags));
}

TEST_CASE("Unit_NoGpuTst_hipSetDeviceFlags") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipSetDeviceFlags));
}

TEST_CASE("Unit_NoGpuTst_hipModuleLoad") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipModuleLoad));
}

TEST_CASE("Unit_NoGpuTst_hipFuncGetAttributes") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipFuncGetAttributes));
}

TEST_CASE("Unit_NoGpuTst_hipModuleLoadData") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipModuleLoadData));
}

TEST_CASE("Unit_NoGpuTst_hipModuleLoadDataEx") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipModuleLoadDataEx));
}

TEST_CASE("Unit_NoGpuTst_hipLaunchCooperativeKernel") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipLaunchCooperativeKernel));
}

TEST_CASE("Unit_NoGpuTst_hipLaunchCooperativeKernelMultiDevice") {
  REQUIRE(NoGpuTst_Common(
          NoGpuTst_hipLaunchCooperativeKernelMultiDevice));
}
#if HT_AMD
TEST_CASE("Unit_NoGpuTst_hipExtLaunchMultiKernelMultiDevice") {
  REQUIRE(NoGpuTst_Common(
          NoGpuTst_hipExtLaunchMultiKernelMultiDevice));
}
#endif
TEST_CASE("Unit_NoGpuTst_hipOccupancyMaxActiveBlocksPerMultiprocessor") {
  REQUIRE(NoGpuTst_Common(
  NoGpuTst_hipOccupancyMaxActiveBlocksPerMultiprocessor));
}

TEST_CASE("Unit_NoGpuTst_hipOccupancyMaxActiveBlocksPerMultiprocessorFlags") {
  REQUIRE(NoGpuTst_Common(
          NoGpuTst_hipOccupancyMaxActiveBlocksPerMultiprocessorFlags));
}

TEST_CASE("Unit_NoGpuTst_hipOccupancyMaxPotentialBlockSize") {
  REQUIRE(NoGpuTst_Common(
          NoGpuTst_hipOccupancyMaxPotentialBlockSize));
}

TEST_CASE("Unit_NoGpuTst_hiprtcVersion") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hiprtcVersion));
}

TEST_CASE("Unit_NoGpuTst_hiprtcCreateProgram_DestroyProg") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hiprtcCreateProgram_DestroyProg));
}

TEST_CASE("Unit_NoGpuTst_hiprtcAddNameExpression") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hiprtcAddNameExpression));
}

TEST_CASE("Unit_NoGpuTst_hipMallocPitch") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipMallocPitch));
}

#if HT_AMD
TEST_CASE("Unit_NoGpuTst_hipMipmappedArrayCreate") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipMipmappedArrayCreate));
}
#endif
TEST_CASE("Unit_NoGpuTst_hipMalloc3D") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipMalloc3D));
}

TEST_CASE("Unit_NoGpuTst_hipGraphCreate") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipGraphCreate));
}

TEST_CASE("Unit_NoGpuTst_hipStreamBeginCapture") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipStreamBeginCapture));
}

TEST_CASE("Unit_NoGpuTst_hipStreamIsCapturing") {
  REQUIRE(NoGpuTst_Common(NoGpuTst_hipStreamIsCapturing));
}
#endif  // #ifdef __linux__
