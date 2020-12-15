/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD_CMD: tex2d_kernel.code %hc --genco %S/tex2d_kernel.cpp -o tex2d_kernel.code
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t --tests 0x01
 * TEST: %t --tests 0x02
 * TEST: %t --tests 0x03
 * TEST: %t --tests 0x04
 * TEST: %t --tests 0x05
 * TEST: %t --tests 0x06
 * TEST: %t --tests 0x07
 * TEST: %t --tests 0x10
 * TEST: %t --tests 0x11
 * TEST: %t --tests 0x12
 * TEST: %t --tests 0x13
 * TEST: %t --tests 0x14 EXCLUDE_HIP_PLATFORM amd
 * TEST: %t --tests 0x15
 * HIT_END
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <type_traits>
#include <limits>
#include <atomic>
#include "test_common.h"

#define CODEOBJ_FILE "tex2d_kernel.code"
#define NON_EXISTING_TEX_NAME "xyz"
#define EMPTY_TEX_NAME ""
#define GLOBAL_KERNEL_VAR "deviceGlobalFloat"
#define TEX_REF "ftex"
#define WIDTH 256
#define HEIGHT 256
#define MAX_STREAMS 4
#define GRIDDIMX 16
#define GRIDDIMY 16
#define GRIDDIMZ 1
#define BLOCKDIMZ 1

#ifdef __HIP_PLATFORM_NVIDIA__

#define CTX_CREATE() \
  hipCtx_t context;\
  initHipCtx(&context);

#define CTX_DESTROY() HIPCHECK(hipCtxDestroy(context));
#define ARRAY_DESTROY(array) HIPCHECK(hipArrayDestroy(array));
#define HIP_TEX_REFERENCE hipTexRef
#define HIP_ARRAY hiparray
/**
 * Internal Function
 */
void initHipCtx(hipCtx_t *pcontext) {
  HIPCHECK(hipInit(0));
  hipDevice_t device;
  HIPCHECK(hipDeviceGet(&device, 0));
  HIPCHECK(hipCtxCreate(pcontext, 0, device));
}

#else  // __HIP_PLATFORM_NVIDIA__

#define CTX_CREATE()
#define CTX_DESTROY()
#define ARRAY_DESTROY(array) HIPCHECK(hipFreeArray(array));
#define HIP_TEX_REFERENCE textureReference*
#define HIP_ARRAY hipArray*
#endif  // __HIP_PLATFORM_NVIDIA__

std::atomic<int> g_thTestPassed(1);

/**
 * Validates negative scenarios for hipModuleGetTexRef
 * texRef = nullptr
 */
bool testTexRefEqNullPtr() {
  bool TestPassed = false;
  hipModule_t Module;
  CTX_CREATE()
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  if (hipSuccess != hipModuleGetTexRef(nullptr, Module, "tex")) {
    TestPassed = true;
  } else {
    printf("Test Failed as texRef = nullptr returns hipSuccess \n");
  }
  CTX_DESTROY()
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetTexRef
 * name = nullptr
 */
bool testNameEqNullPtr() {
  bool TestPassed = false;
  hipModule_t Module;
  HIP_TEX_REFERENCE texref;
  CTX_CREATE()
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  if (hipSuccess != hipModuleGetTexRef(&texref, Module, nullptr)) {
    TestPassed = true;
  } else {
    printf("Test Failed as name = nullptr returns hipSuccess \n");
  }
  CTX_DESTROY()
  return TestPassed;
}
/**
 * Validates negative scenarios for hipModuleGetTexRef
 * name = Non Existing Tex Name
 */
bool testInvalidTexName() {
  bool TestPassed = false;
  hipModule_t Module;
  HIP_TEX_REFERENCE texref;
  CTX_CREATE()
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  if (hipSuccess != hipModuleGetTexRef(&texref, Module,
                                       NON_EXISTING_TEX_NAME)) {
    TestPassed = true;
  } else {
    printf("Test Failed as invalid tex ref returns hipSuccess \n");
  }
  CTX_DESTROY()
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetTexRef
 * name = Empty Tex Name
 */
bool testEmptyTexName() {
  bool TestPassed = false;
  hipModule_t Module;
  HIP_TEX_REFERENCE texref;
  CTX_CREATE()
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  if (hipSuccess != hipModuleGetTexRef(&texref, Module, EMPTY_TEX_NAME)) {
    TestPassed = true;
  } else {
    printf("Test Failed as empty tex ref returns hipSuccess \n");
  }
  CTX_DESTROY()
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetTexRef
 * name = Global Kernel Variable
 */
bool testWrongTexRef() {
  bool TestPassed = false;
  hipModule_t Module;
  HIP_TEX_REFERENCE texref;
  CTX_CREATE()
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  if (hipSuccess != hipModuleGetTexRef(&texref, Module, GLOBAL_KERNEL_VAR)) {
    TestPassed = true;
  } else {
    printf("Test Failed as global tex ref returns hipSuccess \n");
  }
  CTX_DESTROY()
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetTexRef
 * module = unloaded module
 */
bool testUnloadedMod() {
  bool TestPassed = false;
  hipModule_t Module;
  HIP_TEX_REFERENCE texref;
  CTX_CREATE()
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIPCHECK(hipModuleUnload(Module));
  if (hipSuccess != hipModuleGetTexRef(&texref, Module, TEX_REF)) {
    TestPassed = true;
  } else {
    printf("Test Failed as unloaded module returns hipSuccess \n");
  }
  CTX_DESTROY()
  return TestPassed;
}
/**
 * Internal Functions
 *
 */
std::vector<char> load_file() {
  std::ifstream file(CODEOBJ_FILE, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    failed("could not open code object '%s'\n", CODEOBJ_FILE);
  }
  return buffer;
}

template <class T> void fillTestBuffer(unsigned int width,
                                       unsigned int height,
                                       T* hData) {
  if (std::is_same<T, float>::value) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        hData[i * width + j] = i * width + j + 0.5;
      }
    }
  } else if (std::is_same<T, int>::value) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        hData[i * width + j] = i * width + j;
      }
    }
  } else if (std::is_same<T, short>::value) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        hData[i * width + j] = (i * width + j)%
           (std::numeric_limits<short>::max());
      }
    }
  } else if (std::is_same<T, char>::value) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        hData[i * width + j] = (i * width + j)%
           (std::numeric_limits<char>::max());
      }
    }
  }
}

void allocInitArray(unsigned int width,
                     unsigned int height,
                     hipArray_Format format,
                     HIP_ARRAY* array
                     ) {
  HIP_ARRAY_DESCRIPTOR desc;
  desc.Format = format;
  desc.NumChannels = 1;
  desc.Width = width;
  desc.Height = height;
  HIPCHECK(hipArrayCreate(array, &desc));
}

template <class T, class T1> void copyBuffer2Array(unsigned int width,
                                                   unsigned int height,
                                                   T* hData,
                                                   T1 array
                                                   ) {
  hip_Memcpy2D copyParam;
  memset(&copyParam, 0, sizeof(copyParam));
#ifdef __HIP_PLATFORM_NVIDIA__
  copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
  copyParam.dstArray = *array;
#else
  copyParam.dstMemoryType = hipMemoryTypeArray;
  copyParam.srcMemoryType = hipMemoryTypeHost;
  copyParam.dstArray = array;
#endif
  copyParam.srcHost = hData;
  copyParam.srcPitch = width * sizeof(T);
  copyParam.WidthInBytes = copyParam.srcPitch;
  copyParam.Height = height;
  HIPCHECK(hipMemcpyParam2D(&copyParam));
}

template <class T> void assignArray2TexRef(hipArray_Format format,
                                           const char* texRefName,
                                           hipModule_t Module,
                                           T array
                                           ) {
  HIP_TEX_REFERENCE texref;
#ifdef __HIP_PLATFORM_NVIDIA__
  HIPCHECK(hipModuleGetTexRef(&texref, Module, texRefName));
  HIPCHECK(hipTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_WRAP));
  HIPCHECK(hipTexRefSetAddressMode(texref, 1, CU_TR_ADDRESS_MODE_WRAP));
  HIPCHECK(hipTexRefSetFilterMode(texref, HIP_TR_FILTER_MODE_POINT));
  HIPCHECK(hipTexRefSetFlags(texref, CU_TRSF_READ_AS_INTEGER));
  HIPCHECK(hipTexRefSetFormat(texref, format, 1));
  HIPCHECK(hipTexRefSetArray(texref, *array, CU_TRSA_OVERRIDE_FORMAT));
#else
  HIPCHECK(hipModuleGetTexRef(&texref, Module, texRefName));
  HIPCHECK(hipTexRefSetAddressMode(texref, 0, hipAddressModeWrap));
  HIPCHECK(hipTexRefSetAddressMode(texref, 1, hipAddressModeWrap));
  HIPCHECK(hipTexRefSetFilterMode(texref, hipFilterModePoint));
  HIPCHECK(hipTexRefSetFlags(texref, HIP_TRSF_READ_AS_INTEGER));
  HIPCHECK(hipTexRefSetFormat(texref, format, 1));
  HIPCHECK(hipTexRefSetArray(texref, array, HIP_TRSA_OVERRIDE_FORMAT));
#endif
}

template <class T> bool validateOutput(unsigned int width,
                                       unsigned int height,
                                       T* hData,
                                       T* hOutputData) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (hData[i * width + j] != hOutputData[i * width + j]) {
        std::cout << "Difference [ " << i << " " << j << "]:" <<
        (int)hData[i * width + j] << "---" << (int)hOutputData[i * width + j]
        << std::endl;
        return false;
      }
    }
  }
  return true;
}
/**
 * Validates texture type data functionality for hipModuleGetTexRef
 *
 */
template <class T> bool testTexType(hipArray_Format format,
                                    const char* texRefName,
                                    const char* kerFuncName) {
  bool TestPassed = true;
  unsigned int width = WIDTH;
  unsigned int height = HEIGHT;
  unsigned int size = width * height * sizeof(T);
  T* hData = reinterpret_cast<T*>(malloc(size));
  if (NULL == hData) {
    printf("Failed to allocate using malloc in testTexType.\n");
    return false;
  }
  CTX_CREATE()
  fillTestBuffer<T>(width, height, hData);
  // Load Kernel File and create hipArray
  hipModule_t Module;
  HIPCHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIP_ARRAY array;
  allocInitArray(width, height, format, &array);
#ifdef __HIP_PLATFORM_NVIDIA__
  // Copy from hData to array using hipMemcpyParam2D
  copyBuffer2Array<T, HIP_ARRAY*>(width, height, hData, &array);
  // Get tex reference from the loaded kernel file
  // Assign array to the tex reference
  assignArray2TexRef<HIP_ARRAY*>(format, texRefName, Module, &array);
#else
  // Copy from hData to array using hipMemcpyParam2D
  copyBuffer2Array<T, HIP_ARRAY>(width, height, hData, array);
  // Get tex reference from the loaded kernel file
  // Assign array to the tex reference
  assignArray2TexRef<HIP_ARRAY>(format, texRefName, Module, array);
#endif
  hipFunction_t Function;
  HIPCHECK(hipModuleGetFunction(&Function, Module, kerFuncName));

  T* dData = NULL;
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dData), size));

  struct {
    void* _Ad;
    unsigned int _Bd;
    unsigned int _Cd;
  } args;
  args._Ad = reinterpret_cast<void*>(dData);
  args._Bd = width;
  args._Cd = height;

  size_t sizeTemp = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                    &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &sizeTemp,
                    HIP_LAUNCH_PARAM_END};

  int temp1 = width / GRIDDIMX;
  int temp2 = height / GRIDDIMY;
  HIPCHECK(
    hipModuleLaunchKernel(Function, GRIDDIMX, GRIDDIMY, GRIDDIMZ,
                          temp1, temp2, BLOCKDIMZ, 0, 0,
                          NULL, reinterpret_cast<void**>(&config)));
  HIPCHECK(hipDeviceSynchronize());
  T* hOutputData = reinterpret_cast<T*>(malloc(size));
  if (NULL == hOutputData) {
    printf("Failed to allocate using malloc in testTexType.\n");
    TestPassed = false;
  } else {
    memset(hOutputData, 0, size);
    HIPCHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));
    TestPassed = validateOutput<T>(width, height, hData, hOutputData);
  }
  free(hOutputData);
  HIPCHECK(hipFree(dData));
  ARRAY_DESTROY(array)
  HIPCHECK(hipModuleUnload(Module));
  free(hData);
  CTX_DESTROY()
  return TestPassed;
}

/**
 * Validates texture functionality with multiple streams for hipModuleGetTexRef
 *
 */
template <class T> bool testTexMultStream(const std::vector<char>& buffer,
                                        hipArray_Format format,
                                        const char* texRefName,
                                        const char* kerFuncName,
                                        unsigned int numOfStreams) {
  bool TestPassed = true;
  unsigned int width = WIDTH;
  unsigned int height = HEIGHT;
  unsigned int size = width * height * sizeof(T);
  T* hData = reinterpret_cast<T*>(malloc(size));
  if (NULL == hData) {
    printf("Failed to allocate using malloc in testTexMultStream.\n");
    return false;
  }
  CTX_CREATE()
  fillTestBuffer<T>(width, height, hData);

  // Load Kernel File and create hipArray
  hipModule_t Module;
  HIPCHECK(hipModuleLoadData(&Module, &buffer[0]));
  HIP_ARRAY array;
  allocInitArray(width, height, format, &array);
#ifdef __HIP_PLATFORM_NVIDIA__
  // Copy from hData to array using hipMemcpyParam2D
  copyBuffer2Array<T, HIP_ARRAY*>(width, height, hData, &array);
  // Get tex reference from the loaded kernel file
  // Assign array to the tex reference
  assignArray2TexRef<HIP_ARRAY*>(format, texRefName, Module, &array);
#else
  // Copy from hData to array using hipMemcpyParam2D
  copyBuffer2Array<T, HIP_ARRAY>(width, height, hData, array);
  // Get tex reference from the loaded kernel file
  // Assign array to the tex reference
  assignArray2TexRef<HIP_ARRAY>(format, texRefName, Module, array);
#endif
  hipFunction_t Function;
  HIPCHECK(hipModuleGetFunction(&Function, Module, kerFuncName));

  // Create Multiple Strings
  hipStream_t streams[MAX_STREAMS]={0};
  T* dData[MAX_STREAMS] = {NULL};
  T* hOutputData[MAX_STREAMS] = {NULL};
  if (numOfStreams > MAX_STREAMS) {
    numOfStreams = MAX_STREAMS;
  }
  unsigned int totalStreamsCreated = 0;
  for (int stream_num = 0; stream_num < numOfStreams; stream_num++) {
    hOutputData[stream_num] = reinterpret_cast<T*>(malloc(size));
    if (NULL == hOutputData[stream_num]) {
      printf("Failed to allocate using malloc in testTexMultStream.\n");
      TestPassed &= false;
      break;
    }
    HIPCHECK(hipStreamCreate(&streams[stream_num]));
    HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dData[stream_num]), size));
    memset(hOutputData[stream_num], 0, size);
    struct {
      void* _Ad;
      unsigned int _Bd;
      unsigned int _Cd;
    } args;
    args._Ad = reinterpret_cast<void*>(dData[stream_num]);
    args._Bd = width;
    args._Cd = height;

    size_t sizeTemp = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &sizeTemp,
                      HIP_LAUNCH_PARAM_END};

    int temp1 = width / GRIDDIMX;
    int temp2 = height / GRIDDIMY;
    HIPCHECK(
      hipModuleLaunchKernel(Function, GRIDDIMX, GRIDDIMY, GRIDDIMZ,
                          temp1, temp2, BLOCKDIMZ, 0, streams[stream_num],
                          NULL, reinterpret_cast<void**>(&config)));
    totalStreamsCreated++;
  }
  // Check the kernel results separately
  for (int stream_num = 0; stream_num < totalStreamsCreated; stream_num++) {
    HIPCHECK(hipStreamSynchronize(streams[stream_num]));
    HIPCHECK(hipMemcpy(hOutputData[stream_num], dData[stream_num], size,
              hipMemcpyDeviceToHost));
    TestPassed &= validateOutput<T>(width, height, hData,
                                    hOutputData[stream_num]);
  }
  for (int i = 0; i < totalStreamsCreated; i++) {
    HIPCHECK(hipFree(dData[i]));
    HIPCHECK(hipStreamDestroy(streams[i]));
    free(hOutputData[i]);
  }
  ARRAY_DESTROY(array)
  HIPCHECK(hipModuleUnload(Module));
  free(hData);
  CTX_DESTROY()
  return TestPassed;
}

/**
 * Internal Thread Functions
 *
 */
void launchSingleStreamMultGPU(int gpu, const std::vector<char>& buffer) {
  bool TestPassed = true;
  HIPCHECK(hipSetDevice(gpu));
  TestPassed = testTexMultStream<float>(buffer,
                                        HIP_AD_FORMAT_FLOAT,
                                        "ftex",
                                        "tex2dKernelFloat", 1);
  g_thTestPassed &= static_cast<int>(TestPassed);
}

void launchMultStreamMultGPU(int gpu, const std::vector<char>& buffer) {
  bool TestPassed = true;
  HIPCHECK(hipSetDevice(gpu));
  TestPassed = testTexMultStream<float>(buffer,
                                        HIP_AD_FORMAT_FLOAT,
                                        "ftex",
                                        "tex2dKernelFloat", 3);
  g_thTestPassed &= static_cast<int>(TestPassed);
}
/**
 * Validates texture functionality with Multiple Streams on multuple GPU
 * for hipModuleGetTexRef
 *
 */
bool testTexMultStreamMultGPU(unsigned int numOfGPUs,
                              const std::vector<char>& buffer) {
  bool TestPassed = true;
  std::thread T[numOfGPUs];

  for (int gpu = 0; gpu < numOfGPUs; gpu++) {
    T[gpu] = std::thread(launchMultStreamMultGPU, gpu, buffer);
  }
  for (int gpu = 0; gpu < numOfGPUs; gpu++) {
    T[gpu].join();
  }

  if (g_thTestPassed) {
    TestPassed = true;
  } else {
    TestPassed = false;
  }
  return TestPassed;
}
/**
 * Validates texture functionality with Single Stream on multuple GPU
 * for hipModuleGetTexRef
 *
 */
bool testTexSingleStreamMultGPU(unsigned int numOfGPUs,
                                const std::vector<char>& buffer) {
  bool TestPassed = true;
  std::thread T[numOfGPUs];

  for (int gpu = 0; gpu < numOfGPUs; gpu++) {
    T[gpu] = std::thread(launchSingleStreamMultGPU, gpu, buffer);
  }
  for (int gpu = 0; gpu < numOfGPUs; gpu++) {
    T[gpu].join();
  }

  if (g_thTestPassed) {
    TestPassed = true;
  } else {
    TestPassed = false;
  }
  return TestPassed;
}

int main(int argc, char** argv) {
  HipTest::parseStandardArguments(argc, argv, true);
  bool TestPassed = true;
  if (p_tests == 0x01) {
    TestPassed = testTexType<float>(HIP_AD_FORMAT_FLOAT,
                                    "ftex",
                                    "tex2dKernelFloat");
  } else if (p_tests == 0x02) {
    TestPassed = testTexType<int>(HIP_AD_FORMAT_SIGNED_INT32,
                                  "itex",
                                  "tex2dKernelInt");
  } else if (p_tests == 0x03) {
    TestPassed = testTexType<short>(HIP_AD_FORMAT_SIGNED_INT16,
                                    "stex",
                                    "tex2dKernelInt16");
  } else if (p_tests == 0x04) {
    TestPassed = testTexType<char>(HIP_AD_FORMAT_SIGNED_INT8,
                                   "ctex",
                                   "tex2dKernelInt8");
  } else if (p_tests == 0x05) {
    auto buffer = load_file();
    TestPassed = testTexMultStream<float>(buffer,
                                          HIP_AD_FORMAT_FLOAT,
                                          "ftex",
                                          "tex2dKernelFloat",
                                          MAX_STREAMS);
  } else if (p_tests == 0x06) {
    int gpu_cnt = 0;
    auto buffer = load_file();
    HIPCHECK(hipGetDeviceCount(&gpu_cnt));
    TestPassed = testTexSingleStreamMultGPU(gpu_cnt, buffer);
  } else if (p_tests == 0x07) {
    int gpu_cnt = 0;
    auto buffer = load_file();
    HIPCHECK(hipGetDeviceCount(&gpu_cnt));
    TestPassed = testTexMultStreamMultGPU(gpu_cnt, buffer);
  } else if (p_tests == 0x10) {
    TestPassed = testTexRefEqNullPtr();
  } else if (p_tests == 0x11) {
    TestPassed = testNameEqNullPtr();
  } else if (p_tests == 0x12) {
    TestPassed = testInvalidTexName();
  } else if (p_tests == 0x13) {
    TestPassed = testEmptyTexName();
  } else if (p_tests == 0x14) {
    TestPassed = testWrongTexRef();
  } else if (p_tests == 0x15) {
    TestPassed = testUnloadedMod();
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
