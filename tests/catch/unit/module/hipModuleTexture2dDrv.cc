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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
/*
This testcase verifies the following scenarios of hipModuleGetTexRef API
1. Negative
2. Basic functionality using different data types
3. Multiple streams
4. MultiThreaded - MultStreamMultGPU
5. MultiThreaded - SingleStreamMultGPU
*/

#include <fstream>
#include <vector>
#include <type_traits>
#include <limits>
#include <atomic>
#include "hip_test_common.hh"
#include "hip_test_checkers.hh"

#define CODEOBJ_FILE "module_kernels.code"
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
#define MAX_GPU 16

std::atomic<int> g_thTestPassed(1);


/**
 * Internal Functions
 * Loads the kernel file
 */
static std::vector<char> load_file() {
  std::ifstream file(CODEOBJ_FILE, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    INFO("could not open code object " << CODEOBJ_FILE);
    REQUIRE(false);
  }
  return buffer;
}

/*
Initializes the array
*/
template<typename T>
void allocInitArray(unsigned int width,
                     unsigned int height,
                     hipArray_Format format,
                     HIP_ARRAY* array
                     ) {
  HIP_ARRAY_DESCRIPTOR desc;
  desc.Format = format;
  desc.NumChannels = 1;
  desc.Width = width * sizeof(T);
  desc.Height = height;
  HIPCHECK(hipArrayCreate(array, &desc));
}

/*
Copies buffer to array using hipMemcpyParam2D API
*/
template <class T, class T1> void copyBuffer2Array(unsigned int width,
                                                   unsigned int height,
                                                   T* hData,
                                                   T1 array
                                                   ) {
  hip_Memcpy2D copyParam;
  memset(&copyParam, 0, sizeof(copyParam));
#if HT_NVIDIA
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
  copyParam.WidthInBytes = width * sizeof(T);
  copyParam.Height = height;
  HIPCHECK(hipMemcpyParam2D(&copyParam));
}

/*
Assigns array to texture ref
*/
template <class T> void assignArray2TexRef(hipArray_Format format,
                                           const char* texRefName,
                                           hipModule_t Module,
                                           T array
                                           ) {
  HIP_TEX_REFERENCE texref;
#if HT_NVIDIA
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
  for (unsigned int i = 0; i < height; i++) {
    for (unsigned int j = 0; j < width; j++) {
      if (hData[i * width + j] != hOutputData[i * width + j]) {
        return false;
      }
    }
  }
  return true;
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
  CTX_CREATE()
  HipTest::setDefaultData<T>(width * height, hData, nullptr, nullptr);

  // Load Kernel File and create hipArray
  hipModule_t Module;
  HIPCHECK(hipModuleLoadData(&Module, &buffer[0]));
  HIP_ARRAY array;
  allocInitArray<T>(width, height, format, &array);
#if HT_NVIDIA
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
  for (unsigned int stream_num = 0; stream_num < numOfStreams; stream_num++) {
    hOutputData[stream_num] = reinterpret_cast<T*>(malloc(size));
    if (NULL == hOutputData[stream_num]) {
      WARN("Failed to allocate using malloc in testTexMultStream");
      TestPassed &= false;
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
    HIPCHECK(hipModuleLaunchKernel(Function, GRIDDIMX, GRIDDIMY, GRIDDIMZ,
                                   temp1, temp2, BLOCKDIMZ, 0,
                                   streams[stream_num],
                                   NULL, reinterpret_cast<void**>(&config)));
    totalStreamsCreated++;
  }
  // Check the kernel results separately
  for (unsigned int stream_num = 0; stream_num < totalStreamsCreated;
       stream_num++) {
    HIPCHECK(hipStreamSynchronize(streams[stream_num]));
    HIPCHECK(hipMemcpy(hOutputData[stream_num], dData[stream_num], size,
                            hipMemcpyDeviceToHost));
    TestPassed &= validateOutput<T>(width, height, hData,
                                    hOutputData[stream_num]);
  }
  for (unsigned int i = 0; i < totalStreamsCreated; i++) {
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
  std::thread T[MAX_GPU];

  for (unsigned int gpu = 0; gpu < numOfGPUs; gpu++) {
    T[gpu] = std::thread(launchMultStreamMultGPU, gpu, buffer);
  }
  for (unsigned int gpu = 0; gpu < numOfGPUs; gpu++) {
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
  std::thread T[MAX_GPU];

  for (unsigned int gpu = 0; gpu < numOfGPUs; gpu++) {
    T[gpu] = std::thread(launchSingleStreamMultGPU, gpu, buffer);
  }
  for (unsigned int gpu = 0; gpu < numOfGPUs; gpu++) {
    T[gpu].join();
  }

  if (g_thTestPassed) {
    TestPassed = true;
  } else {
    TestPassed = false;
  }
  return TestPassed;
}

/*
This testcase verifies the negative scenarios of hipModuleGetTexRef API
*/
TEST_CASE("Unit_hipModuleGetTexRef_Negative") {
  hipModule_t Module;
  HIP_TEX_REFERENCE texref;
  CTX_CREATE()
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));

  SECTION("TexRef as nullptr") {
    REQUIRE(hipModuleGetTexRef(nullptr, Module, "tex") != hipSuccess);
  }

  SECTION("Name as nullptr") {
    REQUIRE(hipModuleGetTexRef(&texref, Module, nullptr) != hipSuccess);
  }

  SECTION("Name as non existing TexName") {
    REQUIRE(hipModuleGetTexRef(&texref, Module,
                               NON_EXISTING_TEX_NAME) != hipSuccess);
  }

  SECTION("Empty tex name") {
    REQUIRE(hipModuleGetTexRef(&texref, Module, EMPTY_TEX_NAME) != hipSuccess);
  }
#if HT_NVIDIA
  SECTION("Name as Global kernel Var") {
    REQUIRE(hipModuleGetTexRef(&texref, Module,
                               GLOBAL_KERNEL_VAR) != hipSuccess);
  }
#endif

  SECTION("Unload Module") {
    HIP_CHECK(hipModuleUnload(Module));
    REQUIRE(hipModuleGetTexRef(&texref, Module, TEX_REF) != hipSuccess);
  }

  CTX_DESTROY()
}
/**
 * Validates texture type data functionality for hipModuleGetTexRef
 * 1.Loads the code object file
 * 2.Based on the template type texRefName,KernelFuncName and format are assigned.
 * 3.Allocate array based on format.
 * 4.Assigns array to texRef
 * 5.Launches the kernel based on the template type which invokes text2D API
     and copies the data to output variable.
 * 6.Validates the data.
 */
TEMPLATE_TEST_CASE("Unit_hipModuleGetTexRef_Basic", "", int,
                   char, uint16_t, float) {
  bool TestPassed = true;
  constexpr unsigned int width = WIDTH;
  constexpr unsigned int height = HEIGHT;
  constexpr unsigned int size = width * height * sizeof(TestType);
  const char *texRefName, *kerFuncName;
  hipArray_Format format;

  TestType* hData = reinterpret_cast<TestType*>(malloc(size));
  if (NULL == hData) {
    INFO("Failed to allocate using malloc in testTexType.\n");
    REQUIRE(false);
  }
  CTX_CREATE()
  HipTest::setDefaultData<TestType>(width * height, hData, nullptr, nullptr);
  // Load Kernel File and create hipArray
  hipModule_t Module;
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIP_ARRAY array;

  if (std::is_same<TestType, char>::value) {
    texRefName = "ctex";
    kerFuncName = "tex2dKernelInt8";
    format = HIP_AD_FORMAT_SIGNED_INT8;
  } else if (std::is_same<TestType, uint16_t>::value)  {
    texRefName = "stex";
    kerFuncName = "tex2dKernelInt16";
    format = HIP_AD_FORMAT_SIGNED_INT16;
  } else if (std::is_same<TestType, int>::value)  {
    texRefName = "itex";
    kerFuncName = "tex2dKernelInt";
    format = HIP_AD_FORMAT_SIGNED_INT32;
  } else if (std::is_same<TestType, float>::value)  {
    texRefName = "ftex";
    kerFuncName = "tex2dKernelFloat";
    format = HIP_AD_FORMAT_FLOAT;
  }
  allocInitArray<TestType>(width, height, format, &array);

#if HT_NVIDIA
  // Copy from hData to array using hipMemcpyParam2D
  copyBuffer2Array<TestType, HIP_ARRAY*>(width, height, hData, &array);
  // Get tex reference from the loaded kernel file
  // Assign array to the tex reference
  assignArray2TexRef<HIP_ARRAY*>(format, texRefName, Module, &array);
#else
  // Copy from hData to array using hipMemcpyParam2D
  copyBuffer2Array<TestType, HIP_ARRAY>(width, height, hData, array);
  // Get tex reference from the loaded kernel file
  // Assign array to the tex reference
  assignArray2TexRef<HIP_ARRAY>(format, texRefName, Module, array);
#endif
  hipFunction_t Function;
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kerFuncName));

  TestType* dData = NULL;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dData), size));

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
  HIP_CHECK(
    hipModuleLaunchKernel(Function, GRIDDIMX, GRIDDIMY, GRIDDIMZ,
                          temp1, temp2, BLOCKDIMZ, 0, 0,
                          NULL, reinterpret_cast<void**>(&config)));
  HIP_CHECK(hipDeviceSynchronize());
  TestType* hOutputData = reinterpret_cast<TestType*>(malloc(size));
  if (NULL == hOutputData) {
    INFO("Failed to allocate using malloc in testTexType");
    REQUIRE(false);
  } else {
    memset(hOutputData, 0, size);
    HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));
    TestPassed = validateOutput<TestType>(width, height, hData, hOutputData);
    REQUIRE(TestPassed);
  }
  free(hOutputData);
  HIP_CHECK(hipFree(dData));
  ARRAY_DESTROY(array)
  HIP_CHECK(hipModuleUnload(Module));
  free(hData);
  CTX_DESTROY()
}

/*
This testcase verifies hipModuleGetTexRef on multiple streams
where
 * 1..Loads the code object file
 * 2.Allocate array and initializes it with hData
 * 3.Assigns array to texRef
   4.Creates multiple streams
 * 4.Launches the kernel on each stream which invokes text2D API
     and copies the data to output variable
 * 5.Validates the hData with output data in each stream.
*/
TEST_CASE("Unit_hipModuleGetTexRef_TexMultStream") {
  bool TestPassed = true;
  auto buffer = load_file();
  TestPassed = testTexMultStream<float>(buffer,
                                        HIP_AD_FORMAT_FLOAT,
                                        "ftex",
                                        "tex2dKernelFloat",
                                        MAX_STREAMS);
  REQUIRE(TestPassed);
}
/*
This testcase verifies hipModuleGetTexRef Multithreaded scenario on
single stream and multi GPU machine.
1. Gets the device count.
2. Create the threads based on device count.
3. Each thread calls the testTexMultStream which performs the same
   above funtionality on single Stream
4. The threads are executed in parallel and are joined later.

This testcase ensures that the multi thread execution on single stream
in parallel is successful
*/
TEST_CASE("Unit_hipModuleGetTexRef_MultiThreadTexSingleStreamMultiGPU") {
  bool TestPassed = true;
  // Testcase skipped on nvidia with CUDA API version 11.2,
  // as hipModuleLoadData returning error code
  // 'a PTX JIT compilation failed'(218), which is invalid
  // behavior. Test passes with AMD and previous CUDA versions.
#if HT_NVIDIA
  INFO("Testcase skipped on CUDA version 11.2\n");
  REQUIRE(true);
#else
  int gpu_cnt = 0;
  auto buffer = load_file();
  HIP_CHECK(hipGetDeviceCount(&gpu_cnt));
  TestPassed = testTexSingleStreamMultGPU(gpu_cnt, buffer);
  REQUIRE(TestPassed);
#endif
}


/*
This testcase verifies hipModuleGetTexRef Multithreaded scenario on
single stream and multi GPU machine.
1. Gets the device count.
2. Create the threads based on device count.
3. Each thread calls the testTexMultStream which performs the same
   above funtionality on multiple Stream
4. The threads are executed in parallel and are joined later.

This testcase ensures that the multi thread execution on multiple streams
in parallel is successful
*/
TEST_CASE("Unit_hipModuleGetTexRef_MultiThreadTexMultiStreamMultiGPU") {
  bool TestPassed = true;
  // Testcase skipped on nvidia with CUDA API version 11.2,
  // as hipModuleLoadData returning error code
  // 'a PTX JIT compilation failed'(218), which is invalid
  // behavior. Test passes with AMD and previous CUDA versions.
#if HT_NVIDIA
  INFO("Testcase skipped on CUDA version 11.2\n");
  REQUIRE(true);
#else
  int gpu_cnt = 0;
  auto buffer = load_file();
  HIP_CHECK(hipGetDeviceCount(&gpu_cnt));
  TestPassed = testTexMultStreamMultGPU(gpu_cnt, buffer);
  REQUIRE(TestPassed);
#endif
}
