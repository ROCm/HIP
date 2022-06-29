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
 * Test Scenarios
 * 1. Verifying hipDrvMemcpy3DAsync API for H2A,A2A,A2H scenarios
 * 2. Verifying hipDrvMemcpy3DAsync API for H2D,D2D,D2H scenarios
 * 3. Verifying Negative Scenarios
 * 4. Verifying Extent validation scenarios by passing 0
 * 5. Verifying hipDrvMemcpy3DAsync API by allocating Memory in
 *    one GPU and trigger hipDrvMemcpy3DAsync from peer GPU for
 *    H2D,D2D,D2H scenarios
 * 6. Verifying hipDrvMemcpy3DAsync API by allocating Memory in
 *    one GPU and trigger hipDrvMemcpy3DAsync from peer GPU for
 *    H2A,A2A,A2H scenarios
 *
 *    Scenarios 3 is temporarily excluded in AMD platform
 *    Scenario 5&6 are excluded in CUDA platform
 */

#include "hip_test_common.hh"
#include "hip_test_checkers.hh"

template<typename T>
class DrvMemcpy3DAsync {
  int width, height, depth;
  unsigned int size;
  hipArray_Format formatKind;
  hiparray arr, arr1;
  hipStream_t stream;
  size_t pitch_D, pitch_E;
  HIP_MEMCPY3D myparms;
  hipDeviceptr_t D_m, E_m;
  T* hData{nullptr};
 public:
  DrvMemcpy3DAsync(int l_width, int l_height, int l_depth,
      hipArray_Format l_format);
  DrvMemcpy3DAsync() = delete;
  void AllocateMemory();
  void SetDefaultData();
  void HostArray_DrvMemcpy3DAsync(bool device_context_change = false);
  void HostDevice_DrvMemcpy3DAsync(bool device_context_change = false);
  void Extent_Validation();
  void NegativeTests();
  void DeAllocateMemory();
};

/* Intializes class variables */
template <typename T>
DrvMemcpy3DAsync<T>::DrvMemcpy3DAsync(int l_width, int l_height, int l_depth,
    hipArray_Format l_format) {
  width = l_width;
  height = l_height;
  depth = l_depth;
  formatKind = l_format;
}

/* Allocating Memory */
template <typename T>
void DrvMemcpy3DAsync<T>::AllocateMemory() {
  size = width * height * depth * sizeof(T);
  hData = reinterpret_cast<T*>(malloc(size));
  memset(hData, 0, size);
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        hData[i*width*height + j*width +k] = i*width*height + j*width + k;
      }
    }
  }
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&D_m),
                          &pitch_D, width*sizeof(T), height));
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&E_m),
                          &pitch_E, width*sizeof(T), height));
  HIP_ARRAY3D_DESCRIPTOR *desc;
  desc = reinterpret_cast<HIP_ARRAY3D_DESCRIPTOR*>
         (malloc(sizeof(HIP_ARRAY3D_DESCRIPTOR)));
  desc->Format = formatKind;
  desc->NumChannels = 1;
  desc->Width = width;
  desc->Height = height;
  desc->Depth = depth;
  desc->Flags = hipArrayDefault;
  HIP_CHECK(hipArray3DCreate(&arr, desc));
  HIP_CHECK(hipArray3DCreate(&arr1, desc));
}

/* Setting the default data */
template <typename T>
void DrvMemcpy3DAsync<T>::SetDefaultData() {
  memset(&myparms, 0x0, sizeof(HIP_MEMCPY3D));
  myparms.srcXInBytes = 0;
  myparms.srcY = 0;
  myparms.srcZ = 0;
  myparms.srcLOD = 0;
  myparms.dstXInBytes = 0;
  myparms.dstY = 0;
  myparms.dstZ = 0;
  myparms.dstLOD = 0;
  myparms.WidthInBytes = width*sizeof(T);
  myparms.Height = height;
  myparms.Depth = depth;
}

/*
This function verifies the negative scenarios of
hipDrvMemcpy3DAsync API
*/
template <typename T>
void DrvMemcpy3DAsync<T>::NegativeTests() {
  HIP_CHECK(hipSetDevice(0));
  AllocateMemory();
  SetDefaultData();
  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  unsigned int MaxPitch;
  HIP_CHECK(hipDeviceGetAttribute(reinterpret_cast<int *>(&MaxPitch),
                                  hipDeviceAttributeMaxPitch, deviceId));
  myparms.srcHost = hData;
  myparms.dstArray = arr;
  myparms.srcPitch = width * sizeof(T);
  myparms.srcHeight = height;
#if HT_NVIDIA
  myparms.srcMemoryType = CU_MEMORYTYPE_HOST;
  myparms.dstMemoryType = CU_MEMORYTYPE_ARRAY;
#else
  myparms.srcMemoryType = hipMemoryTypeHost;
  myparms.dstMemoryType = hipMemoryTypeArray;
#endif

  SECTION("Passing nullptr to Source Host") {
    myparms.srcHost = nullptr;
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("Passing both dst host and device") {
    myparms.dstHost = hData;
    myparms.dstArray = nullptr;
    myparms.dstDevice = D_m;
    myparms.WidthInBytes = pitch_D;
#if HT_NVIDIA
    myparms.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
    myparms.dstMemoryType = hipMemoryTypeDevice;
#endif
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("Passing max value to WidthInBytes") {
    myparms.WidthInBytes = std::numeric_limits<int>::max();
    myparms.Height = std::numeric_limits<int>::max();
    myparms.Depth = std::numeric_limits<int>::max();
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("Passing width > max width size") {
    myparms.WidthInBytes = width*sizeof(T) + 1;
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("Passing height > max height size") {
    myparms.Height = height + 1;
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("Passing depth > max depth size") {
    myparms.Depth = depth + 1;
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("widthinbytes + srcXinBytes is out of bound") {
    myparms.srcXInBytes = 1;
    myparms.dstArray = nullptr;
    myparms.dstDevice = hipDeviceptr_t(D_m);
    myparms.dstPitch = pitch_D;
    myparms.dstHeight = height;
#if HT_NVIDIA
    myparms.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
    myparms.dstMemoryType = hipMemoryTypeDevice;
#endif
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("widthinbytes + dstXinBytes is out of bound") {
    myparms.dstXInBytes = pitch_D;
    myparms.dstArray = nullptr;
    myparms.dstDevice = hipDeviceptr_t(D_m);
    myparms.dstPitch = pitch_D;
    myparms.dstHeight = height;
#if HT_NVIDIA
    myparms.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
    myparms.dstMemoryType = hipMemoryTypeDevice;
#endif
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("srcY + height is out of bound") {
    myparms.srcY = 1;
    myparms.dstArray = nullptr;
    myparms.dstDevice = hipDeviceptr_t(D_m);
    myparms.dstPitch = pitch_D;
    myparms.dstHeight = height;
#if HT_NVIDIA
    myparms.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
    myparms.dstMemoryType = hipMemoryTypeDevice;
#endif
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("dstY + height out of bounds") {
    myparms.dstY = 1;
    myparms.dstArray = nullptr;
    myparms.dstDevice = hipDeviceptr_t(D_m);
    myparms.dstPitch = pitch_D;
    myparms.dstHeight = height;
#if HT_NVIDIA
    myparms.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
    myparms.dstMemoryType = hipMemoryTypeDevice;
#endif
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("src pitch greater than Max allowed pitch") {
#if HT_NVIDIA
    myparms.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    myparms.dstMemoryType = CU_MEMORYTYPE_HOST;
#else
    myparms.srcMemoryType = hipMemoryTypeDevice;
    myparms.dstMemoryType = hipMemoryTypeHost;
#endif
    myparms.srcDevice = D_m;
    myparms.srcHost = nullptr;
    myparms.srcPitch = MaxPitch;
    myparms.srcHeight = height;
    myparms.dstHost = hData;
    myparms.dstArray = nullptr;
    myparms.dstPitch = width*sizeof(T);
    myparms.dstHeight = height;
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("dst pitch greater than Max allowed pitch") {
    myparms.dstDevice = hipDeviceptr_t(D_m);
    myparms.dstArray = nullptr;
    myparms.dstPitch = MaxPitch+1;
    myparms.dstHeight = height;
#if HT_NVIDIA
    myparms.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
    myparms.dstMemoryType = hipMemoryTypeDevice;
#endif
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("Nullptr to src/dst device") {
    myparms.dstDevice = hipDeviceptr_t(nullptr);
    myparms.dstArray = nullptr;
    myparms.dstPitch = pitch_D;
    myparms.dstHeight = height;
#if HT_NVIDIA
    myparms.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
    myparms.dstMemoryType = hipMemoryTypeDevice;
#endif
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("Nullptr to src/dst array") {
    myparms.dstArray = nullptr;
    REQUIRE(hipDrvMemcpy3DAsync(&myparms, stream) != hipSuccess);
  }

  SECTION("Nullptr to hipDrvMemcpy3DAsync") {
    REQUIRE(hipDrvMemcpy3DAsync(nullptr, stream) != hipSuccess);
  }

  DeAllocateMemory();
}
/*
This function verifies the Extent validation scenarios of
hipDrvMemcpy3DAsync API
*/
template <typename T>
void DrvMemcpy3DAsync<T>::Extent_Validation() {
  HIP_CHECK(hipSetDevice(0));
  // Allocating the memory
  AllocateMemory();

  // Setting default data
  SetDefaultData();
#if HT_NVIDIA
  myparms.srcMemoryType = CU_MEMORYTYPE_HOST;
  myparms.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
  myparms.srcMemoryType = hipMemoryTypeHost;
  myparms.dstMemoryType = hipMemoryTypeDevice;
#endif
  myparms.srcHost = hData;
  myparms.srcPitch = width * sizeof(T);
  myparms.srcHeight = height;
  myparms.dstDevice = D_m;
  myparms.dstPitch = pitch_D;
  myparms.dstHeight = height;

  SECTION("WidthInBytes is 0") {
    myparms.WidthInBytes = 0;
    HIP_CHECK(hipDrvMemcpy3DAsync(&myparms, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  SECTION("Height is 0") {
    myparms.Height = 0;
    HIP_CHECK(hipDrvMemcpy3DAsync(&myparms, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  SECTION("Depth is 0") {
    myparms.Depth = 0;
    HIP_CHECK(hipDrvMemcpy3DAsync(&myparms, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  DeAllocateMemory();
}
/*
This Function verifies following functionalities of hipDrvMemcpy3DAsync API
1. Host to Device copy
2. Device to Device
3. Device to Host
In the end  validates the results.

This functionality is verified in 2 scenarios
1. Basic scenario on same GPU device
2. Device context change scenario where memory is allocated in 1 GPU
   and hipDrvMemcpy3DAsync API is trigerred from another GPU
*/
template <typename T>
void DrvMemcpy3DAsync<T>::HostDevice_DrvMemcpy3DAsync
     (bool device_context_change) {
  HIP_CHECK(hipSetDevice(0));
  bool skip_test = false;
  int peerAccess = 0;
  AllocateMemory();
  if (device_context_change) {
    HIP_CHECK(hipDeviceCanAccessPeer(&peerAccess, 0, 1));
    if (!peerAccess) {
      WARN("skipped the testcase as no peer access");
      skip_test = true;
    } else {
      HIP_CHECK(hipSetDevice(1));
    }
  }
  if (!skip_test) {
    SetDefaultData();
#if HT_NVIDIA
    myparms.srcMemoryType = CU_MEMORYTYPE_HOST;
    myparms.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
    myparms.srcMemoryType = hipMemoryTypeHost;
    myparms.dstMemoryType = hipMemoryTypeDevice;
#endif
    myparms.srcHost = hData;
    myparms.srcPitch = width * sizeof(T);
    myparms.srcHeight = height;
    myparms.dstDevice = hipDeviceptr_t(D_m);
    myparms.dstPitch = pitch_D;
    myparms.dstHeight = height;
    HIP_CHECK(hipDrvMemcpy3DAsync(&myparms, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Device to Device
    SetDefaultData();
#if HT_NVIDIA
    myparms.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    myparms.dstMemoryType = CU_MEMORYTYPE_DEVICE;
#else
    myparms.srcMemoryType = hipMemoryTypeDevice;
    myparms.dstMemoryType = hipMemoryTypeDevice;
#endif
    myparms.srcDevice = hipDeviceptr_t(D_m);
    myparms.srcPitch = pitch_D;
    myparms.srcHeight = height;
    myparms.dstDevice = hipDeviceptr_t(E_m);
    myparms.dstPitch = pitch_E;
    myparms.dstHeight = height;
    HIP_CHECK(hipDrvMemcpy3DAsync(&myparms, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    T *hOutputData = reinterpret_cast<T*>(malloc(size));
    memset(hOutputData, 0,  size);

    // Device to host
    SetDefaultData();
#if HT_NVIDIA
    myparms.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    myparms.dstMemoryType = CU_MEMORYTYPE_HOST;
#else
    myparms.srcMemoryType = hipMemoryTypeDevice;
    myparms.dstMemoryType = hipMemoryTypeHost;
#endif
    myparms.srcDevice = hipDeviceptr_t(E_m);
    myparms.srcPitch = pitch_E;
    myparms.srcHeight = height;
    myparms.dstHost = hOutputData;
    myparms.dstPitch = width * sizeof(T);
    myparms.dstHeight = height;
    HIP_CHECK(hipDrvMemcpy3DAsync(&myparms, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HipTest::checkArray(hData, hOutputData, width, height, depth);
    free(hOutputData);
  }
  DeAllocateMemory();
}

/*
This Function verifies following functionalities of hipDrvMemcpy3DAsync API
1. Host to Array copy
2. Array to Array
3. Array to Host
In the end  validates the results.

This functionality is verified in 2 scenarios
1. Basic scenario on same GPU device
2. Device context change scenario where memory is allocated in 1 GPU
   and hipDrvMemcpy3DAsync API is trigerred from another GPU
*/
template <typename T>
void DrvMemcpy3DAsync<T>::HostArray_DrvMemcpy3DAsync
                          (bool device_context_change) {
  HIP_CHECK(hipSetDevice(0));
  bool skip_test = false;
  int peerAccess = 0;
  AllocateMemory();
  if (device_context_change) {
    HIP_CHECK(hipDeviceCanAccessPeer(&peerAccess, 0, 1));
    if (!peerAccess) {
      WARN("skipped the testcase as no peer access");
      skip_test = true;
    } else {
      HIP_CHECK(hipSetDevice(1));
    }
  }
  if (!skip_test) {
    SetDefaultData();
#if HT_NVIDIA
    myparms.srcMemoryType = CU_MEMORYTYPE_HOST;
    myparms.dstMemoryType = CU_MEMORYTYPE_ARRAY;
#else
    myparms.srcMemoryType = hipMemoryTypeHost;
    myparms.dstMemoryType = hipMemoryTypeArray;
#endif
    myparms.srcHost = hData;
    myparms.srcPitch = width * sizeof(T);
    myparms.srcHeight = height;
    myparms.dstArray = arr;
    HIP_CHECK(hipDrvMemcpy3DAsync(&myparms, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    // Array to Array
    SetDefaultData();
#if HT_NVIDIA
    myparms.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    myparms.dstMemoryType = CU_MEMORYTYPE_ARRAY;
#else
    myparms.srcMemoryType = hipMemoryTypeArray;
    myparms.dstMemoryType = hipMemoryTypeArray;
#endif
    myparms.srcArray = arr;
    myparms.dstArray = arr1;
    HIP_CHECK(hipDrvMemcpy3DAsync(&myparms, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    T *hOutputData = reinterpret_cast<T*>(malloc(size));
    memset(hOutputData, 0,  size);
    SetDefaultData();
    // Device to host
#if HT_NVIDIA
    myparms.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    myparms.dstMemoryType = CU_MEMORYTYPE_HOST;
#else
    myparms.srcMemoryType = hipMemoryTypeArray;
    myparms.dstMemoryType = hipMemoryTypeHost;
#endif
    myparms.srcArray = arr1;
    myparms.dstHost = hOutputData;
    myparms.dstPitch = width * sizeof(T);
    myparms.dstHeight = height;
    HIP_CHECK(hipDrvMemcpy3DAsync(&myparms, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HipTest::checkArray(hData, hOutputData, width, height, depth);
    free(hOutputData);
  }
  DeAllocateMemory();
}

/* DeAllocating the memory */
template <typename T>
void DrvMemcpy3DAsync<T>::DeAllocateMemory() {
  HIP_CHECK(hipArrayDestroy(arr));
  HIP_CHECK(hipArrayDestroy(arr1));
  HIP_CHECK(hipStreamDestroy(stream));
  free(hData);
}

/* Verifying hipDrvMemcpy3DAsync API Host to Array for different datatypes */
TEMPLATE_TEST_CASE("Unit_hipDrvMemcpy3DAsync_MultipleDataTypes", "",
                   uint8_t, int, float) {
  for (int i = 1; i < 25; i++) {
    if (std::is_same<TestType, float>::value) {
      DrvMemcpy3DAsync<TestType> memcpy3d_float(i, i, i,
          HIP_AD_FORMAT_FLOAT);
      memcpy3d_float.HostArray_DrvMemcpy3DAsync();
    } else if (std::is_same<TestType, uint8_t>::value)  {
      DrvMemcpy3DAsync<TestType> memcpy3d_intx(i, i, i,
          HIP_AD_FORMAT_UNSIGNED_INT8);
      memcpy3d_intx.HostArray_DrvMemcpy3DAsync();
    } else if (std::is_same<TestType, int>::value)  {
      DrvMemcpy3DAsync<TestType> memcpy3d_inty(i, i, i,
          HIP_AD_FORMAT_SIGNED_INT32);
      memcpy3d_inty.HostArray_DrvMemcpy3DAsync();
    }
  }
}

/* This testcase verifies H2D copy of hipDrvMemcpy3DAsync API */
TEST_CASE("Unit_hipDrvMemcpy3DAsync_HosttoDevice") {
  DrvMemcpy3DAsync<float> memcpy3d_D2H_float(10, 10, 1, HIP_AD_FORMAT_FLOAT);
  memcpy3d_D2H_float.HostDevice_DrvMemcpy3DAsync();
}

/* This testcase verifies negative scenarios of hipDrvMemcpy3DAsync API */
#if HT_NVIDIA
TEST_CASE("Unit_hipDrvMemcpy3DAsync_Negative") {
  DrvMemcpy3DAsync<float> memcpy3d(10, 10, 1, HIP_AD_FORMAT_FLOAT);
  memcpy3d.NegativeTests();
}
#endif

/* This testcase verifies extent validation scenarios of
   hipDrvMemcpy3DAsync API */
TEST_CASE("Unit_hipDrvMemcpy3DAsync_ExtentValidation") {
  DrvMemcpy3DAsync<float> memcpy3d(10, 10, 1, HIP_AD_FORMAT_FLOAT);
  memcpy3d.Extent_Validation();
}

/* This testcase verifies H2D copy in device context
change scenario for hipDrvMemcpy3DAsync API */
#if HT_AMD
TEST_CASE("Unit_hipDrvMemcpy3DAsync_H2DDeviceContextChange") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    DrvMemcpy3DAsync<float> memcpy3d(10, 10, 1, HIP_AD_FORMAT_FLOAT);
    memcpy3d.HostDevice_DrvMemcpy3DAsync(true);
  } else {
    SUCCEED("skipped testcase as Device count is < 2");
  }
}


/* This testcase verifies Host to Array copy in device context
change scenario for hipDrvMemcpy3DAsync API */
TEST_CASE("Unit_hipDrvMemcpy3DAsync_Host2ArrayDeviceContextChange") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    DrvMemcpy3DAsync<float> memcpy3d(10, 10, 10, HIP_AD_FORMAT_FLOAT);
    memcpy3d.HostArray_DrvMemcpy3DAsync(true);
  } else {
    SUCCEED("skipped testcase as Device count is < 2");
  }
}
#endif


