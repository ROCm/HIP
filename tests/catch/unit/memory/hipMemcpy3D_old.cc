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

/*
 * This testfile verifies the following scenarios of hipMemcpy3D API
 *
 * 1. Verifying hipMemcpy3D API for H2D,D2D and D2H scenarios for
      different datatypes and sizes.
 * 2. Verifying Negative Scenarios
 * 3. Verifying Extent validation scenarios by passing 0
 * 4. Verifying hipMemcpy3D API by allocating Memory in
 *    one GPU and trigger hipMemcpy3D from peer GPU
 *
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

static constexpr auto width{10};
static constexpr auto height{10};
static constexpr auto depth{10};

template <typename T>
class Memcpy3D {
  int width, height, depth;
  unsigned int size;
  hipArray *arr, *arr1;
  hipChannelFormatKind formatKind;
  hipMemcpy3DParms myparms;
  T* hData;
 public:
  Memcpy3D(int l_width, int l_height, int l_depth,
      hipChannelFormatKind l_format);
  void simple_Memcpy3D();
  void Extent_Validation();
  void NegativeTests();
  void AllocateMemory();
  void DeAllocateMemory();
  void SetDefaultData();
  void D2D_DeviceMem_OnDiffDevice();
  void D2H_H2D_DeviceMem_OnDiffDevice();
};

/*
 * This API sets the default values of hipMemcpy3DParms structure
 */
template <typename T>
void Memcpy3D<T>::SetDefaultData() {
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.extent = make_hipExtent(width , height, depth);
}

/*
 * Constructor initalized width,depth and height
 */
template <typename T>
Memcpy3D<T>::Memcpy3D(int l_width, int l_height, int l_depth,
                      hipChannelFormatKind l_format) {
  width = l_width;
  height = l_height;
  depth = l_depth;
  formatKind = l_format;
}

/*
 * Allocating Memory and initalizing data for both
 * device and host variables
 */
template <typename T>
void Memcpy3D<T>::AllocateMemory() {
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
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(T)*8,
                                                          0, 0, 0, formatKind);
  HIP_CHECK(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width, height,
                                                depth), hipArrayDefault));
  HIP_CHECK(hipMalloc3DArray(&arr1, &channelDesc, make_hipExtent(width, height,
                                                 depth), hipArrayDefault));
}

/*
 * DeAllocates the Memory of device and host variables
 */
template <typename T>
void Memcpy3D<T>::DeAllocateMemory() {
    HIP_CHECK(hipFreeArray(arr));
    HIP_CHECK(hipFreeArray(arr1));
    free(hData);
}

/*
 * This API verifies both H2D & D2H functionalities of hipMemcpy3D API
 * by allocating memory in one GPU and calling the hipMemcpy3D API
 * from another GPU.
 * H2D case:
 * Input : "hData" is  initialized with the respective offset value
 * Output: Destination array "arr" variable.
 *
 * D2H case:
 * Input: "arr" array variable from the above output
 * Output: "hOutputData" variable data is copied from "arr" variable
 *
 * Validating the result by comparing "hData" and "hOutputData" variables
 */
template <typename T>
void Memcpy3D<T>::D2H_H2D_DeviceMem_OnDiffDevice() {
  HIP_CHECK(hipSetDevice(0));
  int peerAccess = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&peerAccess, 1, 0));
  if (peerAccess) {
    AllocateMemory();
    // Memory is allocated on device 0 and Memcpy3DAsync triggered from device 1
    HIP_CHECK(hipSetDevice(1));

    // H2D Scenario
    memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
    SetDefaultData();
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                        width, height);
    myparms.dstArray = arr;
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyHostToDevice;
#else
    myparms.kind = hipMemcpyHostToDevice;
#endif
    REQUIRE(hipMemcpy3D(&myparms) == hipSuccess);
    HIP_CHECK(hipDeviceSynchronize());

    // Device to host
    memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
    T *hOutputData = reinterpret_cast<T*>(malloc(size));
    memset(hOutputData, 0,  size);
    SetDefaultData();
    myparms.dstPtr = make_hipPitchedPtr(hOutputData,
                                        width * sizeof(T),
                                        width, height);
    myparms.srcArray = arr;
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyDeviceToHost;
#else
    myparms.kind = hipMemcpyDeviceToHost;
#endif
    REQUIRE(hipMemcpy3D(&myparms) == hipSuccess);
    HIP_CHECK(hipDeviceSynchronize());

    // Validating the result
    HipTest::checkArray(hData, hOutputData, width, height, depth);
    free(hOutputData);
    DeAllocateMemory();
  } else {
    SUCCEED("Skipped the test as there is no peer access\n");
  }
}
/*
 * This API verifies both D2D functionalities of hipMemcpy3D API
 * by allocating memory in one GPU and calling the hipMemcpy3D API
 * from another GPU.
 *
 * D2D case:
 * Input : "arr" variable  is  initialized with the "hData" variable in GPU-0
 * Output: "arr2" variable in GPU-0
 *
 * hipMemcpy3D API is triggered from GPU-1
 * The "arr2" variable is then copied to "hOutputData" for validating 
 * the result
 *
 * Validating the result by comparing "hData" and "hOutputData" variables
 */
template <typename T>
void Memcpy3D<T>::D2D_DeviceMem_OnDiffDevice() {
  HIP_CHECK(hipSetDevice(0));
  int peerAccess = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&peerAccess, 0, 1));
  if (peerAccess) {
    AllocateMemory();
    memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
    SetDefaultData();

    // Host to device copy
    myparms.srcPtr = make_hipPitchedPtr(hData,
        width * sizeof(T),
        width, height);
    myparms.dstArray = arr;
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyHostToDevice;
#else
    myparms.kind = hipMemcpyHostToDevice;
#endif
    REQUIRE(hipMemcpy3D(&myparms) == hipSuccess);
    hipArray *arr2;
    hipChannelFormatDesc channelDesc1 = hipCreateChannelDesc(sizeof(T)*8,
        0, 0, 0, formatKind);
    HIP_CHECK(hipMalloc3DArray(&arr2, &channelDesc1,
          make_hipExtent(width, height,
            depth), hipArrayDefault));

    // Allocating Mem on GPU device 0 and trigger hipMemcpy3D from GPU 1
    HIP_CHECK(hipSetDevice(1));

    // D2D Scenario
    memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
    SetDefaultData();
    myparms.srcArray = arr;
    myparms.dstArray = arr2;
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyDeviceToDevice;
#else
    myparms.kind = hipMemcpyDeviceToDevice;
#endif
    REQUIRE(hipMemcpy3D(&myparms) == hipSuccess);
    HIP_CHECK(hipDeviceSynchronize());

    // For validating the D2D copy copying it again to hOutputData and
    // verifying it with iniital data hData
    memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
    T *hOutputData = reinterpret_cast<T*>(malloc(size));
    memset(hOutputData, 0, size);
    SetDefaultData();

    // Device to host
    myparms.dstPtr = make_hipPitchedPtr(hOutputData,
        width * sizeof(T),
        width, height);
    myparms.srcArray = arr2;
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyDeviceToHost;
#else
    myparms.kind = hipMemcpyDeviceToHost;
#endif
    REQUIRE(hipMemcpy3D(&myparms) == hipSuccess);
    HIP_CHECK(hipDeviceSynchronize());
    HipTest::checkArray(hData, hOutputData, width, height, depth);

    // DeAllocating Memory
    free(hOutputData);
    DeAllocateMemory();
  } else {
    SUCCEED("Skipped the test as there is no peer access\n");
  }
}
/*
 * This API verifies all the negative scenarios of hipMemcpy3D API
 */
template <typename T>
void Memcpy3D<T>::NegativeTests() {
  HIP_CHECK(hipSetDevice(0));
  AllocateMemory();

  // Initialization of data
  memset(&myparms, 0, sizeof(myparms));
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.extent = make_hipExtent(width , height, depth);
#ifdef __HIP_PLATFORM_NVCC__
  myparms.kind = cudaMemcpyHostToDevice;
#else
  myparms.kind = hipMemcpyHostToDevice;
#endif

  SECTION("Nullptr to destination array") {
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                        width, height);
    myparms.dstArray = nullptr;
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Nullptr to source array") {
    myparms.srcArray = nullptr;
    myparms.dstPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                        width, height);
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing both Source ptr and array") {
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                        width, height);
    myparms.srcArray = arr;
    myparms.dstArray = arr1;
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing both destination ptr and array") {
    myparms.dstPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                        width, height);
    myparms.dstArray = arr;
    myparms.srcArray = arr1;
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing Max value to extent") {
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                        width, height);
    myparms.dstArray = arr;
    myparms.extent = make_hipExtent(std::numeric_limits<int>::max(),
                                    std::numeric_limits<int>::max(),
                                    std::numeric_limits<int>::max());
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing Source pitchedPtr as nullptr") {
    myparms.srcPtr = make_hipPitchedPtr(nullptr, width * sizeof(T),
                                        width, height);
    myparms.dstArray = arr;
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing Dst pitchedPtr as nullptr") {
    myparms.dstPtr = make_hipPitchedPtr(nullptr, width * sizeof(T),
                                        width, height);
    myparms.srcArray = arr;
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing width > max width size in extent") {
    myparms.extent = make_hipExtent(width+1 , height, depth);
    myparms.dstArray = arr;
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                        width, height);
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing hgt > max width size in extent") {
    myparms.extent = make_hipExtent(width , height+1, depth);
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                        width, height);
    myparms.dstArray = arr;
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing depth > max width size in extent") {
    myparms.extent = make_hipExtent(width , height, depth+1);
    myparms.dstArray = arr;
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                        width, height);
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing dst width pos > max allocated width") {
    myparms.dstPos = make_hipPos(width+1, 0, 0);
    myparms.dstArray = arr;
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                        width, height);
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing dst height pos > max allocated hgt") {
    myparms.dstPos = make_hipPos(0, height+1, 0);
    myparms.dstArray = arr;
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                        width, height);
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing dst depth pos > max allocated depth") {
    myparms.dstPos = make_hipPos(0, 0, depth+1);
    myparms.dstArray = arr;
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                        width, height);
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing src width pos > max allocated width") {
    myparms.srcPos = make_hipPos(width+1, 0, 0);
    myparms.srcArray = arr;
    myparms.dstArray = arr1;
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyDeviceToDevice;
#else
    myparms.kind = hipMemcpyDeviceToDevice;
#endif
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing src height pos > max allocated hgt") {
    myparms.srcPos = make_hipPos(0, height+1, 0);
    myparms.srcArray = arr;
    myparms.dstArray = arr1;
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyDeviceToDevice;
#else
    myparms.kind = hipMemcpyDeviceToDevice;
#endif
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing src height pos > max allocated hgt") {
    myparms.srcPos = make_hipPos(0, 0, depth+1);
    myparms.srcArray = arr;
    myparms.dstArray = arr1;
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyDeviceToDevice;
#else
    myparms.kind = hipMemcpyDeviceToDevice;
#endif
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  SECTION("Passing src array size  > dst array size") {
    // Passing src array size greater than destination array size
    hipArray *arr2;
    hipChannelFormatDesc channelDesc1 = hipCreateChannelDesc(sizeof(T)*8,
                                                    0, 0, 0, formatKind);
    HIP_CHECK(hipMalloc3DArray(&arr2, &channelDesc1,
                              make_hipExtent(3, 3
                              , 3), hipArrayDefault));
    myparms.srcArray = arr;
    myparms.dstArray = arr2;
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyDeviceToDevice;
#else
    myparms.kind = hipMemcpyDeviceToDevice;
#endif
    REQUIRE(hipMemcpy3D(&myparms) != hipSuccess);
  }

  // DeAllocation of memory
  DeAllocateMemory();
}

/*
 * This API verifies the Extent validation Scenarios
 */
template <typename T>
void Memcpy3D<T>::Extent_Validation() {
  HIP_CHECK(hipSetDevice(0));
  AllocateMemory();
  memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                      width, height);
  myparms.dstArray = arr;
#ifdef __HIP_PLATFORM_NVCC__
  myparms.kind = cudaMemcpyHostToDevice;
#else
  myparms.kind = hipMemcpyHostToDevice;
#endif
  SECTION("Passing Extent as 0") {
    myparms.extent = make_hipExtent(0 , 0, 0);
    REQUIRE(hipMemcpy3D(&myparms) == hipSuccess);
  }
  SECTION("Passing Width 0 in Extent") {
    myparms.extent = make_hipExtent(0 , height, depth);
    REQUIRE(hipMemcpy3D(&myparms) == hipSuccess);
  }
  SECTION("Passing Height 0 in Extent") {
    myparms.extent = make_hipExtent(width , 0, depth);
    REQUIRE(hipMemcpy3D(&myparms) == hipSuccess);
  }
  SECTION("Passing Depth 0 in Extent") {
    myparms.extent = make_hipExtent(width , height, 0);
    REQUIRE(hipMemcpy3D(&myparms) == hipSuccess);
  }
  SECTION("Passing Depth 0 in Extent") {
    REQUIRE(hipMemcpy3D(nullptr) != hipSuccess);
  }
  DeAllocateMemory();
}

/*
 * This API verifies H2H-D2D-D2H functionalities of hipMemcpy3D API
 *
 * Input : "arr" variable  is  initialized with the "hData" variable in GPU-0
 * Output: "arr1" variable in GPU-0
 *
 * The "arr1" variable is then copied to "hOutputData" for validating
 * the result
 *
 * Validating the result by comparing "hData" and "hOutputData" variables
 */

template <typename T>
void Memcpy3D<T>::simple_Memcpy3D() {
  HIP_CHECK(hipSetDevice(0));
  AllocateMemory();

  // Host to Device
  memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
  SetDefaultData();
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T),
                                      width, height);
  myparms.dstArray = arr;
#ifdef __HIP_PLATFORM_NVCC__
  myparms.kind = cudaMemcpyHostToDevice;
#else
  myparms.kind = hipMemcpyHostToDevice;
#endif
  REQUIRE(hipMemcpy3D(&myparms) == hipSuccess);

  // Array to Array
  memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
  SetDefaultData();
  myparms.srcArray = arr;
  myparms.dstArray = arr1;
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyDeviceToDevice;
#else
    myparms.kind = hipMemcpyDeviceToDevice;
#endif
  REQUIRE(hipMemcpy3D(&myparms) == hipSuccess);
  T *hOutputData = reinterpret_cast<T*>(malloc(size));
  memset(hOutputData, 0,  size);

  // Device to host
  memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
  SetDefaultData();
  myparms.dstPtr = make_hipPitchedPtr(hOutputData,
                   width * sizeof(T), width, height);
  myparms.srcArray = arr1;
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyDeviceToHost;
#else
    myparms.kind = hipMemcpyDeviceToHost;
#endif
  REQUIRE(hipMemcpy3D(&myparms) == hipSuccess);

  // Validating the result
  HipTest::checkArray(hData, hOutputData, width, height, depth);

  // DeAllocating the Memory
  free(hOutputData);
  DeAllocateMemory();
}
/*
 This testcase performs hipMemcpy3D API validation for 
 different datatypes and different sizes
*/
TEMPLATE_TEST_CASE("Unit_hipMemcpy3D_Basic", "[hipMemcpy3D]",
    int, unsigned int, float) {
  int device = -1;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop,device));
  auto i = GENERATE_COPY(10, 100, 1024, prop.maxTexture3D[0]);
  auto j = GENERATE(10, 100);
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    if (std::is_same<TestType, float>::value) {
      Memcpy3D<TestType> memcpy3d_obj(i, j, j, hipChannelFormatKindFloat);
      memcpy3d_obj.simple_Memcpy3D();
    } else if (std::is_same<TestType, unsigned int>::value)  {
      Memcpy3D<TestType> memcpy3d_obj(i, j, j, hipChannelFormatKindUnsigned);
      memcpy3d_obj.simple_Memcpy3D();
    } else if (std::is_same<TestType, int>::value)  {
      Memcpy3D<TestType> memcpy3d_obj(i, j, j, hipChannelFormatKindSigned);
      memcpy3d_obj.simple_Memcpy3D();
    }
  } else {
    SUCCEED("skipping the testcases as numDevices < 2");
  }
}

/*
This testcase performs the extent validation scenarios of
hipMemcpy3D API
*/
TEST_CASE("Unit_hipMemcpy3D_ExtentValidation") {
  Memcpy3D<int> memcpy3d(width, height, depth,
                         hipChannelFormatKindSigned);
  memcpy3d.Extent_Validation();
}

/*
This testcase performs the negative scenarios of
hipMemcpy3D API
*/
TEST_CASE("Unit_hipMemcpy3D_multiDevice-Negative") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    Memcpy3D<int> memcpy3d(width, height, depth,
                           hipChannelFormatKindSigned);
    memcpy3d.NegativeTests();
  } else {
    SUCCEED("skipping the testcases as numDevices < 2");
  }
}

/*
This testcase performs the D2H,H2D and D2D on peer
GPU device
*/
TEST_CASE("Unit_hipMemcpy3D_multiDevice-OnPeerDevice") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    SECTION("D2H & H2D On DiffDevice") {
      Memcpy3D<float> memcpy3d_d2h_obj(width, height, depth,
                                       hipChannelFormatKindFloat);
      memcpy3d_d2h_obj.D2H_H2D_DeviceMem_OnDiffDevice();
    }

    SECTION("D2D On DiffDevice") {
      Memcpy3D<float> memcpy3d_d2d_obj(width, height, depth,
                                       hipChannelFormatKindFloat);
      memcpy3d_d2d_obj.D2D_DeviceMem_OnDiffDevice();
    }
  } else {
    SUCCEED("skipping the testcases as numDevices < 2");
  }
}
