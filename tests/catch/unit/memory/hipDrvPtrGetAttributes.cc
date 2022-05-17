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
 *  Tests for hipDrvPointerGetAttributes API
 Functional Tests:
   1. Pass multiple device related attributes in the attributes of hipDrvPointerGetAttributes API
      for the device pointer and check the behaviour
   2. Pass device and host attributes in the attributes of hipDrvPointerGetAttributes API and validate the behaviour
   3. Pass invalid pointer to hipDrvPointerGetAttributes API and validate the behaviour.

 Negative Tests:
  1. Pass invalid numAttributes
  2. Pass nullptr to attributes
  3. Pass nullptr to data
  4. Pass nullptr to device pointer
*/
#include <hip_test_common.hh>

static size_t Nbytes = 0;
constexpr size_t N {1000000};

/* This testcase verifies Negative Scenarios of
 * hipDrvPointerGetAttributes API */
TEST_CASE("Unit_hipDrvPtrGetAttributes_Negative") {
  HIP_CHECK(hipSetDevice(0));
  Nbytes = N * sizeof(int);
  int deviceId;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int* A_d;
  int* A_Pinned_h;

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_Pinned_h), Nbytes,
                         hipHostMallocDefault));
  HIP_CHECK(hipGetDevice(&deviceId));
  unsigned int device_ordinal;
  int *dev_ptr;
  void *data[2];
  data[0] = &dev_ptr;
  data[1] = &device_ordinal;

  hipPointer_attribute attributes[] = {HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                       HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL};

  SECTION("Passing nullptr to attributes") {
    REQUIRE(hipDrvPointerGetAttributes(2, nullptr, data,
            reinterpret_cast<hipDeviceptr_t>(A_d)) == hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to data") {
    REQUIRE(hipDrvPointerGetAttributes(2, attributes, nullptr,
              reinterpret_cast<hipDeviceptr_t>(A_d)) == hipErrorInvalidValue);
  }

#if HT_AMD
  SECTION("Passing nullptr to device Pointer") {
    hipDeviceptr_t ptr = 0;
    REQUIRE(hipDrvPointerGetAttributes(2, attributes, data,
            ptr) == hipErrorInvalidValue);
  }
#endif
#if HT_NVIDIA
  SECTION("Passing invalid dependencies") {
    hipPointer_attribute attributes1[] = {HIP_POINTER_ATTRIBUTE_DEVICE_POINTER};
    REQUIRE(hipDrvPointerGetAttributes(2, attributes1, data,
            reinterpret_cast<hipDeviceptr_t>(A_d)) == hipErrorInvalidValue);
  }
#endif
}

// Testcase verifies functional scenarios of hipDrvPointerGetAttributes API
TEST_CASE("Unit_hipDrvPtrGetAttributes_Functional") {
  HIP_CHECK(hipSetDevice(0));
  Nbytes = N * sizeof(int);
  int deviceId;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int* A_d;
  int* A_Pinned_h;

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_Pinned_h), Nbytes,
                         hipHostMallocDefault));
  HIP_CHECK(hipGetDevice(&deviceId));

  SECTION("Passing device attributes to device pointer") {
    unsigned int memory_type;
    int device_ordinal;
    int *dev{nullptr};
    int *dev_ptr{nullptr};
    int *dev_ptr1{nullptr};
    unsigned int range_size;
    int *start_addr{nullptr};
    void *data[5];
    data[0] = (&memory_type);
    data[1] = (&device_ordinal);
    data[2] = (&dev_ptr);
    data[3] = (&range_size);
    data[4] = (&start_addr);

    // Device memory
    hipPointer_attribute attributes[] = {HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                         HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                         HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                         HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                                        HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR};
    HIP_CHECK(hipPointerGetAttribute(&dev_ptr1,
              HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
              reinterpret_cast<hipDeviceptr_t>(A_d + 100)));
    HIP_CHECK(hipPointerGetAttribute(&dev,
	      HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
              reinterpret_cast<hipDeviceptr_t>(A_d)));
    HIP_CHECK(hipDrvPointerGetAttributes(5, attributes, data,
              reinterpret_cast<hipDeviceptr_t>(A_d + 100)));

    REQUIRE(dev_ptr == dev_ptr1);
#if HT_NVIDIA
    REQUIRE(memory_type == CU_MEMORYTYPE_DEVICE);
#else
    REQUIRE(memory_type == hipMemoryTypeDevice);
#endif
    REQUIRE(device_ordinal == deviceId);
    REQUIRE(range_size == Nbytes);
    REQUIRE(start_addr == dev);
  }

  SECTION("Passing device and host attributes to device pointer") {
    int device_ordinal;
    int *host_ptr;
    void *data[2];
    data[0] = (&host_ptr);
    data[1] = (&device_ordinal);

    hipPointer_attribute attributes[] = {HIP_POINTER_ATTRIBUTE_HOST_POINTER,
                                         HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL};
    HIP_CHECK(hipDrvPointerGetAttributes(2, attributes, data,
              reinterpret_cast<hipDeviceptr_t>(A_d)));
    REQUIRE(host_ptr == nullptr);
    REQUIRE(device_ordinal == deviceId);
  }

  SECTION("Passing host related attributes to host pointer") {
    int device_ordinal;
    void *data[2];
    int *host_ptr;
    data[0] = (&host_ptr);
    data[1] = (&device_ordinal);
    hipPointer_attribute attributes[] = {HIP_POINTER_ATTRIBUTE_HOST_POINTER,
                                         HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL};
    HIP_CHECK(hipDrvPointerGetAttributes(2, attributes, data,
              reinterpret_cast<hipDeviceptr_t>(A_Pinned_h)));
    REQUIRE(host_ptr == A_Pinned_h);
    REQUIRE(device_ordinal == deviceId);
  }
}
