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
Following scenarios are verified for hipPointerGetAttributes API
1. Run through a couple simple cases to test lookups host pointer arithmetic
2. Allocates memory across all devices withing the specified size range
3. Allocates tiny memory across all devices
4. Multi-threaded test with many simul allocs.

*/
#include<hip_test_common.hh>
#include <vector>
#include <iostream>
#include <string>

size_t Nbytes = 0;
constexpr size_t N{1000000};



//=================================================================================================
// Utility Functions:
//=================================================================================================

bool operator==(const hipPointerAttribute_t& lhs,
                const hipPointerAttribute_t& rhs) {
  return ((lhs.hostPointer == rhs.hostPointer) &&
          (lhs.devicePointer == rhs.devicePointer) &&
         (lhs.memoryType == rhs.memoryType) && (lhs.device == rhs.device) &&
         (lhs.allocationFlags == rhs.allocationFlags));
}


bool operator!=(const hipPointerAttribute_t& lhs,
                const hipPointerAttribute_t& rhs) {
  return !(lhs == rhs);
}


const char* memoryTypeToString(hipMemoryType memoryType) {
  switch (memoryType) {
    case hipMemoryTypeHost:
      return "[Host]";
    case hipMemoryTypeDevice:
      return "[Device]";
    default:
      return "[Unknown]";
  }
}


void resetAttribs(hipPointerAttribute_t* attribs) {
    attribs->hostPointer = reinterpret_cast<void*>(-1);
    attribs->devicePointer = reinterpret_cast<void*>(-1);
    attribs->memoryType = hipMemoryTypeHost;
    attribs->device = -2;
    attribs->isManaged = -1;
    attribs->allocationFlags = 0xffff;
}


void printAttribs(const hipPointerAttribute_t* attribs) {
  printf(
        "hostPointer:%p devicePointer:%p  memType:%s deviceId:%d isManaged:%d "
        "allocationFlags:%u\n",
        attribs->hostPointer, attribs->devicePointer,
        memoryTypeToString(attribs->memoryType),
        attribs->device, attribs->isManaged, attribs->allocationFlags);
}


inline int zrand(int max) { return rand() % max; }



// Store the hipPointer attrib and some extra info
// so can later compare the looked-up info against
// the reference expectation
struct SuperPointerAttribute {
    void* _pointer;
    size_t _sizeBytes;
    hipPointerAttribute_t _attrib;
};


// Support function to check result against a reference:
void checkPointer(const SuperPointerAttribute& ref, int major,
                  int minor, void* pointer) {
    hipPointerAttribute_t attribs;
    resetAttribs(&attribs);

    hipError_t e = hipPointerGetAttributes(&attribs, pointer);
    if ((e != hipSuccess) || (attribs != ref._attrib)) {
        HIP_CHECK(e);
        REQUIRE(attribs != ref._attrib);
    } else {
        printf("#%4d.%d GOOD:%p getattr ::  ", major, minor, pointer);
        printAttribs(&attribs);
    }
}


// Test that allocates memory across all devices withing the
// specified size range
// (minSize...maxSize). Then does lookups to make sure the
// info reported by the tracker matches
// expecations Then deallocates it all.
// Multiple threads can call this function and in fact
// we do this in the testMultiThreaded_1 test.
void clusterAllocs(int numAllocs, size_t minSize, size_t maxSize) {
  Nbytes = N * sizeof(char);
  printf("clusterAllocs numAllocs=%d size=%lu..%lu\n",
         numAllocs, minSize, maxSize);
  const int Max_Devices = 256;
  std::vector<SuperPointerAttribute> reference(numAllocs);

  REQUIRE(minSize > 0);
  REQUIRE(maxSize >= minSize);

  int numDevices;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  //---
  // Populate with device and host allocations.
  size_t totalDeviceAllocated[Max_Devices];
  for (int i = 0; i < numDevices; i++) {
    totalDeviceAllocated[i] = 0;
  }
  for (int i = 0; i < numAllocs; i++) {
    unsigned rand_seed = time(NULL);
    bool isDevice = HipTest::RAND_R(&rand_seed) & 0x1;
    reference[i]._sizeBytes = zrand(maxSize - minSize) + minSize;

    reference[i]._attrib.device = zrand(numDevices);
    HIP_CHECK(hipSetDevice(reference[i]._attrib.device));
    reference[i]._attrib.isManaged = 0;

    void* ptr;
    if (isDevice) {
      totalDeviceAllocated[reference[i]._attrib.device] +=
                           reference[i]._sizeBytes;
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr),
                         reference[i]._sizeBytes));
      reference[i]._attrib.memoryType = hipMemoryTypeDevice;
      reference[i]._attrib.devicePointer = ptr;
      reference[i]._attrib.hostPointer = NULL;
      reference[i]._attrib.allocationFlags = 0;
    } else {
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&ptr),
                             reference[i]._sizeBytes,
                             hipHostMallocDefault));
      reference[i]._attrib.memoryType = hipMemoryTypeHost;
      reference[i]._attrib.devicePointer = ptr;
      reference[i]._attrib.hostPointer = ptr;
      reference[i]._attrib.allocationFlags = 0;
    }
    reference[i]._pointer = ptr;
  }

  for (int i = 0; i < numDevices; i++) {
    size_t free, total;
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipMemGetInfo(&free, &total));
    printf(
           "  device#%d: hipMemGetInfo: "
           "free=%zu (%4.2fMB) totalDevice=%lu (%4.2fMB) total=%zu "
           "(%4.2fMB)\n",
           i, free, (free / 1024.0 / 1024.0), totalDeviceAllocated[i],
           (totalDeviceAllocated[i]) / 1024.0 / 1024.0, total,
           (total / 1024.0 / 1024.0));
    REQUIRE(free + totalDeviceAllocated[i] <= total);
  }

  // Now look up each pointer we inserted and verify we can find it:
  char * ptr;
  for (int i = 0; i < numAllocs; i++) {
    SuperPointerAttribute& ref = reference[i];
    ptr = static_cast<char *>(ref._pointer);
    checkPointer(ref, i, 0, ref._pointer);
    checkPointer(ref, i, 1, (ptr +
                 ref._sizeBytes / 2));
    if (ref._sizeBytes > 1) {
      checkPointer(ref, i, 2, (ptr +
                   ref._sizeBytes - 1));
    }

    if (ref._attrib.memoryType == hipMemoryTypeDevice) {
      hipFree(ref._pointer);
    } else {
      hipHostFree(ref._pointer);
    }
  }
}

//========================================================================
// Functions to run tests
//=======================================================================
//--
// Run through a couple simple cases to test lookups host pointer arithmetic:
TEST_CASE("Unit_hipPointerGetAttributes_Basic") {
  HIP_CHECK(hipSetDevice(0));
  Nbytes = N * sizeof(char);
  printf("\n");
  printf("=============================================================\n");
  printf("Simple Tests\n");
  printf("=============================================================\n");

  char* A_d;
  char* A_Pinned_h;
  char* A_OSAlloc_h;
  hipError_t e;

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_Pinned_h), Nbytes,
                         hipHostMallocDefault));
  A_OSAlloc_h = reinterpret_cast<char*>(malloc(Nbytes));

  size_t free, total;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  printf("hipMemGetInfo: free=%zu (%4.2f) Nbytes=%lu total=%zu (%4.2f)\n", free,
         (free / 1024.0 / 1024.0), Nbytes, total,
         (total / 1024.0 / 1024.0));
  REQUIRE(free + Nbytes <= total);


  hipPointerAttribute_t attribs;
  hipPointerAttribute_t attribs2;

  // Device memory
  printf("\nDevice memory (hipMalloc)\n");
  HIP_CHECK(hipPointerGetAttributes(&attribs, A_d));

  // Check pointer arithmetic cases:
  resetAttribs(&attribs2);
  HIP_CHECK(hipPointerGetAttributes(&attribs2, A_d + 100));
  char *ptr = reinterpret_cast<char *>(attribs.devicePointer);
  REQUIRE(ptr + 100 ==
            reinterpret_cast<char*>(attribs2.devicePointer));

  // Corner case at end of array:
  resetAttribs(&attribs2);
  HIP_CHECK(hipPointerGetAttributes(&attribs2, A_d + Nbytes - 1));
  REQUIRE((ptr + Nbytes - 1) ==
            reinterpret_cast<char*>(attribs2.devicePointer));

  // Pointer just beyond array must be invalid or at least a different pointer
  resetAttribs(&attribs2);
  e = hipPointerGetAttributes(&attribs2, A_d + Nbytes + 1);
  if (e != hipErrorInvalidValue) {
    // We might have strayed into another pointer area.
    REQUIRE(reinterpret_cast<char*>(ptr) !=
              reinterpret_cast<char*>(attribs2.devicePointer));
  }


  resetAttribs(&attribs2);
  e = hipPointerGetAttributes(&attribs2, A_d + Nbytes);
  if (e != hipErrorInvalidValue) {
    REQUIRE(attribs.devicePointer != attribs2.devicePointer);
  }
  hipFree(A_d);
  e = hipPointerGetAttributes(&attribs, A_d);
  REQUIRE(e == hipErrorInvalidValue);

  // Device-visible host memory
  printf("\nDevice-visible host memory (hipHostMalloc)\n");
  HIP_CHECK(hipPointerGetAttributes(&attribs, A_Pinned_h));

  resetAttribs(&attribs2);
  HIP_CHECK(hipPointerGetAttributes(&attribs2, A_Pinned_h + Nbytes / 2));
  char *ptr1 = reinterpret_cast<char *>(attribs.hostPointer);
  REQUIRE((ptr1 + Nbytes / 2)
            == reinterpret_cast<char*>(attribs2.hostPointer));


  hipHostFree(A_Pinned_h);
  e = hipPointerGetAttributes(&attribs, A_Pinned_h);
  REQUIRE(e == hipErrorInvalidValue);

  // OS memory
  printf("\nOS-allocated memory (malloc)\n");
  e = hipPointerGetAttributes(&attribs, A_OSAlloc_h);
  REQUIRE(e == hipErrorInvalidValue);
}

TEST_CASE("Unit_hipPointerGetAttributes_ClusterAlloc") {
  srand(0x100);
  printf("\n=============================================\n");
  clusterAllocs(100, 1024 * 1, 1024 * 1024);
}

TEST_CASE("Unit_hipPointerGetAttributes_TinyClusterAlloc") {
  srand(0x200);
  printf("\n=============================================\n");
  clusterAllocs(1000, 1, 10);  //  Many tiny allocations;
}

// Multi-threaded test with many simul allocs.
// IN : serialize will force the test to run in serial fashion.
TEST_CASE("Unit_hipPointerGetAttributes_MultiThread") {
    srand(0x300);
    auto serialize = 1;
    printf("\n=============================================\n");
    printf("MultiThreaded_1\n");
    if (serialize) printf("[SERIALIZE]\n");
    printf("===============================================\n");
    std::thread t1(clusterAllocs, 1000, 101, 1000);
    if (serialize) t1.join();

    std::thread t2(clusterAllocs, 1000, 11, 100);
    if (serialize) t2.join();

    std::thread t3(clusterAllocs, 1000, 5, 10);
    if (serialize) t3.join();

    std::thread t4(clusterAllocs, 1000, 1, 4);
    if (serialize) t4.join();
}
