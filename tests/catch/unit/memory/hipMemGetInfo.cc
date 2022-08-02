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


#include <hip_test_common.hh>
#include <thread>
#include <vector>

/*
 * This testcase verifies hipMemGetInfo API
 * 1. Different memory chunk allocation
 *  1.1. hipMalloc
 *  1.2. hipMallocArray
 *  1.3. hipMalloc3D
 *  1.3. hipMalloc3DArray
 * 2. Allocation using different threads
 * 3. Negative: Invalid args
 */

struct MinAlloc {
 private:
  int value;
  MinAlloc() {
    size_t freeMemInit;
    size_t totalMemInit;

    unsigned int* A_mem{nullptr};
    size_t mallocSize{1};

    HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));
    // allocate 1 byte
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_mem), mallocSize));

    size_t freeMemRet;
    size_t totalMemRet;
    // actual allocation should be bigger to reflect the minimum allocation on device
    HIP_CHECK(hipMemGetInfo(&freeMemRet, &totalMemRet));
    REQUIRE(freeMemInit >= freeMemRet);
    HIP_CHECK(hipFree(A_mem));

    // store the size of minimum allocation
    value = (freeMemInit - freeMemRet);
  }

 public:
  static int Get() {
    static MinAlloc instance;
    return instance.value;
  }
};

// if the memory being allocated is not divisible by the minimum allocation add an extra minimum
// allocation AddedAllocation = InitialAllocation + (MinAllocation - divisionRemainer)
void fixAllocSize(size_t& allocation) {
  REQUIRE(MinAlloc::Get() != 0);
  if (allocation % MinAlloc::Get() != 0) {
    auto adjustment = allocation % MinAlloc::Get();
    adjustment = MinAlloc::Get() - adjustment;
    allocation = allocation + adjustment;
  }
}

// Print information about memory
#define MEMINFO(totalMem, freeMemInit, freeMemRet, usedMem)                                        \
  INFO("Total memory: \t\t\t" << totalMem << "\n"                                                  \
                              << "Memory used: \t\t\t\t" << freeMemInit - freeMemRet << "\n"       \
                              << "Free memory after alloc: \t\t" << freeMemRet << "\n"             \
                              << "Free memory initally: \t\t" << freeMemInit << "\n"               \
                              << "Memory assumed to be used: \t\t" << usedMem);


TEST_CASE("Unit_hipMemGetInfo_DifferentMallocLarge") {
  size_t freeMemInit;
  size_t totalMemInit;
  HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));

  unsigned int* A_mem{nullptr};
  unsigned int* B_mem{nullptr};

  size_t freeMemRet;
  size_t totalMemRet;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  auto totalMemory = prop.totalGlobalMem;


  // allocate half of free mem
  auto Malloc1Size = freeMemInit >> 1;
  // if the allocation is not divisible by the MinAllocation
  // take into account and add padding
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_mem), Malloc1Size));

  // allocate an extra quarter of free mem
  auto Malloc2Size = Malloc1Size >> 1;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&B_mem), Malloc2Size));

  HIP_CHECK(hipMemGetInfo(&freeMemRet, &totalMemRet));

  MEMINFO(totalMemRet, freeMemInit, freeMemRet, Malloc1Size + Malloc2Size);
  // check if device property total memory is the same as
  // total memory returned from hipMemGetInfo
  REQUIRE(totalMemory == totalMemRet);
  auto allocSize = Malloc1Size + Malloc2Size;
  auto assumedFreeMem = freeMemInit - allocSize;

  REQUIRE(freeMemRet <= assumedFreeMem);
  HIP_CHECK(hipFree(A_mem));
  HIP_CHECK(hipFree(B_mem));
}

TEST_CASE("Unit_hipMemGetInfo_DifferentMallocSmall") {
  size_t freeMemInit;
  size_t totalMemInit;
  HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));

  unsigned int* A_mem{nullptr};
  size_t freeMemRet;
  size_t totalMemRet;
  // allocate smaller chunk than minimum
  size_t Malloc1Size = 1;

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_mem), Malloc1Size));

  HIP_CHECK(hipMemGetInfo(&freeMemRet, &totalMemRet));
  MEMINFO(totalMemRet, freeMemInit, freeMemRet, Malloc1Size);

  auto assumedFreeMem = freeMemInit - Malloc1Size;
  // Free memory should be less than assumed for
  // single allocation smaller than min allocation chunk
  REQUIRE(freeMemRet <= assumedFreeMem);

  HIP_CHECK(hipFree(A_mem));
}

TEST_CASE("Unit_hipMemGetInfo_DifferentMallocMultiSmall") {
  size_t freeMemInit;
  size_t totalMemInit;
  HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));

  unsigned int* A_mem{nullptr};
  unsigned int* B_mem{nullptr};
  size_t freeMemRet;
  size_t totalMemRet;

  // Allocate memory that is a quarter of the min allocation
  // Expected behaviour is to reuse the min allocation memory
  size_t MallocSize = MinAlloc::Get() >> 2;

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_mem), MallocSize));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&B_mem), MallocSize));

  HIP_CHECK(hipMemGetInfo(&freeMemRet, &totalMemRet));
  MEMINFO(totalMemRet, freeMemInit, freeMemRet, MallocSize * 2);


  auto assumedFreeMem = freeMemInit - (MallocSize * 2);

  // Confirm mem alocation results
  REQUIRE(freeMemRet <= assumedFreeMem);
  HIP_CHECK(hipFree(A_mem));
  HIP_CHECK(hipFree(B_mem));
}

TEMPLATE_TEST_CASE("Unit_hipMemGetInfo_MallocArray", "", int, int4, char) {
  // get initial mem data
  size_t freeMemInit;
  size_t totalMemInit;
  HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));

  // create and allocate an Array
  hipArray_t arrayPtr{};

  auto bytesPerItem = sizeof(TestType);
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();
  hipExtent extent{};
  extent.width = GENERATE(32, 128, 256, 512, 1024);

  extent.height = GENERATE(0, 32, 128, 256, 512, 1024);

  HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));

  // check if memory is correct
  size_t freeMemRet;
  size_t totalMemRet;
  HIP_CHECK(hipMemGetInfo(&freeMemRet, &totalMemRet));

  // calculate used memory, take into account 1D array (height = 0)
  size_t usedMem = bytesPerItem * extent.width * (extent.height != 0 ? extent.height : 1);

  // ensure we allocate at least the min allocation for the array
  MEMINFO(totalMemRet, freeMemInit, freeMemRet, usedMem);

  size_t assumedFreeMem = freeMemInit - usedMem;

  REQUIRE(freeMemRet <= assumedFreeMem);

  HIP_CHECK(hipFreeArray(arrayPtr));
}

TEST_CASE("Unit_hipMemGetInfo_Malloc3D") {
  // Get initial memory
  size_t freeMemInit;
  size_t totalMemInit;
  HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));

  // Allocate 3D object
  hipExtent extent{};
  // extent is given in bytes for with
  extent.width = GENERATE(32, 128, 256);
  extent.height = GENERATE(32, 128, 256);
  extent.depth = GENERATE(32, 128, 256);
  hipPitchedPtr A_mem{};
  HIP_CHECK(hipMalloc3D(&A_mem, extent));

  // Get memory after allocation
  size_t freeMemRet;
  size_t totalMemRet;
  HIP_CHECK(hipMemGetInfo(&freeMemRet, &totalMemRet));

  // Verify result
  size_t mallocSize = A_mem.pitch * extent.height * extent.depth;

  size_t assumedFreeMem = freeMemInit - mallocSize;
  MEMINFO(totalMemRet, freeMemInit, freeMemRet, mallocSize);

  REQUIRE(freeMemRet <= assumedFreeMem);

  HIP_CHECK(hipFree(A_mem.ptr));
}

TEMPLATE_TEST_CASE("Unit_hipMemGetInfo_Malloc3DArray", "", char, int, int4) {
  // Get initial memory
  size_t freeMemInit;
  size_t totalMemInit;
  HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));
  // Allocate 3D object
  hipArray_t arrayPtr{};
  size_t sizeInBytes = (size_t)sizeof(TestType);
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  int device;
  HIP_CHECK(hipGetDevice(&device));
  int allignSize{0};
  hipDeviceGetAttribute(&allignSize, hipDeviceAttributeTextureAlignment, device);

#if HT_NVIDIA
  auto flag = GENERATE(hipArrayDefault, hipArrayLayered, hipArrayCubemap,
                       hipArrayLayered | hipArrayCubemap);
#else
  // hipArrayCubemap not supported on AMD
  auto flag = GENERATE(hipArrayDefault, hipArrayLayered);
#endif

  hipExtent extent{};
  extent.width = GENERATE(32, 128, 256, 512);
  extent.height = GENERATE(0, 32, 128, 256, 512);
  if (flag == hipArrayCubemap) {
    // width must be equal to height, and depth must be six.
    extent.height = extent.width;
    extent.depth = 6;
  } else if (flag == hipArrayLayered | hipArrayCubemap) {
    // width must be equal to height, and depth must be a multiple six.
    extent.height = extent.width;
    extent.depth = 6 * GENERATE(4, 8, 16, 32);
  } else if (extent.height == 0 && flag != hipArrayLayered) {
    // if height = 0 the depth must be 0 unless using hipArrayLayered flag
    extent.depth = 0;
  } else {
    extent.depth = GENERATE(32, 128, 256, 512);
  }


  // Get memory after allocation
  auto h = extent.height == 0 ? 1 : extent.height;
  auto d = extent.depth == 0 ? 1 : extent.depth;
  auto w = extent.width * sizeInBytes;
  size_t mallocSize = w * h * d;

  HIP_CHECK(hipMalloc3DArray(&arrayPtr, &desc, extent, flag));

  // Verify result
  size_t freeMemRet;
  size_t totalMemRet;
  HIP_CHECK(hipMemGetInfo(&freeMemRet, &totalMemRet));

  // Sometimes hipMemGetInfo reports that no new memory has be allocated for testcase
  // take this into account
  if (freeMemInit == freeMemRet) {
    // no new memory allocation has occured verify that memory trying
    // to be allocated is less than a min allocation block
    MEMINFO(totalMemRet, freeMemInit, freeMemRet, mallocSize);
    REQUIRE(mallocSize <= static_cast<size_t>(MinAlloc::Get()));

  } else {
    MEMINFO(totalMemRet, freeMemInit, freeMemRet, mallocSize);
    size_t assumedFreeMem = freeMemInit - mallocSize;
    REQUIRE(freeMemRet <= assumedFreeMem);
  }
  HIP_CHECK(hipFreeArray(arrayPtr));
}


TEST_CASE("Unit_hipMemGetInfo_ParaLarge") {
  size_t freeMemInit;
  size_t totalMemInit;
  HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));
  unsigned int* A_mem{nullptr};
  unsigned int* B_mem{nullptr};

  // allocate half of free mem
  auto Malloc1Size = freeMemInit >> 1;
  // if the allocation is not divisible by the MinAllocation
  // take into account and add padding
  std::thread t1(
      [&] { HIP_CHECK_THREAD(hipMalloc(reinterpret_cast<void**>(&A_mem), Malloc1Size)); });

  // allocate an extra quarter of free mem
  auto Malloc2Size = Malloc1Size >> 1;
  std::thread t2(
      [&] { HIP_CHECK_THREAD(hipMalloc(reinterpret_cast<void**>(&B_mem), Malloc2Size)); });

  t1.join();
  t2.join();
  HIP_CHECK_THREAD_FINALIZE();

  size_t freeMemRet;
  size_t totalMemRet;
  HIP_CHECK(hipMemGetInfo(&freeMemRet, &totalMemRet));

  MEMINFO(totalMemRet, freeMemInit, freeMemRet, Malloc1Size + Malloc2Size);
  auto allocSize = Malloc1Size + Malloc2Size;
  REQUIRE(freeMemRet <= freeMemInit - allocSize);

  HIP_CHECK(hipFree(A_mem));
  HIP_CHECK(hipFree(B_mem));
}

TEST_CASE("Unit_hipMemGetInfo_ParaSmall") {
  size_t freeMemInit;
  size_t totalMemInit;
  HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));
  unsigned int* A_mem{nullptr};
  // allocate smaller chunk than minimum
  size_t Malloc1Size = 1;

  std::thread t1(
      [&] { HIP_CHECK_THREAD(hipMalloc(reinterpret_cast<void**>(&A_mem), Malloc1Size)) });
  t1.join();
  HIP_CHECK_THREAD_FINALIZE();
  size_t freeMemRet;
  size_t totalMemRet;
  HIP_CHECK(hipMemGetInfo(&freeMemRet, &totalMemRet));
  MEMINFO(totalMemRet, freeMemInit, freeMemRet, Malloc1Size);


  auto assumedFreeMem = freeMemInit - Malloc1Size;
  // Free memory should be less than assumed for
  // single allocation smaller than min allocation chunk
  REQUIRE(freeMemRet <= assumedFreeMem);

  HIP_CHECK(hipFree(A_mem));
}


TEST_CASE("Unit_hipMemGetInfo_Negative") {
  size_t freeMemInit;
  size_t totalMemInit;
  HIP_CHECK(hipMemGetInfo(&freeMemInit, &totalMemInit));

  unsigned int* A_mem{nullptr};
  auto MallocSize = MinAlloc::Get();

  SECTION("Zero allocation") {
    size_t freeMemRet;
    size_t totalMemRet;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_mem), 0));
    HIP_CHECK(hipMemGetInfo(&freeMemRet, &totalMemRet));

    REQUIRE(freeMemRet == freeMemInit);
  }
  SECTION("Nullptr as first param passed to hipMemGetInfo") {
    size_t* freeMemRet = nullptr;
    size_t totalMemRet;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_mem), MallocSize));
    // Segfaults on AMD and returns hipSuccess on Nvidia
    HIP_CHECK(hipMemGetInfo(freeMemRet, &totalMemRet));
  }
  SECTION("Nullptr as second param passed to hipMemGetInfo") {
    size_t freeMemRet;
    size_t* totalMemRet = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_mem), MallocSize));
    // Segfaults on AMD and returns hipSuccess on Nvidia
    HIP_CHECK(hipMemGetInfo(&freeMemRet, totalMemRet));
  }
  SECTION("Nullptr as both params passed to hipMemGetInfo") {
#if HT_AMD
    HipTest::HIP_SKIP_TEST("EXSWCPHIPT-135");
    return;
#endif
    size_t* freeMemRet = nullptr;
    size_t* totalMemRet = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_mem), MallocSize));
    // Segfaults on AMD and returns hipSuccess on Nvidia
    HIP_CHECK(hipMemGetInfo(freeMemRet, totalMemRet));
  }

  HIP_CHECK(hipFree(A_mem));
}
