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
hipMallocMipmappedArray API test scenarios
1. Basic Functionality
2. Negative Scenarios
3. Allocating Small and big chunk data
4. Multithreaded scenario
*/

#include <array>
#include <hip_test_common.hh>
#include "hipArrayCommon.hh"

static constexpr auto ARRAY_SIZE{4};
static constexpr auto BIG_ARRAY_SIZE{100};
static constexpr auto ARRAY_LOOP{100};


/*
 * This API verifies  memory allocations for small and
 * bigger chunks of data.
 * Two scenarios are verified in this API
 * 1. SmallArray: Allocates ARRAY_SIZE in a loop and
 *    releases the memory and verifies the meminfo.
 * 2. BigArray: Allocates BIG_ARRAY_SIZE in a loop and
 *    releases the memory and verifies the meminfo
 *
 * In both cases, the memory info before allocation and
 * after releasing the memory should be the same
 *
 */
static void MallocMipmappedArray_DiffSizes(int gpu) {
  HIP_CHECK_THREAD(hipSetDevice(gpu));
  //Use of GENERATE in thead function causes random failures with multithread condition.
  std::vector<size_t> runs {ARRAY_SIZE, BIG_ARRAY_SIZE};
  for (const auto& size : runs) {
    auto numLevelsLimit = floor(log2(size));
    for (unsigned int numLevels = 0; numLevels < numLevelsLimit; numLevels++) {
      size_t width{size}, height{size}, depth{size};
      hipChannelFormatDesc channelDesc = hipCreateChannelDesc<float>();
      std::array<hipMipmappedArray_t, ARRAY_LOOP> arr;
      size_t pavail, avail, total;
      HIP_CHECK_THREAD(hipMemGetInfo(&pavail, &total));

      for (int i = 0; i < ARRAY_LOOP; i++) {
        HIP_CHECK_THREAD(hipMallocMipmappedArray(&arr[i], &channelDesc, make_hipExtent(width, height, depth),
                                        (1 + numLevels), hipArrayDefault));
      }
      for (int i = 0; i < ARRAY_LOOP; i++) {
        HIP_CHECK_THREAD(hipFreeMipmappedArray(arr[i]));
      }

      HIP_CHECK_THREAD(hipMemGetInfo(&avail, &total));
      REQUIRE_THREAD(pavail == avail);
    }
  }
}

TEST_CASE("Unit_hipMallocMipmappedArray_DiffSizes") {
  MallocMipmappedArray_DiffSizes(0);
  HIP_CHECK_THREAD_FINALIZE();
}

/*
This testcase verifies the hipMallocMipmappedArray API in multithreaded
scenario by launching threads in parallel on multiple GPUs
and verifies the hipMallocMipmappedArray API with small and big chunks data
*/
TEST_CASE("Unit_hipMallocMipmappedArray_MultiThread") {
  std::vector<std::thread> threadlist;
  int devCnt = 0;
  devCnt = HipTest::getDeviceCount();
  const auto pavail = getFreeMem();
  for (int i = 0; i < devCnt; i++) {
    threadlist.push_back(std::thread(MallocMipmappedArray_DiffSizes, i));
  }

  for (auto& t : threadlist) {
    t.join();
  }
  HIP_CHECK_THREAD_FINALIZE();
  const auto avail = getFreeMem();

  if (pavail != avail) {
    WARN("Memory leak of hipMallocMipmappedArray API in multithreaded scenario");
    REQUIRE(false);
  }
}

namespace {
void checkMipmappedArrayIsExpected(hipMipmappedArray_t array, hipArray level_array, const hipChannelFormatDesc& expected_desc,
                          const hipExtent& expected_extent, const unsigned int expected_flags) {
// hipArrayGetInfo doesn't currently exist (EXSWCPHIPT-87)
#if HT_AMD
  std::ignore = array;
  std::ignore = expected_desc;
  std::ignore = expected_extent;
  std::ignore = expected_flags;
#else
  cudaArraySparseProperties sparseProperties;
  cudaChannelFormatDesc queried_desc;
  cudaExtent queried_extent;
  unsigned int queried_flags;

  cudaMipmappedArrayGetSparseProperties(&sparseProperties, array);
  cudaArrayGetInfo(&queried_desc, &queried_extent, &queried_flags, level_array);

  REQUIRE(expected_desc.x == queried_desc.x);
  REQUIRE(expected_desc.y == queried_desc.y);
  REQUIRE(expected_desc.z == queried_desc.z);
  REQUIRE(expected_desc.f == queried_desc.f);

  REQUIRE(expected_extent.width == sparseProperties.width);
  REQUIRE(expected_extent.height == sparseProperties.height);
  REQUIRE(expected_extent.depth == sparseProperties.depth);

  REQUIRE(expected_extent.width == queried_extent.width);
  REQUIRE(expected_extent.height == queried_extent.height);
  REQUIRE(expected_extent.depth == queried_extent.depth);

  REQUIRE(expected_flags == queried_flags);
#endif
}
}  // namespace

TEMPLATE_TEST_CASE("Unit_hipMallocMipmappedArray_happy", "", char, uint2, int4, short4, float) {
  hipMipmappedArray_t array;
  const auto desc = hipCreateChannelDesc<TestType>();
#if HT_AMD
  const unsigned int flags = hipArrayDefault;
#else
  const unsigned int flags =
      GENERATE(hipArrayDefault, hipArraySurfaceLoadStore, hipArrayTextureGather);
#endif
  constexpr size_t size = 64;
  const unsigned int numLevels = GENERATE(1, 3, 5, 7);

  std::vector<hipExtent> extents;
  extents.reserve(3);
  extents.push_back({size, size, 0});  // 2D array
  if (flags != hipArrayTextureGather) {
    extents.push_back({size, 0, 0});        // 1D array
    extents.push_back({size, size, size});  // 3D array
  };

  for (const auto extent : extents) {
    CAPTURE(flags, extent.width, extent.height, extent.depth);

    HIP_CHECK(hipMallocMipmappedArray(&array, &desc, extent, numLevels, flags));
    hipArray* hipArray = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&hipArray, array, 1));
    checkMipmappedArrayIsExpected(array, hipArray, desc, extent, flags);
    HIP_CHECK(hipFreeMipmappedArray(array));
  }
}

TEMPLATE_TEST_CASE("Unit_hipMallocMipmappedArray_MaxTexture", "", int, uint4, ushort2, float) {
  hipMipmappedArray_t array;
  const hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();
#if HT_AMD
  const unsigned int flag = hipArrayDefault;
#else
  const unsigned int flag = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore);
#endif
  unsigned int numLevels = GENERATE(1, 5, 9, 13);
  if (flag == hipArraySurfaceLoadStore) {
    HipTest::HIP_SKIP_TEST("EXSWCPHIPT-58");
    return;
  }
  CAPTURE(flag);
  const Sizes sizes(flag);
  CAPTURE(sizes.max1D, sizes.max2D, sizes.max3D);

  const size_t s = 64;
  SECTION("Happy") {
    // stored in a vector so some values can be ifdef'd out
    std::vector<hipExtent> extentsToTest{
        make_hipExtent(sizes.max1D, 0, 0),                              // 1D max
        make_hipExtent(sizes.max2D[0], s, 0),                           // 2D max width
        make_hipExtent(s, sizes.max2D[1], 0),                           // 2D max height
        make_hipExtent(sizes.max2D[0], sizes.max2D[1], 0),              // 2D max
        make_hipExtent(sizes.max3D[0], s, s),                           // 3D max width
        make_hipExtent(s, sizes.max3D[1], s),                           // 3D max height
        make_hipExtent(s, s, sizes.max3D[2]),                           // 3D max depth
        make_hipExtent(s, sizes.max3D[1], sizes.max3D[2]),              // 3D max height and depth
        make_hipExtent(sizes.max3D[0], s, sizes.max3D[2]),              // 3D max width and depth
        make_hipExtent(sizes.max3D[0], sizes.max3D[1], s),              // 3D max width and height
        make_hipExtent(sizes.max3D[0], sizes.max3D[1], sizes.max3D[2])  // 3D max
    };
    const auto extent =
        GENERATE_COPY(from_range(std::begin(extentsToTest), std::end(extentsToTest)));
    CAPTURE(extent.width, extent.height, extent.depth);
    auto maxArrayCreateError = hipMallocMipmappedArray(&array, &desc, extent, numLevels, flag);
    // this can try to alloc many GB of memory, so out of memory is acceptable
    if (maxArrayCreateError == hipErrorOutOfMemory) return;
    HIP_CHECK(maxArrayCreateError);
    hipArray* hipArray = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&hipArray, array, 1));
    checkMipmappedArrayIsExpected(array, hipArray, desc, extent, flags);
    HIP_CHECK(hipFreeMipmappedArray(array));
  }
  SECTION("Negative") {
    std::vector<hipExtent> extentsToTest {
      make_hipExtent(sizes.max1D + 1, 0, 0),                          // 1D max
          make_hipExtent(sizes.max2D[0] + 1, s, 0),                   // 2D max width
          make_hipExtent(s, sizes.max2D[1] + 1, 0),                   // 2D max height
          make_hipExtent(sizes.max2D[0] + 1, sizes.max2D[1] + 1, 0),  // 2D max
          make_hipExtent(sizes.max3D[0] + 1, s, s),                   // 3D max width
          make_hipExtent(s, sizes.max3D[1] + 1, s),                   // 3D max height
#if !HT_NVIDIA                                       // leads to hipSuccess on NVIDIA
          make_hipExtent(s, s, sizes.max3D[2] + 1),  // 3D max depth
#endif
          make_hipExtent(s, sizes.max3D[1] + 1, sizes.max3D[2] + 1),  // 3D max height and depth
          make_hipExtent(sizes.max3D[0] + 1, s, sizes.max3D[2] + 1),  // 3D max width and depth
          make_hipExtent(sizes.max3D[0] + 1, sizes.max3D[1] + 1, s),  // 3D max width and height
          make_hipExtent(sizes.max3D[0] + 1, sizes.max3D[1] + 1, sizes.max3D[2] + 1)  // 3D max
    };
    const auto extent =
        GENERATE_COPY(from_range(std::begin(extentsToTest), std::end(extentsToTest)));
    CAPTURE(extent.width, extent.height, extent.depth);
    HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, extent, numLevels, flag), hipErrorInvalidValue);
  }
}


#if HT_AMD
constexpr std::array<unsigned int, 1> validFlags{hipArrayDefault};
#else
constexpr std::array<unsigned int, 9> validFlags{
    hipArrayDefault,
    hipArrayDefault | hipArraySurfaceLoadStore,
    hipArrayLayered,
    hipArrayLayered | hipArraySurfaceLoadStore,
    hipArrayCubemap,
    hipArrayCubemap | hipArrayLayered,
    hipArrayCubemap | hipArraySurfaceLoadStore,
    hipArrayCubemap | hipArrayLayered | hipArraySurfaceLoadStore,
    hipArrayTextureGather};
#endif

hipExtent makeMipmappedExtent(unsigned int flag, size_t s) {
  if (flag == hipArrayTextureGather) {
    return make_hipExtent(s, s, 0);
  }
  return make_hipExtent(s, s, s);
}


// Providing the array pointer as nullptr should return an error
TEST_CASE("Unit_hipMallocMipmappedArray_Negative_NullArrayPtr") {
  hipChannelFormatDesc desc = hipCreateChannelDesc<float4>();
  unsigned int numLevels = 1;
  constexpr size_t s = 6;

  const auto flag = GENERATE(from_range(std::begin(validFlags), std::end(validFlags)));
  HIP_CHECK_ERROR(hipMallocMipmappedArray(nullptr, &desc, makeMipmappedExtent(flag, s), numLevels, flag),
                  hipErrorInvalidValue);
}

// Providing the description pointer as nullptr should return an error
TEST_CASE("Unit_hipMallocMipmappedArray_Negative_NullDescPtr") {
  constexpr size_t s = 6;  // 6 to keep cubemap happy
  unsigned int numLevels = 1;
  hipMipmappedArray_t array;

  const auto flag = GENERATE(from_range(std::begin(validFlags), std::end(validFlags)));

  HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, nullptr, makeMipmappedExtent(flag, s), numLevels, flag),
                  hipErrorInvalidValue);
}

// Zero width arrays are not allowed
TEST_CASE("Unit_hipMallocMipmappedArray_Negative_ZeroWidth") {
  constexpr size_t s = 6;  // 6 to keep cubemap happy
  unsigned int numLevels = 1;
  hipMipmappedArray_t array;
  hipChannelFormatDesc desc = hipCreateChannelDesc<float4>();

  const auto flag = GENERATE(from_range(std::begin(validFlags), std::end(validFlags)));

  HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, make_hipExtent(0, s, s), numLevels, flag),
                  hipErrorInvalidValue);
}

// Zero height arrays are only allowed for 1D arrays and layered arrays
TEST_CASE("Unit_hipMallocMipmappedArray_Negative_ZeroHeight") {
  constexpr size_t s = 6;  // 6 to keep cubemap happy
  unsigned int numLevels = 1;
  hipMipmappedArray_t array;
  hipChannelFormatDesc desc = hipCreateChannelDesc<float4>();
  std::array<unsigned int, 2> exceptions{hipArrayLayered,
                                         hipArrayLayered | hipArraySurfaceLoadStore};

  const auto flag = GENERATE(from_range(std::begin(validFlags), std::end(validFlags)));

  if (std::find(std::begin(exceptions), std::end(exceptions), flag) == std::end(exceptions)) {
    // flag is not in list of exceptions
    HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, make_hipExtent(s, 0, s), numLevels, flag),
                    hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipMallocMipmappedArray_Negative_InvalidFlags") {
  constexpr size_t s = 6;  // 6 to keep cubemap happy
  unsigned int numLevels = 1;
  hipMipmappedArray_t array;
  hipChannelFormatDesc desc = hipCreateChannelDesc<float4>();

#if HT_AMD
  const unsigned int flag = 0xDEADBEEF;
#else
  const unsigned int flag =
      GENERATE(0xDEADBEEF, hipArrayTextureGather | hipArraySurfaceLoadStore,
               hipArrayTextureGather | hipArrayCubemap,
               hipArrayTextureGather | hipArraySurfaceLoadStore | hipArrayCubemap);
#endif

  CAPTURE(flag);

  REQUIRE(std::find(std::begin(validFlags), std::end(validFlags), flag) == std::end(validFlags));

  HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, makeMipmappedExtent(flag, s), numLevels, flag), hipErrorInvalidValue);
}

void testInvalidDescriptionMipmapped(hipChannelFormatDesc desc) {
  constexpr size_t s = 6;  // 6 to keep cubemap happy
  unsigned int numLevels = 1;
  hipMipmappedArray_t array;

#if HT_NVIDIA
  hipError_t expectedError = hipErrorUnknown;
#else
  hipError_t expectedError = hipErrorInvalidValue;
#endif

  const auto flag = GENERATE(from_range(std::begin(validFlags), std::end(validFlags)));
  HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, makeMipmappedExtent(flag, s), numLevels, flag), expectedError);
}

TEST_CASE("Unit_hipMallocMipmappedArray_Negative_InvalidFormat") {
  hipChannelFormatDesc desc = hipCreateChannelDesc<float4>();
  desc.f = GENERATE(hipChannelFormatKindNone, 0xBEEF);
  testInvalidDescriptionMipmapped(desc);
}

TEST_CASE("Unit_hipMallocMipmappedArray_Negative_BadChannelLayout") {
  const int bits = GENERATE(8, 16, 32);
  const hipChannelFormatKind formatKind =
      GENERATE(hipChannelFormatKindSigned, hipChannelFormatKindUnsigned, hipChannelFormatKindFloat);
  if (bits == 8 && formatKind == hipChannelFormatKindFloat) return;


  hipChannelFormatDesc desc = GENERATE_COPY(hipCreateChannelDesc(bits, bits, bits, 0, formatKind),
                                            hipCreateChannelDesc(0, bits, bits, 0, formatKind),
                                            hipCreateChannelDesc(0, bits, bits, bits, formatKind),
                                            hipCreateChannelDesc(bits, 0, bits, 0, formatKind),
                                            hipCreateChannelDesc(bits, bits, 0, bits, formatKind),
                                            hipCreateChannelDesc(0, 0, bits, 0, formatKind),
                                            hipCreateChannelDesc(0, 0, bits, bits, formatKind));

  INFO("kind: " << channelFormatString(formatKind));
  INFO("x: " << desc.x << ", y: " << desc.y << ", z: " << desc.z << ", w: " << desc.w);

  testInvalidDescriptionMipmapped(desc);
}

TEST_CASE("Unit_hipMallocMipmappedArray_Negative_8BitFloat") {
  hipChannelFormatDesc desc = GENERATE(hipCreateChannelDesc(8, 0, 0, 0, hipChannelFormatKindFloat),
                                       hipCreateChannelDesc(8, 8, 0, 0, hipChannelFormatKindFloat),
                                       hipCreateChannelDesc(8, 8, 8, 8, hipChannelFormatKindFloat));

  testInvalidDescriptionMipmapped(desc);
}

TEST_CASE("Unit_hipMallocMipmappedArray_Negative_DifferentChannelSizes") {
  const int bitsX = GENERATE(8, 16, 32);
  const int bitsY = GENERATE(8, 16, 32);
  const int bitsZ = GENERATE(8, 16, 32);
  const int bitsW = GENERATE(8, 16, 32);
  if (bitsX == bitsY && bitsY == bitsZ && bitsZ == bitsW) return;  // skip when they are equal

  const hipChannelFormatKind channelFormat =
      GENERATE(hipChannelFormatKindSigned, hipChannelFormatKindUnsigned, hipChannelFormatKindFloat);

  if (channelFormat == hipChannelFormatKindFloat &&
      (bitsX == 8 || bitsY == 8 || bitsZ == 8 || bitsW == 8))
    return;  // 8 bit floats aren't allowed

  hipChannelFormatDesc desc = hipCreateChannelDesc(bitsX, bitsY, bitsZ, bitsW, channelFormat);

  INFO("format: " << channelFormatString(channelFormat) << ", x bits: " << bitsX
                  << ", y bits: " << bitsY << ", z bits: " << bitsZ << ", w bits: " << bitsW);


  testInvalidDescriptionMipmapped(desc);
}

TEST_CASE("Unit_hipMallocMipmappedArray_Negative_BadChannelSize") {
  const int badBits = GENERATE(-1, 0, 10, 100);
  const hipChannelFormatKind formatKind =
      GENERATE(hipChannelFormatKindSigned, hipChannelFormatKindUnsigned, hipChannelFormatKindFloat);
  hipChannelFormatDesc desc = hipCreateChannelDesc(badBits, badBits, badBits, badBits, formatKind);

  INFO("Number of bits: " << badBits);

  testInvalidDescriptionMipmapped(desc);
}


// hipMallocMipmappedArray should handle the max numeric value gracefully.
TEST_CASE("Unit_hipMallocMipmappedArray_Negative_NumericLimit") {
  hipMipmappedArray_t arrayPtr;
  unsigned int numLevels = 1;
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();

  size_t size = std::numeric_limits<size_t>::max();
  const auto flag = GENERATE(from_range(std::begin(validFlags), std::end(validFlags)));
  HIP_CHECK_ERROR(hipMallocMipmappedArray(&arrayPtr, &desc, makeMipmappedExtent(flag, size), numLevels, flag),
                  hipErrorInvalidValue);
}

// texture gather arrays are only allowed to be 2D
TEMPLATE_TEST_CASE("Unit_hipMallocMipmappedArray_Negative_Non2DTextureGather", "", char, uchar2, float2) {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("Texture Gather arrays not supported using AMD backend");
  return;
#endif
  hipMipmappedArray_t array;
  unsigned int numLevels = 1;
  const auto desc = hipCreateChannelDesc<TestType>();

  constexpr unsigned int flags = hipArrayTextureGather;
  constexpr size_t size = 64;
  const hipExtent extent = GENERATE(make_hipExtent(size, 0, 0), make_hipExtent(size, size, size));

  HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, extent, numLevels, flags), hipErrorInvalidValue);
}

TEST_CASE("Unit_hipMallocMipmappedArray_Negative_NumLevels") {
  hipMipmappedArray_t array;
  constexpr size_t size = 6;
  unsigned int numLevels = floor(log2(size)) + 2;
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();

  const auto flag = GENERATE(from_range(std::begin(validFlags), std::end(validFlags)));
  HIP_CHECK_ERROR(hipMallocMipmappedArray(&array, &desc, makeMipmappedExtent(flag, size), numLevels, flag),
                  hipErrorInvalidValue);
}

TEST_CASE("Unit_hipGetMipmappedArrayLevel_Negative") {
  constexpr size_t s = 6;
  unsigned int numLevels = 1;
  hipMipmappedArray_t array;
  hipArray level_array;

  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();

  HIP_CHECK(hipMallocMipmappedArray(&array, &desc, make_hipExtent(s, s, s), numLevels, hipArrayDefault));
  SECTION("Level is invalid")
  {
    HIP_CHECK_ERROR(hipGetMipmappedArrayLevel(&level_array, array, 3), hipErrorInvalidValue);
  }
  SECTION("Mipmapped array is nullptr")
  {
    HIP_CHECK_ERROR(hipGetMipmappedArrayLevel(&level_array, nullptr, 1), hipErrorInvalidResourceHandle);
  }
  HIP_CHECK(hipFreeMipmappedArray(array));
}
