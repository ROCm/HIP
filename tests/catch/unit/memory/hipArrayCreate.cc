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
hipArrayCreate API test scenarios
1. Negative Scenarios
2. Allocating Small and big chunk data
3. Multithreaded scenario
*/

#include <array>
#include <numeric>
#include <hip_test_common.hh>
#include "hipArrayCommon.hh"
#include "DriverContext.hh"

static constexpr auto NUM_W{4};
static constexpr auto BIGNUM_W{100};
static constexpr auto NUM_H{4};
static constexpr auto BIGNUM_H{100};
static constexpr auto ARRAY_LOOP{100};

/*
 * This API verifies  memory allocations for small and
 * bigger chunks of data.
 * Two scenarios are verified in this API
 * 1. SmallArray: Allocates NUM_W*NUM_H in a loop and
 *    releases the memory and verifies the meminfo.
 * 2. BigArray: Allocates BIGNUM_W*BIGNUM_H in a loop and
 *    releases the memory and verifies the meminfo
 *
 * In both cases, the memory info before allocation and
 * after releasing the memory should be the same.
 *
 */

static void ArrayCreate_DiffSizes(int gpu) {
  HIP_CHECK(hipSetDevice(gpu));
  std::vector<std::pair<size_t, size_t>> array_size{{NUM_W, NUM_H}, {BIGNUM_W, BIGNUM_H}};
  for (auto& size : array_size) {
    std::array<HIP_ARRAY, ARRAY_LOOP> array;
    const size_t pavail = getFreeMem();
    HIP_ARRAY_DESCRIPTOR desc;
    desc.NumChannels = 1;
    desc.Width = std::get<0>(size);
    desc.Height = std::get<1>(size);
    desc.Format = HIP_AD_FORMAT_FLOAT;
    for (int i = 0; i < ARRAY_LOOP; i++) {
      HIP_CHECK(hipArrayCreate(&array[i], &desc));
    }
    for (int i = 0; i < ARRAY_LOOP; i++) {
      HIP_CHECK(hipArrayDestroy(array[i]));
    }
    const size_t avail = getFreeMem();
    if (pavail != avail) {
      HIPASSERT(false);
    }
  }
}

/* This testcase verifies hipArrayCreate API for small and big chunks data*/
TEST_CASE("Unit_hipArrayCreate_DiffSizes") { ArrayCreate_DiffSizes(0); }

/*
This testcase verifies the hipArrayCreate API in multithreaded
scenario by launching threads in parallel on multiple GPUs
and verifies the hipArrayCreate API with small and big chunks data
*/
TEST_CASE("Unit_hipArrayCreate_MultiThread") {
  std::vector<std::thread> threadlist;
  int devCnt = 0;

  devCnt = HipTest::getDeviceCount();

  const size_t pavail = getFreeMem();
  for (int i = 0; i < devCnt; i++) {
    // FIXME: the HIP_CHECK and HIPASSERT are not threadsafe so this test is broken.
    threadlist.push_back(std::thread(ArrayCreate_DiffSizes, i));
  }

  for (auto& t : threadlist) {
    t.join();
  }
  const size_t avail = getFreeMem();

  if (pavail != avail) {
    WARN("Memory leak of hipMalloc3D API in multithreaded scenario");
    REQUIRE(false);
  }
}


// All the possible formats for channel data in an array.
static const std::vector<hipArray_Format> formats{
    HIP_AD_FORMAT_UNSIGNED_INT8, HIP_AD_FORMAT_UNSIGNED_INT16, HIP_AD_FORMAT_UNSIGNED_INT32,
    HIP_AD_FORMAT_SIGNED_INT8,   HIP_AD_FORMAT_SIGNED_INT16,   HIP_AD_FORMAT_SIGNED_INT32,
    HIP_AD_FORMAT_HALF,          HIP_AD_FORMAT_FLOAT};

// Helpful for printing errors
const char* formatToString(hipArray_Format f) {
  switch (f) {
    case HIP_AD_FORMAT_UNSIGNED_INT8:
      return "Unsigned Int 8";
    case HIP_AD_FORMAT_UNSIGNED_INT16:
      return "Unsigned Int 16";
    case HIP_AD_FORMAT_UNSIGNED_INT32:
      return "Unsigned Int 32";
    case HIP_AD_FORMAT_SIGNED_INT8:
      return "Signed Int 8";
    case HIP_AD_FORMAT_SIGNED_INT16:
      return "Signed Int 16";
    case HIP_AD_FORMAT_SIGNED_INT32:
      return "Signed Int 32";
    case HIP_AD_FORMAT_HALF:
      return "Float 16";
    case HIP_AD_FORMAT_FLOAT:
      return "Float 32";
    default:
      return "not found";
  }
}

// Tests /////////////////////////////////////////

#if HT_AMD
constexpr auto MemoryTypeHost = hipMemoryTypeHost;
constexpr auto MemoryTypeArray = hipMemoryTypeArray;
constexpr auto NORMALIZED_COORDINATES = HIP_TRSF_NORMALIZED_COORDINATES;
constexpr auto READ_AS_INTEGER = HIP_TRSF_READ_AS_INTEGER;
#else
constexpr auto MemoryTypeHost = CU_MEMORYTYPE_HOST;
constexpr auto MemoryTypeArray = CU_MEMORYTYPE_ARRAY;
// (EXSWCPHIPT-92) HIP equivalents not defined for CUDA backend.
constexpr auto NORMALIZED_COORDINATES = CU_TRSF_NORMALIZED_COORDINATES;
constexpr auto READ_AS_INTEGER = CU_TRSF_READ_AS_INTEGER;
#endif

// Copy data from host to the hiparray, accounting 1D or 2D arrays
template <typename T>
void copyToArray(hiparray dst, const std::vector<T>& src, const size_t height) {
  const auto sizeInBytes = src.size() * sizeof(T);
  if (height == 0) {
    // FIXME(EXSWCPHIPT-64) remove cast when API is fixed (will require major version change)
    HIP_CHECK(hipMemcpyHtoA(reinterpret_cast<hipArray*>(dst), 0, src.data(), sizeInBytes));
  } else {
    const auto pitch = sizeInBytes / height;
    hip_Memcpy2D copyParams{};
    copyParams.srcMemoryType = MemoryTypeHost;
    copyParams.srcXInBytes = 0;  // x offset
    copyParams.srcY = 0;         // y offset
    copyParams.srcHost = src.data();
    copyParams.srcPitch = pitch;


    copyParams.dstMemoryType = MemoryTypeArray;
    copyParams.dstXInBytes = 0;  // x offset
    copyParams.dstY = 0;         // y offset
    copyParams.dstArray = dst;

    copyParams.WidthInBytes = pitch;
    copyParams.Height = height;

    HIP_CHECK(hipMemcpyParam2D(&copyParams));
  }
}

// Test the allocated array by generating a texture from it then reading from that texture.
// Textures are read-only, so write to the array then copy that into normal device memory.
template <typename T>
void testArrayAsTexture(hiparray array, const size_t width, const size_t height) {
  using vec_info = vector_info<T>;
  using scalar_type = typename vec_info::type;
  const auto h = height ? height : 1;
  const auto size = sizeof(T) * width * h;

  // set hip array
  std::vector<scalar_type> hostData(width * h * vec_info::size);
  // assigned ascending values to the data array to show indexing is working
  std::iota(std::begin(hostData), std::end(hostData), 0);

  copyToArray(array, hostData, height);

  // create texture
  hipTextureObject_t textObj{};

  HIP_RESOURCE_DESC resDesc{};
  memset(&resDesc, 0, sizeof(HIP_RESOURCE_DESC));
  resDesc.resType = HIP_RESOURCE_TYPE_ARRAY;
  resDesc.res.array.hArray = array;
  resDesc.flags = 0;

  HIP_TEXTURE_DESC texDesc{};
  memset(&texDesc, 0, sizeof(HIP_TEXTURE_DESC));
  // use the actual values in the texture, not normalized data
  texDesc.filterMode = HIP_TR_FILTER_MODE_POINT;
  // Use normalized coordinates and also read the data in the original data type
  texDesc.flags |= NORMALIZED_COORDINATES | READ_AS_INTEGER;

  HIP_CHECK(hipTexObjectCreate(&textObj, &resDesc, &texDesc, nullptr));

  // run kernel
  T* device_data{};
  HIP_CHECK(hipMalloc(&device_data, size));
  readFromTexture<<<dim3(width / BlockSize, height ? height / BlockSize : 1, 1),
                    dim3(BlockSize, height ? BlockSize : 1, 1)>>>(device_data, textObj, width,
                                                                  height, false);
  HIP_CHECK(hipGetLastError());  // check for errors when running the kernel

  // copy data back and then test it
  std::fill(std::begin(hostData), std::end(hostData), 0);
  HIP_CHECK(hipMemcpy(hostData.data(), device_data, size, hipMemcpyDeviceToHost));

  checkDataIsAscending(hostData);

  // clean up
  HIP_CHECK(hipTexObjectDestroy(textObj));
  HIP_CHECK(hipFree(device_data));
}

// Selection of types chosen since trying all types would be slow to compile
// Test the happy path of the hipArrayCreate
TEMPLATE_TEST_CASE("Unit_hipArrayCreate_happy", "", uint, int, int4, ushort, short2, char, uchar2,
                   char4, float, float2, float4) {
#if HT_AMD
  if (std::is_same<uint, TestType>::value || std::is_same<ushort, TestType>::value ||
      std::is_same<uchar2, TestType>::value) {
    HipTest::HIP_SKIP_TEST("Probably EXSWCPHIPT-62");
    return;
  }
#endif
  using vec_info = vector_info<TestType>;
  DriverContext ctx;

  HIP_ARRAY_DESCRIPTOR desc;
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;
  desc.Width = 1024;
  desc.Height = GENERATE(0, 1024);

  size_t initFree = getFreeMem();

  // pointer to the array in device memory
  hiparray array{};

  HIP_CHECK(hipArrayCreate(&array, &desc));

  testArrayAsTexture<TestType>(array, desc.Width, desc.Height);

  size_t finalFree = getFreeMem();

  const size_t allocSize = sizeof(TestType) * desc.Width * (desc.Height ? desc.Height : 1);
  // will be aligned to some size, so this is not exact
  REQUIRE(initFree - finalFree >= allocSize);

  HIP_CHECK(hipArrayDestroy(array));
}


// Only widths and Heights up to the maxTexture size is supported
TEMPLATE_TEST_CASE("Unit_hipArrayCreate_maxTexture", "", uint, int, int4, ushort, short2, char,
                   uchar2, char4, float, float2, float4) {
  using vec_info = vector_info<TestType>;
  DriverContext ctx;

  HIP_ARRAY_DESCRIPTOR desc;
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;

  int device;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  hiparray array{};
  SECTION("Happy") {
    SECTION("1D - Max") {
      desc.Width = prop.maxTexture1D;
      desc.Height = 0;
    }
    SECTION("2D - Max Width") {
      desc.Width = prop.maxTexture2D[0];
      desc.Height = 64;
    }
    SECTION("2D - Max Height") {
      desc.Width = 64;
      desc.Height = prop.maxTexture2D[1];
    }
    SECTION("2D - Max Width and Height") {
      desc.Width = prop.maxTexture2D[0];
      desc.Height = prop.maxTexture2D[1];
    }
    auto maxArrayCreateError = hipArrayCreate(&array, &desc);
    // this can try to alloc many GB of memory, so out of memory is acceptable
    // return to avoid destroy
    if (maxArrayCreateError == hipErrorOutOfMemory) return;
    HIP_CHECK(maxArrayCreateError);
    HIP_CHECK(hipArrayDestroy(array));
  }
  SECTION("Negative") {
    SECTION("1D - More Than Max") {
      desc.Width = prop.maxTexture1D + 1;
      desc.Height = 0;
    }
    SECTION("2D - More Than Max Width") {
      desc.Width = prop.maxTexture2D[0] + 1;
      desc.Height = 64;
    }
    SECTION("2D - More Than Max Height") {
      desc.Width = 64;
      desc.Height = prop.maxTexture2D[1] + 1;
    }
    SECTION("2D - More Than Max Width and Height") {
      desc.Width = prop.maxTexture2D[0] + 1;
      desc.Height = prop.maxTexture2D[1] + 1;
    }
    HIP_CHECK_ERROR(hipArrayCreate(&array, &desc), hipErrorInvalidValue);
  }
}

// zero-width array is not supported
TEST_CASE("Unit_hipArrayCreate_ZeroWidth") {
  DriverContext ctx;
  HIP_ARRAY_DESCRIPTOR desc;
  desc.Format = formats[0];
  desc.NumChannels = 4;
  desc.Width = 0;
  desc.Height = GENERATE(0, 1024);

  // pointer to the array in device memory
  hiparray array;
  HIP_CHECK_ERROR(hipArrayCreate(&array, &desc), hipErrorInvalidValue);
}

// HipArrayCreate will return an error when nullptr is used as the array argument
TEST_CASE("Unit_hipArrayCreate_Nullptr") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("Probably EXSWCPHIPT-45");
  return;
#endif
  DriverContext ctx;
  SECTION("Null array") {
    HIP_ARRAY_DESCRIPTOR desc;
    desc.Format = formats[0];
    desc.NumChannels = 4;
    desc.Width = 1024;
    desc.Height = 1024;

    HIP_CHECK_ERROR(hipArrayCreate(nullptr, &desc), hipErrorInvalidValue);
  }
  SECTION("Null Description") {
    hiparray array;
    HIP_CHECK_ERROR(hipArrayCreate(&array, nullptr), hipErrorInvalidValue);
  }
}

// Only elements with 1,2, or 4 channels is supported
TEST_CASE("Unit_hipArrayCreate_BadNumberChannelElement") {
  DriverContext ctx;
  HIP_ARRAY_DESCRIPTOR desc;
  desc.Format = GENERATE(from_range(std::begin(formats), std::end(formats)));
  desc.NumChannels = GENERATE(-1, 0, 3, 5, 8);
  desc.Width = 1024;
  desc.Height = GENERATE(0, 1024);

  hiparray array;

  INFO("Format: " << formatToString(desc.Format) << " NumChannels: " << desc.NumChannels
                  << " Height: " << desc.Height)
  HIP_CHECK_ERROR(hipArrayCreate(&array, &desc), hipErrorInvalidValue);
}

// Only certain channel formats are acceptable.
TEST_CASE("Unit_hipArrayCreate_BadChannelFormat") {
  DriverContext ctx;
  HIP_ARRAY_DESCRIPTOR desc;

  // create a bad format
  desc.Format =
      std::accumulate(std::begin(formats), std::end(formats), formats[0],
                      [](auto i, auto f) { return static_cast<decltype(desc.Format)>(i + f); });
  for (auto&& format : formats) {
    REQUIRE(desc.Format != format);
  }

  desc.NumChannels = 4;
  desc.Width = 1024;
  desc.Height = GENERATE(0, 1024);

  hiparray array;

  INFO("Format: " << formatToString(desc.Format) << " Height: " << desc.Height)
  HIP_CHECK_ERROR(hipArrayCreate(&array, &desc), hipErrorInvalidValue);
}
