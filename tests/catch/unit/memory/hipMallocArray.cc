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
hipMallocArray API test scenarios
1. Basic Functionality
2. Negative Scenarios
3. Allocating Small and big chunk data
4. Multithreaded scenario
*/

#include <hip_test_common.hh>
#include <limits>
#if defined(_WIN32) || defined(_WIN64)
#include <numeric>
#endif

static constexpr auto NUM_W{4};
static constexpr auto BIGNUM_W{100};
static constexpr auto BIGNUM_H{100};
static constexpr auto NUM_H{4};
static constexpr auto ARRAY_LOOP{100};

/*
 * This API verifies  memory allocations for small and
 * bigger chunks of data.
 * Two scenarios are verified in this API
 * 1. NUM_W(small Data): Allocates NUM_W*NUM_H in a loop and
 *    releases the memory and verifies the meminfo.
 * 2. BIGNUM_W(big data): Allocates BIGNUM_W*BIGNUM_H in a loop and
 *    releases the memory and verifies the meminfo
 *
 * In both cases, the memory info before allocation and
 * after releasing the memory should be the same
 *
 */
static void MallocArray_DiffSizes(int gpu) {
  HIP_CHECK(hipSetDevice(gpu));
  std::vector<std::pair<size_t, size_t>> array_size{{NUM_W, NUM_H}, {BIGNUM_W, BIGNUM_H}};
  for (auto& size : array_size) {
    std::array<hipArray_t, ARRAY_LOOP> A_d;
    size_t tot, avail, ptot, pavail;
    hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
    HIP_CHECK(hipMemGetInfo(&pavail, &ptot));
    for (int i = 0; i < ARRAY_LOOP; i++) {
      HIP_CHECK(
          hipMallocArray(&A_d[i], &desc, std::get<0>(size), std::get<1>(size), hipArrayDefault));
    }
    for (int i = 0; i < ARRAY_LOOP; i++) {
      HIP_CHECK(hipFreeArray(A_d[i]));
    }
    HIP_CHECK(hipMemGetInfo(&avail, &tot));
    if ((pavail != avail)) {
      HIPASSERT(false);
    }
  }
}

TEST_CASE("Unit_hipMallocArray_DiffSizes") { MallocArray_DiffSizes(0); }

/*
This testcase verifies the hipMallocArray API in multithreaded
scenario by launching threads in parallel on multiple GPUs
and verifies the hipMallocArray API with small and big chunks data
*/
TEST_CASE("Unit_hipMallocArray_MultiThread") {
  std::vector<std::thread> threadlist;
  int devCnt = 0;
  devCnt = HipTest::getDeviceCount();
  size_t tot, avail, ptot, pavail;
  HIP_CHECK(hipMemGetInfo(&pavail, &ptot));
  for (int i = 0; i < devCnt; i++) {
    // TODO the HIP_CHECK and HIPASSERT are not threadsafe so this test is broken.
    threadlist.push_back(std::thread(MallocArray_DiffSizes, i));
  }

  for (auto& t : threadlist) {
    t.join();
  }
  HIP_CHECK(hipMemGetInfo(&avail, &tot));

  if (pavail != avail) {
    WARN("Memory leak of hipMalloc3D API in multithreaded scenario");
    REQUIRE(false);
  }
}


constexpr size_t BlockSize = 16;

template <class T, size_t N> struct type_and_size {
  using type = T;
  static constexpr size_t size = N;
};

// scalars are interpreted as a vector of 1 length.
// template <size_t N> using int_constant = std::integral_constant<size_t, N>;
template <typename T> struct vector_info;
template <> struct vector_info<int> : type_and_size<int, 1> {};
template <> struct vector_info<float> : type_and_size<float, 1> {};
template <> struct vector_info<short> : type_and_size<short, 1> {};
template <> struct vector_info<char> : type_and_size<char, 1> {};
template <> struct vector_info<unsigned int> : type_and_size<unsigned int, 1> {};
template <> struct vector_info<unsigned short> : type_and_size<unsigned short, 1> {};
template <> struct vector_info<unsigned char> : type_and_size<unsigned char, 1> {};

template <> struct vector_info<int2> : type_and_size<int, 2> {};
template <> struct vector_info<float2> : type_and_size<float, 2> {};
template <> struct vector_info<short2> : type_and_size<short, 2> {};
template <> struct vector_info<char2> : type_and_size<char, 2> {};
template <> struct vector_info<uint2> : type_and_size<unsigned int, 2> {};
template <> struct vector_info<ushort2> : type_and_size<unsigned short, 2> {};
template <> struct vector_info<uchar2> : type_and_size<unsigned char, 2> {};

template <> struct vector_info<int4> : type_and_size<int, 4> {};
template <> struct vector_info<float4> : type_and_size<float, 4> {};
template <> struct vector_info<short4> : type_and_size<short, 4> {};
template <> struct vector_info<char4> : type_and_size<char, 4> {};
template <> struct vector_info<uint4> : type_and_size<unsigned int, 4> {};
template <> struct vector_info<ushort4> : type_and_size<unsigned short, 4> {};
template <> struct vector_info<uchar4> : type_and_size<unsigned char, 4> {};

// Kernels ///////////////////////////////////////

// read from a texture using normalized coordinates
constexpr size_t ChannelToRead = 1;
template <typename T>
__global__ void readFromTexture(T* output, hipTextureObject_t texObj, size_t width, size_t height,
                                bool textureGather) {
  // Calculate normalized texture coordinates
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const float u = x / (float)width;

  // Read from texture and write to global memory
  if (height == 0) {
    output[x] = tex1D<T>(texObj, u);
  } else {
    const float v = y / (float)height;
    output[y * width + x] =
        textureGather ? tex2Dgather<T>(texObj, u, v, ChannelToRead) : tex2D<T>(texObj, u, v);
  }
}

template <typename T> __device__ void addOne(T* a) {
  using scalar_type = typename vector_info<T>::type;
  auto as = reinterpret_cast<scalar_type*>(a);
  for (size_t i = 0; i < vector_info<T>::size; ++i) {
    as[i] = as[i] + static_cast<scalar_type>(1);
  }
}

// read from a surface and write to another
template <typename T> __global__ void incSurface(hipSurfaceObject_t surf, size_t height) {
  // Calculate surface coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (height == 0) {
    T data;
    surf1Dread(&data, surf, x * sizeof(T));
    addOne(&data);  // change the value to show that write works
    surf1Dwrite(data, surf, x * sizeof(T));
  } else {
    T data;
    surf2Dread(&data, surf, x * sizeof(T), y);
    addOne(&data);  // change the value to show that write works
    surf2Dwrite(data, surf, x * sizeof(T), y);
  }
}

// Helpers ///////////////////////////////////////

template <typename T> size_t getAllocSize(const size_t width, const size_t height) noexcept {
  return sizeof(T) * width * (height ? height : 1);
}

template <typename T> void checkDataIsAscending(const std::vector<T>& hostData) {
  bool allMatch = true;
  size_t i = 0;
  for (; i < hostData.size(); ++i) {
    allMatch = allMatch && hostData[i] == static_cast<T>(i);
    if (!allMatch) break;
  }
  INFO("hostData[" << i << "] == " << static_cast<T>(hostData[i]));
  REQUIRE(allMatch);
}

const char* channelFormatString(hipChannelFormatKind formatKind) noexcept {
  switch (formatKind) {
    case hipChannelFormatKindFloat:
      return "float";
    case hipChannelFormatKindSigned:
      return "signed";
    case hipChannelFormatKindUnsigned:
      return "unsigned";
    default:
      return "error";
  }
}

// Tests /////////////////////////////////////////

// Test the default array by generating a texture from it then reading from that texture.
// Textures are read-only so write to the array then copy from the texture into normal device memory
template <typename T>
void testArrayAsTexture(hipArray_t arrayPtr, const size_t width, const size_t height) {
  using scalar_type = typename vector_info<T>::type;
  constexpr auto vec_size = vector_info<T>::size;

  const auto h = height ? height : 1;
  const size_t pitch = width * sizeof(T);  // no padding
  const auto size = pitch * h;

  // create an array to initialize the hip array, then later use it to hold the result
  std::vector<scalar_type> hostData(width * h * vec_size);

  // Setup backing array
  // assign ascending values to the data array to show indexing is working.
  std::iota(std::begin(hostData), std::end(hostData), 0);
  HIP_CHECK(
      hipMemcpy2DToArray(arrayPtr, 0, 0, hostData.data(), pitch, pitch, h, hipMemcpyHostToDevice));


  // create texture
  hipTextureObject_t textObj{};
  hipResourceDesc resDesc{};
  memset(&resDesc, 0, sizeof(hipResourceDesc));
  // enum to store how to resDesc.res union is being used
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = arrayPtr;

  hipTextureDesc textDesc{};
  memset(&textDesc, 0, sizeof(hipTextureDesc));
  textDesc.filterMode =
      hipFilterModePoint;  // use the actual values in the texture, not normalized data
  textDesc.readMode = hipReadModeElementType;  // don't convert the data to floats
  textDesc.normalizedCoords = 1;               // use normalized coordinates (0.0-1.0)

  HIP_CHECK(hipCreateTextureObject(&textObj, &resDesc, &textDesc, nullptr));


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
  HIP_CHECK(hipDestroyTextureObject(textObj));
  HIP_CHECK(hipFree(device_data));
}

// Test an array created with the TextureGather flag.
// First generating a texture from the array then reading from that texture.
// Textures are read-only so first write to the array then copy from the texture into normal device
// memory. Texture Gather works by taking the nth channel from the 4 elements used for sampling from
// the texture using bilinear filtering (bilinear interpolation)
//
//  Example
//
//  |
//  | A     B
//  |     x
//  |
//  | C     D
//  |___________
//
// if `x` is the point sampled, texture gather is set to query the 3nd channel, and A=(1,2,3,4),
// B=(5,6,7,8), C=(9,a,b,c) D=(d,e,f,0) then the output of the sample would be (3,7,b,f) (assuming
// the points are chosen in that order)
// when the channel queried doesn't exist, the value 0 should be returned.
template <typename T>
void testArrayAsTextureWithGather(hipArray_t arrayPtr, const size_t width, const size_t height) {
  REQUIRE(height != 0);  // 1D TextureGather isn't allowed
  using scalar_type = typename vector_info<T>::type;
  constexpr auto vec_size = vector_info<T>::size;

  const size_t pitch = width * sizeof(T);  // no padding
  const auto size = pitch * height;

  std::vector<scalar_type> hostData(width * height * vec_size);

  // Setup backing array
  // assign ascending values to the data array to show indexing is working.
  std::iota(std::begin(hostData), std::end(hostData), 0);
  HIP_CHECK(hipMemcpy2DToArray(arrayPtr, 0, 0, hostData.data(), pitch, pitch, height,
                               hipMemcpyHostToDevice));

  // create texture
  hipTextureObject_t textObj{};
  hipResourceDesc resDesc{};
  memset(&resDesc, 0, sizeof(hipResourceDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = arrayPtr;

  hipTextureDesc textDesc{};
  memset(&textDesc, 0, sizeof(hipTextureDesc));
  textDesc.filterMode =
      hipFilterModePoint;  // use the actual values in the texture, not normalized data
  textDesc.readMode = hipReadModeElementType;    // don't convert the data to floats
  textDesc.addressMode[0] = hipAddressModeWrap;  // for queries outside the texture...
  textDesc.addressMode[1] = hipAddressModeWrap;  // wrap around in all dimensions
  textDesc.addressMode[2] = hipAddressModeWrap;
  textDesc.normalizedCoords = 1;  // use normalized coordinates (0.0 - 1.0)

  HIP_CHECK(hipCreateTextureObject(&textObj, &resDesc, &textDesc, nullptr));

  // run kernel
  T* device_data{};
  HIP_CHECK(hipMalloc(&device_data, size));
  readFromTexture<T>
      <<<dim3(width / BlockSize, height / BlockSize, 1), dim3(BlockSize, BlockSize, 1)>>>(
          device_data, textObj, width, height, true);
  HIP_CHECK(hipGetLastError());

  // copy data back
  std::fill(std::begin(hostData), std::end(hostData), 0);
  HIP_CHECK(hipMemcpy(hostData.data(), device_data, size, hipMemcpyDeviceToHost));

  if (ChannelToRead >= vec_size) {
    // we expect all the values to be zero
    auto not_zero_idx = std::find_if(std::begin(hostData), std::end(hostData), [](scalar_type& x) {
      return x != static_cast<scalar_type>(0);
    });
    CAPTURE(std::distance(std::begin(hostData), not_zero_idx));
    REQUIRE(not_zero_idx == std::end(hostData));
  } else {
    // convert a row and column of the element into the index of the first channel of the element
    // also accounts for the wrap-around
    // use int to deal with negative indexes
    auto toIndex = [width, height](int row, int column) -> size_t {
      auto wrap = [](int value, int wrapSize) {
        auto v = value % wrapSize;
        return v < 0 ? wrapSize + v : v;
      };
      const auto c = wrap(column, width);
      const auto r = wrap(row, height);
      return vec_size * (width * r + c);
    };

    // calculate the index of the values that would have been used for bilinear filtering
    // then check that the values in the element are those indexes
    bool allMatch = true;
    size_t dataIdx = 0;
    for (size_t row = 0; allMatch && row < height; ++row) {
      for (size_t col = 0; allMatch && col < width; ++col) {
        // coordinates of the elements used for bilinear filtering
        std::array<scalar_type, 4> elementIndexes = {
            static_cast<scalar_type>(toIndex(row, static_cast<int>(col) - 1)),
            static_cast<scalar_type>(toIndex(row, col)),
            static_cast<scalar_type>(toIndex(static_cast<int>(row) - 1, col)),
            static_cast<scalar_type>(
                toIndex(static_cast<int>(row) - 1, static_cast<int>(col) - 1))};

        // add offset for the channel that is selected
        std::for_each(std::begin(elementIndexes), std::end(elementIndexes),
                      [](scalar_type& x) { x += static_cast<scalar_type>(ChannelToRead); });

        // calculate the output we are looking at
        dataIdx = vec_size * (width * row + col);

        // test each value sampled
        for (int channel = 0; channel < vec_size; ++channel) {
          allMatch = allMatch && hostData[dataIdx + channel] == elementIndexes[channel];
        }
      }
    }
    CAPTURE(dataIdx, hostData[dataIdx], hostData[dataIdx + 1], hostData[dataIdx + 2],
            hostData[dataIdx + 3],
            static_cast<scalar_type>(toIndex(0, -1)) + static_cast<scalar_type>(ChannelToRead),
            static_cast<scalar_type>(toIndex(0, 0)) + static_cast<scalar_type>(ChannelToRead),
            static_cast<scalar_type>(toIndex(-1, 0)) + static_cast<scalar_type>(ChannelToRead),
            static_cast<scalar_type>(toIndex(-1, -1)) + static_cast<scalar_type>(ChannelToRead));
    REQUIRE(allMatch);
  }

  // clean up
  HIP_CHECK(hipDestroyTextureObject(textObj));
  HIP_CHECK(hipFree(device_data));
}

// Test the an array created with the SurfaceLoadStore flag by generating a surface and reading from
// it and writing to it.
template <typename T>
void testArrayAsSurface(hipArray_t arrayPtr, const size_t width, const size_t height) {
  using scalar_type = typename vector_info<T>::type;
  constexpr auto vec_size = vector_info<T>::size;

  const auto h = height ? height : 1;
  const size_t pitch = width * sizeof(T);  // no padding
  const auto size = pitch * h;

  std::vector<scalar_type> hostData(width * h * vec_size);

  // Setup backing array
  // assign ascending values to the data array to show indexing is working.
  std::iota(std::begin(hostData), std::end(hostData), 0);
  HIP_CHECK(
      hipMemcpy2DToArray(arrayPtr, 0, 0, hostData.data(), pitch, pitch, h, hipMemcpyHostToDevice));


  // create surface
  hipSurfaceObject_t surfObj{};
  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(hipResourceDesc));
  resDesc.resType = hipResourceTypeArray;

  resDesc.res.array.array = arrayPtr;
  HIP_CHECK(hipCreateSurfaceObject(&surfObj, &resDesc));


  // run kernel
  T* device_data{};
  HIP_CHECK(hipMalloc(&device_data, size));
  // This will increment the values of the surface, so this is undone later
  incSurface<T><<<dim3(width / BlockSize, height ? height / BlockSize : 1, 1),
                  dim3(BlockSize, height ? BlockSize : 1, 1)>>>(surfObj, height);
  HIP_CHECK(hipGetLastError());  // check for errors when running the kernel


  // copy data back and then test it
  std::fill(std::begin(hostData), std::end(hostData), 0);
  HIP_CHECK(hipMemcpy2DFromArray(hostData.data(), pitch, arrayPtr, 0, 0, pitch, h,
                                 hipMemcpyDeviceToHost));


  // undo the increment
  std::for_each(std::begin(hostData), std::end(hostData),
                [](scalar_type& x) { x -= static_cast<scalar_type>(1); });
  checkDataIsAscending(hostData);

  // clean up
  HIP_CHECK(hipDestroySurfaceObject(surfObj));
  HIP_CHECK(hipFree(device_data));
}

size_t getFreeMem() {
  size_t free = 0, total = 0;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  return free;
}

// The happy path of a default array and a SurfaceLoadStore array should work
// Selection of types chosen to reduce compile times
TEMPLATE_TEST_CASE("Unit_hipMallocArray_happy", "", uint, int, int4, ushort, short2, char, uchar2,
                   char4, float, float2, float4) {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-62");
  return;
#endif

  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  size_t init_free = getFreeMem();

  // pointer to the array in device memory
  hipArray_t arrayPtr{};
  size_t width = 1024;
  size_t height;

  SECTION("hipArrayDefault") {
    height = GENERATE(0, 1024);
    INFO("flag is hipArrayDefault");
    INFO("height: " << height);

    HIP_CHECK(hipMallocArray(&arrayPtr, &desc, width, height, hipArrayDefault));
    testArrayAsTexture<TestType>(arrayPtr, width, height);
  }
#if HT_NVIDIA  // surfaces not supported on AMD
  SECTION("hipArraySurfaceLoadStore") {
    height = GENERATE(0, 1024);
    INFO("flag is hipArraySurfaceLoadStore");
    INFO("height: " << height);

    HIP_CHECK(hipMallocArray(&arrayPtr, &desc, width, height, hipArraySurfaceLoadStore));
    testArrayAsSurface<TestType>(arrayPtr, width, height);
  }
  SECTION("hipArrayTextureGather") {
    height = 1024;
    INFO("flag is hipArrayTextureGather");
    INFO("height: " << height);

    HIP_CHECK(hipMallocArray(&arrayPtr, &desc, width, height, hipArrayTextureGather));
    testArrayAsTextureWithGather<TestType>(arrayPtr, width, height);
  }
#endif

  size_t final_free = getFreeMem();

  const size_t alloc_size = getAllocSize<TestType>(width, height);
  // alloc will be chunked, so this is not exact
  REQUIRE(init_free - final_free >= alloc_size);

  HIP_CHECK(hipFreeArray(arrayPtr));
}

// Arrays can be up to the size of maxTexture* but no bigger
// EXSWCPHIPT-71 - no equivalent value for maxSurface and maxTexture2DGather.
TEMPLATE_TEST_CASE("Unit_hipMallocArray_MaxTexture_Default", "", uint, int4, ushort, short2, char,
                   char4, float2, float4) {
  int device;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  size_t width, height;
  hipArray_t array{};
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();
  const unsigned int flag = hipArrayDefault;

  SECTION("Happy") {
    SECTION("1D - Max") {
      width = prop.maxTexture1D;
      height = 0;
    }
    SECTION("2D - Max Width") {
      width = prop.maxTexture2D[0];
      height = 64;
    }
    SECTION("2D - Max Height") {
      width = 64;
      height = prop.maxTexture2D[1];
    }
    SECTION("2D - Max Width and Height") {
      width = prop.maxTexture2D[0];
      height = prop.maxTexture2D[1];
    }
    auto maxArrayCreateError = hipMallocArray(&array, &desc, width, height, flag);
    // this can try to alloc many GB of memory, so out of memory is fair
    if (maxArrayCreateError == hipErrorOutOfMemory) return;
    HIP_CHECK(maxArrayCreateError);
    HIP_CHECK(hipFreeArray(array));
  }
  SECTION("Negative") {
    SECTION("1D - More Than Max") {
      width = prop.maxTexture1D + 1;
      height = 0;
    }
    SECTION("2D - More Than Max Width") {
      width = prop.maxTexture2D[0] + 1;
      height = 64;
    }
    SECTION("2D - More Than Max Height") {
      width = 64;
      height = prop.maxTexture2D[1] + 1;
    }
    SECTION("2D - More Than Max Width and Height") {
      width = prop.maxTexture2D[0] + 1;
      height = prop.maxTexture2D[1] + 1;
    }
    HIP_CHECK_ERROR(hipMallocArray(&array, &desc, width, height, flag), hipErrorInvalidValue);
  }
}


// Arrays with channels of different size are not allowed.
TEST_CASE("Unit_hipMallocArray_Negative_DifferentChannelSizes") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-59");
  return;
#endif
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
  REQUIRE(desc.x == bitsX);
  REQUIRE(desc.y == bitsY);
  REQUIRE(desc.z == bitsZ);
  REQUIRE(desc.w == bitsW);

  hipArray_t arrayPtr{};
  size_t width = 1024;
  size_t height = 1024;

  INFO("format: " << channelFormatString(channelFormat) << ", x bits: " << bitsX
                  << ", y bits: " << bitsY << ", z bits: " << bitsZ << ", w bits: " << bitsW);

#if HT_AMD
  unsigned int flag = hipArrayDefault;
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, width, height, flag), hipErrorInvalidValue);
#else
  unsigned int flag = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore, hipArrayTextureGather);
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, width, height, flag), hipErrorUnknown);
#endif
}

// Zero-width array is not supported
TEST_CASE("Unit_hipMallocArray_Negative_ZeroWidth") {
  hipChannelFormatDesc desc = hipCreateChannelDesc<float4>();

  // pointer to the array in device memory
  hipArray_t arrayPtr;

  size_t width = 0;
  size_t height = GENERATE(0, 32);

  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, width, height, hipArrayDefault),
                  hipErrorInvalidValue);
}

// Providing the array pointer as nullptr should return an error
TEST_CASE("Unit_hipMallocArray_Negative_NullArrayPtr") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-45");
  return;
#endif
  hipChannelFormatDesc desc = hipCreateChannelDesc<float4>();

  HIP_CHECK_ERROR(hipMallocArray(nullptr, &desc, 1024, 0, hipArrayDefault), hipErrorInvalidValue);
}

// Providing the desc pointer as nullptr should return an error
TEST_CASE("Unit_hipMallocArray_Negative_NullDescPtr") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-83");
  return;
#endif
  hipArray_t arrayPtr;
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, nullptr, 1024, 0, hipArrayDefault),
                  hipErrorInvalidValue);
}

// Inappropriate but related flags should still return an error
TEST_CASE("Unit_hipMallocArray_Negative_BadFlags") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-72");
  return;
#endif
  hipChannelFormatDesc desc = hipCreateChannelDesc<float4>();

  hipArray_t arrayPtr;
  SECTION("Flags that dont work with 1D") {
#if HT_AMD
    // * cudaArrayLayered           0x01 - 1
    // * cudaArrayCubemap           0x04 - 4
    unsigned int flag =
        GENERATE(hipArrayLayered, hipArrayCubemap, hipArrayLayered | hipArrayCubemap);
#else
    // * cudaArrayTextureGather     0x08 - 8 (2D only)
    unsigned int flag = GENERATE(hipArrayTextureGather, hipArrayLayered, hipArrayCubemap,
                                 hipArrayLayered | hipArrayCubemap);
#endif
    INFO("Using flag " << flag);
    HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, 1024, 0, flag), hipErrorInvalidValue);
  }
  SECTION("Flags that dont work with 2D") {
    unsigned int flag = GENERATE(hipArrayCubemap, hipArrayLayered | hipArrayCubemap);
    INFO("Using flag " << flag);
    HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, 1024, 1024, flag), hipErrorInvalidValue);
  }
}

// 8-bit float channels are not supported
TEMPLATE_TEST_CASE("Unit_hipMallocArray_Negative_8bitFloat", "", float, float2, float4) {
  hipChannelFormatDesc desc = GENERATE(hipCreateChannelDesc(8, 0, 0, 0, hipChannelFormatKindFloat),
                                       hipCreateChannelDesc(8, 8, 0, 0, hipChannelFormatKindFloat),
                                       hipCreateChannelDesc(8, 8, 8, 8, hipChannelFormatKindFloat));

  // pointer to the array in device memory
  hipArray_t arrayPtr;

#if HT_AMD
  unsigned int flags = hipArrayDefault;
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, 1024, 1024, flags), hipErrorInvalidValue);
#else
  unsigned int flags = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore, hipArrayTextureGather);
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, 1024, 1024, flags), hipErrorUnknown);
#endif
}

// Only 8, 16, and 32 bit channels are supported
TEST_CASE("Unit_hipMallocArray_Negative_BadNumberOfBits") {
  const int badBits = GENERATE(-1, 0, 10, 100);
  const hipChannelFormatKind formatKind =
      GENERATE(hipChannelFormatKindSigned, hipChannelFormatKindUnsigned, hipChannelFormatKindFloat);
  hipChannelFormatDesc desc = hipCreateChannelDesc(badBits, badBits, badBits, badBits, formatKind);

  REQUIRE(desc.x == badBits);
  REQUIRE(desc.y == badBits);
  REQUIRE(desc.z == badBits);
  REQUIRE(desc.w == badBits);

  // pointer to the array in device memory
  hipArray_t arrayPtr;

  INFO("Number of bits: " << badBits);
#if HT_AMD
  unsigned int flag = hipArrayDefault;
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, 1024, 1024, flag), hipErrorInvalidValue);
#else
  unsigned int flag = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore, hipArrayTextureGather);
  INFO("flag: " << flag);
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, 1024, 1024, flag), hipErrorUnknown);
#endif
}

// creating elements with 3 channels is not supported.
TEST_CASE("Unit_hipMallocArray_Negative_3ChannelElement") {
  const int bits = GENERATE(8, 16, 32);
  hipChannelFormatKind formatKind =
      GENERATE(hipChannelFormatKindSigned, hipChannelFormatKindUnsigned, hipChannelFormatKindFloat);
  if (bits == 8 && formatKind == hipChannelFormatKindFloat) return;

  hipChannelFormatDesc desc = hipCreateChannelDesc(bits, bits, bits, 0, formatKind);

  REQUIRE(desc.x == bits);
  REQUIRE(desc.y == bits);
  REQUIRE(desc.z == bits);
  REQUIRE(desc.w == 0);

  // pointer to the array in device memory
  hipArray_t arrayPtr;

#if HT_AMD
  unsigned int flag = hipArrayDefault;
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, 1024, 1024, flag), hipErrorInvalidValue);
#else
  unsigned int flag = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore, hipArrayTextureGather);
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, 1024, 1024, flag), hipErrorUnknown);
#endif
}

// The bit channel description should not allow any channels after a zero channel
TEST_CASE("Unit_hipMallocArray_Negative_ChannelAfterZeroChannel") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-59");
  return;
#endif
  const int bits = GENERATE(8, 16, 32);
  const hipChannelFormatKind formatKind =
      GENERATE(hipChannelFormatKindSigned, hipChannelFormatKindUnsigned, hipChannelFormatKindFloat);
  if (bits == 8 && formatKind == hipChannelFormatKindFloat) return;

  hipChannelFormatDesc desc = GENERATE_COPY(hipCreateChannelDesc(0, bits, bits, 0, formatKind),
                                            hipCreateChannelDesc(0, bits, bits, bits, formatKind),
                                            hipCreateChannelDesc(bits, 0, bits, 0, formatKind),
                                            hipCreateChannelDesc(bits, bits, 0, bits, formatKind),
                                            hipCreateChannelDesc(0, 0, bits, 0, formatKind),
                                            hipCreateChannelDesc(0, 0, bits, bits, formatKind));

  INFO("kind: " << channelFormatString(formatKind));
  INFO("x: " << desc.x << ", y: " << desc.y << ", z: " << desc.z << ", w: " << desc.w);

  hipArray_t arrayPtr;
#if HT_AMD
  unsigned int flag = hipArrayDefault;
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, 1024, 1024, flag), hipErrorInvalidValue);
#else
  unsigned int flag = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore, hipArrayTextureGather);
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, 1024, 1024, flag), hipErrorUnknown);
#endif
}

// The channel format should be one of the defined formats
TEST_CASE("Unit_hipMallocArray_Negative_InvalidChannelFormat") {
  const int bits = 32;
  hipChannelFormatKind formatKind = static_cast<hipChannelFormatKind>(0xFF);
  hipChannelFormatDesc desc = hipCreateChannelDesc(bits, bits, bits, bits, formatKind);

  REQUIRE(desc.f != hipChannelFormatKindFloat);
  REQUIRE(desc.f != hipChannelFormatKindUnsigned);
  REQUIRE(desc.f != hipChannelFormatKindSigned);

  hipArray_t arrayPtr;

  CAPTURE(formatKind);

#if HT_AMD
  unsigned int flag = hipArrayDefault;
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, 1024, 1024, flag), hipErrorInvalidValue);
#else
  unsigned int flag = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore);
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, 1024, 1024, flag), hipErrorUnknown);
#endif
}


// hipMallocArray should handle the max numeric value gracefully.
TEST_CASE("Unit_hipMallocArray_Negative_NumericLimit") {
  hipArray_t arrayPtr;
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();

  size_t size = std::numeric_limits<size_t>::max();
#if HT_AMD
  unsigned int flag = hipArrayDefault;
#else
  unsigned int flag = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore, hipArrayTextureGather);
#endif
  HIP_CHECK_ERROR(hipMallocArray(&arrayPtr, &desc, size, size, flag), hipErrorInvalidValue);
}
