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
  std::vector<size_t> array_size;
  array_size.push_back(NUM_W);
  array_size.push_back(BIGNUM_W);
  for (auto& size : array_size) {
    hipArray* A_d[ARRAY_LOOP];
    size_t tot, avail, ptot, pavail;
    hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
    HIP_CHECK(hipMemGetInfo(&pavail, &ptot));
    for (int i = 0; i < ARRAY_LOOP; i++) {
      if (size == NUM_W) {
        HIP_CHECK(hipMallocArray(&A_d[i], &desc, NUM_W, NUM_H, hipArrayDefault));
      } else {
        HIP_CHECK(hipMallocArray(&A_d[i], &desc, BIGNUM_W, BIGNUM_H, hipArrayDefault));
      }
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

/*
 * This testcase verifies the negative scenarios of
 * hipMallocArray API
 */
TEST_CASE("Unit_hipMallocArray_Negative") {
  hipArray* A_d;
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
#if HT_NVIDIA
  SECTION("NullPointer to Array") {
    REQUIRE(hipMallocArray(nullptr, &desc, NUM_W, NUM_H, hipArrayDefault) != hipSuccess);
  }

  SECTION("NullPointer to Channel Descriptor") {
    REQUIRE(hipMallocArray(&A_d, nullptr, NUM_W, NUM_H, hipArrayDefault) != hipSuccess);
  }
#endif
  SECTION("Width 0 in hipMallocArray") {
    REQUIRE(hipMallocArray(&A_d, &desc, 0, NUM_H, hipArrayDefault) != hipSuccess);
  }

  SECTION("Height 0 in hipMallocArray") {
    REQUIRE(hipMallocArray(&A_d, &desc, NUM_W, 0, hipArrayDefault) == hipSuccess);
  }

  SECTION("Invalid Flag") { REQUIRE(hipMallocArray(&A_d, &desc, NUM_W, NUM_H, 100) != hipSuccess); }

  SECTION("Max int values") {
    REQUIRE(hipMallocArray(&A_d, &desc, std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max(), hipArrayDefault) != hipSuccess);
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
#endif

  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  size_t init_free = getFreeMem();

  // pointer to the array in device memory
  hipArray_t arrayPtr{};
  size_t width = 1024;
  size_t height = GENERATE(0, 1024);

  SECTION("hipArrayDefault") {
    INFO("flag is hipArrayDefault");
    INFO("height: " << height);

    HIP_CHECK(hipMallocArray(&arrayPtr, &desc, width, height, hipArrayDefault));
    testArrayAsTexture<TestType>(arrayPtr, width, height);
  }
#if HT_NVIDIA  // surfaces and texture gather not supported on AMD
  SECTION("hipArraySurfaceLoadStore") {
    INFO("flag is hipArraySurfaceLoadStore");
    INFO("height: " << height);

    HIP_CHECK(hipMallocArray(&arrayPtr, &desc, width, height, hipArraySurfaceLoadStore));
    testArrayAsSurface<TestType>(arrayPtr, width, height);
  }
#endif

  size_t final_free = getFreeMem();

  const size_t alloc_size = getAllocSize<TestType>(width, height);
  // alloc will be chunked, so this is not exact
  REQUIRE(init_free - final_free >= alloc_size);

  HIP_CHECK(hipFreeArray(arrayPtr));
}
