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

#include <hip_test_common.hh>
#include <vector>
#include <iostream>

template <typename T>
__global__ void tex1dKernelFetch(T *val, hipTextureObject_t obj, int N) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    val[k] = tex1Dfetch<T>(obj, k);
  }
}

template <typename T>
static inline __host__ __device__ constexpr int rank() {
  return sizeof(T) / sizeof(decltype(T::x));
}

template<typename T>
static inline T getRandom() {
  double r = 0;
  if (std::is_signed < T > ::value) {
    r = (std::rand() - RAND_MAX / 2.0) / (RAND_MAX / 2.0 + 1.);
  } else {
    r = std::rand() / (RAND_MAX + 1.);
  }
  return static_cast<T>(std::numeric_limits < T > ::max() * r);
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 1>::type* = nullptr>
static inline void initVal(T &val) {
  val.x = getRandom<decltype(T::x)>();
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 2>::type* = nullptr>
static inline void initVal(T &val) {
  val.x = getRandom<decltype(T::x)>();
  val.y = getRandom<decltype(T::x)>();
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 4>::type* = nullptr>
static inline void initVal(T &val) {
  val.x = getRandom<decltype(T::x)>();
  val.y = getRandom<decltype(T::x)>();
  val.z = getRandom<decltype(T::x)>();
  val.w = getRandom<decltype(T::x)>();
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 1>::type* = nullptr>
static inline void printVector(T &val) {
  using B = decltype(T::x);
  constexpr bool isChar = std::is_same<B, char>::value
      || std::is_same<B, unsigned char>::value;
  std::cout << "(";
  std::cout << (isChar ? static_cast<int>(val.x) : val.x);
  std::cout << ")";
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 2>::type* = nullptr>
static inline void printVector(T &val) {
  using B = decltype(T::x);
  constexpr bool isChar = std::is_same<B, char>::value
      || std::is_same<B, unsigned char>::value;
  std::cout << "(";
  std::cout << (isChar ? static_cast<int>(val.x) : val.x);
  std::cout << ", " << (isChar ? static_cast<int>(val.y) : val.y);
  std::cout << ")";
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 4>::type* = nullptr>
static inline void printVector(T &val) {
  using B = decltype(T::x);
  constexpr bool isChar = std::is_same<B, char>::value
      || std::is_same<B, unsigned char>::value;
  std::cout << "(";
  std::cout << (isChar ? static_cast<int>(val.x) : val.x);
  std::cout << ", " << (isChar ? static_cast<int>(val.y) : val.y);
  std::cout << ", " << (isChar ? static_cast<int>(val.z) : val.z);
  std::cout << ", " << (isChar ? static_cast<int>(val.w) : val.w);
  std::cout << ")";
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 1>::type* = nullptr>
static inline bool isEqual(const T &val0, const T &val1) {
  return val0.x == val1.x;
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 2>::type* = nullptr>
static inline bool isEqual(const T &val0, const T &val1) {
  return val0.x == val1.x &&
         val0.y == val1.y;
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 4>::type* = nullptr>
static inline bool isEqual(const T &val0, const T &val1) {
  return val0.x == val1.x &&
         val0.y == val1.y &&
         val0.z == val1.z &&
         val0.w == val1.w;
}

template<typename T>
bool runTest(const char *description) {
  const int N = 1024;
  bool testResult = true;
  // Allocating the required buffer on gpu device
  T *texBuf, *texBufOut;
  T val[N], output[N];
  printf("%s<%s>(): size: %zu, %zu\n", __FUNCTION__, description,
         sizeof(T), sizeof(decltype(T::x)));

  memset(output, 0, sizeof(output));
  std::srand(std::time(nullptr)); // use current time as seed for random generator

  for (int i = 0; i < N; i++) {
    initVal<T>(val[i]);
  }

  HIP_CHECK(hipMalloc(&texBuf, N * sizeof(T)));
  HIP_CHECK(hipMalloc(&texBufOut, N * sizeof(T)));
  HIP_CHECK(hipMemcpy(texBuf, val, N * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(texBufOut, 0, N * sizeof(T)));
  hipResourceDesc resDescLinear;

  memset(&resDescLinear, 0, sizeof(resDescLinear));
  resDescLinear.resType = hipResourceTypeLinear;
  resDescLinear.res.linear.devPtr = texBuf;
  resDescLinear.res.linear.desc = hipCreateChannelDesc<T>();
  resDescLinear.res.linear.sizeInBytes = N * sizeof(T);

  hipTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = hipReadModeElementType;
  texDesc.addressMode[0] = hipAddressModeClamp;

  // Creating texture object
  hipTextureObject_t texObj = 0;
  HIP_CHECK(hipCreateTextureObject(&texObj, &resDescLinear, &texDesc, NULL));

  dim3 dimBlock(64, 1, 1);
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);

  hipLaunchKernelGGL(tex1dKernelFetch<T>, dimGrid, dimBlock, 0, 0, texBufOut,
                     texObj, N);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(output, texBufOut, N * sizeof(T), hipMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    if (!isEqual(output[i], val[i])) {
      std::cout << "output[" << i << "]= ";
      printVector<T>(output[i]);
      std::cout << ", expected val[" << i << "]= ";
      printVector<T>(val[i]);
      std::cout << "\n";
      testResult = false;
      break;
    }
  }
  HIP_CHECK(hipDestroyTextureObject(texObj));
  HIP_CHECK(hipFree(texBuf));
  HIP_CHECK(hipFree(texBufOut));

  printf(": %s\n", testResult ? "succeeded" : "failed");
  REQUIRE(testResult == true);
  return testResult;
}

TEST_CASE("Unit_hipTextureFetch_vector") {
  // test for char
  runTest<char1>("char1");
  runTest<char2>("char2");
  runTest<char4>("char4");

  // test for uchar
  runTest<uchar1>("uchar1");
  runTest<uchar2>("uchar2");
  runTest<uchar4>("uchar4");

  // test for short
  runTest<short1>("short1");
  runTest<short2>("short2");
  runTest<short4>("short4");

  // test for ushort
  runTest<ushort1>("ushort1");
  runTest<ushort2>("ushort2");
  runTest<ushort4>("ushort4");

  // test for int
  runTest<int1>("int1");
  runTest<int2>("int2");
  runTest<int4>("int4");

  // test for unsigned int
  runTest<uint1>("uint1");
  runTest<uint2>("uint2");
  runTest<uint4>("uint4");

  // test for float
  runTest<float1>("float1");
  runTest<float2>("float2");
  runTest<float4>("float4");
}
