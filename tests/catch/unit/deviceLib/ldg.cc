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


constexpr int WIDTH = 16;
constexpr int HEIGHT = 16;
constexpr int NUM = WIDTH * HEIGHT;

constexpr int THREADS_PER_BLOCK_X = 8;
constexpr int THREADS_PER_BLOCK_Y = 8;

template <typename T>
__global__ void vectoradd_float(T* a, const T* bm, int width, int height)

{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int i = y * width + x;
  if (i < (width * height)) {
    a[i] = __ldg(&bm[i]);
  }
}

int2 make_vector2(int a) { return make_int2(a, a); }

char2 make_vector2(signed char a) { return make_char2(a, a); }

char4 make_vector4(signed char a) { return make_char4(a, a, a, a); }

short2 make_vector2(short a) { return make_short2(a, a); }

ushort2 make_vector2(unsigned short a) { return make_ushort2(a, a); }

short4 make_vector4(short a) { return make_short4(a, a, a, a); }

int4 make_vector4(int a) { return make_int4(a, a, a, a); }

uint2 make_vector2(unsigned int a) { return make_uint2(a, a); }

uint4 make_vector4(unsigned int a) { return make_uint4(a, a, a, a); }

float2 make_vector2(float a) { return make_float2(a, a); }

float4 make_vector4(float a) { return make_float4(a, a, a, a); }

uchar2 make_vector2(unsigned char a) { return make_uchar2(a, a); }

uchar4 make_vector4(unsigned char a) { return make_uchar4(a, a, a, a); }

double2 make_vector2(double a) { return make_double2(a, a); }


template <typename T, typename U> int dataTypesRun() {
  T* hostA;
  T* hostB;

  T* deviceA;
  T* deviceB;

  int i;
  int errors;

  hostA = (T*)malloc(NUM * sizeof(T));
  hostB = (T*)malloc(NUM * sizeof(T));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (U)i;
  }

  HIP_CHECK(hipMalloc((void**)&deviceA, NUM * sizeof(T)));
  HIP_CHECK(hipMalloc((void**)&deviceB, NUM * sizeof(T)));

  HIP_CHECK(hipMemcpy(deviceB, hostB, NUM * sizeof(T), hipMemcpyHostToDevice));


  hipLaunchKernelGGL(vectoradd_float,
                     dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA,
                     static_cast<const T*>(deviceB), WIDTH, HEIGHT);


  HIP_CHECK(hipMemcpy(hostA, deviceA, NUM * sizeof(T), hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i])) {
      errors++;
    }
  }

  HIP_CHECK(hipFree(deviceA));
  HIP_CHECK(hipFree(deviceB));

  free(hostA);
  free(hostB);

  return errors;
}


template <typename T, typename U> int dataTypesRun2() {
  T* hostA;
  T* hostB;

  T* deviceA;
  T* deviceB;

  int i;
  int errors;

  hostA = (T*)malloc(NUM * sizeof(T));
  hostB = (T*)malloc(NUM * sizeof(T));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = make_vector2((U)i);
  }

  HIP_CHECK(hipMalloc((void**)&deviceA, NUM * sizeof(T)));
  HIP_CHECK(hipMalloc((void**)&deviceB, NUM * sizeof(T)));

  HIP_CHECK(hipMemcpy(deviceB, hostB, NUM * sizeof(T), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(vectoradd_float,
                     dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA,
                     static_cast<const T*>(deviceB), WIDTH, HEIGHT);


  HIP_CHECK(hipMemcpy(hostA, deviceA, NUM * sizeof(T), hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i].x != (hostB[i].x) && hostA[i].y != (hostB[i].y)) {
      errors++;
    }
  }

  HIP_CHECK(hipFree(deviceA));
  HIP_CHECK(hipFree(deviceB));

  free(hostA);
  free(hostB);

  return errors;
}


template <typename T, typename U> int dataTypesRun4() {
  T* hostA;
  T* hostB;

  T* deviceA;
  T* deviceB;

  int i;
  int errors;

  hostA = (T*)malloc(NUM * sizeof(T));
  hostB = (T*)malloc(NUM * sizeof(T));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = make_vector4((U)i);
  }

  HIP_CHECK(hipMalloc((void**)&deviceA, NUM * sizeof(T)));
  HIP_CHECK(hipMalloc((void**)&deviceB, NUM * sizeof(T)));

  HIP_CHECK(hipMemcpy(deviceB, hostB, NUM * sizeof(T), hipMemcpyHostToDevice));


  hipLaunchKernelGGL(vectoradd_float,
                     dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                     dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, deviceA,
                     static_cast<const T*>(deviceB), WIDTH, HEIGHT);


  HIP_CHECK(hipMemcpy(hostA, deviceA, NUM * sizeof(T), hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i].x != (hostB[i].x) && hostA[i].y != (hostB[i].y) && hostA[i].z != (hostB[i].z) &&
        hostA[i].w != (hostB[i].w)) {
      errors++;
    }
  }

  HIP_CHECK(hipFree(deviceA));
  HIP_CHECK(hipFree(deviceB));

  free(hostA);
  free(hostB);

  return errors;
}

TEST_CASE("Unit_ldg") {
  using namespace std;

  int errors;

  errors = dataTypesRun<char, char>() | dataTypesRun<short, short>() | dataTypesRun<int, int>() |
      dataTypesRun<long, long>() | dataTypesRun<long long, long long>() |
      dataTypesRun<signed char, signed char>() | dataTypesRun<unsigned char, unsigned char>() |
      dataTypesRun<unsigned short, unsigned short>() | dataTypesRun<unsigned int, unsigned int>() |
      dataTypesRun<unsigned long, unsigned long>() |
      dataTypesRun<unsigned long long, unsigned long long>() | dataTypesRun<float, float>() |
      dataTypesRun<double, double>();

  REQUIRE(errors == 0);

  errors = dataTypesRun2<int2, int>() | dataTypesRun2<short2, short>() |
      dataTypesRun2<ushort2, unsigned short>() | dataTypesRun2<char2, signed char>() |
      dataTypesRun2<uchar2, unsigned char>() | dataTypesRun2<uint2, unsigned int>() |
      dataTypesRun2<float2, float>() | dataTypesRun2<double2, double>();

  REQUIRE(errors == 0);

  errors = dataTypesRun4<int4, int>() | dataTypesRun4<char4, signed char>() |
      dataTypesRun4<uchar4, unsigned char>() | dataTypesRun4<short4, short>() |
      dataTypesRun4<uint4, unsigned int>() | dataTypesRun4<float4, float>();

  REQUIRE(errors == 0);
}
