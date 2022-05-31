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
 Test Scenarios :
 1) Calling hipMemcpyTo/FromSymbolAsync() using user declared stream obj and hipStreamPerThread.
 2) Validate get symbol address/size for global const array.
 3) Validate get symbol address/size for static const variable.
*/

#include <hip_test_common.hh>

constexpr size_t NUM = 1024;
constexpr size_t SIZE = 1024 * 4;

__device__ int globalIn[NUM];
__device__ int globalOut[NUM];

__global__ void Assign(int* Out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Out[tid] = globalIn[tid];
  globalOut[tid] = globalIn[tid];
}

__device__ __constant__ int globalConst[NUM];

__global__ void checkAddress(int* addr, bool* out) { *out = (globalConst == addr); }

TEST_CASE("Unit_hipMemcpyToSymbolAsync_ToNFrom") {
  int *A{nullptr}, *Am{nullptr}, *B{nullptr}, *Ad{nullptr}, *C{nullptr}, *Cm{nullptr};
  A = new int[NUM];
  B = new int[NUM];
  C = new int[NUM];

  HIP_CHECK(hipMalloc((void**)&Ad, SIZE));
  HIP_CHECK(hipHostMalloc((void**)&Am, SIZE));
  HIP_CHECK(hipHostMalloc((void**)&Cm, SIZE));

  for (size_t i = 0; i < NUM; i++) {
    A[i] = -1 * static_cast<int>(i);
    B[i] = 0;
    C[i] = 0;
    Am[i] = -1 * static_cast<int>(i);
    Cm[i] = 0;
  }


  SECTION("Calling hipMemcpyTo/FromSymbol using stream") {
    hipStream_t stream{};
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(
        hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), Am, SIZE, 0, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpyFromSymbolAsync(Cm, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost,
                                       stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
    for (size_t i = 0; i < NUM; i++) {
      REQUIRE(Am[i] == B[i]);
      REQUIRE(Am[i] == Cm[i]);
    }
  }

  SECTION("Calling hipMemcpyTo/FromSymbol - validate value in host memory") {
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(globalIn), A, SIZE, 0, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpyFromSymbol(C, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < NUM; i++) {
      REQUIRE(A[i] == B[i]);
      REQUIRE(A[i] == C[i]);
    }
  }

  SECTION("Calling hipMemcpyTo/FromSymbol using user declared stream obj") {
    hipStream_t stream{};
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(
        hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), A, SIZE, 0, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpyFromSymbolAsync(C, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));

    for (size_t i = 0; i < NUM; i++) {
      REQUIRE(A[i] == B[i]);
      REQUIRE(A[i] == C[i]);
    }
  }

  SECTION("Calling hipMemcpyTo/FromSymbol using hipStreamPerThread") {
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), A, SIZE, 0, hipMemcpyHostToDevice,
                                     hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpyFromSymbolAsync(C, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost,
                                       hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    for (size_t i = 0; i < NUM; i++) {
      REQUIRE(A[i] == B[i]);
      REQUIRE(A[i] == C[i]);
    }
  }

  // Check for address on GPU and CPU side and compare it
  // If address mismatch report error
  // Validate size of symbol as well, compare it with output of hipGetSymbolSize
  SECTION("Validate address on GPU") {
    bool* checkOkD{nullptr};
    bool checkOk = false;
    size_t symbolSize = 0;
    int* symbolAddress{nullptr};
    HIP_CHECK(hipGetSymbolSize(&symbolSize, HIP_SYMBOL(globalConst)));
    HIP_CHECK(hipGetSymbolAddress((void**)&symbolAddress, HIP_SYMBOL(globalConst)));
    HIP_CHECK(hipMalloc((void**)&checkOkD, sizeof(bool)));
    hipLaunchKernelGGL(checkAddress, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, symbolAddress, checkOkD);
    HIP_CHECK(hipMemcpy(&checkOk, checkOkD, sizeof(bool), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(checkOkD));
    HIP_ASSERT(checkOk);
    HIP_ASSERT((symbolSize == SIZE));
  }

  HIP_CHECK(hipHostFree(Am));
  HIP_CHECK(hipHostFree(Cm));
  HIP_CHECK(hipFree(Ad));
  delete[] A;
  delete[] B;
  delete[] C;
}

/**
 1) Validate get symbol address/size for global const array.
 2) Validate get symbol address/size for static const variable.
 */
TEST_CASE("Unit_hipGetSymbolAddressAndSize_Validation") {
  bool* checkOkD{nullptr};
  bool checkOk = false;
  size_t symbolSize{};
  int* symbolArrAddress{};
  float* symbolVarAddress{};

  SECTION("Validate symbol size/address of global const array") {
    HIP_CHECK(hipGetSymbolSize(&symbolSize, HIP_SYMBOL(globalConstArr)));
    HIP_CHECK(hipGetSymbolAddress(reinterpret_cast<void**>(&symbolArrAddress),
                                  HIP_SYMBOL(globalConstArr)));
    HIP_CHECK(hipMalloc(&checkOkD, sizeof(bool)));
    hipLaunchKernelGGL(checkGlobalConstAddress, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                       symbolArrAddress, checkOkD);
    HIP_CHECK(hipMemcpy(&checkOk, checkOkD, sizeof(bool), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(checkOkD));
    HIP_ASSERT(checkOk);
    HIP_ASSERT(symbolSize == SIZE);
  }

  SECTION("Validate symbol size/address of static const variable") {
    HIP_CHECK(hipGetSymbolSize(&symbolSize, HIP_SYMBOL(statConstVar)));
    HIP_CHECK(
        hipGetSymbolAddress(reinterpret_cast<void**>(&symbolVarAddress), HIP_SYMBOL(statConstVar)));
    HIP_CHECK(hipMalloc(&checkOkD, sizeof(bool)));
    hipLaunchKernelGGL(checkStaticConstVarAddress, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                       symbolVarAddress, checkOkD);
    HIP_CHECK(hipMemcpy(&checkOk, checkOkD, sizeof(bool), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(checkOkD));
    HIP_ASSERT(checkOk);
    HIP_ASSERT(symbolSize == sizeof(float));
  }
}

TEST_CASE("Unit_hipGetSymbolAddress_Negative") {
  SECTION("Invalid symbol") {
    int notADeviceSymbol{0};
    int* addr{nullptr};
    HIP_CHECK_ERROR(
        hipGetSymbolAddress(reinterpret_cast<void**>(&addr), HIP_SYMBOL(notADeviceSymbol)),
        hipErrorInvalidSymbol);
  }

  SECTION("Nullptr symbol") {
    int* addr{nullptr};
    HIP_CHECK_ERROR(hipGetSymbolAddress(reinterpret_cast<void**>(&addr), nullptr),
                    hipErrorInvalidSymbol);
  }
}

TEST_CASE("Unit_hipGetSymbolSize_Negative") {
  SECTION("Invalid symbol") {
    int notADeviceSymbol{0};
    size_t dsize{0};
    HIP_CHECK_ERROR(hipGetSymbolSize(&dsize, HIP_SYMBOL(notADeviceSymbol)), hipErrorInvalidSymbol);
  }

  SECTION("Nullptr symbol") {
    size_t size{0};
    HIP_CHECK_ERROR(hipGetSymbolSize(&size, nullptr), hipErrorInvalidSymbol);
  }
}
