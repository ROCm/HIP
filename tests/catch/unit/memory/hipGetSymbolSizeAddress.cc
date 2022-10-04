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

#include <tuple>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>

namespace {
constexpr size_t kArraySize = 5;
}  // anonymous namespace

#define HIP_GET_SYMBOL_SIZE_ADDRESS_DEFINE_GLOBALS(type)                                           \
  __device__ type type##_var = 0;                                                                  \
  __device__ type type##_arr[kArraySize] = {};                                                     \
  extern "C" {                                                                                     \
  __global__ void type##_var_address_validation_kernel(void* ptr, bool* out) {                     \
    *out = static_cast<void*>(&type##_var) == ptr;                                                 \
  }                                                                                                \
  __global__ void type##_arr_address_validation_kernel(void* ptr, bool* out) {                     \
    *out = static_cast<void*>(type##_arr) == ptr;                                                  \
  }                                                                                                \
  }

HIP_GET_SYMBOL_SIZE_ADDRESS_DEFINE_GLOBALS(int)
HIP_GET_SYMBOL_SIZE_ADDRESS_DEFINE_GLOBALS(float)
HIP_GET_SYMBOL_SIZE_ADDRESS_DEFINE_GLOBALS(char)
HIP_GET_SYMBOL_SIZE_ADDRESS_DEFINE_GLOBALS(double)

#define HIP_GET_SYMBOL_SIZE_ADDRESS_SYMBOLS(type) HIP_SYMBOL(type##_var), HIP_SYMBOL(type##_arr)

template <typename T> struct ValidationKernel {
  void (*kernel)(void*, bool*);
};

#define HIP_GET_SYMBOL_SIZE_VALIDATION_KERNELS(type)                                               \
  ValidationKernel<type*>{type##_var_address_validation_kernel},                                   \
      ValidationKernel<type(*)[kArraySize]> {                                                      \
    type##_arr_address_validation_kernel                                                           \
  }

namespace {
constexpr auto kTestSymbols = std::make_tuple(
    HIP_GET_SYMBOL_SIZE_ADDRESS_SYMBOLS(int), HIP_GET_SYMBOL_SIZE_ADDRESS_SYMBOLS(float),
    HIP_GET_SYMBOL_SIZE_ADDRESS_SYMBOLS(char), HIP_GET_SYMBOL_SIZE_ADDRESS_SYMBOLS(double));

constexpr auto kValidationKernels = std::make_tuple(
    HIP_GET_SYMBOL_SIZE_VALIDATION_KERNELS(int), HIP_GET_SYMBOL_SIZE_VALIDATION_KERNELS(float),
    HIP_GET_SYMBOL_SIZE_VALIDATION_KERNELS(char), HIP_GET_SYMBOL_SIZE_VALIDATION_KERNELS(double));
}  // anonymous namespace

template <typename T, size_t N> constexpr const void* GetSymbol() {
  if constexpr (N == 1) {
    return std::get<T*>(kTestSymbols);
  } else {
    return std::get<T(*)[kArraySize]>(kTestSymbols);
  }
}

template <typename T, size_t N> constexpr auto GetValidationKernel() {
  if constexpr (N == 1) {
    return std::get<ValidationKernel<T*>>(kValidationKernels).kernel;
  } else {
    return std::get<ValidationKernel<T(*)[kArraySize]>>(kValidationKernels).kernel;
  }
}

template <typename T, size_t N> static void HipGetSymbolSizeAddressTest() {
  constexpr auto size = N * sizeof(T);

  T* symbol_ptr = nullptr;
  size_t symbol_size = 0;
  constexpr auto symbol = GetSymbol<T, N>();
  HIP_CHECK(hipGetSymbolAddress(reinterpret_cast<void**>(&symbol_ptr), symbol));
  HIP_CHECK(hipGetSymbolSize(&symbol_size, symbol));
  REQUIRE(symbol_ptr != nullptr);
  REQUIRE(symbol_size == size);

  LinearAllocGuard<bool> equal_addresses(LinearAllocs::hipMalloc, sizeof(bool));
  HIP_CHECK(hipMemset(equal_addresses.ptr(), false, sizeof(*equal_addresses.ptr())))
  constexpr auto kernel = GetValidationKernel<T, N>();
  kernel<<<1, 1>>>(symbol_ptr, equal_addresses.ptr());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(nullptr));
  bool ok = false;
  HIP_CHECK(hipMemcpy(&ok, equal_addresses.ptr(), sizeof(ok), hipMemcpyDeviceToHost));
  REQUIRE(ok);
}

template <typename T> static void HipGetSymbolSizeAddressTest() {
  SECTION("scalar") { HipGetSymbolSizeAddressTest<T, 1>(); }
  SECTION("array") { HipGetSymbolSizeAddressTest<T, kArraySize>(); }
}

TEMPLATE_TEST_CASE("Unit_hipGetSymbolSizeAddress_Basic", "", int, float, char, double) {
  HipGetSymbolSizeAddressTest<TestType>();
}

TEST_CASE("Unit_hipGetSymbolAddress_Negative_Parameters") {
  SECTION("devPtr == nullptr") {
    HIP_CHECK_ERROR(hipGetSymbolAddress(nullptr, HIP_SYMBOL(int_var)), hipErrorInvalidValue);
  }
  SECTION("symbolName == nullptr") {
    void* ptr = nullptr;
    HIP_CHECK_ERROR(hipGetSymbolAddress(&ptr, nullptr), hipErrorInvalidSymbol);
  }
}

TEST_CASE("Unit_hipGetSymbolSize_Negative_Parameters") {
  SECTION("size == nullptr") {
    HIP_CHECK_ERROR(hipGetSymbolSize(nullptr, HIP_SYMBOL(int_var)), hipErrorInvalidValue);
  }
  SECTION("symbolName == nullptr") {
    size_t size = 0;
    HIP_CHECK_ERROR(hipGetSymbolSize(&size, nullptr), hipErrorInvalidSymbol);
  }
}