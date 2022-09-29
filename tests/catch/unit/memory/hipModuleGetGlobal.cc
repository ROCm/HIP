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

#include <algorithm>
#include <array>
#include <string>
#include <type_traits>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

#include "hipModuleGetGlobal.hh"

template <typename T> constexpr const char* TypeNameToString() {
  if constexpr (std::is_same<int, T>::value) {
    return "int";
  } else if constexpr (std::is_same<float, T>::value) {
    return "float";
  } else if constexpr (std::is_same<char, T>::value) {
    return "char";
  } else if constexpr (std::is_same<double, T>::value) {
    return "double";
  } else {
    static_assert(!sizeof(T), "Stringify not implemented for this type");
  }
}

template <typename T, size_t N> static void HipModuleGetGlobalTest(hipModule_t module) {
  constexpr auto size = N * sizeof(T);

  hipDeviceptr_t global = nullptr;
  size_t global_size = 0;
  const std::string global_name = TypeNameToString<T>() + std::string(1 == N ? "_var" : "_arr");
  HIP_CHECK(hipModuleGetGlobal(&global, &global_size, module, global_name.c_str()));
  REQUIRE(global != nullptr);
  REQUIRE(size == global_size);

  hipFunction_t kernel = nullptr;
  const auto kernel_name = global_name + "_address_validation_kernel";
  HIP_CHECK(hipModuleGetFunction(&kernel, module, kernel_name.c_str()));
  bool* equal_addresses;
  HIP_CHECK(hipMalloc(&equal_addresses, sizeof(*equal_addresses)));
  HIP_CHECK(hipMemset(equal_addresses, false, sizeof(*equal_addresses)));
  void* kernel_args[2] = {&global, &equal_addresses};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args, nullptr));
  HIP_CHECK(hipStreamSynchronize(nullptr));
  bool ok;
  HIP_CHECK(hipMemcpy(&ok, equal_addresses, sizeof(ok), hipMemcpyDeviceToHost));
  REQUIRE(ok);

  constexpr T expected_value = 42;
  std::array<T, N> fill_buffer;
  std::fill_n(fill_buffer.begin(), N, expected_value);
  HIP_CHECK(hipMemcpyHtoD(global, fill_buffer.data(), size));


  std::array<T, N> read_buffer;
  HIP_CHECK(hipMemcpyDtoH(read_buffer.data(), global, size));
  const auto it = std::find_if_not(std::begin(read_buffer), std::end(read_buffer),
                                   [](const T element) { return expected_value == element; });
  REQUIRE(it == std::end(read_buffer));
}

template <typename T> static void HipModuleGetGlobalTest(hipModule_t module) {
  SECTION("array") { HipModuleGetGlobalTest<T, kArraySize>(module); }
  SECTION("scalar") { HipModuleGetGlobalTest<T, 1>(module); }
}

static hipModule_t GetModule() {
  static hipModule_t module = nullptr;
  if (!module) {
    HIP_CHECK(hipModuleLoad(&module, "test_module.code"));
  }
  return module;
}

TEST_CASE("Unit_hipModuleGetGlobal_Basic") {
  hipModule_t module = GetModule();
  // Listed like this instead of using a templated test because a separate test case will be created
  // for each type, so loading the module for each type can't be avoided. Sections are used to
  // retain a modicum of organization, which leads to using the singleton, to avoid loading the
  // module before each section.
  SECTION("int") { HipModuleGetGlobalTest<int>(module); }
  SECTION("float") { HipModuleGetGlobalTest<float>(module); }
  SECTION("char") { HipModuleGetGlobalTest<char>(module); }
  SECTION("double") { HipModuleGetGlobalTest<double>(module); }
}

TEST_CASE("Unit_hipModuleGetGlobal_Negative_Parameters") {
  hipModule_t module = GetModule();
  hipDeviceptr_t global = nullptr;
  size_t global_size = 0;
  SECTION("dptr == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetGlobal(nullptr, &global_size, module, "int_var"),
                    hipErrorInvalidValue);
  }
  SECTION("bytes == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetGlobal(&global, nullptr, module, "int_var"), hipErrorInvalidValue);
  }
  SECTION("hmod == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetGlobal(&global, &global_size, nullptr, "int_var"),
                    hipErrorNotFound);
  }
  SECTION("name == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetGlobal(&global, &global_size, module, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("name == empty string") {
    HIP_CHECK_ERROR(hipModuleGetGlobal(&global, &global_size, module, ""), hipErrorNotFound);
  }
  SECTION("name == invalid name") {
    HIP_CHECK_ERROR(hipModuleGetGlobal(&global, &global_size, module, "dummy"), hipErrorNotFound);
  }
}