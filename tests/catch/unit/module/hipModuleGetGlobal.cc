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
#include <resource_guards.hh>
#include <utils.hh>

#include "hip_module_common.hh"
#include "hipModuleGetGlobal.hh"

template <typename T, size_t N>
static void HipModuleGetGlobalTest(hipModule_t module, const std::string global_name) {
  constexpr auto size = N * sizeof(T);

  hipDeviceptr_t global;
  size_t global_size = 0;
  HIP_CHECK(hipModuleGetGlobal(&global, &global_size, module, global_name.c_str()));
  REQUIRE(global != 0);
  REQUIRE(size == global_size);

  hipFunction_t kernel = nullptr;
  const auto kernel_name = global_name + "_address_validation_kernel";
  HIP_CHECK(hipModuleGetFunction(&kernel, module, kernel_name.c_str()));
  LinearAllocGuard<bool> equal_addresses(LinearAllocs::hipMalloc, sizeof(bool));
  HIP_CHECK(hipMemset(equal_addresses.ptr(), false, sizeof(*equal_addresses.ptr())));
  bool* equal_addresses_ptr = equal_addresses.ptr();
  void* kernel_args[2] = {&global, &equal_addresses_ptr};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args, nullptr));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  bool ok;
  HIP_CHECK(hipMemcpy(&ok, equal_addresses_ptr, sizeof(ok), hipMemcpyDeviceToHost));
  REQUIRE(ok);

  constexpr T expected_value = 42;
  std::array<T, N> fill_buffer;
  std::fill_n(fill_buffer.begin(), N, expected_value);
  HIP_CHECK(hipMemcpyHtoD(global, fill_buffer.data(), size));


  std::array<T, N> read_buffer;
  HIP_CHECK(hipMemcpyDtoH(read_buffer.data(), global, size));
  ArrayFindIfNot(read_buffer.data(), expected_value, read_buffer.size());
}

#define HIP_MODULE_GET_GLOBAL_S(expr) #expr
#define HIP_MODULE_GET_GLOBAL_TEST(type, module)                                                   \
  SECTION("array") {                                                                               \
    HipModuleGetGlobalTest<type, kArraySize>(module, HIP_MODULE_GET_GLOBAL_S(type##_arr));         \
  }                                                                                                \
  SECTION("scalar") {                                                                              \
    HipModuleGetGlobalTest<type, 1>(module, HIP_MODULE_GET_GLOBAL_S(type##_var));                  \
  }

static inline hipModule_t GetModule() {
  HIP_CHECK(hipFree(nullptr));
  const static auto mg = ModuleGuard::LoadModule("get_global_test_module.code");
  return mg.module();
}

TEST_CASE("Unit_hipModuleGetGlobal_Positive_Basic") {
  hipModule_t module = GetModule();

  SECTION("int") { HIP_MODULE_GET_GLOBAL_TEST(int, module); }

  SECTION("float") { HIP_MODULE_GET_GLOBAL_TEST(float, module); }

  SECTION("char") { HIP_MODULE_GET_GLOBAL_TEST(char, module); }

  SECTION("double") { HIP_MODULE_GET_GLOBAL_TEST(double, module); }
}

TEST_CASE("Unit_hipModuleGetGlobal_Positive_Parameters") {
  hipModule_t module = GetModule();
  hipDeviceptr_t global = 0;
  size_t global_size = 0;

  SECTION("dptr == nullptr") {
    HIP_CHECK(hipModuleGetGlobal(nullptr, &global_size, module, "int_var"));
  }

  SECTION("bytes == nullptr") {
    HIP_CHECK(hipModuleGetGlobal(&global, nullptr, module, "int_var"));
  }
}

TEST_CASE("Unit_hipModuleGetGlobal_Negative_Parameters") {
  hipModule_t module = GetModule();
  hipDeviceptr_t global = 0;
  size_t global_size = 0;

// Disabled on AMD due to defect - EXSWHTEC-165
#if HT_NVIDIA
  SECTION("dptr == nullptr and bytes == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetGlobal(nullptr, nullptr, module, "int_var"), hipErrorInvalidValue);
  }
#endif

// Disabled on AMD due to defect - EXSWHTEC-163
#if HT_NVIDIA
  SECTION("hmod == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetGlobal(&global, &global_size, nullptr, "int_var"),
                    hipErrorInvalidResourceHandle);
  }
#endif

  SECTION("name == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetGlobal(&global, &global_size, module, nullptr),
                    hipErrorInvalidValue);
  }

// Disabled on AMD due to defect - EXSWHTEC-164
#if HT_NVIDIA
  SECTION("name == empty string") {
    HIP_CHECK_ERROR(hipModuleGetGlobal(&global, &global_size, module, ""), hipErrorInvalidValue);
  }
#endif

  SECTION("name == invalid name") {
    HIP_CHECK_ERROR(hipModuleGetGlobal(&global, &global_size, module, "dummy"), hipErrorNotFound);
  }
}