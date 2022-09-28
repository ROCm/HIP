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

#include <array>
#include <algorithm>
#include <type_traits>
#include <string_view>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

#include "hipModuleGetGlobal.hh"

template<typename T> const char * TypeNameToString() {
    if(std::is_same<int, T>::value) {
        return "int";
    } else if(std::is_same<float, T>::value) {
        return "float";
    } else {
        return "idk";
    }
}

template<typename T> void HipModuleGetGlobalTest(hipModule_t module) {
  hipDeviceptr_t var = nullptr;
  size_t var_size = 0;
  const std::string global_name = TypeNameToString<T>() + std::string("_arr");
  HIP_CHECK(hipModuleGetGlobal(&var, &var_size, module, global_name.c_str()));
  constexpr size_t expected_size = array_size * sizeof(T);
  REQUIRE(expected_size == var_size);
  std::array<T, array_size> arr;
  HIP_CHECK(hipMemcpyDtoH(arr.data(), var, array_size * sizeof(T)));
  const auto it = std::find_if_not(std::begin(arr), std::end(arr), [](const T element) {
    return expected_value<T> == element;
  });
  REQUIRE(it == std::end(arr));
}

TEST_CASE("Blahem") {
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, "test_module.code"));
  // Listed like this instead of using a templated test case to avoid loading the module separately
  // for each test case
  HipModuleGetGlobalTest<int>(module);
  HipModuleGetGlobalTest<float>(module);
}