
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
#include <utility>

#define fileName "module_kernels.code"
#define kernel_name "hello_world"

namespace testhipFuncGetAttributesApi {
__global__
void fn(float* px, float* py) {
  bool a[42];
  __shared__ double b[69];
  for (auto&& x : b) x = *py++;
  for (auto&& x : a) x = *px++ > 0.0;
  for (auto&& x : a) if (x) *--py = *--px;
}
template <int WGSIZE, int LDS>
__launch_bounds__(WGSIZE, 1) __global__ void kernelfn(int *x) {
  __shared__ int lds[LDS];
  for (int i = 0; i < LDS; ++i) {
    lds[i] = x[i];
  }
  x[LDS - 1] = lds[0] / lds[LDS - 1];
}
template <int WGSIZE, int LDS> bool test_Attributes_Values() {
  bool TestPassed = true;
  hipFuncAttributes attr{};
  hipFuncGetAttributes(&attr,
     reinterpret_cast<void const *>(kernelfn<WGSIZE, LDS>));
  if (attr.maxThreadsPerBlock != WGSIZE) {
    TestPassed = false;
  }
  if (attr.sharedSizeBytes != LDS * sizeof(int)) {
    TestPassed = false;
  }
  return TestPassed;
}
}  // namespace testhipFuncGetAttributesApi
/**
 * hipFuncGetAttributes and hipModuleGetFunction functional tests
 * Scenario1: Validates the value of attribute "maxThreadsPerBlock" should be non zero.
 * Scenario2: Validates the value of attribute
 *            "HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK" should be non zero.
 */
// scenario 1
TEST_CASE("Unit_hipFuncGetAttributes_FuncTst") {
  hipFuncAttributes attr{};
  auto r = hipFuncGetAttributes(&attr,
  reinterpret_cast<const void*>(&testhipFuncGetAttributesApi::fn));
  REQUIRE_FALSE(r != hipSuccess);
  REQUIRE_FALSE(attr.maxThreadsPerBlock == 0);
}
// scenario 2
TEST_CASE("Unit_hipFuncGetAttribute_FuncTst") {
  hipModule_t Module;
  int attrib_val;
  CTX_CREATE()
  hipFunction_t Function;
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  auto r = hipFuncGetAttribute(&attrib_val,
          HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, Function);
  REQUIRE_FALSE(r != hipSuccess);
  REQUIRE_FALSE(attrib_val == 0);
  HIP_CHECK(hipModuleUnload(Module));
  CTX_DESTROY()
}
/**
 * hipFuncGetAttributes negative tests
 * Scenario1: Validates returned error code for attr = nullptr
 * Scenario2: Validates returned error code for function = nullptr
 */
TEST_CASE("Unit_hipFuncGetAttributes_NegTst") {
  SECTION("attr is nullptr") {
    REQUIRE_FALSE(hipSuccess == hipFuncGetAttributes(nullptr,
    reinterpret_cast<const void*>(&testhipFuncGetAttributesApi::fn)));
  }
  SECTION("function is nullptr") {
    hipFuncAttributes attr{};
    REQUIRE_FALSE(hipSuccess == hipFuncGetAttributes(&attr, nullptr));
  }
}
/**
 * hipFuncGetAttribute negative tests
 * Scenario1: Validates returned error code for attrib_val = nullptr
 * Scenario2: Validates returned error code for attrib = invalid = 0xff
 */
TEST_CASE("Unit_hipFuncGetAttribute_NegTst") {
  hipModule_t Module;
  CTX_CREATE()
  hipFunction_t Function;
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  SECTION("attr is nullptr") {
    REQUIRE_FALSE(hipSuccess == hipFuncGetAttribute(nullptr,
        HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, Function));
  }
  SECTION("attr is invalid") {
    int attrib_val;
    REQUIRE_FALSE(hipSuccess == hipFuncGetAttribute(&attrib_val,
           static_cast<hipFunction_attribute>(0xff), Function));
  }
  HIP_CHECK(hipModuleUnload(Module));
  CTX_DESTROY()
}
/**
 * hipFuncGetAttributes
 * Scenario4: Validates the value of attribute "maxThreadsPerBlock".
 * Scenario5: Validates the value of attribute "sharedSizeBytes".
 */
TEST_CASE("Unit_hipFuncGetAttributes_AttrTest") {
  bool TestPassed = true;
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<64, 64>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<128, 64>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<256, 64>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<512, 64>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<1024, 64>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<64, 128>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<128, 128>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<256, 128>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<512, 128>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<1024, 128>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<64, 256>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<128, 256>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<256, 256>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<512, 256>();
  TestPassed &= testhipFuncGetAttributesApi::
                test_Attributes_Values<1024, 256>();
  REQUIRE(TestPassed);
}

