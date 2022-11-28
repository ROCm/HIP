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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
/*
Testcase Scenarios :
Unit_hipRuntimeGetVersion_Positive - Test simple reading of HIP runtime version with hipRuntimeGetVersion api
Unit_hipRuntimeGetVersion_Negative - Test unsuccessful execution of hipRuntimeGetVersion when nullptr is set as input parameter
*/

/*
 * Conformance test for checking functionality of
 * hipError_t hipRuntimeGetVersion(int* runtimeVersion);
 * On HIP/HCC path this function returns HIP runtime patch version
 * (a 5 digit code) however on
 * HIP/NVCC path this function return CUDA runtime version.
 */
#include <hip_test_common.hh>

TEST_CASE("Unit_hipRuntimeGetVersion_Positive") {
  int runtimeVersion = -1;
  HIP_CHECK(hipRuntimeGetVersion(&runtimeVersion));
  REQUIRE(runtimeVersion > 0);
  INFO("Runtime version " << runtimeVersion);
}

TEST_CASE("Unit_hipRuntimeGetVersion_Negative") {
  // If initialization is attempted with nullptr, error shall be reported
  CHECK_FALSE(hipRuntimeGetVersion(nullptr) == hipSuccess);
}
