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

/*
This testcase verifies the basic scenario of hipHostGetFlags API
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

static constexpr auto LEN{1024*1024};

/*
This testcase verifies hipHostGetFlags API basic scenario
1. Allocates the memory using different flags
2. Gets the flags of the respective variable using
   hipHostGetFlags API
3. Validates it with the initial flags used while allocating
   memory
*/
TEMPLATE_TEST_CASE("Unit_hipHostGetFlags_Basic", "", int,
                    float, double) {
  constexpr auto SIZE{LEN * sizeof(TestType)};
  TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  TestType *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  unsigned int FlagA, FlagB, FlagC;

  FlagA = hipHostMallocWriteCombined | hipHostMallocMapped;
  FlagB = hipHostMallocWriteCombined | hipHostMallocMapped;
  FlagC = hipHostMallocMapped;

  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  if (prop.canMapHostMemory != 1) {
    SUCCEED("Device Property canMapHostMemory is not set");
  } else {
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), SIZE,
                            hipHostMallocWriteCombined | hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&B_h), SIZE,
                            hipHostMallocWriteCombined | hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&C_h), SIZE,
                            hipHostMallocMapped));

    unsigned int flagA, flagB, flagC;
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), A_h, 0));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&B_d), B_h, 0));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&C_d), C_h, 0));
    HIP_CHECK(hipHostGetFlags(&flagA, A_h));
    HIP_CHECK(hipHostGetFlags(&flagB, B_h));
    HIP_CHECK(hipHostGetFlags(&flagC, C_h));

    HipTest::setDefaultData<TestType>(LEN, A_h, B_h, C_h);

    dim3 dimGrid(LEN / 512, 1, 1);
    dim3 dimBlock(512, 1, 1);
    hipLaunchKernelGGL(HipTest::vectorADD, dimGrid, dimBlock,
                       0, 0, static_cast<const TestType*>(A_d),
                       static_cast<const TestType*>(B_d), C_d, LEN);

    HIP_CHECK(hipMemcpy(C_h, C_d, SIZE, hipMemcpyDeviceToHost));
    // Note this really HostToHost not
    // DeviceToHost, since memory is mapped...
    HipTest::checkVectorADD(A_h, B_h, C_h, LEN);

    REQUIRE(flagA == FlagA);
    REQUIRE(flagB == FlagB);
    REQUIRE(flagC == FlagC);
    HIP_CHECK(hipHostFree(A_h));
    HIP_CHECK(hipHostFree(B_h));
    HIP_CHECK(hipHostFree(C_h));
  }
}
