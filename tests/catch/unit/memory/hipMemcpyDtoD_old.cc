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
This testcase verifies the hipMemcpyDtoD basic scenario
1. H2D-KernelLaunch-D2H scenario
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>

static constexpr auto NUM_ELM{1024};

/*
This testcase verifies hipMemcpyDtoD API
1.Initializes device variables
2.Launches kernel and performs the sum of device variables
3.Copies the result to host variable and validates the result.
4.Sets the peer device
5.D2D copy from GPU-0 to GPU-1
6.Kernel Launch
7.DtoH copy and validating the result
*/
TEMPLATE_TEST_CASE("Unit_hipMemcpyDtoD_Basic", "",
                   int, float, double) {
  size_t Nbytes = NUM_ELM * sizeof(TestType);
  int numDevices = 0;
  TestType *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr},
           *X_d{nullptr}, *Y_d{nullptr}, *Z_d{nullptr};
  TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};

  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    int canAccessPeer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      HIP_CHECK(hipSetDevice(0));
      HipTest::initArrays<TestType>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h,
                                    NUM_ELM, false);
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipMalloc(&X_d, Nbytes));
      HIP_CHECK(hipMalloc(&Y_d, Nbytes));
      HIP_CHECK(hipMalloc(&Z_d, Nbytes));

      HIP_CHECK(hipSetDevice(0));
      HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));
      hipLaunchKernelGGL(HipTest::vectorADD, dim3(1),
                         dim3(1), 0, 0,
                         static_cast<const TestType *>(A_d),
                         static_cast<const TestType *>(B_d), C_d, NUM_ELM);
      HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
      HIP_CHECK(hipDeviceSynchronize());
      HipTest::checkVectorADD<TestType>(A_h, B_h, C_h, NUM_ELM);

      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipMemcpyDtoD((hipDeviceptr_t)X_d, (hipDeviceptr_t)A_d,
                              Nbytes));
      HIP_CHECK(hipMemcpyDtoD((hipDeviceptr_t)Y_d, (hipDeviceptr_t)B_d,
                              Nbytes));

      hipLaunchKernelGGL(HipTest::vectorADD, dim3(1),
                         dim3(1), 0, 0,
                         static_cast<const TestType*>(X_d),
                         static_cast<const TestType*>(Y_d), Z_d, NUM_ELM);
      HIP_CHECK(hipMemcpyDtoH(C_h, (hipDeviceptr_t)Z_d, Nbytes));
      HIP_CHECK(hipDeviceSynchronize());
      HipTest::checkVectorADD<TestType>(A_h, B_h, C_h, NUM_ELM);

      HipTest::freeArrays<TestType>(A_d, B_d, C_d, A_h, B_h, C_h, false);
      HIP_CHECK(hipFree(X_d));
      HIP_CHECK(hipFree(Y_d));
      HIP_CHECK(hipFree(Z_d));
    } else {
      SUCCEED("Machine does not seem to have P2P Capabilities");
    }
  }
}
