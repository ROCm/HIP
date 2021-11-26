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

/**
Testcase Scenarios :
 1) Validates parameter list of hipStreamAddCallback.
 2) Validates hipStreamAddCallback functionality with default stream.
 3) Validates hipStreamAddCallback functionality with defined stream.
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <chrono>
#include <thread>

#define UNUSED(expr) do { (void)(expr); } while (0)

#ifdef __HIP_PLATFORM_AMD__
#define HIPRT_CB
#endif

namespace hipStreaAddCallbackTest {
size_t NSize = 4 * 1024 * 1024;
float *A_h, *C_h;
bool gcbDone = false;
bool gPassed = true;
void *ptr0xff = reinterpret_cast<void *>(0xffffffff);
void *gusrptr;
hipStream_t gstream;

void HIPRT_CB Callback(hipStream_t stream, hipError_t status,
                       void* userData) {
  UNUSED(stream);
  HIP_CHECK(status);
  REQUIRE(userData == NULL);
  gPassed = true;
  for (size_t i = 0; i < NSize; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      UNSCOPED_INFO("Data mismatch :" << i);
      gPassed = false;
      break;
    }
  }
  gcbDone = true;
}
/**
 * Validates functionality of hipStreamAddCallback with default/created stream.
 */
bool testStreamCallbackFunctionality(bool isDefault) {
  float *A_d, *C_d;
  size_t Nbytes = NSize * sizeof(float);
  gcbDone = false;
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(C_h != nullptr);

  // Fill with Phi + i
  for (size_t i = 0; i < NSize; i++) {
    A_h[i] = 1.618f + i;
  }

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  if (isDefault) {
    HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice,
                            0));

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;
    hipLaunchKernelGGL((HipTest::vector_square), dim3(blocks),
                        dim3(threadsPerBlock), 0, 0, A_d, C_d, NSize);

    HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost,
                            0));
    HIP_CHECK(hipStreamAddCallback(0, Callback, nullptr, 0));
    while (!gcbDone) std::this_thread::sleep_for(std::chrono::microseconds(100000));  // Sleep for 100 ms
  } else {
    hipStream_t mystream;
    HIP_CHECK(hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));

    HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice,
                            mystream));

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;
    hipLaunchKernelGGL((HipTest::vector_square), dim3(blocks),
                      dim3(threadsPerBlock), 0, mystream, A_d, C_d, NSize);

    HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost,
                            mystream));
    HIP_CHECK(hipStreamAddCallback(mystream, Callback, nullptr, 0));
    while (!gcbDone) std::this_thread::sleep_for(std::chrono::microseconds(100000));  // Sleep for 100 ms
    HIP_CHECK(hipStreamDestroy(mystream));
  }
  HIP_CHECK(hipFree(reinterpret_cast<void*>(C_d)));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
  free(C_h);
  free(A_h);
  return gPassed;
}
/**
 * Scenario1: Validates if callback = nullptr returns error code for created stream.
 * Scenario2: Validates if callback = nullptr returns error code for default stream.
 * Scenario3: Validates if flag != 0 returns error code for created stream.
 * Scenario4: Validates if flag != 0 returns error code for default stream.
 * Scenario5: Validates if userData pointer is passed properly to callback.
 * Scenario6: Validates if stream value is passed properly to callback.
 */
void Callback_ChkUsrdataPtr(hipStream_t stream, hipError_t status,
                            void* userData) {
  REQUIRE(stream == gstream);
  HIP_CHECK(status);
  gPassed = true;
  if (gusrptr != userData) {
    gPassed = false;
  }
  gcbDone = true;
}

void Callback_ChkStreamValue(hipStream_t stream, hipError_t status,
                             void* userData) {
  REQUIRE(userData == nullptr);
  HIP_CHECK(status);
  gPassed = true;
  if (stream != gstream) {
    gPassed = false;
  }
  gcbDone = true;
}
}  // namespace hipStreaAddCallbackTest


using hipStreaAddCallbackTest::gcbDone;
using hipStreaAddCallbackTest::gPassed;
using hipStreaAddCallbackTest::ptr0xff;
using hipStreaAddCallbackTest::gusrptr;
using hipStreaAddCallbackTest::gstream;
using hipStreaAddCallbackTest::testStreamCallbackFunctionality;
using hipStreaAddCallbackTest::Callback;
using hipStreaAddCallbackTest::Callback_ChkUsrdataPtr;
using hipStreaAddCallbackTest::Callback_ChkStreamValue;


/*
 * Validates parameter list of hipStreamAddCallback.
 */
TEST_CASE("Unit_hipStreamAddCallback_ParamTst") {
  hipStream_t mystream;
  HIP_CHECK(hipStreamCreate(&mystream));
  // Scenario1
  SECTION("callback is nullptr") {
    REQUIRE_FALSE(hipSuccess == hipStreamAddCallback(mystream, nullptr,
                                         nullptr, 0));
  }
  // Scenario2
  SECTION("stream is default") {
    REQUIRE_FALSE(hipSuccess == hipStreamAddCallback(0, nullptr,
                                         nullptr, 0));
  }
  // Scenario3
  SECTION("flag is nonzero for non-default stream") {
    REQUIRE_FALSE(hipSuccess == hipStreamAddCallback(mystream, Callback,
                                         nullptr, 10));
  }
  // Scenario4
  SECTION("flag is nonzero for default stream") {
    REQUIRE_FALSE(hipSuccess == hipStreamAddCallback(0, Callback,
                                         nullptr, 10));
  }
  // Scenario5
  SECTION("userData pointer value validation") {
    gstream = mystream;
    gusrptr = ptr0xff;
    gPassed = true;
    gcbDone = false;
    HIP_CHECK(hipStreamAddCallback(mystream, Callback_ChkUsrdataPtr,
                                  gusrptr, 0));
    while (!gcbDone) {
      std::this_thread::sleep_for(std::chrono::microseconds(100000));  // Sleep for 100 ms
    }
    REQUIRE_FALSE(!gPassed);
  }
  // Scenario6
  SECTION("stream value validation") {
    gstream = mystream;
    gPassed = true;
    gcbDone = false;
    HIP_CHECK(hipStreamAddCallback(mystream, Callback_ChkStreamValue,
                                  nullptr, 0));
    while (!gcbDone) {
      std::this_thread::sleep_for(std::chrono::microseconds(100000));  // Sleep for 100 ms
    }
    REQUIRE_FALSE(!gPassed);
  }
  HIP_CHECK(hipStreamDestroy(mystream));
}

/*
 * Validates hipStreamAddCallback functionality with default stream.
 */
TEST_CASE("Unit_hipStreamAddCallback_WithDefaultStream") {
  bool TestPassed = true;
  TestPassed = testStreamCallbackFunctionality(true);
  REQUIRE(TestPassed);
}

/*
 * Validates hipStreamAddCallback functionality with defined stream.
 */
TEST_CASE("Unit_hipStreamAddCallback_WithCreatedStream") {
  bool TestPassed = true;
  TestPassed = testStreamCallbackFunctionality(false);
  REQUIRE(TestPassed);
}

