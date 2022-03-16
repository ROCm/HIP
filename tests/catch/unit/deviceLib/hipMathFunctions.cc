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
 1) Verification of absolute int64 operation performed at device.
 2) Verification of __fp16 operation performed at device.
 3) Verification of pow operations performed at device.
*/

#include <hip_test_common.hh>

__global__ void kernel_abs_int64(long long *input, long long *output) {  // NOLINT
    int tx = threadIdx.x;
    output[tx] = abs(input[tx]);
}


#define CHECK_ABS_INT64(IN, OUT, EXP)    \
  {                                      \
    if (OUT != EXP)  {                   \
      INFO("check_abs_int64 failed on " << IN << ", output " << OUT << \
                  ", expected " << EXP); \
      REQUIRE(false);                    \
    }                                    \
  }

template<class T, class F>
__global__ void kernel_simple(F f, T *out) {
  *out = f();
}

template<class T, class F>
void check_simple(F f, T expected, const char* file, unsigned line) {
  auto memsize = sizeof(T);
  T *outputCPU = reinterpret_cast<T *>(malloc(memsize));
  T *outputGPU = nullptr;
  REQUIRE(outputCPU != nullptr);
  HIP_CHECK(hipMalloc(&outputGPU, memsize));
  hipLaunchKernelGGL(kernel_simple, 1, 1, 0, 0, f, outputGPU);
  HIP_CHECK(hipMemcpy(outputCPU, outputGPU, memsize, hipMemcpyDeviceToHost));
  if (*outputCPU != expected) {
    INFO("File " << file << ", line " << line << " check failed." <<
    " output = " << static_cast<double>(*outputCPU) << " expected "
    << static_cast<double>(expected));
    REQUIRE(false);
  }
  HIP_CHECK(hipFree(outputGPU));
  free(outputCPU);
}

#define CHECK_SIMPLE(lambda, expected) \
    check_simple(lambda, expected, __FILE__, __LINE__);


/**
  Verification of absolute int64 operation performed at device.
 */
TEST_CASE("Unit_abs_int64_Verification") {
  using datatype_t = long long;  // NOLINT

  datatype_t *inputCPU{}, *outputCPU{};
  datatype_t *inputGPU{}, *outputGPU{};
  const int NUM_INPUTS = 8;
  auto memsize = NUM_INPUTS * sizeof(datatype_t);

  // allocate memories
  inputCPU = reinterpret_cast<datatype_t *>(malloc(memsize));
  outputCPU = reinterpret_cast<datatype_t *>(malloc(memsize));
  REQUIRE(inputCPU != nullptr);
  REQUIRE(outputCPU != nullptr);
  HIP_CHECK(hipMalloc(&inputGPU, memsize));
  HIP_CHECK(hipMalloc(&outputGPU, memsize));

  // populate input with constants
  inputCPU[0] = -81985529216486895ll;
  inputCPU[1] =  81985529216486895ll;
  inputCPU[2] = -1250999896491ll;
  inputCPU[3] =  1250999896491ll;
  inputCPU[4] = -19088743ll;
  inputCPU[5] =  19088743ll;
  inputCPU[6] = -291ll;
  inputCPU[7] =  291ll;

  // copy inputs to device
  HIP_CHECK(hipMemcpy(inputGPU, inputCPU, memsize, hipMemcpyHostToDevice));

  // launch kernel
  hipLaunchKernelGGL(kernel_abs_int64, dim3(1), dim3(NUM_INPUTS), 0, 0,
                                                      inputGPU, outputGPU);
  // copy outputs from device
  HIP_CHECK(hipMemcpy(outputCPU, outputGPU, memsize, hipMemcpyDeviceToHost));

  // check outputs
  CHECK_ABS_INT64(inputCPU[0], outputCPU[0], outputCPU[1]);
  CHECK_ABS_INT64(inputCPU[1], outputCPU[1], outputCPU[1]);
  CHECK_ABS_INT64(inputCPU[2], outputCPU[2], outputCPU[3]);
  CHECK_ABS_INT64(inputCPU[3], outputCPU[3], outputCPU[3]);
  CHECK_ABS_INT64(inputCPU[4], outputCPU[4], outputCPU[5]);
  CHECK_ABS_INT64(inputCPU[5], outputCPU[5], outputCPU[5]);
  CHECK_ABS_INT64(inputCPU[6], outputCPU[6], outputCPU[7]);
  CHECK_ABS_INT64(inputCPU[7], outputCPU[7], outputCPU[7]);

  // free memories
  HIP_CHECK(hipFree(inputGPU));
  HIP_CHECK(hipFree(outputGPU));
  free(inputCPU);
  free(outputCPU);
}

/**
  Verification of __fp16 operation performed at device.
 */
TEST_CASE("Unit__fp16_Verification") {
  CHECK_SIMPLE([]__device__(){ return max<__fp16>(1.0f, 2.0f); }, 2.0f);
  CHECK_SIMPLE([]__device__(){ return min<__fp16>(1.0f, 2.0f); }, 1.0f);
}

/**
  Verification of pow operations performed at device.
 */
TEST_CASE("Unit_pown_Verification") {
  CHECK_SIMPLE([]__device__(){ return powif(2.0f, 2); }, 4.0f);
  CHECK_SIMPLE([]__device__(){ return powi(2.0, 2); }, 4.0);
  CHECK_SIMPLE([]__device__(){ return pow(2.0f, 2); }, 4.0f);
  CHECK_SIMPLE([]__device__(){ return pow(2.0, 2); }, 4.0);
  CHECK_SIMPLE([]__device__(){ return pow(2.0f16, 2); }, 4.0f16);
}
