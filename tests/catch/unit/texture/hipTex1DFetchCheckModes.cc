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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

#define N 16
#define offset 3

static __global__ void tex1dKernel(float *val, hipTextureObject_t obj) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < (N - offset))
      val[k] = tex1Dfetch<float>(obj, k+offset);
}


static void runTest(hipTextureAddressMode addressMode,
                                         hipTextureFilterMode filterMode) {
  hipCtx_t HipContext;
  hipDevice_t HipDevice;
  int deviceID = 0;
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipDeviceGet(&HipDevice, deviceID));
  HIP_CHECK(hipCtxCreate(&HipContext, 0, HipDevice));

  // Allocating the required buffer on gpu device
  float *texBuf, *texBufOut;
  float val[N], output[N];

  for (int i = 0; i < N; i++) {
      val[i] = i+1;
      output[i] = 0.0;
  }

  HIP_CHECK(hipMalloc(&texBuf, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&texBufOut, N * sizeof(float)));
  HIP_CHECK(hipMemcpy(texBuf, val, N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(texBufOut, 0, N * sizeof(float)));
  hipResourceDesc resDescLinear;

  memset(&resDescLinear, 0, sizeof(resDescLinear));
  resDescLinear.resType = hipResourceTypeLinear;
  resDescLinear.res.linear.devPtr = texBuf;
  resDescLinear.res.linear.desc = hipCreateChannelDesc(32, 0, 0, 0,
                                                  hipChannelFormatKindFloat);
  resDescLinear.res.linear.sizeInBytes = N * sizeof(float);

  hipTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = hipReadModeElementType;
  texDesc.addressMode[0] = addressMode;
  texDesc.addressMode[1] = addressMode;
  texDesc.filterMode = filterMode;
  texDesc.normalizedCoords = false;

  // Creating texture object
  hipTextureObject_t texObj = 0;
  HIP_CHECK(hipCreateTextureObject(&texObj, &resDescLinear, &texDesc, NULL));

  dim3 dimBlock(1, 1, 1);
  dim3 dimGrid(N, 1, 1);

  hipLaunchKernelGGL(tex1dKernel, dim3(dimGrid), dim3(dimBlock), 0, 0,
                     texBufOut, texObj);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(output, texBufOut, N * sizeof(float),
                                                    hipMemcpyDeviceToHost));

  for (int i = 0; i < (N - offset); i++) {
      if (output[i] != val[i + offset]) {
          INFO("Output not matching at index " << i);
          REQUIRE(false);
      }
  }

  for (int i = (N - offset); i < N; i++) {
     if (output[i] != 0) {
          INFO("Output found to be updated at index " << i);
          REQUIRE(false);
     }
  }

  HIP_CHECK(hipDestroyTextureObject(texObj));
  HIP_CHECK(hipFree(texBuf));
  HIP_CHECK(hipFree(texBufOut));
}


TEST_CASE("Unit_tex1Dfetch_CheckModes") {
    SECTION("hipAddressModeClamp AND hipFilterModePoint") {
      runTest(hipAddressModeClamp, hipFilterModePoint);
    }
    SECTION("hipAddressModeClamp AND hipFilterModeLinear") {
      runTest(hipAddressModeClamp, hipFilterModeLinear);
    }
    SECTION("hipAddressModeWrap AND hipFilterModePoint") {
      runTest(hipAddressModeWrap, hipFilterModePoint);
    }
    SECTION("hipAddressModeWrap AND hipFilterModeLinear") {
      runTest(hipAddressModeWrap, hipFilterModeLinear);
    }
}
