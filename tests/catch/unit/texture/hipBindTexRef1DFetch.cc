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

#define N 512

texture<float, 1, hipReadModeElementType> tex;

static __global__ void kernel(float *out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
      out[x] = tex1Dfetch(tex, x);
  }
}


TEST_CASE("Unit_hipBindTexture_tex1DfetchVerification") {
  float *texBuf;
  float val[N], output[N];
  size_t offset = 0;
  float *devBuf;
  for (int i = 0; i < N; i++) {
      val[i] = i;
      output[i] = 0.0;
  }
  hipChannelFormatDesc chanDesc =
      hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);

  HIP_CHECK(hipMalloc(&texBuf, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&devBuf, N * sizeof(float)));
  HIP_CHECK(hipMemcpy(texBuf, val, N * sizeof(float), hipMemcpyHostToDevice));

  tex.addressMode[0] = hipAddressModeClamp;
  tex.addressMode[1] = hipAddressModeClamp;
  tex.filterMode = hipFilterModePoint;
  tex.normalized = 0;

  HIP_CHECK(hipBindTexture(&offset, tex, reinterpret_cast<void *>(texBuf),
                                               chanDesc, N * sizeof(float)));
  HIP_CHECK(hipGetTextureAlignmentOffset(&offset, &tex));

  dim3 dimBlock(64, 1, 1);
  dim3 dimGrid(N / dimBlock.x, 1, 1);

  hipLaunchKernelGGL(kernel, dim3(dimGrid), dim3(dimBlock), 0, 0, devBuf);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(output, devBuf, N * sizeof(float),
                                                  hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
      if (output[i] != val[i]) {
        INFO("Mismatch at index : " << i << ", output[i] " << output[i]
                                               << ", val[i] " << val[i]);
        REQUIRE(false);
      }
  }

  HIP_CHECK(hipUnbindTexture(&tex));
  HIP_CHECK(hipFree(texBuf));
  HIP_CHECK(hipFree(devBuf));
}
