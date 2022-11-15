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
#include <hip/hip_runtime_api.h>

#include "gl_interop_common.hh"

#include "GLContextScopeGuard.hh"

TEST_CASE("Unit_hipGraphicsResourceGetMappedPointer_Positive_Basic") {
  GLContextScopeGuard gl_context;

  CreateGLBufferObject();

  hipGraphicsResource* vbo_resource;

  HIP_CHECK(hipGraphicsGLRegisterBuffer(&vbo_resource, vbo, hipGraphicsRegisterFlagsNone));

  HIP_CHECK(hipGraphicsMapResources(1, &vbo_resource, 0));

  float* buffer_devptr = nullptr;
  size_t size = 0;

  HIP_CHECK(hipGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr), &size,
                                                vbo_resource));

  REQUIRE(buffer_devptr != nullptr);
  REQUIRE(size == kWidth * kHeight * 4 * sizeof(float));

  HIP_CHECK(hipGraphicsUnmapResources(1, &vbo_resource, 0));

  HIP_CHECK(hipGraphicsUnregisterResource(vbo_resource));
}

TEST_CASE("Unit_hipGraphicsResourceGetMappedPointer_Positive_Parameters") {
  GLContextScopeGuard gl_context;

  CreateGLBufferObject();

  hipGraphicsResource* vbo_resource;

  HIP_CHECK(hipGraphicsGLRegisterBuffer(&vbo_resource, vbo, hipGraphicsRegisterFlagsNone));

  HIP_CHECK(hipGraphicsMapResources(1, &vbo_resource, 0));

  float* buffer_devptr = nullptr;
  size_t size = 0;

  SECTION("devPtr == nullptr") {
    HIP_CHECK(hipGraphicsResourceGetMappedPointer(nullptr, &size, vbo_resource));
    REQUIRE(size == kWidth * kHeight * 4 * sizeof(float));
  }

  SECTION("size == nullptr") {
    HIP_CHECK(hipGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr), nullptr,
                                                  vbo_resource));
    REQUIRE(buffer_devptr != nullptr);
  }

  HIP_CHECK(hipGraphicsUnmapResources(1, &vbo_resource, 0));

  HIP_CHECK(hipGraphicsUnregisterResource(vbo_resource));
}

TEST_CASE("Unit_hipGraphicsResourceGetMappedPointer_Negative_Parameters") {
  GLContextScopeGuard gl_context;

  HIP_CHECK(hipFree(0));  // necessary for correct initialization on NVIDIA

  float* buffer_devptr = nullptr;
  size_t size = 0;

  SECTION("invalid resource") {
    hipGraphicsResource* invalid_resource;
    HIP_CHECK_ERROR(hipGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr),
                                                        &size, invalid_resource),
                    hipErrorInvalidHandle);
  }
}