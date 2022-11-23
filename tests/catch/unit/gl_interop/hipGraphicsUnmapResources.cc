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

TEST_CASE("Unit_hipGraphicsUnmapResources_Negative_Parameters") {
  GLContextScopeGuard gl_context;

  GLBufferObject vbo;

  hipGraphicsResource* vbo_resource;

  HIP_CHECK(hipGraphicsGLRegisterBuffer(&vbo_resource, vbo, hipGraphicsRegisterFlagsNone));

  HIP_CHECK(hipGraphicsMapResources(1, &vbo_resource, 0));

  SECTION("count == 0") {
    HIP_CHECK_ERROR(hipGraphicsUnmapResources(0, &vbo_resource, 0), hipErrorInvalidValue);
  }

  SECTION("resources == nullptr") {
    HIP_CHECK_ERROR(hipGraphicsUnmapResources(1, nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("not mapped resource") {
    hipGraphicsResource* not_mapped_resource;
    HIP_CHECK(hipGraphicsGLRegisterBuffer(&not_mapped_resource, vbo, hipGraphicsRegisterFlagsNone));
    HIP_CHECK_ERROR(hipGraphicsUnmapResources(1, &not_mapped_resource, 0), hipErrorNotMapped);
    HIP_CHECK(hipGraphicsUnregisterResource(not_mapped_resource));
  }

  SECTION("invalid stream") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipGraphicsUnmapResources(1, &vbo_resource, stream),
                    hipErrorContextIsDestroyed);
  }

  HIP_CHECK(hipGraphicsUnmapResources(1, &vbo_resource, 0));

  HIP_CHECK(hipGraphicsUnregisterResource(vbo_resource));
}