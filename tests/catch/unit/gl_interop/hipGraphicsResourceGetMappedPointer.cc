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

TEST_CASE("Unit_hipGraphicsResourceGetMappedPointer_Positive_Basic") {
  GLContextScopeGuard gl_context;

  GLBufferObject vbo;

  hipGraphicsResource* vbo_resource;

  HIP_CHECK(hipGraphicsGLRegisterBuffer(&vbo_resource, vbo, hipGraphicsRegisterFlagsNone));

  HIP_CHECK(hipGraphicsMapResources(1, &vbo_resource, 0));

  float* buffer_devptr = nullptr;
  size_t size = 0;

  HIP_CHECK(hipGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr), &size,
                                                vbo_resource));

  REQUIRE(buffer_devptr != nullptr);
  REQUIRE(size == vbo.kSize);

  HIP_CHECK(hipGraphicsUnmapResources(1, &vbo_resource, 0));

  HIP_CHECK(hipGraphicsUnregisterResource(vbo_resource));
}

TEST_CASE("Unit_hipGraphicsResourceGetMappedPointer_Positive_Parameters") {
  GLContextScopeGuard gl_context;

  GLBufferObject vbo;

  hipGraphicsResource* vbo_resource;

  HIP_CHECK(hipGraphicsGLRegisterBuffer(&vbo_resource, vbo, hipGraphicsRegisterFlagsNone));

  HIP_CHECK(hipGraphicsMapResources(1, &vbo_resource, 0));

  float* buffer_devptr = nullptr;
  size_t size = 0;

  SECTION("devPtr == nullptr") {
    HIP_CHECK(hipGraphicsResourceGetMappedPointer(nullptr, &size, vbo_resource));
    REQUIRE(size == vbo.kSize);
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

  GLBufferObject vbo;

  hipGraphicsResource* vbo_resource;

  HIP_CHECK(hipGraphicsGLRegisterBuffer(&vbo_resource, vbo, hipGraphicsRegisterFlagsNone));

  HIP_CHECK(hipGraphicsMapResources(1, &vbo_resource, 0));

  float* buffer_devptr = nullptr;
  size_t size = 0;

  SECTION("non-pointer resource") {
    GLImageObject tex;
    hipGraphicsResource* tex_resource;

    HIP_CHECK(hipGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D,
                                         hipGraphicsRegisterFlagsNone));
    HIP_CHECK(hipGraphicsMapResources(1, &tex_resource, 0));

    HIP_CHECK_ERROR(hipGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr),
                                                        &size, tex_resource),
                    hipErrorNotMappedAsPointer);

    HIP_CHECK(hipGraphicsUnmapResources(1, &tex_resource, 0));
    HIP_CHECK(hipGraphicsUnregisterResource(tex_resource));
  }

  SECTION("unregistered resource") {
    hipGraphicsResource* unregistered_resource;
    HIP_CHECK(
        hipGraphicsGLRegisterBuffer(&unregistered_resource, vbo, hipGraphicsRegisterFlagsNone));
    HIP_CHECK(hipGraphicsUnregisterResource(unregistered_resource));
    HIP_CHECK_ERROR(hipGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr),
                                                        &size, unregistered_resource),
                    hipErrorContextIsDestroyed);
  }

  SECTION("not mapped resource") {
    hipGraphicsResource* not_mapped_resource;
    HIP_CHECK(hipGraphicsGLRegisterBuffer(&not_mapped_resource, vbo, hipGraphicsRegisterFlagsNone));
    HIP_CHECK_ERROR(hipGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr),
                                                        &size, not_mapped_resource),
                    hipErrorNotMapped);
    HIP_CHECK(hipGraphicsUnregisterResource(not_mapped_resource));
  }

  SECTION("unmapped resource") {
    hipGraphicsResource* unmapped_resource;

    HIP_CHECK(hipGraphicsGLRegisterBuffer(&unmapped_resource, vbo, hipGraphicsRegisterFlagsNone));

    HIP_CHECK(hipGraphicsMapResources(1, &unmapped_resource, 0));
    HIP_CHECK(hipGraphicsUnmapResources(1, &unmapped_resource, 0));

    HIP_CHECK_ERROR(hipGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr),
                                                        &size, unmapped_resource),
                    hipErrorNotMapped);

    HIP_CHECK(hipGraphicsUnregisterResource(unmapped_resource));
  }

  HIP_CHECK(hipGraphicsUnmapResources(1, &vbo_resource, 0));

  HIP_CHECK(hipGraphicsUnregisterResource(vbo_resource));
}