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

TEST_CASE("Unit_hipGraphicsSubResourceGetMappedArray_Positive_Basic") {
  GLContextScopeGuard gl_context;

  GLImageObject tex;

  hipGraphicsResource* tex_resource;

  HIP_CHECK(
      hipGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D, hipGraphicsRegisterFlagsNone));

  HIP_CHECK(hipGraphicsMapResources(1, &tex_resource, 0));

  hipArray* image_devptr = nullptr;
  HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&image_devptr, tex_resource, 0, 0));

  REQUIRE(image_devptr != nullptr);

  HIP_CHECK(hipGraphicsUnmapResources(1, &tex_resource, 0));

  HIP_CHECK(hipGraphicsUnregisterResource(tex_resource));
}

TEST_CASE("Unit_hipGraphicsSubResourceGetMappedArray_Negative_Parameters") {
  GLContextScopeGuard gl_context;

  GLImageObject tex;

  hipGraphicsResource* tex_resource;

  HIP_CHECK(
      hipGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D, hipGraphicsRegisterFlagsNone));

  HIP_CHECK(hipGraphicsMapResources(1, &tex_resource, 0));

  hipArray* image_devptr = nullptr;

  SECTION("array == nullptr") {
    HIP_CHECK(hipGraphicsSubResourceGetMappedArray(nullptr, tex_resource, 0, 0));
  }

  SECTION("non-texture resource") {
    GLBufferObject vbo;
    hipGraphicsResource* vbo_resource;

    HIP_CHECK(hipGraphicsGLRegisterBuffer(&vbo_resource, vbo, hipGraphicsRegisterFlagsNone));
    HIP_CHECK(hipGraphicsMapResources(1, &vbo_resource, 0));

    HIP_CHECK_ERROR(hipGraphicsSubResourceGetMappedArray(&image_devptr, vbo_resource, 0, 0),
                    hipErrorNotMappedAsArray);

    HIP_CHECK(hipGraphicsUnmapResources(1, &vbo_resource, 0));
    HIP_CHECK(hipGraphicsUnregisterResource(vbo_resource));
  }

  SECTION("unregistered resource") {
    hipGraphicsResource* unregistered_resource;
    HIP_CHECK(hipGraphicsGLRegisterImage(&unregistered_resource, tex, GL_TEXTURE_2D,
                                         hipGraphicsRegisterFlagsNone));
    HIP_CHECK(hipGraphicsUnregisterResource(unregistered_resource));
    HIP_CHECK_ERROR(
        hipGraphicsSubResourceGetMappedArray(&image_devptr, unregistered_resource, 0, 0),
        hipErrorContextIsDestroyed);
  }

  SECTION("not mapped resource") {
    hipGraphicsResource* not_mapped_resource;
    HIP_CHECK(hipGraphicsGLRegisterImage(&not_mapped_resource, tex, GL_TEXTURE_2D,
                                         hipGraphicsRegisterFlagsNone));
    HIP_CHECK_ERROR(hipGraphicsSubResourceGetMappedArray(&image_devptr, not_mapped_resource, 0, 0),
                    hipErrorNotMapped);
    HIP_CHECK(hipGraphicsUnregisterResource(not_mapped_resource));
  }

  SECTION("unmapped resource") {
    hipGraphicsResource* unmapped_resource;

    HIP_CHECK(hipGraphicsGLRegisterImage(&unmapped_resource, tex, GL_TEXTURE_2D,
                                         hipGraphicsRegisterFlagsNone));

    HIP_CHECK(hipGraphicsMapResources(1, &unmapped_resource, 0));
    HIP_CHECK(hipGraphicsUnmapResources(1, &unmapped_resource, 0));

    HIP_CHECK_ERROR(hipGraphicsSubResourceGetMappedArray(&image_devptr, unmapped_resource, 0, 0),
                    hipErrorNotMapped);

    HIP_CHECK(hipGraphicsUnregisterResource(unmapped_resource));
  }

  SECTION("invalid arrayIndex") {
    HIP_CHECK_ERROR(hipGraphicsSubResourceGetMappedArray(&image_devptr, tex_resource,
                                                         std::numeric_limits<int>::max(), 0),
                    hipErrorInvalidValue);
  }

  SECTION("invalid mipLevel") {
    HIP_CHECK_ERROR(hipGraphicsSubResourceGetMappedArray(&image_devptr, tex_resource, 0,
                                                         std::numeric_limits<int>::max()),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphicsUnmapResources(1, &tex_resource, 0));

  HIP_CHECK(hipGraphicsUnregisterResource(tex_resource));
}