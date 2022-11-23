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

namespace {
constexpr std::array<unsigned int, 5> kFlags{
    hipGraphicsRegisterFlagsNone, hipGraphicsRegisterFlagsReadOnly,
    hipGraphicsRegisterFlagsWriteDiscard, hipGraphicsRegisterFlagsSurfaceLoadStore,
    hipGraphicsRegisterFlagsTextureGather};
}  // anonymous namespace

TEST_CASE("Unit_hipGraphicsGLRegisterImage_Positive_Basic") {
  GLContextScopeGuard gl_context;

  const auto flags = GENERATE(from_range(begin(kFlags), end(kFlags)));

  GLImageObject tex;

  hipGraphicsResource* tex_resource;

  HIP_CHECK(hipGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D, flags));

  HIP_CHECK(hipGraphicsUnregisterResource(tex_resource));
}

TEST_CASE("Unit_hipGraphicsGLRegisterImage_Positive_Register_Twice") {
  GLContextScopeGuard gl_context;

  GLImageObject tex;

  hipGraphicsResource *tex_resource_1, *tex_resource_2;

  HIP_CHECK(hipGraphicsGLRegisterImage(&tex_resource_1, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsNone));
  HIP_CHECK(hipGraphicsGLRegisterImage(&tex_resource_2, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsNone));

  HIP_CHECK(hipGraphicsUnregisterResource(tex_resource_1));
  HIP_CHECK(hipGraphicsUnregisterResource(tex_resource_2));
}

TEST_CASE("Unit_hipGraphicsGLRegisterImage_Negative_Parameters") {
  GLContextScopeGuard gl_context;

  GLImageObject tex;

  hipGraphicsResource* tex_resource;

  SECTION("resource == nullptr") {
    HIP_CHECK_ERROR(
        hipGraphicsGLRegisterImage(nullptr, tex, GL_TEXTURE_2D, hipGraphicsRegisterFlagsNone),
        hipErrorInvalidValue);
  }

  SECTION("invalid image") {
    HIP_CHECK_ERROR(hipGraphicsGLRegisterImage(&tex_resource, GLuint{}, GL_TEXTURE_2D,
                                               hipGraphicsRegisterFlagsNone),
                    hipErrorInvalidValue);
  }

  SECTION("invalid target") {
    HIP_CHECK_ERROR(
        hipGraphicsGLRegisterImage(&tex_resource, tex, GL_BUFFER, hipGraphicsRegisterFlagsNone),
        hipErrorInvalidValue);
  }

  SECTION("target does not match the object") {
    HIP_CHECK_ERROR(hipGraphicsGLRegisterImage(&tex_resource, tex, GL_RENDERBUFFER,
                                               hipGraphicsRegisterFlagsNone),
                    hipErrorInvalidValue);
  }

  SECTION("invalid flags") {
    HIP_CHECK_ERROR(hipGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D,
                                               std::numeric_limits<unsigned int>::max()),
                    hipErrorInvalidValue);
  }
}