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

TEST_CASE("Unit_hipGraphicsUnregisterResource_Negative_Parameters") {
  GLContextScopeGuard gl_context;

  GLBufferObject vbo;

  SECTION("already unregistered resource") {
    hipGraphicsResource* unregistered_resource;
    HIP_CHECK(
        hipGraphicsGLRegisterBuffer(&unregistered_resource, vbo, hipGraphicsRegisterFlagsNone));
    HIP_CHECK(hipGraphicsUnregisterResource(unregistered_resource));
    HIP_CHECK_ERROR(hipGraphicsUnregisterResource(unregistered_resource), hipErrorInvalidContext);
  }

  SECTION("mapped resource") {
    hipGraphicsResource* mapped_resource;
    HIP_CHECK(hipGraphicsGLRegisterBuffer(&mapped_resource, vbo, hipGraphicsRegisterFlagsNone));
    HIP_CHECK(hipGraphicsMapResources(1, &mapped_resource, 0));
    HIP_CHECK_ERROR(hipGraphicsUnregisterResource(mapped_resource), hipErrorAlreadyMapped);
  }
}