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

#pragma once

#include <EGL/egl.h>

#include <hip_test_common.hh>

class GLContextScopeGuard {
 public:
  GLContextScopeGuard() {
    // 1. Initialize EGL
    egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    REQUIRE(eglInitialize(egl_display_, &major_, &minor_));

    // 2. Select an appropriate configuration
    REQUIRE(eglChooseConfig(egl_display_, kConfigAttribs, &egl_config_, 1, &num_configs_));

    // 3. Create a surface
    egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config_, kPbufferAttribs);

    // 4. Bind the API
    REQUIRE(eglBindAPI(EGL_OPENGL_API));

    // 5. Create a context and make it current
    egl_context_ = eglCreateContext(egl_display_, egl_config_, EGL_NO_CONTEXT, NULL);

    REQUIRE(eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_));
  }

  ~GLContextScopeGuard() {
    // 6. Terminate EGL when finished
    eglTerminate(egl_display_);
  }

  GLContextScopeGuard(const GLContextScopeGuard&) = delete;
  GLContextScopeGuard& operator=(const GLContextScopeGuard&) = delete;

  GLContextScopeGuard(GLContextScopeGuard&&) = delete;
  GLContextScopeGuard& operator=(GLContextScopeGuard&&) = delete;

 private:
  static constexpr EGLint kConfigAttribs[] = {EGL_SURFACE_TYPE,
                                              EGL_PBUFFER_BIT,
                                              EGL_BLUE_SIZE,
                                              8,
                                              EGL_GREEN_SIZE,
                                              8,
                                              EGL_RED_SIZE,
                                              8,
                                              EGL_DEPTH_SIZE,
                                              8,
                                              EGL_RENDERABLE_TYPE,
                                              EGL_OPENGL_BIT,
                                              EGL_NONE};

  static constexpr int kPbufferWidth = 9;
  static constexpr int kPbufferHeight = 9;

  static constexpr EGLint kPbufferAttribs[] = {
      EGL_WIDTH, kPbufferWidth, EGL_HEIGHT, kPbufferHeight, EGL_NONE,
  };

  EGLDisplay egl_display_;
  EGLint major_;
  EGLint minor_;
  EGLint num_configs_;
  EGLConfig egl_config_;
  EGLSurface egl_surface_;
  EGLContext egl_context_;
};