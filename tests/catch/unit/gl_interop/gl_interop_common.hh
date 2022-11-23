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

#include <variant>

#define GL_GLEXT_PROTOTYPES
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <hip_test_common.hh>

class GLBufferObject {
 public:
  static constexpr size_t kSize = 512 * 512 * 4 * sizeof(float);

  GLBufferObject() {
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, kSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    REQUIRE(glGetError() == GL_NO_ERROR);
  }

  ~GLBufferObject() { glDeleteBuffers(1, &vbo_); }

  operator GLuint() const { return vbo_; }

 private:
  GLuint vbo_;
};

class GLImageObject {
 public:
  static constexpr size_t kWidth = 512, kHeight = 512;

  GLImageObject() {
    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, kWidth, kHeight, 0, GL_RGBA_INTEGER_EXT,
                 GL_UNSIGNED_BYTE, NULL);
    REQUIRE(glGetError() == GL_NO_ERROR);
  }

  ~GLImageObject() { glDeleteTextures(1, &tex_); }

  operator GLuint() const { return tex_; }

 private:
  GLuint tex_;
};

static std::once_flag glut_init_flag;

class GLUTContextScopeGuard {
 public:
  GLUTContextScopeGuard() {
    std::call_once(glut_init_flag, &GLUTContextScopeGuard::init);
    glut_window_ = glutCreateWindow("");
  }

  ~GLUTContextScopeGuard() { glutDestroyWindow(glut_window_); }

  GLUTContextScopeGuard(const GLUTContextScopeGuard&) = delete;
  GLUTContextScopeGuard& operator=(const GLUTContextScopeGuard&) = delete;

  GLUTContextScopeGuard(GLUTContextScopeGuard&&) = delete;
  GLUTContextScopeGuard& operator=(GLUTContextScopeGuard&&) = delete;

 private:
  int glut_window_;

  static void init() {
    static char proc_name[] = "";
    static std::array<char*, 2> glut_argv = {proc_name, nullptr};
    static int glut_argc = 1;

    glutInit(&glut_argc, glut_argv.data());
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(512, 512);
  }
};

class EGLContextScopeGuard {
 public:
  EGLContextScopeGuard() {
    // 1. Initialize EGL
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");

    eglQueryDevicesEXT(egl_devices_.max_size(), egl_devices_.data(), &num_devices_);

    INFO("Detected " << num_devices_ << " devices");

    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

    egl_display_ = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, egl_devices_.at(0), 0);

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

  ~EGLContextScopeGuard() {
    // 6. Terminate EGL when finished
    eglTerminate(egl_display_);
  }

  EGLContextScopeGuard(const EGLContextScopeGuard&) = delete;
  EGLContextScopeGuard& operator=(const EGLContextScopeGuard&) = delete;

  EGLContextScopeGuard(EGLContextScopeGuard&&) = delete;
  EGLContextScopeGuard& operator=(EGLContextScopeGuard&&) = delete;

 private:
  // clang-format off
  static constexpr EGLint kConfigAttribs[] = {
      EGL_SURFACE_TYPE,
      EGL_PBUFFER_BIT,
      EGL_BLUE_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_RED_SIZE, 8,
      EGL_DEPTH_SIZE, 8,
      EGL_RENDERABLE_TYPE,
      EGL_OPENGL_BIT,
      EGL_NONE
  };
  // clang-format on

  static constexpr int kPbufferWidth = 9;
  static constexpr int kPbufferHeight = 9;

  static constexpr EGLint kPbufferAttribs[] = {
      EGL_WIDTH, kPbufferWidth, EGL_HEIGHT, kPbufferHeight, EGL_NONE,
  };

  std::array<EGLDeviceEXT, 8> egl_devices_;
  EGLint num_devices_;
  EGLDisplay egl_display_;
  EGLint major_, minor_;
  EGLint num_configs_;
  EGLConfig egl_config_;
  EGLSurface egl_surface_;
  EGLContext egl_context_;
};

class GLContextScopeGuard {
 public:
  using GLUTContextScopeGuardPtr = std::unique_ptr<GLUTContextScopeGuard>;
  using EGLContextScopeGuardPtr = std::unique_ptr<EGLContextScopeGuard>;
  using GLContextScopeGuardVariant =
      std::variant<GLUTContextScopeGuardPtr, EGLContextScopeGuardPtr>;

  static constexpr char kEnvarName[] = "GL_CONTEXT_TYPE";

  GLContextScopeGuard() {
    char* val = std::getenv(kEnvarName);
    std::string val_str = val == NULL ? "" : val;

    if (val_str.empty() || val_str == "GLUT") {
      gl_context_ = std::make_unique<GLUTContextScopeGuard>();
    } else if (val_str == "EGL") {
      gl_context_ = std::make_unique<EGLContextScopeGuard>();
    } else {
      INFO("Unsupported " << kEnvarName << " value '" << val_str << "'");
      INFO("Supported values are ['GLUT', 'EGL']");
      REQUIRE(false);
    }
  }

  GLContextScopeGuard(const GLContextScopeGuard&) = delete;
  GLContextScopeGuard& operator=(const GLContextScopeGuard&) = delete;

  GLContextScopeGuard(GLContextScopeGuard&&) = delete;
  GLContextScopeGuard& operator=(GLContextScopeGuard&&) = delete;

 private:
  GLContextScopeGuardVariant gl_context_;
};