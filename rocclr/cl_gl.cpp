/* Copyright (c) 2010-present Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "top.hpp"

#ifdef _WIN32
#include <d3d10_1.h>
#include <d3d9.h>
#include <dxgi.h>
// This is necessary since there are common GL/D3D10 functions
#include "cl_d3d9_amd.hpp"
#include "cl_d3d10_amd.hpp"
#include "cl_d3d11_amd.hpp"
#endif  //_WIN32

#include <GL/gl.h>
#include <GL/glext.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglplatform.h>

#include "cl_common.hpp"
#include "cl_gl_amd.hpp"

#include "device/device.hpp"

/* The pixel internal format for DOPP texture defined in gl_enum.h */
#define GL_BGR8_ATI 0x8083
#define GL_BGRA8_ATI 0x8088

#include <cstring>
#include <vector>


/*! \addtogroup API
 *  @{
 *
 *  \addtogroup CL_GL_Interops
 *
 * This section discusses OpenCL functions that allow applications to
 * use OpenGL buffer/texture/render-buffer objects as OpenCL memory
 * objects. This allows efficient sharing of data between these OpenCL
 * and OpenGL. The OpenCL API can be used to execute kernels that read
 * and/or write memory objects that are also an OpenGL buffer object
 * or a texture.  An OpenCL image object can be created from an OpenGL
 * texture or renderbuffer object. An OpenCL buffer object can be
 * created from an OpenGL buffer object.  An OpenCL memory object can
 * be created from an OpenGL texture/buffer/render-buffer object or
 * the default system provided framebuffer if any only if the OpenCL
 * clContext has been created from a GL clContext. OpenGL contexts are
 * created using platform specific APIs (EGL, CGL, WGL, GLX are some
 * of the platform specific APIs that allow applications to create GL
 * contexts). The appropriate platform API (such as EGL, CGL, WGL,
 * GLX) will be extended to allow a CL clContext to be created from a
 * GL clContext. Creating an OpenCL memory object from the default
 * system provided framebuffer will also require an appropriate
 * extension to the platform API. Refer to the appropriate platform
 * API documentation to understand how to create a CL clContext from a
 * GL clContext and creating a CL memory object from the default
 * system provided framebuffer.
 *
 *  @{
 *
 *  \addtogroup clCreateFromGLBuffer
 *
 *  @{
 */

/*! \brief Creates an OpenCL buffer object from an OpenGL buffer object.
 *
 *  \param clContext is a valid OpenCL clContext created from an OpenGL clContext.
 *
 *  \param clFlags is a bit-field that is used to specify usage information. Only
 *  CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY and CL_MEM_READ_WRITE can be used.
 *
 *  \param glBufferName is a GL buffer object name. The GL buffer
 *  object must have a data store created though it does not need to
 *  be initialized. The size of the data store will be used to
 *  determine the size of the CL buffer object.
 *
 *  \param pCpuMem is a pointer to the buffer data that may already be
 *  allocated by the application. The size of the buffer that pCpuMem points
 *  to must be >= \a size bytes. Passing in a pointer to an already allocated
 *  buffer on the host and using it as a buffer object allows applications to
 *  share data efficiently with kernels and the host.
 *
 *  \param errcode_ret will return an appropriate error code. If errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return valid non-zero OpenCL buffer object and errcode_ret is set
 *  to CL_SUCCESS if the buffer object is created successfully. It
 *  returns a NULL value with one of the following error values
 *  returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a clContext is not a valid clContext.
 *  - CL_INVALID_VALUE if values specified in \a clFlags are not valid.
 *  - CL_INVALID_GL_OBJECT if glBufferName is not a GL buffer object or is a
 *    GL buffer object but does not have a data store created.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r29
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateFromGLBuffer,
                  (cl_context context, cl_mem_flags flags, GLuint bufobj, cl_int* errcode_ret)) {
  cl_mem clMemObj = NULL;

  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return clMemObj;
  }

  if (!(((flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY) ||
        ((flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY) ||
        ((flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE))) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return clMemObj;
  }

  return (amd::clCreateFromGLBufferAMD(*as_amd(context), flags, bufobj, errcode_ret));
}
RUNTIME_EXIT

/*! \brief creates the following:
 *  - an OpenCL 2D image object from an OpenGL 2D texture object
 *    or a single face of an OpenGL cubemap texture object,
 *  - an OpenCL 2D image array object from an OpenGL 2D texture array object,
 *  - an OpenCL 1D image object from an OpenGL 1D texture object,
 *  - an OpenCL 1D image buffer object from an OpenGL texture buffer object,
 *  - an OpenCL 1D image array object from an OpenGL 1D texture array object,
 *  - an OpenCL 3D image object from an OpenGL 3D texture object.
 *
 *  \param clContext is a valid OpenCL clContext created from an OpenGL clContext.
 *
 *  \param clFlags is a bit-field that is used to specify usage information.
 *  Only CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY and CL_MEM_READ_WRITE values
 *  can be used.
 *
 *  \param texture_target must be GL_TEXTURE_1D, GL_TEXTURE_1D_ARRAY,
 *  GL_TEXTURE_BUFFER, GL_TEXTURE_2D_ARRAY, GL_TEXTURE_3D,
 *  GL_TEXTURE_2D, GL_TEXTURE_CUBE_MAP_POSITIVE_X,
 *  GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
 *  GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
 *  GL_TEXTURE_CUBE_MAP_NEGATIVE_Z or GL_TEXTURE_RECTANGLE_ARB.
 *
 *  \param miplevel is the mipmap level to be used. If \a texture_target
 *  is GL_TEXTURE_BUFFER, \a miplevel must be 0.
 *
 *  \param texture is a GL 1D, 2D, 3D, 1D array, 2D array, cubemap,
 *  rectangle or buffer texture object.
 *  The texture object must be a complete texture as per
 *  OpenGL rules on texture completeness. The texture format and dimensions
 *  defined by OpenGL for the specified miplevel of the texture will be
 *  used to create the OpenCL image memory object. Only GL texture formats
 *  that map to appropriate image channel order and data type can be used
 *  to create the the OpenCL image memory object.
 *
 *  \param errcode_ret will return an appropriate error code. If \a
 *  errcode_ret is NULL, no error code is returned.
 *
 *  \return A valid non-zero OpenCL image object and \a errcode_ret is set to
 *  CL_SUCCESS if the image object is created successfully. It returns a NULL value
 *  with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a clContext is not a valid clContext or was not
 *    created from a GL clContext.
 *  - CL_INVALID_VALUE if values specified in \a clFlags are not valid.
 *  - CL_INVALID_MIP_LEVEL if \a miplevel is not a valid mip-level for \a texture.
 *  - CL_INVALID_GL_OBJECT if \a texture is not an appropriate GL 2D texture,
 *    cubemap or texture rectangle.
 *  - CL_INVALID_IMAGE_FORMAT_DESCRIPTOR if the OpenGL texture format does not
 *    map to an appropriate OpenCL image format.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateFromGLTexture,
                  (cl_context context, cl_mem_flags flags, GLenum texture_target, GLint miplevel,
                   GLuint texture, cl_int* errcode_ret)) {
  cl_mem clMemObj = NULL;

  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return clMemObj;
  }

  if (!(((flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY) ||
        ((flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY) ||
        ((flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE))) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return clMemObj;
  }

  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  bool supportPass = false;
  bool sizePass = false;
  for (const auto& it : devices) {
    if (it->info().imageSupport_) {
      supportPass = true;
    }
  }
  if (!supportPass) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    LogWarning("there are no devices in context to support images");
    return static_cast<cl_mem>(0);
  }

  return amd::clCreateFromGLTextureAMD(*as_amd(context), flags, texture_target, miplevel, texture,
                                       errcode_ret);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateFromGLTexture2D
 *  @{
 */

/*! \brief Create an OpenCL 2D image object from an OpenGL 2D texture object.
 *
 *  \param clContext is a valid OpenCL clContext created from an OpenGL clContext.
 *
 *  \param clFlags is a bit-field that is used to specify usage information.
 *  Only CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY and CL_MEM_READ_WRITE values
 *  can be used.
 *
 *  \param target must be GL_TEXTURE_2D, GL_TEXTURE_CUBE_MAP_POSITIVE_X,
 *  GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
 *  GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
 *  GL_TEXTURE_CUBE_MAP_NEGATIVE_Z or GL_TEXTURE_RECTANGLE_ARB.
 *
 *  \param miplevel is the mipmap level to be used.
 *
 *  \param texture is a GL 2D texture, cubemap or texture rectangle
 *  object name.  The texture object must be a complete texture as per
 *  OpenGL rules on texture completeness. The \a texture format and
 *  dimensions specified using appropriate glTexImage2D call for \a
 *  miplevel will be used to create the 2D image object.  Only GL
 *  texture formats that map to appropriate image channel order and
 *  data type can be used to create the 2D image object.
 *
 *  \param errcode_ret will return an appropriate error code. If \a
 *  errcode_ret is NULL, no error code is returned.
 *
 *  \return A valid non-zero OpenCL image object and \a errcode_ret is set to
 *  CL_SUCCESS if the image object is created successfully. It returns a NULL value
 *  with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a clContext is not a valid clContext or was not
 *    created from a GL clContext.
 *  - CL_INVALID_VALUE if values specified in \a clFlags are not valid.
 *  - CL_INVALID_MIP_LEVEL if \a miplevel is not a valid mip-level for \a texture.
 *  - CL_INVALID_GL_OBJECT if \a texture is not an appropriate GL 2D texture,
 *    cubemap or texture rectangle.
 *  - CL_INVALID_IMAGE_FORMAT_DESCRIPTOR if the OpenGL texture format does not
 *    map to an appropriate OpenCL image format.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r29
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateFromGLTexture2D,
                  (cl_context context, cl_mem_flags flags, GLenum target, GLint miplevel,
                   GLuint texture, cl_int* errcode_ret)) {
  cl_mem clMemObj = NULL;

  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return clMemObj;
  }

  if (!(((flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY) ||
        ((flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY) ||
        ((flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE))) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return clMemObj;
  }

  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  bool supportPass = false;
  bool sizePass = false;
  for (const auto& it : devices) {
    if (it->info().imageSupport_) {
      supportPass = true;
    }
  }
  if (!supportPass) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    LogWarning("there are no devices in context to support images");
    return static_cast<cl_mem>(0);
  }

  return amd::clCreateFromGLTextureAMD(*as_amd(context), flags, target, miplevel, texture,
                                       errcode_ret);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateFromGLTexture3D
 *  @{
 */

/*! \brief Create an OpenCL 3D image object from an OpenGL 3D texture object.
 *
 *  \param clContext is a valid OpenCL clContext created from an OpenGL clContext.
 *
 *  \param clFlags is a bit-field that is used to specify usage information.
 *  Only CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY and CL_MEM_READ_WRITE values
 *  can be used.
 *
 *  \param target must be GL_TEXTURE_3D.
 *
 *  \param miplevel is the mipmap level to be used.
 *
 *  \param texture is a GL 3D texture object [name].
 *  The texture object must be a complete texture as per OpenGL rules on texture
 *  completeness. The \a texture format and dimensions specified using appropriate
 *  glTexImage3D call for \a miplevel will be used to create the 3D image object.
 *  Only GL texture formats that map to appropriate image channel order and
 *  data type can be used to create the 3D image object.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero OpenCL image object and \a errcode_ret is set to
 *  CL_SUCCESS if the image object is created successfully. It returns a NULL value
 *  with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a clContext is not a valid clContext or was not
 *    created from a GL clContext.
 *  - CL_INVALID_VALUE if values specified in \a clFlags are not valid.
 *  - CL_INVALID_MIP_LEVEL if \a miplevel is not a valid mip-level for \a texture.
 *  - CL_INVALID_GL_OBJECT if \a texture is not an GL 3D texture.
 *  - CL_INVALID_IMAGE_FORMAT_DESCRIPTOR if the OpenGL texture format does not
 *    map to an appropriate OpenCL image format.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r29
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateFromGLTexture3D,
                  (cl_context context, cl_mem_flags flags, GLenum target, GLint miplevel,
                   GLuint texture, cl_int* errcode_ret)) {
  cl_mem clMemObj = NULL;

  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return clMemObj;
  }

  if (!(((flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY) ||
        ((flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY) ||
        ((flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE))) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return clMemObj;
  }

  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  bool supportPass = false;
  bool sizePass = false;
  for (const auto& it : devices) {
    if (it->info().imageSupport_) {
      supportPass = true;
    }
  }
  if (!supportPass) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    LogWarning("there are no devices in context to support images");
    return static_cast<cl_mem>(0);
  }

  return amd::clCreateFromGLTextureAMD(*as_amd(context), flags, target, miplevel, texture,
                                       errcode_ret);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateFromGLRenderbuffer
 *  @{
 */

/*! \brief Create an OpenCL 2D image object from an OpenGL renderbuffer object.
 *
 *  \param clContext is a valid OpenCL clContext created from an OpenGL clContext.
 *
 *  \param clFlags is a bit-field that is used to specify usage information.
 *  Only CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY and CL_MEM_READ_WRITE values
 *  can be used.
 *
 *  \param renderbuffer is a GL renderbuffer object name. The renderbuffer
 *  storage must be specified before the image object can be created. Only
 *  GL renderbuffer formats that map to appropriate image channel order and
 *  data type can be used to create the 2D image object.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero OpenCL image object and \a errcode_ret is set
 *  to CL_SUCCESS if the image object is created successfully. It returns a
 *  NULL value with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a clContext is not a valid clContext or was not
 *    created from a GL clContext.
 *  - CL_INVALID_VALUE if values specified in \a clFlags are not valid.
 *  - CL_INVALID_GL_OBJECT if \a renderbuffer is not an GL renderbuffer object.
 *  - CL_INVALID_IMAGE_FORMAT_DESCRIPTOR if the OpenGL renderbuffer format
 *    does not map to an appropriate OpenCL image format.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r29
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateFromGLRenderbuffer, (cl_context context, cl_mem_flags flags,
                                                       GLuint renderbuffer, cl_int* errcode_ret)) {
  cl_mem clMemObj = NULL;

  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return clMemObj;
  }

  if (!(((flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY) ||
        ((flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY) ||
        ((flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE))) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return clMemObj;
  }

  return (amd::clCreateFromGLRenderbufferAMD(*as_amd(context), flags, renderbuffer, errcode_ret));
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clGetGLObjectInfo
 *  @{
 */

/*! \brief Query GL object type from a CL memory object.
 *
 *  \param memobj [is a valid cl_mem object created from a GL object].
 *
 *  \param gl_object_type returns the type of GL object attached to memobj
 *  and can be CL_GL_OBJECT_BUFFER, CL_GL_OBJECT_TEXTURE2D,
 *  CL_GL_OBJECT_TEXTURE_RECTANGLE, CL_GL_OBJECT_TEXTURE3D, or
 *  CL_GL_OBJECT_RENDERBUFFER. If \a gl_object_type is NULL, it is ignored.
 *
 *  \param gl_object_name returns the GL object name used to create memobj.
 *  If \a gl_object_name is NULL, it is ignored.
 *
 *  \return One of the following values is returned:
 *  - CL_SUCCESS if the call was executed successfully.
 *  - CL_INVALID_MEM_OBJECT if \a memobj is not a valid OpenCL memory object.
 *  - CL_INVALID_GL_OBJECT if there is no GL object associated with \a memobj.
 *
 *  \version 1.0r29
 */
RUNTIME_ENTRY(cl_int, clGetGLObjectInfo,
              (cl_mem memobj, cl_gl_object_type* gl_object_type, GLuint* gl_object_name)) {
  if (!is_valid(memobj)) {
    LogWarning("\"memobj\" is not a  valid cl_mem object");
    return CL_INVALID_MEM_OBJECT;
  }

  amd::InteropObject* interop = as_amd(memobj)->getInteropObj();
  if (NULL == interop) {
    LogWarning("CL object \"memobj\" is not created from GL object");
    return CL_INVALID_GL_OBJECT;
  }

  amd::GLObject* glObject = interop->asGLObject();
  if (NULL == glObject) {
    LogWarning("CL object \"memobj\" is not created from GL object");
    return CL_INVALID_GL_OBJECT;
  }

  cl_int result;

  cl_gl_object_type clGLType = glObject->getCLGLObjectType();
  result = amd::clGetInfo(clGLType, sizeof(cl_gl_object_type), gl_object_type, NULL);

  GLuint glName = glObject->getGLName();
  result |= amd::clGetInfo(glName, sizeof(GLuint), gl_object_name, NULL);

  return result;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clGetGLTextureInfo
 *  @{
 */

/*! \brief Query additional information about the GL texture object associated
 *  with \a memobj.
 *
 *  \param memobj [is a valid cl_mem object created from a GL object].
 *
 *  \param param_name specifies what additional information about the GL
 *  texture object associated with \a memobj to query:
 *  - CL_GL_TEXTURE_TARGET (GLenum) to query the \a target argument specified
 *    in clCreateGLTexture2D or clCreateGLTexture3D calls.
 *  - CL_GL_MIPMAP_LEVEL (GLint) to query the \a miplevel argument specified
 *    in clCreateGLTexture2D or clCreateGLTexture3D calls.
 *
 *  \param param_value is a pointer to memory where the appropriate result
 *  being queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type as
 *  described for \a param_name argumnet (GLenum or GLint).
 *  \a param_value_size_ret returns the actual size in bytes of data copied to
 *  \a param_value. If \a param_value_size_ret is NULL, it is ignored
 *
 *  \return One of the following values is returned:
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_MEM_OBJECT if \a memobj is not a valid OpenCL memory object.
 *  - CL_INVALID_GL_OBJECT if there is no GL texture object (2D or 3D texture)
 *    associated with \a memobj.
 *  - CL_INVALID_VALUE if \a param_name is not valid, or if size in bytes
 *    specified by \a param_value_size is < size of return type required by
 *    \a param_name and \a param_value is not NULL, or if \a param_value and
 *    \a param_value_size_ret are NULL.
 *
 *  \version 1.0r29
 */
RUNTIME_ENTRY(cl_int, clGetGLTextureInfo,
              (cl_mem memobj, cl_gl_texture_info param_name, size_t param_value_size,
               void* param_value, size_t* param_value_size_ret)) {
  if (!is_valid(memobj)) {
    LogWarning("\"memobj\" is not a  valid cl_mem object");
    return CL_INVALID_MEM_OBJECT;
  }
  amd::InteropObject* interop = as_amd(memobj)->getInteropObj();
  if (NULL == interop) {
    LogWarning("CL object \"memobj\" is not created from GL object");
    return CL_INVALID_GL_OBJECT;
  }
  amd::GLObject* glObject = interop->asGLObject();
  if ((NULL == glObject) || (NULL != glObject->asBufferGL())) {
    LogWarning("CL object \"memobj\" is not created from GL texture");
    return CL_INVALID_GL_OBJECT;
  }

  switch (param_name) {
    case CL_GL_TEXTURE_TARGET: {
      GLenum glTarget = glObject->getGLTarget();
      if (glTarget == GL_TEXTURE_CUBE_MAP) {
        glTarget = glObject->getCubemapFace();
      }
      return amd::clGetInfo(glTarget, param_value_size, param_value, param_value_size_ret);
    }
    case CL_GL_MIPMAP_LEVEL: {
      GLint mipLevel = glObject->getGLMipLevel();
      return amd::clGetInfo(mipLevel, param_value_size, param_value, param_value_size_ret);
    }
    case CL_GL_NUM_SAMPLES: {
      GLsizei numSamples = glObject->getNumSamples();
      return amd::clGetInfo(numSamples, param_value_size, param_value, param_value_size_ret);
    }
    default:
      LogWarning("Unknown param_name in clGetGLTextureInfoAMD");
      break;
  }

  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clEnqueueAcquireExtObjects
 *  @{
 */

/*! \brief Acquire OpenCL memory objects that have been created from external
 *  objects (OpenGL, D3D).
 *
 *  \param command_queue is a valid command-queue.
 *
 *  \param num_objects is the number of memory objects to be acquired
 *  in \a mem_objects.
 *
 *  \param mem_objects is a pointer to a list of CL memory objects that refer
 *  to a GL object (buffer/texture/renderbuffer objects or the framebuffer).
 *
 *  \param event_wait_list specify [is a pointer to] events that need to
 *  complete before this particular command can be executed.
 *  If \a event_wait_list is NULL, then this particular command does not wait
 *  on any event to complete. If \a event_wait_list is NULL,
 *  \a num_events_in_wait_list must be 0. If \a event_wait_list is not NULL,
 *  the list of events pointed to by \a event_wait_list must be valid and
 *  \a num_events_in_wait_list must be greater than 0. The events specified in
 *  \a event_wait_list act as synchronization points.
 *
 *  \param num_events_in_wait_list specify the number of events in
 *  \a event_wait_list. It must be 0 if \a event_wait_list is NULL. It  must be
 *  greater than 0 if \a event_wait_list is not NULL.
 *
 *  \param event returns an event object that identifies this particular
 *  command and can be used to query or queue a wait for this particular
 *  command to complete. \a event can be NULL in which case it will not be
 *  possible for the application to query the status of this command or queue a
 *  wait for this command to complete.
 *
 *  \return One of the following values is returned:
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_SUCCESS if \a num_objects is 0 and \a mem_objects is NULL; the
 *    function does nothing.
 *  - CL_INVALID_VALUE if \a num_objects is zero and \a mem_objects is not a
 *    NULL value or if \a num_objects > 0 and \a mem_objects is NULL.
 *  - CL_INVALID_MEM_OBJECT if memory objects in \a mem_objects are not valid
 *    OpenCL memory objects.
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if clContext associated with \a command_queue was not
 *    created from an OpenGL clContext.
 *  - CL_INVALID_GL_OBJECT if memory objects in \a mem_objects have not been
 *    created from a GL object(s).
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *    required by the OpenCL implementation on the host.
 *
 *  \version 1.0r29
 */
RUNTIME_ENTRY(cl_int, clEnqueueAcquireGLObjects,
              (cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clEnqueueAcquireExtObjectsAMD(command_queue, num_objects, mem_objects,
                                            num_events_in_wait_list, event_wait_list, event,
                                            CL_COMMAND_ACQUIRE_GL_OBJECTS);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clEnqueueReleaseGLObjects
 *  @{
 */

/*! \brief Release OpenCL memory objects that have been created from OpenGL
 *  objects.
 *
 *  \param command_queue is a valid command-queue [which is associated with the
 *  OpenCL clContext releasing the OpenGL objects].
 *
 *  \param num_objects is the number of memory objects to be released
 *  in \a mem_objects.
 *
 *  \param mem_objects is a pointer to a list of CL memory objects that refer
 *  to a GL object (buffer/texture/renderbuffer objects or the framebuffer).
 *
 *  \param event_wait_list specify [is a pointer to] events that need to
 *  complete before this particular command can be executed.
 *  If \a event_wait_list is NULL, then this particular command does not wait
 *  on any event to complete. If \a event_wait_list is NULL,
 *  \a num_events_in_wait_list must be 0. If \a event_wait_list is not NULL,
 *  the list of events pointed to by \a event_wait_list must be valid and
 *  \a num_events_in_wait_list must be greater than 0. The events specified in
 *  \a event_wait_list act as synchronization points.
 *
 *  \param num_events_in_wait_list specify the number of events in
 *  \a event_wait_list. It must be 0 if \a event_wait_list is NULL. It  must be
 *  greater than 0 if \a event_wait_list is not NULL.
 *
 *  \param event returns an event object that identifies this particular
 *  command and can be used to query or queue a wait for this particular
 *  command to complete. \a event can be NULL in which case it will not be
 *  possible for the application to query the status of this command or queue a
 *  wait for this command to complete.
 *
 *  \return One of the following values is returned:
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_SUCCESS if \a num_objects is 0 and \a mem_objects is NULL; the
 *    function does nothing.
 *  - CL_INVALID_VALUE if \a num_objects is zero and \a mem_objects is not a
 *    NULL value or if \a num_objects > 0 and \a mem_objects is NULL.
 *  - CL_INVALID_MEM_OBJECT if memory objects in \a mem_objects are not valid
 *    OpenCL memory objects.
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if clContext associated with \a command_queue was not
 *    created from an OpenGL clContext.
 *  - CL_INVALID_GL_OBJECT if memory objects in \a mem_objects have not been
 *    created from a GL object(s).
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *    required by the OpenCL implementation on the host.
 *
 *  \version 1.0r29
 */
RUNTIME_ENTRY(cl_int, clEnqueueReleaseGLObjects,
              (cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clEnqueueReleaseExtObjectsAMD(command_queue, num_objects, mem_objects,
                                            num_events_in_wait_list, event_wait_list, event,
                                            CL_COMMAND_RELEASE_GL_OBJECTS);
}
RUNTIME_EXIT

/*! @}
*  \addtogroup clCreateEventFromGLsyncKHR
*  @{
*/

/*! \brief Creates an event object linked to an OpenGL sync object.
*  Completion of such an event object is equivalent to waiting for completion
*  of the fence command associated with the linked GL sync object.
*
*  \param context is valid OpenCL context created from an OpenGL context
*  or share group, using the cl_khr_gl_sharing extension.
*
*  \param sync is the 'name' of a sync object in the GL share group associated
*  with context.
*
*  \param errcode_ret Returns an appropriate error code as described below.
*  If errcode_ret is NULL, no error code is returned.
*
*  \return a valid OpenCL event object and errcode_ret is set to CL_SUCCESS
*  if the event object is created successfully.Otherwise, it returns a NULL
*  value with one of the following error values returned in errcode_ret:
*  - CL_INVALID_CONTEXT if context is not a valid context or was not created
*    from a GL context.
*  - CL_INVALID_GL_OBJECT if sync is not the name of a sync object in the
*    GL share group associated with context.
*
*  \version 1.1
*/

RUNTIME_ENTRY_RET(cl_event, clCreateEventFromGLsyncKHR,
                  (cl_context context, cl_GLsync clGLsync, cl_int* errcode_ret)) {
  // create event of fence sync type
  amd::ClGlEvent* clglEvent = new amd::ClGlEvent(*as_amd(context));
  clglEvent->context().glenv()->glFlush_();
  // initially set the status of fence as queued
  clglEvent->setStatus(CL_SUBMITTED);
  // store GLsync id of the fence in event in order to associate them together
  clglEvent->setData(clGLsync);
  amd::Event* evt = dynamic_cast<amd::Event*>(clglEvent);
  evt->retain();
  return as_cl(evt);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clGetGLContextInfoKHR
 *  @{
 */

/*! \brief This f-n is defined in CL extension cl_khr_gl_sharing and serves
 *  the purpose of quering current device and all devices that support
 *  CL-GL interoperability.
 *
 *  \param properties points to an <attribute list>, which is a array of
 *  ordered <attribute name, value> pairs terminated with zero. If an
 *  attribute is not specified in <properties>, then its default value
 *  (listed in table 4.attr) is used (it is said to be specified
 *  implicitly). If <properties> is NULL or empty (points to a list
 *  whose first value is zero), all attributes take on their default
 *  values.
 *
 *  \param param_name may accept one of the following enumerated values:
 *  - CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR  0x2006
 *  - CL_DEVICES_FOR_GL_CONTEXT_KHR         0x2007.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type as
 *  described for \a param_name argumnet (GLenum or GLint).
 *  \a param_value_size_ret returns the actual size in bytes of data copied to
 *  \a param_value. If \a param_value_size_ret is NULL, it is ignored
 *
 *  \param param_value is a pointer to memory where the appropriate result
 *  being queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type as
 *  described for \a param_name argumnet (GLenum or GLint).
 *  \a param_value_size_ret returns the actual size in bytes of data copied to
 *  \a param_value. If \a param_value_size_ret is NULL, it is ignored
 *
 *  \return one of the following values is returned:
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_SUCCESS if \a num_objects is 0 and \a mem_objects is NULL; the
 *    function does nothing.
 *  - CL_INVALID_VALUE if \a num_objects is zero and \a mem_objects is not a
 *    NULL value or if \a num_objects > 0 and \a mem_objects is NULL.
 *  - CL_INVALID_MEM_OBJECT if memory objects in \a mem_objects are not valid
 *    OpenCL memory objects.
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if clContext associated with \a command_queue was not
 *    created from an OpenGL clContext.
 *  - CL_INVALID_GL_OBJECT if memory objects in \a mem_objects have not been
 *    created from a GL object(s).
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *    required by the OpenCL implementation on the host.
 *  - CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR if
 *
 *  \version 1.0r47
 */
RUNTIME_ENTRY(cl_int, clGetGLContextInfoKHR,
              (const cl_context_properties* properties, cl_gl_context_info param_name,
               size_t param_value_size, void* param_value, size_t* param_value_size_ret)) {
  cl_int errcode=0;
  cl_device_id* gpu_devices;
  cl_uint num_gpu_devices = 0;
  amd::Context::Info info;
  static const bool VALIDATE_ONLY = true;

  errcode = amd::Context::checkProperties(properties, &info);
  if (CL_SUCCESS != errcode) {
    return errcode;
  }

  if (!(info.flags_ & amd::Context::GLDeviceKhr)) {
    // No GL context is specified
    return CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR;
  }

  // Get devices
  //errcode = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 0, NULL, &num_gpu_devices);
  if (errcode != CL_SUCCESS && errcode != CL_DEVICE_NOT_FOUND) {
    return CL_INVALID_VALUE;
  }

  if (!num_gpu_devices) {
    return CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR;
  }

  switch (param_name) {
    case CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR:
      // Return the CL device currently associated with the specified OpenGL context.
      if (num_gpu_devices) {
        gpu_devices = (cl_device_id*)alloca(num_gpu_devices * sizeof(cl_device_id));

        //errcode = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, num_gpu_devices, gpu_devices, NULL);
        if (errcode != CL_SUCCESS) {
          return errcode;
        }

        for (cl_uint i = 0; i < num_gpu_devices; ++i) {
          cl_device_id device = gpu_devices[i];
          if (is_valid(device) &&
              as_amd(device)->bindExternalDevice(info.flags_, info.hDev_, info.hCtx_,
                                                 VALIDATE_ONLY)) {
            return amd::clGetInfo(device, param_value_size, param_value, param_value_size_ret);
          }
        }

        *not_null(param_value_size_ret) = 0;
      }
      break;

    case CL_DEVICES_FOR_GL_CONTEXT_KHR: {
      // List of all CL devices that can be associated with the specified OpenGL context.
      cl_uint total_devices = num_gpu_devices;
      size_t size = total_devices * sizeof(cl_device_id);

      cl_device_id* devices = (cl_device_id*)alloca(size);

      //errcode = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, total_devices, devices, NULL);
      if (errcode != CL_SUCCESS) {
        return errcode;
      }

      std::vector<amd::Device*> compatible_devices;

      for (cl_uint i = 0; i < total_devices; ++i) {
        cl_device_id device = devices[i];
        if (is_valid(device) &&
            as_amd(device)->bindExternalDevice(info.flags_, info.hDev_, info.hCtx_,
                                               VALIDATE_ONLY)) {
          compatible_devices.push_back(as_amd(device));
        }
      }

      size_t deviceCount = compatible_devices.size();
      size_t deviceCountSize = deviceCount * sizeof(cl_device_id);

      if (param_value != NULL && param_value_size < deviceCountSize) {
        return CL_INVALID_VALUE;
      }

      *not_null(param_value_size_ret) = deviceCountSize;

      if (param_value != NULL) {
        cl_device_id* deviceList = (cl_device_id*)param_value;
        for (const auto& it : compatible_devices) {
          *deviceList++ = as_cl(it);
        }
      }

      return CL_SUCCESS;
    } break;

    default:
      LogWarning("\"param_name\" is not valid");
      return CL_INVALID_VALUE;
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

//
//
//          namespace amd
//
//
namespace amd {

typedef struct {
  GLenum glBinding;
  GLenum glTarget;
} TargetBindings_t;

/*! @}
 *  \addtogroup CL-GL interop helper functions
 *  @{
 */

//! Function clearGLErrors() to clear all GL error bits, if any
void clearGLErrors(const Context& amdContext) {
  GLenum glErr, glLastErr = GL_NO_ERROR;
  while (1) {
    glErr = amdContext.glenv()->glGetError_();
    if (glErr == GL_NO_ERROR || glErr == glLastErr) {
      break;
    }
    glLastErr = glErr;
    LogWarning("GL error");
  }
}

GLenum checkForGLError(const Context& amdContext) {
  GLenum glRetErr = GL_NO_ERROR;
  GLenum glErr;
  while (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
    glRetErr = glErr;  // Just return the last GL error
    LogWarning("Check GL error");
  }
  return glRetErr;
}

//! Function getCLFormatFromGL returns "true" if GL format
//! is compatible with CL format, "false" otherwise.
bool getCLFormatFromGL(const Context& amdContext, GLint gliInternalFormat,
                       cl_image_format* pclImageFormat, int* piBytesPerPixel, cl_mem_flags flags) {
  bool bRetVal = false;

  /*
  Available values for "image_channel_order"
  ==========================================
  CL_R
  CL_A
  CL_INTENSITY
  CL_LUMINANCE
  CL_RG
  CL_RA
  CL_RGB
  CL_RGBA
  CL_ARGB
  CL_BGRA

  Available values for "image_channel_data_type"
  ==============================================
  CL_SNORM_INT8
  CL_SNORM_INT16
  CL_UNORM_INT8
  CL_UNORM_INT16
  CL_UNORM_SHORT_565
  CL_UNORM_SHORT_555
  CL_UNORM_INT_101010
  CL_SIGNED_INT8
  CL_SIGNED_INT16
  CL_SIGNED_INT32
  CL_UNSIGNED_INT8
  CL_UNSIGNED_INT16
  CL_UNSIGNED_INT32
  CL_HALF_FLOAT
  CL_FLOAT
  */

  switch (gliInternalFormat) {
    case GL_RGB10_EXT:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_UNORM_INT_101010;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_RGB10_A2:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_UNORM_INT_101010;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_BGR8_ATI:
    case GL_BGRA8_ATI:
      pclImageFormat->image_channel_order = CL_BGRA;
      pclImageFormat->image_channel_data_type = CL_UNORM_INT8;  // CL_UNSIGNED_INT8;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_ALPHA8:
      pclImageFormat->image_channel_order = CL_A;
      pclImageFormat->image_channel_data_type = CL_UNORM_INT8;  // CL_UNSIGNED_INT8;
      *piBytesPerPixel = 1;
      bRetVal = true;
      break;

    case GL_R8:
    case GL_R8UI:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_R8) ? CL_UNORM_INT8 : CL_UNSIGNED_INT8;
      *piBytesPerPixel = 1;
      bRetVal = true;
      break;

    case GL_R8I:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT8;
      *piBytesPerPixel = 1;
      bRetVal = true;
      break;

    case GL_RG8:
    case GL_RG8UI:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_RG8) ? CL_UNORM_INT8 : CL_UNSIGNED_INT8;
      *piBytesPerPixel = 2;
      bRetVal = true;
      break;

    case GL_RG8I:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT8;
      *piBytesPerPixel = 2;
      bRetVal = true;
      break;

    case GL_RGB8:
    case GL_RGB8UI:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_RGB8) ? CL_UNORM_INT8 : CL_UNSIGNED_INT8;
      *piBytesPerPixel = 3;
      bRetVal = true;
      break;

    case GL_RGB8I:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT8;
      *piBytesPerPixel = 3;
      bRetVal = true;
      break;

    case GL_RGBA:
    case GL_RGBA8:
    case GL_RGBA8UI:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_RGBA8UI) ? CL_UNSIGNED_INT8 : CL_UNORM_INT8;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_RGBA8I:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT8;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_R16:
    case GL_R16UI:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_R16) ? CL_UNORM_INT16 : CL_UNSIGNED_INT16;
      bRetVal = true;
      *piBytesPerPixel = 2;
      break;

    case GL_R16I:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT16;
      *piBytesPerPixel = 2;
      bRetVal = true;
      break;

    case GL_R16F:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type = CL_HALF_FLOAT;
      *piBytesPerPixel = 2;
      bRetVal = true;
      break;

    case GL_RG16:
    case GL_RG16UI:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_RG16) ? CL_UNORM_INT16 : CL_UNSIGNED_INT16;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_RG16I:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT16;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_RG16F:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type = CL_HALF_FLOAT;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_RGB16:
    case GL_RGB16UI:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_RGB16) ? CL_UNORM_INT16 : CL_UNSIGNED_INT16;
      *piBytesPerPixel = 6;
      bRetVal = true;
      break;

    case GL_RGB16I:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT16;
      *piBytesPerPixel = 6;
      bRetVal = true;
      break;

    case GL_RGB16F:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_HALF_FLOAT;
      *piBytesPerPixel = 6;
      bRetVal = true;
      break;

    case GL_RGBA16:
    case GL_RGBA16UI:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type =
          (gliInternalFormat == GL_RGBA16) ? CL_UNORM_INT16 : CL_UNSIGNED_INT16;
      *piBytesPerPixel = 8;
      bRetVal = true;
      break;

    case GL_RGBA16I:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT16;
      *piBytesPerPixel = 8;
      bRetVal = true;
      break;

    case GL_RGBA16F:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_HALF_FLOAT;
      *piBytesPerPixel = 8;
      bRetVal = true;
      break;

    case GL_R32I:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT32;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_R32UI:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type = CL_UNSIGNED_INT32;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_R32F:
      pclImageFormat->image_channel_order = CL_R;
      pclImageFormat->image_channel_data_type = CL_FLOAT;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;

    case GL_RG32I:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT32;
      *piBytesPerPixel = 8;
      bRetVal = true;
      break;

    case GL_RG32UI:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type = CL_UNSIGNED_INT32;
      *piBytesPerPixel = 8;
      bRetVal = true;
      break;

    case GL_RG32F:
      pclImageFormat->image_channel_order = CL_RG;
      pclImageFormat->image_channel_data_type = CL_FLOAT;
      *piBytesPerPixel = 8;
      bRetVal = true;
      break;

    case GL_RGB32I:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT32;
      *piBytesPerPixel = 12;
      bRetVal = true;
      break;

    case GL_RGB32UI:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_UNSIGNED_INT32;
      *piBytesPerPixel = 12;
      bRetVal = true;
      break;

    case GL_RGB32F:
      pclImageFormat->image_channel_order = CL_RGB;
      pclImageFormat->image_channel_data_type = CL_FLOAT;
      *piBytesPerPixel = 12;
      bRetVal = true;
      break;

    case GL_RGBA32I:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_SIGNED_INT32;
      *piBytesPerPixel = 16;
      bRetVal = true;
      break;

    case GL_RGBA32UI:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_UNSIGNED_INT32;
      *piBytesPerPixel = 16;
      bRetVal = true;
      break;

    case GL_RGBA32F:
      pclImageFormat->image_channel_order = CL_RGBA;
      pclImageFormat->image_channel_data_type = CL_FLOAT;
      *piBytesPerPixel = 16;
      bRetVal = true;
      break;
    case GL_DEPTH_COMPONENT32F:
      pclImageFormat->image_channel_order = CL_DEPTH;
      pclImageFormat->image_channel_data_type = CL_FLOAT;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;
    case GL_DEPTH_COMPONENT16:
      pclImageFormat->image_channel_order = CL_DEPTH;
      pclImageFormat->image_channel_data_type = CL_UNORM_INT16;
      *piBytesPerPixel = 2;
      bRetVal = true;
      break;
    case GL_DEPTH24_STENCIL8:
      pclImageFormat->image_channel_order = CL_DEPTH_STENCIL;
      pclImageFormat->image_channel_data_type = CL_UNORM_INT24;
      *piBytesPerPixel = 4;
      bRetVal = true;
      break;
    case GL_DEPTH32F_STENCIL8:
      pclImageFormat->image_channel_order = CL_DEPTH_STENCIL;
      pclImageFormat->image_channel_data_type = CL_FLOAT;
      *piBytesPerPixel = 5;
      bRetVal = true;
      break;
    default:
      LogWarning("unsupported GL internal format");
      break;
  }
  amd::Image::Format imageFormat(*pclImageFormat);
  if (bRetVal && !imageFormat.isSupported(amdContext, 0, flags)) {
    bRetVal = false;
  }
  return bRetVal;
}

void BufferGL::initDeviceMemory() {
  deviceMemories_ =
      reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(BufferGL));
  memset(deviceMemories_, 0, context_().devices().size() * sizeof(DeviceMemory));
}

static GLenum clChannelDataTypeToGlType(cl_channel_type channel_type) {
  // Pick
  // GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_INT,
  // GL_UNSIGNED_INT, GL_FLOAT, GL_2_BYTES, GL_3_BYTES, GL_4_BYTES
  // or GL_DOUBLE
  switch (channel_type) {
    case CL_SNORM_INT8:
      return GL_BYTE;
    case CL_SNORM_INT16:
      return GL_SHORT;
    case CL_UNORM_INT8:
      return GL_UNSIGNED_BYTE;
    case CL_UNORM_INT16:
      return GL_UNSIGNED_SHORT;
    case CL_SIGNED_INT8:
      return GL_BYTE;
    case CL_SIGNED_INT16:
      return GL_SHORT;
    case CL_SIGNED_INT32:
      return GL_INT;
    case CL_UNSIGNED_INT8:
      return GL_UNSIGNED_BYTE;
    case CL_UNSIGNED_INT16:
      return GL_UNSIGNED_SHORT;
    case CL_UNSIGNED_INT32:
      return GL_UNSIGNED_INT;
    case CL_FLOAT:
      return GL_FLOAT;
    case CL_UNORM_INT_101010:
      return GL_UNSIGNED_INT_10_10_10_2;
    case CL_HALF_FLOAT:
    case CL_UNORM_SHORT_565:
    case CL_UNORM_SHORT_555:
    default:
      guarantee(false, "Unexpected CL type.");
      return 0;
  }
}

static GLenum glInternalFormatToGlFormat(GLenum internalFormat) {
  switch (internalFormat) {
    // Base internal formats
    case GL_RGBA:
    case GL_BGRA:
      return internalFormat;
    // Sized internal formats
    case GL_RGBA8:
    case GL_RGBA16:
    case GL_RGBA16F:
    case GL_RGBA32F:
      return GL_RGBA;
    case GL_RGBA8I:
    case GL_RGBA8UI:
    case GL_RGBA16I:
    case GL_RGBA16UI:
    case GL_RGBA32I:
    case GL_RGBA32UI:
      return GL_RGBA_INTEGER;

    default:
      guarantee(false, "Unexpected GL internal format.");
      return 0;
  }
}

void ImageGL::initDeviceMemory() {
  deviceMemories_ =
      reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(ImageGL));
  memset(deviceMemories_, 0, context_().devices().size() * sizeof(DeviceMemory));
}

//*******************************************************************
//
// Internal implementation of CL API functions
//
//*******************************************************************

//
//      clCreateFromGLBufferAMD
//
cl_mem clCreateFromGLBufferAMD(Context& amdContext, cl_mem_flags flags, GLuint bufobj,
                               cl_int* errcode_ret) {
  BufferGL* pBufferGL = NULL;
  GLenum glErr;
  GLenum glTarget = GL_ARRAY_BUFFER;
  GLint gliSize = 0;
  GLint gliMapped = 0;

  // Verify context init'ed for interop
  if (!amdContext.glenv() || !amdContext.glenv()->isAssociated()) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("\"amdContext\" is not created from GL context or share list");
    return (cl_mem)0;
  }

  // Add this scope to bound the scoped lock
  {
    GLFunctions::SetIntEnv ie(amdContext.glenv());
    if (!ie.isValid()) {
      *not_null(errcode_ret) = CL_INVALID_CONTEXT;
      LogWarning("\"amdContext\" is not created from GL context or share list");
      return as_cl<Memory>(0);
    }

    // Verify GL buffer object
    clearGLErrors(amdContext);
    if ((GL_FALSE == amdContext.glenv()->glIsBuffer_(bufobj)) ||
        (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_()))) {
      *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
      LogWarning("\"bufobj\" is not a GL buffer object");
      return (cl_mem)0;
    }

    // It seems that CL spec is not concerned with GL_BUFFER_USAGE, so skip it

    // Check if size is available - data store is created

    amdContext.glenv()->glBindBuffer_(glTarget, bufobj);
    clearGLErrors(amdContext);
    amdContext.glenv()->glGetBufferParameteriv_(glTarget, GL_BUFFER_SIZE, &gliSize);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
      LogWarning("cannot get the GL buffer size");
      return (cl_mem)0;
    }
    if (gliSize == 0) {
      //@todo - check why sometime the size is zero
      *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
      LogWarning("the GL buffer's data store is not created");
      return (cl_mem)0;
    }

    // Mapping will be done at acquire time (sync point)

  }  // Release scoped lock

  // Now create BufferGL object
  pBufferGL = new (amdContext) BufferGL(amdContext, flags, gliSize, 0, bufobj);

  if (!pBufferGL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    LogWarning("cannot create object of class BufferGL");
    return (cl_mem)0;
  }

  if (!pBufferGL->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pBufferGL->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;

  // Create interop object
  if (pBufferGL->getInteropObj() == NULL) {
    *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
    LogWarning("cannot create object of class BufferGL");
    return (cl_mem)0;
  }

  // Fixme: If more than one device is present in the context, we choose the first device.
  // We should come up with a more elegant solution to handle this.
  assert(amdContext.devices().size() == 1);

  const auto it = amdContext.devices().cbegin();
  const amd::Device& dev = *(*it);

  device::Memory* mem = pBufferGL->getDeviceMemory(dev);
  if (NULL == mem) {
    LogPrintfError("Can't allocate memory size - 0x%08X bytes!", pBufferGL->getSize());
    *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
    return (cl_mem)0;
  }
  mem->processGLResource(device::Memory::GLDecompressResource);

  return as_cl<Memory>(pBufferGL);
}

cl_mem clCreateFromGLTextureAMD(Context& amdContext, cl_mem_flags clFlags, GLenum target,
                                GLint miplevel, GLuint texture, int* errcode_ret) {
  ImageGL* pImageGL = NULL;
  GLenum glErr;
  GLenum glTarget = 0;
  GLenum glInternalFormat;
  cl_image_format clImageFormat;
  uint dim = 1;
  cl_mem_object_type clType;
  cl_gl_object_type clGLType;
  GLsizei numSamples = 1;

  // Verify context init'ed for interop
  if (!amdContext.glenv() || !amdContext.glenv()->isAssociated()) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("\"amdContext\" is not created from GL context or share list");
    return static_cast<cl_mem>(0);
  }

  GLint gliTexWidth = 1;
  GLint gliTexHeight = 1;
  GLint gliTexDepth = 1;

  // Add this scope to bound the scoped lock
  {
    GLFunctions::SetIntEnv ie(amdContext.glenv());
    if (!ie.isValid()) {
      *not_null(errcode_ret) = CL_INVALID_CONTEXT;
      LogWarning("\"amdContext\" is not created from GL context or share list");
      return as_cl<Memory>(0);
    }

    // Verify GL texture object
    clearGLErrors(amdContext);
    if ((GL_FALSE == amdContext.glenv()->glIsTexture_(texture)) ||
        (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_()))) {
      *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
      LogWarning("\"texture\" is not a GL texture object");
      return static_cast<cl_mem>(0);
    }

    bool image = true;

    // Check target value validity
    switch (target) {
      case GL_TEXTURE_BUFFER:
        glTarget = GL_TEXTURE_BUFFER;
        dim = 1;
        clType = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        clGLType = CL_GL_OBJECT_TEXTURE_BUFFER;
        image = false;
        break;

      case GL_TEXTURE_1D:
        glTarget = GL_TEXTURE_1D;
        dim = 1;
        clType = CL_MEM_OBJECT_IMAGE1D;
        clGLType = CL_GL_OBJECT_TEXTURE1D;
        break;

      case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
      case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
      case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
      case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
      case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
      case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
        glTarget = GL_TEXTURE_CUBE_MAP;
        dim = 2;
        clType = CL_MEM_OBJECT_IMAGE2D;
        clGLType = CL_GL_OBJECT_TEXTURE2D;
        break;

      case GL_TEXTURE_1D_ARRAY:
        glTarget = GL_TEXTURE_1D_ARRAY;
        dim = 2;
        clType = CL_MEM_OBJECT_IMAGE1D_ARRAY;
        clGLType = CL_GL_OBJECT_TEXTURE1D_ARRAY;
        break;

      case GL_TEXTURE_2D:
        glTarget = GL_TEXTURE_2D;
        dim = 2;
        clType = CL_MEM_OBJECT_IMAGE2D;
        clGLType = CL_GL_OBJECT_TEXTURE2D;
        break;

      case GL_TEXTURE_2D_MULTISAMPLE:
        glTarget = GL_TEXTURE_2D_MULTISAMPLE;
        dim = 2;
        clType = CL_MEM_OBJECT_IMAGE2D;
        clGLType = CL_GL_OBJECT_TEXTURE2D;
        break;

      case GL_TEXTURE_RECTANGLE_ARB:
        glTarget = GL_TEXTURE_RECTANGLE_ARB;
        dim = 2;
        clType = CL_MEM_OBJECT_IMAGE2D;
        clGLType = CL_GL_OBJECT_TEXTURE2D;
        break;

      case GL_TEXTURE_2D_ARRAY:
        glTarget = GL_TEXTURE_2D_ARRAY;
        dim = 3;
        clType = CL_MEM_OBJECT_IMAGE2D_ARRAY;
        clGLType = CL_GL_OBJECT_TEXTURE2D_ARRAY;
        break;

      case GL_TEXTURE_3D:
        glTarget = GL_TEXTURE_3D;
        dim = 3;
        clType = CL_MEM_OBJECT_IMAGE3D;
        clGLType = CL_GL_OBJECT_TEXTURE3D;
        break;

      default:
        // wrong value
        *not_null(errcode_ret) = CL_INVALID_VALUE;
        LogWarning("invalid \"target\" value");
        return static_cast<cl_mem>(0);
        break;
    }

    amdContext.glenv()->glBindTexture_(glTarget, texture);

    // Check if size is available - data store is created
    if (image) {
      // Check mipmap level for "texture" name
      GLint gliTexBaseLevel;
      GLint gliTexMaxLevel;

      clearGLErrors(amdContext);
      amdContext.glenv()->glGetTexParameteriv_(glTarget, GL_TEXTURE_BASE_LEVEL, &gliTexBaseLevel);
      if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
        *not_null(errcode_ret) = CL_INVALID_MIP_LEVEL;
        LogWarning("Cannot get base mipmap level of a GL \"texture\" object");
        return static_cast<cl_mem>(0);
      }
      clearGLErrors(amdContext);
      amdContext.glenv()->glGetTexParameteriv_(glTarget, GL_TEXTURE_MAX_LEVEL, &gliTexMaxLevel);
      if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
        *not_null(errcode_ret) = CL_INVALID_MIP_LEVEL;
        LogWarning("Cannot get max mipmap level of a GL \"texture\" object");
        return static_cast<cl_mem>(0);
      }
      if ((gliTexBaseLevel > miplevel) || (miplevel > gliTexMaxLevel)) {
        *not_null(errcode_ret) = CL_INVALID_MIP_LEVEL;
        LogWarning("\"miplevel\" is not a valid mipmap level of the GL \"texture\" object");
        return static_cast<cl_mem>(0);
      }

      // Get GL texture format and check if it's compatible with CL format
      clearGLErrors(amdContext);
      amdContext.glenv()->glGetTexLevelParameteriv_(target, miplevel, GL_TEXTURE_INTERNAL_FORMAT,
                                                    (GLint*)&glInternalFormat);
      if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
        *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        LogWarning("Cannot get internal format of \"miplevel\" of GL \"texture\" object");
        return static_cast<cl_mem>(0);
      }

      amdContext.glenv()->glGetTexLevelParameteriv_(target, miplevel, GL_TEXTURE_SAMPLES,
                                                    (GLint*)&numSamples);
      if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
        *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        LogWarning("Cannot get  numbers of samples of GL \"texture\" object");
        return static_cast<cl_mem>(0);
      }
      if (numSamples > 1) {
        *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        LogWarning("MSAA \"texture\" object is not suppoerted for the device");
        return static_cast<cl_mem>(0);
      }

      // Now get CL format from GL format and bytes per pixel
      int iBytesPerPixel = 0;
      if (!getCLFormatFromGL(amdContext, glInternalFormat, &clImageFormat, &iBytesPerPixel,
                             clFlags)) {
        *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        LogWarning("\"texture\" format does not map to an appropriate CL image format");
        return static_cast<cl_mem>(0);
      }

      switch (dim) {
        case 3:
          clearGLErrors(amdContext);
          amdContext.glenv()->glGetTexLevelParameteriv_(target, miplevel, GL_TEXTURE_DEPTH,
                                                        &gliTexDepth);
          if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
            *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
            LogWarning("Cannot get the depth of \"miplevel\" of GL \"texure\"");
            return static_cast<cl_mem>(0);
          }
        // Fall trough to process other dimensions...
        case 2:
          clearGLErrors(amdContext);
          amdContext.glenv()->glGetTexLevelParameteriv_(target, miplevel, GL_TEXTURE_HEIGHT,
                                                        &gliTexHeight);
          if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
            *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
            LogWarning("Cannot get the height of \"miplevel\" of GL \"texure\"");
            return static_cast<cl_mem>(0);
          }
        // Fall trough to process other dimensions...
        case 1:
          clearGLErrors(amdContext);
          amdContext.glenv()->glGetTexLevelParameteriv_(target, miplevel, GL_TEXTURE_WIDTH,
                                                        &gliTexWidth);
          if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
            *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
            LogWarning("Cannot get the width of \"miplevel\" of GL \"texure\"");
            return static_cast<cl_mem>(0);
          }
          break;
        default:
          *not_null(errcode_ret) = CL_INVALID_VALUE;
          LogWarning("invalid \"target\" value");
          return static_cast<cl_mem>(0);
      }
    } else {
      GLint size;

      // In case target is GL_TEXTURE_BUFFER
      GLint backingBuffer;
      clearGLErrors(amdContext);
      amdContext.glenv()->glGetTexLevelParameteriv_(
          glTarget, 0, GL_TEXTURE_BUFFER_DATA_STORE_BINDING, &backingBuffer);
      if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
        *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        LogWarning("Cannot get backing buffer for GL \"texture buffer\" object");
        return static_cast<cl_mem>(0);
      }
      amdContext.glenv()->glBindBuffer_(glTarget, backingBuffer);

      // Get GL texture format and check if it's compatible with CL format
      clearGLErrors(amdContext);
      amdContext.glenv()->glGetIntegerv_(GL_TEXTURE_BUFFER_FORMAT_EXT,
                                         reinterpret_cast<GLint*>(&glInternalFormat));
      if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
        *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        LogWarning("Cannot get internal format of \"miplevel\" of GL \"texture\" object");
        return static_cast<cl_mem>(0);
      }

      // Now get CL format from GL format and bytes per pixel
      int iBytesPerPixel = 0;
      if (!getCLFormatFromGL(amdContext, glInternalFormat, &clImageFormat, &iBytesPerPixel,
                             clFlags)) {
        *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        LogWarning("\"texture\" format does not map to an appropriate CL image format");
        return static_cast<cl_mem>(0);
      }

      clearGLErrors(amdContext);
      amdContext.glenv()->glGetBufferParameteriv_(glTarget, GL_BUFFER_SIZE, &size);
      if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
        *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        LogWarning("Cannot get internal format of \"miplevel\" of GL \"texture\" object");
        return static_cast<cl_mem>(0);
      }

      gliTexWidth = size / iBytesPerPixel;
    }
    size_t imageSize = (clType == CL_MEM_OBJECT_IMAGE1D_ARRAY) ? static_cast<size_t>(gliTexHeight)
                                                               : static_cast<size_t>(gliTexDepth);

    if (!amd::Image::validateDimensions(
            amdContext.devices(), clType, static_cast<size_t>(gliTexWidth),
            static_cast<size_t>(gliTexHeight), static_cast<size_t>(gliTexDepth), imageSize)) {
      *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
      LogWarning("The GL \"texture\" data store is not created or out of supported dimensions");
      return static_cast<cl_mem>(0);
    }

    // PBO and mapping will be done at "acquire" time (sync point)

  }  // Release scoped lock

  target = (glTarget == GL_TEXTURE_CUBE_MAP) ? target : 0;

  pImageGL = new (amdContext)
      ImageGL(amdContext, clType, clFlags, clImageFormat, static_cast<size_t>(gliTexWidth),
              static_cast<size_t>(gliTexHeight), static_cast<size_t>(gliTexDepth), glTarget,
              texture, miplevel, glInternalFormat, clGLType, numSamples, target);

  if (!pImageGL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    LogWarning("Cannot create class ImageGL - out of memory?");
    return static_cast<cl_mem>(0);
  }

  if (!pImageGL->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pImageGL->release();
    return static_cast<cl_mem>(0);
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<Memory>(pImageGL);
}

//
//      clCreateFromGLRenderbufferDAMD
//
cl_mem clCreateFromGLRenderbufferAMD(Context& amdContext, cl_mem_flags clFlags, GLuint renderbuffer,
                                     int* errcode_ret) {
  ImageGL* pImageGL = NULL;
  GLenum glErr;

  GLenum glTarget = GL_RENDERBUFFER;
  GLenum glInternalFormat;
  cl_image_format clImageFormat;

  // Verify context init'ed for interop
  if (!amdContext.glenv() || !amdContext.glenv()->isAssociated()) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("\"amdContext\" is not created from GL context or share list");
    return (cl_mem)0;
  }

  GLint gliRbWidth;
  GLint gliRbHeight;

  // Add this scope to bound the scoped lock
  {
    GLFunctions::SetIntEnv ie(amdContext.glenv());
    if (!ie.isValid()) {
      *not_null(errcode_ret) = CL_INVALID_CONTEXT;
      LogWarning("\"amdContext\" is not created from GL context or share list");
      return as_cl<Memory>(0);
    }

    // Verify GL renderbuffer object
    clearGLErrors(amdContext);
    if ((GL_FALSE == amdContext.glenv()->glIsRenderbufferEXT_(renderbuffer)) ||
        (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_()))) {
      *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
      LogWarning("\"renderbuffer\" is not a GL texture object");
      return (cl_mem)0;
    }

    amdContext.glenv()->glBindRenderbuffer_(glTarget, renderbuffer);

    // Get GL RB format and check if it's compatible with CL format
    clearGLErrors(amdContext);
    amdContext.glenv()->glGetRenderbufferParameterivEXT_(glTarget, GL_RENDERBUFFER_INTERNAL_FORMAT,
                                                         (GLint*)&glInternalFormat);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
      LogWarning("Cannot get internal format of GL \"renderbuffer\" object");
      return (cl_mem)0;
    }

    // Now get CL format from GL format and bytes per pixel
    int iBytesPerPixel = 0;
    if (!getCLFormatFromGL(amdContext, glInternalFormat, &clImageFormat, &iBytesPerPixel,
                           clFlags)) {
      *not_null(errcode_ret) = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
      LogWarning("\"renderbuffer\" format does not map to an appropriate CL image format");
      return (cl_mem)0;
    }

    // Check if size is available - data store is created
    clearGLErrors(amdContext);
    amdContext.glenv()->glGetRenderbufferParameterivEXT_(glTarget, GL_RENDERBUFFER_WIDTH,
                                                         &gliRbWidth);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
      LogWarning("Cannot get the width of GL \"renderbuffer\"");
      return (cl_mem)0;
    }
    if (gliRbWidth == 0) {
      *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
      LogWarning("The GL \"renderbuffer\" data store is not created");
      return (cl_mem)0;
    }
    clearGLErrors(amdContext);
    amdContext.glenv()->glGetRenderbufferParameterivEXT_(glTarget, GL_RENDERBUFFER_HEIGHT,
                                                         &gliRbHeight);
    if (GL_NO_ERROR != (glErr = amdContext.glenv()->glGetError_())) {
      *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
      LogWarning("Cannot get the height of GL \"renderbuffer\"");
      return (cl_mem)0;
    }
    if (gliRbHeight == 0) {
      *not_null(errcode_ret) = CL_INVALID_GL_OBJECT;
      LogWarning("The GL \"renderbuffer\" data store is not created");
      return (cl_mem)0;
    }

    // PBO and mapping will be done at "acquire" time (sync point)

  }  // Release scoped lock

  pImageGL =
      new (amdContext) ImageGL(amdContext, CL_MEM_OBJECT_IMAGE2D, clFlags, clImageFormat,
                               (size_t)gliRbWidth, (size_t)gliRbHeight, 1, glTarget, renderbuffer,
                               0, glInternalFormat, CL_GL_OBJECT_RENDERBUFFER, 0);

  if (!pImageGL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    LogWarning("Cannot create class ImageGL from renderbuffer - out of memory?");
    return (cl_mem)0;
  }

  if (!pImageGL->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pImageGL->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<Memory>(pImageGL);
}

//
//      clEnqueueAcquireExtObjectsAMD
//

static cl_int clSetInteropObjects(cl_uint num_objects, const cl_mem* mem_objects,
                                  std::vector<amd::Memory*>& interopObjects) {
  if ((num_objects == 0 && mem_objects != NULL) || (num_objects != 0 && mem_objects == NULL)) {
    return CL_INVALID_VALUE;
  }

  while (num_objects-- > 0) {
    cl_mem obj = *mem_objects++;
    if (!is_valid(obj)) {
      return CL_INVALID_MEM_OBJECT;
    }

    amd::Memory* mem = as_amd(obj);
    if (mem->getInteropObj() == NULL) {
      return CL_INVALID_GL_OBJECT;
    }

    interopObjects.push_back(mem);
  }
  return CL_SUCCESS;
}

cl_int clEnqueueAcquireExtObjectsAMD(cl_command_queue command_queue, cl_uint num_objects,
                                     const cl_mem* mem_objects, cl_uint num_events_in_wait_list,
                                     const cl_event* event_wait_list, cl_event* event,
                                     cl_command_type cmd_type) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (cmd_type == CL_COMMAND_ACQUIRE_GL_OBJECTS) {
    // Verify context init'ed for interop
    if (!hostQueue.context().glenv() || !hostQueue.context().glenv()->isAssociated()) {
      LogWarning("\"amdContext\" is not created from GL context or share list");
      return CL_INVALID_CONTEXT;
    }
  }

  std::vector<amd::Memory*> memObjects;
  cl_int err = clSetInteropObjects(num_objects, mem_objects, memObjects);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::Command::EventWaitList eventWaitList;
  err = amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list,
                                event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

#ifdef _WIN32
  if ((hostQueue.context().info().flags_ & amd::Context::InteropUserSync) == 0) {
    //! Make sure D3D10 queues are flushed and all commands are finished
    //! before CL side would access interop objects
    if (cmd_type == CL_COMMAND_ACQUIRE_D3D10_OBJECTS_KHR) {
      SyncD3D10Objects(memObjects);
    }
    //! Make sure D3D11 queues are flushed and all commands are finished
    //! before CL side would access interop objects
    if (cmd_type == CL_COMMAND_ACQUIRE_D3D11_OBJECTS_KHR) {
      SyncD3D11Objects(memObjects);
    }
    //! Make sure D3D9 queues are flushed and all commands are finished
    //! before CL side would access interop objects
    if (cmd_type == CL_COMMAND_ACQUIRE_DX9_MEDIA_SURFACES_KHR) {
      SyncD3D9Objects(memObjects);
    }
  }
#endif  //_WIN32

  //! Now create command and enqueue
  amd::AcquireExtObjectsCommand* command = new amd::AcquireExtObjectsCommand(
      hostQueue, eventWaitList, num_objects, memObjects, cmd_type);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}


//
//      clEnqueueReleaseExtObjectsAMD
//
cl_int clEnqueueReleaseExtObjectsAMD(cl_command_queue command_queue, cl_uint num_objects,
                                     const cl_mem* mem_objects, cl_uint num_events_in_wait_list,
                                     const cl_event* event_wait_list, cl_event* event,
                                     cl_command_type cmd_type) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  std::vector<amd::Memory*> memObjects;
  cl_int err = clSetInteropObjects(num_objects, mem_objects, memObjects);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::Command::EventWaitList eventWaitList;
  err = amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list,
                                event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  //! Now create command and enqueue
  amd::ReleaseExtObjectsCommand* command = new amd::ReleaseExtObjectsCommand(
      hostQueue, eventWaitList, num_objects, memObjects, cmd_type);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();

#ifdef _WIN32
  if ((hostQueue.context().info().flags_ & amd::Context::InteropUserSync) == 0) {
    //! Make sure CL command queue is flushed and all commands are finished
    //! before D3D10 side would access interop resources
    if (cmd_type == CL_COMMAND_RELEASE_DX9_MEDIA_SURFACES_KHR ||
        cmd_type == CL_COMMAND_RELEASE_D3D10_OBJECTS_KHR ||
        cmd_type == CL_COMMAND_RELEASE_D3D11_OBJECTS_KHR) {
      command->awaitCompletion();
    }
  }
#endif  //_WIN32

  *not_null(event) = as_cl(&command->event());

  if (event == NULL) {
    command->release();
  }

  return CL_SUCCESS;
}

// Placed here as opposed to command.cpp, as glext.h and cl_gl_amd.hpp will have
// to be included because of the GL calls
bool ClGlEvent::waitForFence() {
  GLenum ret;
  // get fence id associated with fence event
  GLsync gs = reinterpret_cast<GLsync>(command().data());
  if (!gs) return false;

// Try to use DC and GLRC of current thread, if it doesn't exist
// create a new GL context on this thread, which is shared with the original context

#ifdef _WIN32
  HDC tempDC_ = wglGetCurrentDC();
  HGLRC tempGLRC_ = wglGetCurrentContext();
  // Set DC and GLRC
  if (tempDC_ && tempGLRC_) {
    ret = context().glenv()->glClientWaitSync_(gs, GL_SYNC_FLUSH_COMMANDS_BIT,
                                               static_cast<GLuint64>(-1));
    if (!(ret == GL_ALREADY_SIGNALED || ret == GL_CONDITION_SATISFIED)) return false;
  } else {
    tempDC_ = context().glenv()->getDC();
    tempGLRC_ = context().glenv()->getIntGLRC();
    if (!context().glenv()->init(reinterpret_cast<intptr_t>(tempDC_),
                                 reinterpret_cast<intptr_t>(tempGLRC_)))
      return false;

    // Make the newly created GL context current to this thread
    context().glenv()->setIntEnv();
    // If fence has not yet executed, wait till it finishes
    ret = context().glenv()->glClientWaitSync_(gs, GL_SYNC_FLUSH_COMMANDS_BIT,
                                               static_cast<GLuint64>(-1));
    if (!(ret == GL_ALREADY_SIGNALED || ret == GL_CONDITION_SATISFIED)) return false;
    // Since we're done making GL calls, restore whatever context was previously current to this
    // thread
    context().glenv()->restoreEnv();
  }
#else  // Lnx
  Display* tempDpy_ = context().glenv()->glXGetCurrentDisplay_();
  GLXDrawable tempDrawable_ = context().glenv()->glXGetCurrentDrawable_();
  GLXContext tempCtx_ = context().glenv()->glXGetCurrentContext_();
  // Set internal Display and GLXContext
  if (tempDpy_ && tempCtx_) {
    ret = context().glenv()->glClientWaitSync_(gs, GL_SYNC_FLUSH_COMMANDS_BIT,
                                               static_cast<GLuint64>(-1));
    if (!(ret == GL_ALREADY_SIGNALED || ret == GL_CONDITION_SATISFIED)) return false;
  } else {
    if (!context().glenv()->init(reinterpret_cast<intptr_t>(context().glenv()->getIntDpy()),
                                 reinterpret_cast<intptr_t>(context().glenv()->getIntCtx())))
      return false;

    // Make the newly created GL context current to this thread
    context().glenv()->setIntEnv();
    // If fence has not yet executed, wait till it finishes
    ret = context().glenv()->glClientWaitSync_(gs, GL_SYNC_FLUSH_COMMANDS_BIT,
                                               static_cast<GLuint64>(-1));
    if (!(ret == GL_ALREADY_SIGNALED || ret == GL_CONDITION_SATISFIED)) return false;
    // Since we're done making GL calls, restore whatever context was previously current to this
    // thread
    context().glenv()->restoreEnv();
  }
#endif
  // If we reach this point, fence should have completed
  setStatus(CL_COMPLETE);
  return true;
}

//
//  GLFunctions implementation
//

#ifdef _WIN32
#define CONVERT_CHAR_GLUBYTE
#else  //!_WIN32
#define CONVERT_CHAR_GLUBYTE (GLubyte*)
#endif  //!_WIN32

#define GLPREFIX(rtype, fcn, dclargs)                                                              \
  if (!(fcn##_ = (PFN_##fcn)GETPROCADDRESS(libHandle_, #fcn))) {                                   \
    if (!(fcn##_ = (PFN_##fcn)GetProcAddress_(reinterpret_cast<FCN_STR_TYPE>(#fcn)))) ++missed_;   \
  }

GLFunctions::SetIntEnv::SetIntEnv(GLFunctions* env) : env_(env) {
  env_->getLock().lock();

  // Set environment (DC and GLRC)
  isValid_ = env_->setIntEnv();
}

GLFunctions::SetIntEnv::~SetIntEnv() {
  // Restore environment (CL DC and CL GLRC)
  env_->restoreEnv();

  env_->getLock().unlock();
}

GLFunctions::GLFunctions(HMODULE h, bool isEGL)
    : libHandle_(h),
      missed_(0),
      eglDisplay_(EGL_NO_DISPLAY),
      eglOriginalContext_(EGL_NO_CONTEXT),
      eglInternalContext_(EGL_NO_CONTEXT),
      eglTempContext_(EGL_NO_CONTEXT),
      isEGL_(isEGL),
#ifdef _WIN32
      hOrigGLRC_(0),
      hDC_(0),
      hIntGLRC_(0)
#else   //!_WIN32
      Dpy_(0),
      Drawable_(0),
      origCtx_(0),
      intDpy_(0),
      intDrawable_(0),
      intCtx_(0),
      XOpenDisplay_(NULL),
      XCloseDisplay_(NULL),
      glXGetCurrentDrawable_(NULL),
      glXGetCurrentDisplay_(NULL),
      glXGetCurrentContext_(NULL),
      glXChooseVisual_(NULL),
      glXCreateContext_(NULL),
      glXDestroyContext_(NULL),
      glXMakeCurrent_(NULL)
#endif  //!_WIN32
{
#define VERIFY_POINTER(p)                                                                          \
  if (NULL == p) {                                                                                 \
    missed_++;                                                                                     \
  }

  if (isEGL_) {
    GetProcAddress_ = (PFN_xxxGetProcAddress)GETPROCADDRESS(h, "eglGetProcAddress");
  } else {
    GetProcAddress_ = (PFN_xxxGetProcAddress)GETPROCADDRESS(h, API_GETPROCADDR);
  }
#ifndef _WIN32
  // Initialize pointers to X11/GLX functions
  // We can not link with these functions on compile time since we need to support
  // console mode. In console mode X server and X server components may be absent.
  // Hence linking with X11 or libGL will fail module image loading in console mode.-tzachi cohen

  if (!isEGL_) {
    glXGetCurrentDrawable_ = (PFNglXGetCurrentDrawable)GETPROCADDRESS(h, "glXGetCurrentDrawable");
    VERIFY_POINTER(glXGetCurrentDrawable_)
    glXGetCurrentDisplay_ = (PFNglXGetCurrentDisplay)GETPROCADDRESS(h, "glXGetCurrentDisplay");
    VERIFY_POINTER(glXGetCurrentDisplay_)
    glXGetCurrentContext_ = (PFNglXGetCurrentContext)GETPROCADDRESS(h, "glXGetCurrentContext");
    VERIFY_POINTER(glXGetCurrentContext_)
    glXChooseVisual_ = (PFNglXChooseVisual)GETPROCADDRESS(h, "glXChooseVisual");
    VERIFY_POINTER(glXChooseVisual_)
    glXCreateContext_ = (PFNglXCreateContext)GETPROCADDRESS(h, "glXCreateContext");
    VERIFY_POINTER(glXCreateContext_)
    glXDestroyContext_ = (PFNglXDestroyContext)GETPROCADDRESS(h, "glXDestroyContext");
    VERIFY_POINTER(glXDestroyContext_)
    glXMakeCurrent_ = (PFNglXMakeCurrent)GETPROCADDRESS(h, "glXMakeCurrent");
    VERIFY_POINTER(glXMakeCurrent_)

    HMODULE hXModule = (HMODULE)Os::loadLibrary("libX11.so.6");
    if (NULL != hXModule) {
      XOpenDisplay_ = (PFNXOpenDisplay)GETPROCADDRESS(hXModule, "XOpenDisplay");
      VERIFY_POINTER(XOpenDisplay_)
      XCloseDisplay_ = (PFNXCloseDisplay)GETPROCADDRESS(hXModule, "XCloseDisplay");
      VERIFY_POINTER(XCloseDisplay_)
    } else {
      missed_ += 2;
    }
  }
// Initialize pointers to GL functions
#include "gl_functions.hpp"
#else
  if (!isEGL_) {
    wglCreateContext_ = (PFN_wglCreateContext)GETPROCADDRESS(h, "wglCreateContext");
    VERIFY_POINTER(wglCreateContext_)
    wglGetCurrentContext_ = (PFN_wglGetCurrentContext)GETPROCADDRESS(h, "wglGetCurrentContext");
    VERIFY_POINTER(wglGetCurrentContext_)
    wglGetCurrentDC_ = (PFN_wglGetCurrentDC)GETPROCADDRESS(h, "wglGetCurrentDC");
    VERIFY_POINTER(wglGetCurrentDC_)
    wglDeleteContext_ = (PFN_wglDeleteContext)GETPROCADDRESS(h, "wglDeleteContext");
    VERIFY_POINTER(wglDeleteContext_)
    wglMakeCurrent_ = (PFN_wglMakeCurrent)GETPROCADDRESS(h, "wglMakeCurrent");
    VERIFY_POINTER(wglMakeCurrent_)
    wglShareLists_ = (PFN_wglShareLists)GETPROCADDRESS(h, "wglShareLists");
    VERIFY_POINTER(wglShareLists_)
  }
#endif
}

GLFunctions::~GLFunctions() {
#ifdef _WIN32
  if (hIntGLRC_) {
    if (!wglDeleteContext_(hIntGLRC_)) {
      DWORD dwErr = GetLastError();
      LogWarning("Cannot delete GLRC");
    }
  }
#else   //!_WIN32
  if (intDpy_) {
    if (intCtx_) {
      glXDestroyContext_(intDpy_, intCtx_);
      intCtx_ = NULL;
    }
    XCloseDisplay_(intDpy_);
    intDpy_ = NULL;
  }
#endif  //!_WIN32
}

bool GLFunctions::init(intptr_t hdc, intptr_t hglrc) {
  if (isEGL_) {
    eglDisplay_ = (EGLDisplay)hdc;
    eglOriginalContext_ = (EGLContext)hglrc;
    return true;
  }

#ifdef _WIN32
  DWORD err;

  if (missed_) {
    return false;
  }

  if (!hdc) {
    hDC_ = wglGetCurrentDC_();
  } else {
    hDC_ = (HDC)hdc;
  }
  hOrigGLRC_ = (HGLRC)hglrc;
  if (!(hIntGLRC_ = wglCreateContext_(hDC_))) {
    err = GetLastError();
    return false;
  }
  if (!wglShareLists_(hOrigGLRC_, hIntGLRC_)) {
    err = GetLastError();
    return false;
  }

  bool makeCurrentNull = false;

  if (wglGetCurrentContext_() == NULL) {
    wglMakeCurrent_(hDC_, hIntGLRC_);

    makeCurrentNull = true;
  }

// Initialize pointers to GL functions
#include "gl_functions.hpp"

  if (makeCurrentNull) {
    wglMakeCurrent_(NULL, NULL);
  }

  if (missed_ == 0) {
    return true;
  }
#else  //!_WIN32
  if (!missed_) {
    if (!hdc) {
      Dpy_ = glXGetCurrentDisplay_();
    } else {
      Dpy_ = (Display*)hdc;
    }
    Drawable_ = glXGetCurrentDrawable_();
    origCtx_ = (GLXContext)hglrc;

    int attribList[] = {GLX_RGBA, None};
    if (!(intDpy_ = XOpenDisplay_(DisplayString(Dpy_)))) {
#if defined(ATI_ARCH_X86)
      asm("int $3");
#endif
    }
    intDrawable_ = DefaultRootWindow(intDpy_);

    XVisualInfo* vis;
    int defaultScreen = DefaultScreen(intDpy_);
    if (!(vis = glXChooseVisual_(intDpy_, defaultScreen, attribList))) {
      return false;
    }
    if (!(intCtx_ = glXCreateContext_(intDpy_, vis, origCtx_, true))) {
      return false;
    }
    return true;
  }
#endif  //!_WIN32
  return false;
}

bool GLFunctions::setIntEnv() {
  if (isEGL_) {
    return true;
  }
#ifdef _WIN32
  // Save current DC and GLRC
  tempDC_ = wglGetCurrentDC_();
  tempGLRC_ = wglGetCurrentContext_();
  // Set internal DC and GLRC
  if (tempDC_ != getDC() || tempGLRC_ != getIntGLRC()) {
    if (!wglMakeCurrent_(getDC(), getIntGLRC())) {
      DWORD err = GetLastError();
      LogWarning("cannot set internal GL environment");
      return false;
    }
  }
#else   //!_WIN32
  tempDpy_ = glXGetCurrentDisplay_();
  tempDrawable_ = glXGetCurrentDrawable_();
  tempCtx_ = glXGetCurrentContext_();
  // Set internal Display and GLXContext
  if (tempDpy_ != getDpy() || tempCtx_ != getIntCtx()) {
    if (!glXMakeCurrent_(getIntDpy(), getIntDrawable(), getIntCtx())) {
      LogWarning("cannot set internal GL environment");
      return false;
    }
  }
#endif  //!_WIN32

  return true;
}

bool GLFunctions::restoreEnv() {
  if (isEGL_) {
    // eglMakeCurrent( );
    return true;
  }
#ifdef _WIN32
  // Restore original DC and GLRC
  if (!wglMakeCurrent_(tempDC_, tempGLRC_)) {
    DWORD err = GetLastError();
    LogWarning("cannot restore original GL environment");
    return false;
  }
#else   //!_WIN32
  // Restore Display and GLXContext
  if (tempDpy_) {
    if (!glXMakeCurrent_(tempDpy_, tempDrawable_, tempCtx_)) {
      LogWarning("cannot restore original GL environment");
      return false;
    }
  } else {
    // Just release internal context
    if (!glXMakeCurrent_(getIntDpy(), None, NULL)) {
      LogWarning("cannot reelase internal GL environment");
      return false;
    }
  }
#endif  //!_WIN32

  return true;
}

}  // namespace amd
