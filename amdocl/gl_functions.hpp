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

GLPREFIX(GLubyte*, glGetString, (GLenum name))

GLPREFIX(void, glBindBuffer, (GLenum target, GLuint buffer))
//GLPREFIX(void, glBindFramebufferEXT, (GLenum target, GLuint framebuffer))
GLPREFIX(void, glBindRenderbuffer, (GLenum target, GLuint renderbuffer))
GLPREFIX(void, glBindTexture, (GLenum target, GLuint texture))
GLPREFIX(void, glBufferData, (GLenum target, GLsizeiptr size, const GLvoid* data, GLenum usage))

GLPREFIX(GLenum, glCheckFramebufferStatusEXT, (GLenum target))

GLPREFIX(void, glDeleteBuffers, (GLsizei n, const GLuint* buffers))
GLPREFIX(void, glDrawPixels, (GLsizei width, GLsizei height, GLenum format, GLenum type, const GLvoid *pixels))

//GLPREFIX(void, glFramebufferRenderbufferEXT, (GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer))

GLPREFIX(void, glGenBuffers, (GLsizei n, GLuint* buffers))
//GLPREFIX(void, glGenFramebuffersEXT, (GLsizei n, GLuint* framebuffers))
//10
GLPREFIX(void, glGetBufferParameteriv, (GLenum target, GLenum pname, GLint* params))
GLPREFIX(GLenum, glGetError, (void))
GLPREFIX(void, glFinish, (void))
GLPREFIX(void, glFlush, (void))
GLPREFIX(GLenum, glClientWaitSync, (GLsync sync, GLbitfield flags, GLuint64 timeout))
GLPREFIX(void, glGetIntegerv, (GLenum pname, GLint *params))
GLPREFIX(void, glGetRenderbufferParameterivEXT, (GLenum target, GLenum pname, GLint* params))
//GLPREFIX(GLubyte*, glGetString, (GLenum name))
GLPREFIX(void, glGetTexImage, (GLenum target, GLint level, GLenum format, GLenum type, GLvoid *pixels))
GLPREFIX(void, glGetTexLevelParameteriv, (GLenum target, GLint level, GLenum pname, GLint *params))
GLPREFIX(void, glGetTexParameteriv, (GLenum target, GLenum pname, GLint *params))

GLPREFIX(GLboolean, glIsBuffer, (GLuint buffer))
GLPREFIX(GLboolean, glIsRenderbufferEXT, (GLuint renderbuffer))
GLPREFIX(GLboolean, glIsTexture, (GLuint texture))
//20
GLPREFIX(GLvoid*, glMapBuffer, (GLenum target, GLenum access))

GLPREFIX(void, glReadPixels, (GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid *pixels))

GLPREFIX(void, glTexImage2D, (GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *pixels))
GLPREFIX(void, glTexImage3D, (GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *pixels))

GLPREFIX(GLboolean, glUnmapBuffer, (GLenum target))

#undef GLPREFIX
