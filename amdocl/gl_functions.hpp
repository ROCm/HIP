//
// Copyright 2010 Advanced Micro Devices, Inc. All rights reserved.
//
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
