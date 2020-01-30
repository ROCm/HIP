/*******************************************************************************
 * Copyright (c) 2008-2010 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

#ifndef __OPENCL_CL_ICD_H
#define __OPENCL_CL_ICD_H

#include <CL/cl.h>
#include <CL/cl_gl.h>

#define cl_khr_icd 1

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef cl_int(CL_API_CALL* clGetPlatformIDs_fn)(
    cl_uint /* num_entries */, cl_platform_id* /* platforms */,
    cl_uint* /* num_platforms */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetPlatformInfo_fn)(
    cl_platform_id /* platform */, cl_platform_info /* param_name */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetDeviceIDs_fn)(
    cl_platform_id /* platform */, cl_device_type /* device_type */, cl_uint /* num_entries */,
    cl_device_id* /* devices */, cl_uint* /* num_devices */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetDeviceInfo_fn)(
    cl_device_id /* device */, cl_device_info /* param_name */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_context(CL_API_CALL* clCreateContext_fn)(
    const cl_context_properties* /* properties */, cl_uint /* num_devices */,
    const cl_device_id* /* devices */,
    void(CL_CALLBACK* /* pfn_notify */)(const char*, const void*, size_t, void*),
    void* /* user_data */, cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_context(CL_API_CALL* clCreateContextFromType_fn)(
    const cl_context_properties* /* properties */, cl_device_type /* device_type */,
    void(CL_CALLBACK* /* pfn_notify*/)(const char*, const void*, size_t, void*),
    void* /* user_data */, cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clRetainContext_fn)(cl_context /* context */)
    CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clReleaseContext_fn)(cl_context /* context */)
    CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetContextInfo_fn)(
    cl_context /* context */, cl_context_info /* param_name */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_command_queue(CL_API_CALL* clCreateCommandQueue_fn)(
    cl_context /* context */, cl_device_id /* device */,
    cl_command_queue_properties /* properties */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clRetainCommandQueue_fn)(cl_command_queue /* command_queue */)
    CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clReleaseCommandQueue_fn)(cl_command_queue /* command_queue */)
    CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetCommandQueueInfo_fn)(
    cl_command_queue /* command_queue */, cl_command_queue_info /* param_name */,
    size_t /* param_value_size */, void* /* param_value */,
    size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clSetCommandQueueProperty_fn)(
    cl_command_queue /* command_queue */, cl_command_queue_properties /* properties */,
    cl_bool /* enable */,
    cl_command_queue_properties* /* old_properties */) /*CL_EXT_SUFFIX__VERSION_1_0_DEPRECATED*/;

typedef cl_mem(CL_API_CALL* clCreateBuffer_fn)(
    cl_context /* context */, cl_mem_flags /* flags */, size_t /* size */, void* /* host_ptr */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_mem(CL_API_CALL* clCreateSubBuffer_fn)(
    cl_mem /* buffer */, cl_mem_flags /* flags */, cl_buffer_create_type /* buffer_create_type */,
    const void* /* buffer_create_info */, cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_1;

typedef cl_mem(CL_API_CALL* clCreateImage2D_fn)(
    cl_context /* context */, cl_mem_flags /* flags */, const cl_image_format* /* image_format */,
    size_t /* image_width */, size_t /* image_height */, size_t /* image_row_pitch */,
    void* /* host_ptr */, cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_mem(CL_API_CALL* clCreateImage3D_fn)(
    cl_context /* context */, cl_mem_flags /* flags */, const cl_image_format* /* image_format */,
    size_t /* image_width */, size_t /* image_height */, size_t /* image_depth */,
    size_t /* image_row_pitch */, size_t /* image_slice_pitch */, void* /* host_ptr */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clRetainMemObject_fn)(cl_mem /* memobj */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clReleaseMemObject_fn)(cl_mem /* memobj */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetSupportedImageFormats_fn)(
    cl_context /* context */, cl_mem_flags /* flags */, cl_mem_object_type /* image_type */,
    cl_uint /* num_entries */, cl_image_format* /* image_formats */,
    cl_uint* /* num_image_formats */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetMemObjectInfo_fn)(
    cl_mem /* memobj */, cl_mem_info /* param_name */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetImageInfo_fn)(
    cl_mem /* image */, cl_image_info /* param_name */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clSetMemObjectDestructorCallback_fn)(
    cl_mem /* memobj */,
    void(CL_CALLBACK* /*pfn_notify*/)(cl_mem /* memobj */, void* /*user_data*/),
    void* /*user_data */) CL_API_SUFFIX__VERSION_1_1;

/* Sampler APIs  */
typedef cl_sampler(CL_API_CALL* clCreateSampler_fn)(
    cl_context /* context */, cl_bool /* normalized_coords */,
    cl_addressing_mode /* addressing_mode */, cl_filter_mode /* filter_mode */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clRetainSampler_fn)(cl_sampler /* sampler */)
    CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clReleaseSampler_fn)(cl_sampler /* sampler */)
    CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetSamplerInfo_fn)(
    cl_sampler /* sampler */, cl_sampler_info /* param_name */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

/* Program Object APIs  */
typedef cl_program(CL_API_CALL* clCreateProgramWithSource_fn)(
    cl_context /* context */, cl_uint /* count */, const char** /* strings */,
    const size_t* /* lengths */, cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithIL(cl_context /* context */,
    const void * /* strings */, size_t /* lengths */,
    cl_int * /* errcode_ret */) CL_EXT_SUFFIX__VERSION_2_0;

typedef cl_program(CL_API_CALL* clCreateProgramWithILKHR_fn)(
    cl_context /* context */, const void* /* il */, size_t /* length */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;

typedef cl_program(CL_API_CALL* clCreateProgramWithBinary_fn)(
    cl_context /* context */, cl_uint /* num_devices */, const cl_device_id* /* device_list */,
    const size_t* /* lengths */, const unsigned char** /* binaries */, cl_int* /* binary_status */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clRetainProgram_fn)(cl_program /* program */)
    CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clReleaseProgram_fn)(cl_program /* program */)
    CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clBuildProgram_fn)(
    cl_program /* program */, cl_uint /* num_devices */, const cl_device_id* /* device_list */,
    const char* /* options */,
    void(CL_CALLBACK* /* pfn_notify */)(cl_program /* program */, void* /* user_data */),
    void* /* user_data */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clUnloadCompiler_fn)(void) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetProgramInfo_fn)(
    cl_program /* program */, cl_program_info /* param_name */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetProgramBuildInfo_fn)(
    cl_program /* program */, cl_device_id /* device */, cl_program_build_info /* param_name */,
    size_t /* param_value_size */, void* /* param_value */,
    size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

/* Kernel Object APIs */
typedef cl_kernel(CL_API_CALL* clCreateKernel_fn)(
    cl_program /* program */, const char* /* kernel_name */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clCreateKernelsInProgram_fn)(
    cl_program /* program */, cl_uint /* num_kernels */, cl_kernel* /* kernels */,
    cl_uint* /* num_kernels_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clRetainKernel_fn)(cl_kernel /* kernel */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clReleaseKernel_fn)(cl_kernel /* kernel */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clSetKernelArg_fn)(cl_kernel /* kernel */, cl_uint /* arg_index */,
                                               size_t /* arg_size */, const void* /* arg_value */)
    CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetKernelInfo_fn)(
    cl_kernel /* kernel */, cl_kernel_info /* param_name */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetKernelWorkGroupInfo_fn)(
    cl_kernel /* kernel */, cl_device_id /* device */, cl_kernel_work_group_info /* param_name */,
    size_t /* param_value_size */, void* /* param_value */,
    size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

/* Event Object APIs  */
typedef cl_int(CL_API_CALL* clWaitForEvents_fn)(
    cl_uint /* num_events */, const cl_event* /* event_list */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetEventInfo_fn)(
    cl_event /* event */, cl_event_info /* param_name */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_event(CL_API_CALL* clCreateUserEvent_fn)(
    cl_context /* context */, cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_1;

typedef cl_int(CL_API_CALL* clRetainEvent_fn)(cl_event /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clReleaseEvent_fn)(cl_event /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clSetUserEventStatus_fn)(
    cl_event /* event */, cl_int /* execution_status */) CL_API_SUFFIX__VERSION_1_1;

typedef cl_int(CL_API_CALL* clSetEventCallback_fn)(
    cl_event /* event */, cl_int /* command_exec_callback_type */,
    void(CL_CALLBACK* /* pfn_notify */)(cl_event, cl_int, void*),
    void* /* user_data */) CL_API_SUFFIX__VERSION_1_1;

/* Profiling APIs  */
typedef cl_int(CL_API_CALL* clGetEventProfilingInfo_fn)(
    cl_event /* event */, cl_profiling_info /* param_name */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

/* Flush and Finish APIs */
typedef cl_int(CL_API_CALL* clFlush_fn)(cl_command_queue /* command_queue */)
    CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clFinish_fn)(cl_command_queue /* command_queue */)
    CL_API_SUFFIX__VERSION_1_0;

/* Enqueued Commands APIs */
typedef cl_int(CL_API_CALL* clEnqueueReadBuffer_fn)(
    cl_command_queue /* command_queue */, cl_mem /* buffer */, cl_bool /* blocking_read */,
    size_t /* offset */, size_t /* cb */, void* /* ptr */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueReadBufferRect_fn)(
    cl_command_queue /* command_queue */, cl_mem /* buffer */, cl_bool /* blocking_read */,
    const size_t* /* buffer_offset */, const size_t* /* host_offset */, const size_t* /* region */,
    size_t /* buffer_row_pitch */, size_t /* buffer_slice_pitch */, size_t /* host_row_pitch */,
    size_t /* host_slice_pitch */, void* /* ptr */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_1;

typedef cl_int(CL_API_CALL* clEnqueueWriteBuffer_fn)(
    cl_command_queue /* command_queue */, cl_mem /* buffer */, cl_bool /* blocking_write */,
    size_t /* offset */, size_t /* cb */, const void* /* ptr */,
    cl_uint /* num_events_in_wait_list */, const cl_event* /* event_wait_list */,
    cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueWriteBufferRect_fn)(
    cl_command_queue /* command_queue */, cl_mem /* buffer */, cl_bool /* blocking_read */,
    const size_t* /* buffer_offset */, const size_t* /* host_offset */, const size_t* /* region */,
    size_t /* buffer_row_pitch */, size_t /* buffer_slice_pitch */, size_t /* host_row_pitch */,
    size_t /* host_slice_pitch */, const void* /* ptr */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_1;

typedef cl_int(CL_API_CALL* clEnqueueCopyBuffer_fn)(
    cl_command_queue /* command_queue */, cl_mem /* src_buffer */, cl_mem /* dst_buffer */,
    size_t /* src_offset */, size_t /* dst_offset */, size_t /* cb */,
    cl_uint /* num_events_in_wait_list */, const cl_event* /* event_wait_list */,
    cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueCopyBufferRect_fn)(
    cl_command_queue /* command_queue */, cl_mem /* src_buffer */, cl_mem /* dst_buffer */,
    const size_t* /* src_origin */, const size_t* /* dst_origin */, const size_t* /* region */,
    size_t /* src_row_pitch */, size_t /* src_slice_pitch */, size_t /* dst_row_pitch */,
    size_t /* dst_slice_pitch */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_1;

typedef cl_int(CL_API_CALL* clEnqueueReadImage_fn)(
    cl_command_queue /* command_queue */, cl_mem /* image */, cl_bool /* blocking_read */,
    const size_t* /* origin[3] */, const size_t* /* region[3] */, size_t /* row_pitch */,
    size_t /* slice_pitch */, void* /* ptr */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueWriteImage_fn)(
    cl_command_queue /* command_queue */, cl_mem /* image */, cl_bool /* blocking_write */,
    const size_t* /* origin[3] */, const size_t* /* region[3] */, size_t /* input_row_pitch */,
    size_t /* input_slice_pitch */, const void* /* ptr */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueCopyImage_fn)(
    cl_command_queue /* command_queue */, cl_mem /* src_image */, cl_mem /* dst_image */,
    const size_t* /* src_origin[3] */, const size_t* /* dst_origin[3] */,
    const size_t* /* region[3] */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueCopyImageToBuffer_fn)(
    cl_command_queue /* command_queue */, cl_mem /* src_image */, cl_mem /* dst_buffer */,
    const size_t* /* src_origin[3] */, const size_t* /* region[3] */, size_t /* dst_offset */,
    cl_uint /* num_events_in_wait_list */, const cl_event* /* event_wait_list */,
    cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueCopyBufferToImage_fn)(
    cl_command_queue /* command_queue */, cl_mem /* src_buffer */, cl_mem /* dst_image */,
    size_t /* src_offset */, const size_t* /* dst_origin[3] */, const size_t* /* region[3] */,
    cl_uint /* num_events_in_wait_list */, const cl_event* /* event_wait_list */,
    cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef void*(CL_API_CALL* clEnqueueMapBuffer_fn)(
    cl_command_queue /* command_queue */, cl_mem /* buffer */, cl_bool /* blocking_map */,
    cl_map_flags /* map_flags */, size_t /* offset */, size_t /* cb */,
    cl_uint /* num_events_in_wait_list */, const cl_event* /* event_wait_list */,
    cl_event* /* event */, cl_int* /* errcode_ret */)CL_API_SUFFIX__VERSION_1_0;

typedef void*(CL_API_CALL* clEnqueueMapImage_fn)(
    cl_command_queue /* command_queue */, cl_mem /* image */, cl_bool /* blocking_map */,
    cl_map_flags /* map_flags */, const size_t* /* origin[3] */, const size_t* /* region[3] */,
    size_t* /* image_row_pitch */, size_t* /* image_slice_pitch */,
    cl_uint /* num_events_in_wait_list */, const cl_event* /* event_wait_list */,
    cl_event* /* event */, cl_int* /* errcode_ret */)CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueUnmapMemObject_fn)(
    cl_command_queue /* command_queue */, cl_mem /* memobj */, void* /* mapped_ptr */,
    cl_uint /* num_events_in_wait_list */, const cl_event* /* event_wait_list */,
    cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueNDRangeKernel_fn)(
    cl_command_queue /* command_queue */, cl_kernel /* kernel */, cl_uint /* work_dim */,
    const size_t* /* global_work_offset */, const size_t* /* global_work_size */,
    const size_t* /* local_work_size */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueTask_fn)(cl_command_queue /* command_queue */,
                                              cl_kernel /* kernel */,
                                              cl_uint /* num_events_in_wait_list */,
                                              const cl_event* /* event_wait_list */,
                                              cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueNativeKernel_fn)(
    cl_command_queue /* command_queue */, void(CL_CALLBACK* user_func)(void*), void* /* args */,
    size_t /* cb_args */, cl_uint /* num_mem_objects */, const cl_mem* /* mem_list */,
    const void** /* args_mem_loc */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueMarker_fn)(cl_command_queue /* command_queue */,
                                                cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueWaitForEvents_fn)(
    cl_command_queue /* command_queue */, cl_uint /* num_events */,
    const cl_event* /* event_list */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueBarrier_fn)(cl_command_queue /* command_queue */)
    CL_API_SUFFIX__VERSION_1_0;

typedef void*(CL_API_CALL* clGetExtensionFunctionAddress_fn)(const char* /* func_name */)
    CL_API_SUFFIX__VERSION_1_0;

typedef cl_mem(CL_API_CALL* clCreateFromGLBuffer_fn)(
    cl_context /* context */, cl_mem_flags /* flags */, cl_GLuint /* bufobj */,
    int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_mem(CL_API_CALL* clCreateFromGLTexture2D_fn)(
    cl_context /* context */, cl_mem_flags /* flags */, cl_GLenum /* target */,
    cl_GLint /* miplevel */, cl_GLuint /* texture */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_mem(CL_API_CALL* clCreateFromGLTexture3D_fn)(
    cl_context /* context */, cl_mem_flags /* flags */, cl_GLenum /* target */,
    cl_GLint /* miplevel */, cl_GLuint /* texture */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_mem(CL_API_CALL* clCreateFromGLRenderbuffer_fn)(
    cl_context /* context */, cl_mem_flags /* flags */, cl_GLuint /* renderbuffer */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetGLObjectInfo_fn)(
    cl_mem /* memobj */, cl_gl_object_type* /* gl_object_type */,
    cl_GLuint* /* gl_object_name */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clGetGLTextureInfo_fn)(
    cl_mem /* memobj */, cl_gl_texture_info /* param_name */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_event(CL_API_CALL* clCreateEventFromGLsyncKHR_fn)(
    cl_context /* context */, cl_GLsync /* cl_GLsync */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_1;

typedef cl_int(CL_API_CALL* clEnqueueAcquireGLObjects_fn)(
    cl_command_queue /* command_queue */, cl_uint /* num_objects */,
    const cl_mem* /* mem_objects */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clEnqueueReleaseGLObjects_fn)(
    cl_command_queue /* command_queue */, cl_uint /* num_objects */,
    const cl_mem* /* mem_objects */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int(CL_API_CALL* clCreateSubDevices_fn)(
    cl_device_id /* in_device */, const cl_device_partition_property* /* properties */,
    cl_uint /* num_entries */, cl_device_id* /* out_devices */,
    cl_uint* /* num_devices */) CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL* clRetainDevice_fn)(cl_device_id /* device */)
    CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL* clReleaseDevice_fn)(cl_device_id /* device */)
    CL_API_SUFFIX__VERSION_1_2;

typedef cl_mem(CL_API_CALL* clCreateImage_fn)(cl_context /* context */, cl_mem_flags /* flags */,
                                              const cl_image_format* /* image_format*/,
                                              const cl_image_desc* /* image_desc*/,
                                              void* /* host_ptr */,
                                              cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;

typedef cl_program(CL_API_CALL* clCreateProgramWithBuiltInKernels_fn)(
    cl_context /* context */, cl_uint /* num_devices */, const cl_device_id* /* device_list */,
    const char* /* kernel_names */, cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL* clCompileProgram_fn)(
    cl_program /* program */, cl_uint /* num_devices */, const cl_device_id* /* device_list */,
    const char* /* options */, cl_uint /* num_input_headers */,
    const cl_program* /* input_headers */, const char** /* header_include_names */,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* /* user_data */) CL_API_SUFFIX__VERSION_1_2;

typedef cl_program(CL_API_CALL* clLinkProgram_fn)(
    cl_context /* context */, cl_uint /* num_devices */, const cl_device_id* /* device_list */,
    const char* /* options */, cl_uint /* num_input_programs */,
    const cl_program* /* input_programs */,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* /* user_data */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL* clUnloadPlatformCompiler_fn)(cl_platform_id /* platform */)
    CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL* clGetKernelArgInfo_fn)(
    cl_kernel /* kernel */, cl_uint /* arg_indx */, cl_kernel_arg_info /* param_name */,
    size_t /* param_value_size */, void* /* param_value */,
    size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL* clEnqueueFillBuffer_fn)(
    cl_command_queue /* command_queue */, cl_mem /* buffer */, const void* /* pattern */,
    size_t /* pattern_size */, size_t /* offset */, size_t /* size */,
    cl_uint /* num_events_in_wait_list */, const cl_event* /* event_wait_list */,
    cl_event* /* event */) CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL* clEnqueueFillImage_fn)(
    cl_command_queue /* command_queue */, cl_mem /* image */, const void* /* fill_color */,
    const size_t* /* origin */, const size_t* /* region */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL* clEnqueueMigrateMemObjects_fn)(
    cl_command_queue /* command_queue */, cl_uint /* num_mem_objects */,
    const cl_mem* /* mem_objects */, cl_mem_migration_flags /* flags */,
    cl_uint /* num_events_in_wait_list */, const cl_event* /* event_wait_list */,
    cl_event* /* event */) CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL* clEnqueueMarkerWithWaitList_fn)(
    cl_command_queue /* command_queue */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_2;

typedef cl_int(CL_API_CALL* clEnqueueBarrierWithWaitList_fn)(
    cl_command_queue /* command_queue */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_1_2;

typedef void*(CL_API_CALL* clGetExtensionFunctionAddressForPlatform_fn)(
    cl_platform_id /* platform */, const char* /* funcname */)CL_API_SUFFIX__VERSION_1_2;

typedef cl_mem(CL_API_CALL* clCreateFromGLTexture_fn)(
    cl_context /* context */, cl_mem_flags /* flags */, cl_GLenum /* texture_target */,
    cl_GLint /* miplevel */, cl_GLuint /* texture */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;

typedef cl_command_queue(CL_API_CALL* clCreateCommandQueueWithProperties_fn)(
    cl_context /* context */, cl_device_id /* device */,
    const cl_queue_properties* /* properties */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_2_0;

typedef cl_sampler(CL_API_CALL* clCreateSamplerWithProperties_fn)(
    cl_context /* context */, const cl_sampler_properties* /* properties */,
    cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_2_0;

typedef void*(CL_API_CALL* clSVMAlloc_fn)(cl_context /* context */, cl_svm_mem_flags /* flags */,
                                          size_t /* size */,
                                          cl_uint /* alignment */)CL_API_SUFFIX__VERSION_2_0;

typedef void(CL_API_CALL* clSVMFree_fn)(cl_context /* context */,
                                        void* /* svm_pointer */) CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL* clSetKernelArgSVMPointer_fn)(
    cl_kernel /* kernel */, cl_uint /*  arg_index */,
    const void* /* arg_value */) CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL* clSetKernelExecInfo_fn)(
    cl_kernel /* kernel */, cl_kernel_exec_info /* param_name */, size_t /* param_value_size */,
    const void* /* param_value */) CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL* clEnqueueSVMFree_fn)(
    cl_command_queue /* command_queue */, cl_uint /* num_svm_pointers */,
    void* [] /* svm_pointers */,
    void(CL_CALLBACK* /* pfn_free_func */)(cl_command_queue /* queue */,
                                           cl_uint /* num_svm_pointers */,
                                           void* [] /* svm_pointers */, void* /* user_data */),
    void* /* user_data */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL* clEnqueueSVMMemcpy_fn)(
    cl_command_queue /* command_queue */, cl_bool /* blocking_copy */, void* /* dst_ptr */,
    const void* /* src_ptr */, size_t /* size */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL* clEnqueueSVMMemFill_fn)(
    cl_command_queue /* command_queue */, void* /* svm_ptr */, const void* /* pattern */,
    size_t /* pattern_size */, size_t /* size */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL* clEnqueueSVMMap_fn)(
    cl_command_queue /* command_queue */, cl_bool /* blocking_map */, cl_map_flags /* flags */,
    void* /* svm_ptr */, size_t /* size */, cl_uint /* num_events_in_wait_list */,
    const cl_event* /* event_wait_list */, cl_event* /* event */) CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL* clEnqueueSVMUnmap_fn)(cl_command_queue /* command_queue */,
                                                  void* /* svm_ptr */,
                                                  cl_uint /* num_events_in_wait_list */,
                                                  const cl_event* /* event_wait_list */,
                                                  cl_event* /* event */) CL_API_SUFFIX__VERSION_2_0;

typedef cl_mem(CL_API_CALL* clCreatePipe_fn)(cl_context /* context */, cl_mem_flags /* flags */,
                                             cl_uint /* pipe_packet_size */,
                                             cl_uint /* pipe_max_packets */,
                                             const cl_pipe_properties* /* properties */,
                                             cl_int* /* errcode_ret */) CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL* clGetPipeInfo_fn)(
    cl_mem /* pipe */, cl_pipe_info /* param_name */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_2_0;

typedef cl_int(CL_API_CALL* clGetKernelSubGroupInfoKHR_fn)(
    cl_kernel /* kernel */, cl_device_id /* device */, cl_kernel_sub_group_info /* param_name */,
    size_t /* input_value_size */, const void* /* input_value */, size_t /* param_value_size */,
    void* /* param_value */, size_t* /* param_value_size_ret */) CL_API_SUFFIX__VERSION_2_0;


typedef cl_int(CL_API_CALL* clSetDefaultDeviceCommandQueue_fn)(
    cl_context /* context */, cl_device_id /* device */,
    cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_2_1;

typedef cl_kernel(CL_API_CALL* clCloneKernel_fn)(
    cl_kernel /* source_kernel */, cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_2_1;

typedef cl_int (CL_API_CALL* clEnqueueSVMMigrateMem_fn)(
    cl_command_queue /* command_queue */, cl_uint /* num_svm_pointers */,
    const void ** /* svm_pointers */, const size_t * /* sizes */,
    cl_mem_migration_flags /* flags */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */, cl_event * /* event */) CL_API_SUFFIX__VERSION_2_1;

typedef cl_int (CL_API_CALL* clGetDeviceAndHostTimer_fn)(
    cl_device_id /* device */, cl_ulong * /* device_timestamp */,
    cl_ulong * /* host_timestamp */) CL_API_SUFFIX__VERSION_2_1;

typedef cl_int (CL_API_CALL* clGetHostTimer_fn)(
    cl_device_id /* device */, cl_ulong * /* host_timestamp */) CL_API_SUFFIX__VERSION_2_1;

typedef cl_int (CL_API_CALL* clSetProgramSpecializationConstant_fn)(
    cl_program /* program */, cl_uint /* spec_id */, size_t /* spec_size */,
    const void* /* spec_value */) CL_API_SUFFIX__VERSION_2_2;

typedef cl_int (CL_API_CALL* clSetProgramReleaseCallback_fn)(
    cl_program /* program */,
    void (CL_CALLBACK *  /* pfn_notify */)(cl_program program, void * user_data),
    void * /* user_data */) CL_API_SUFFIX__VERSION_2_2;

typedef struct _cl_icd_dispatch_table {
  /* OpenCL 1.0 */
  clGetPlatformIDs_fn GetPlatformIDs;
  clGetPlatformInfo_fn GetPlatformInfo;
  clGetDeviceIDs_fn GetDeviceIDs;
  clGetDeviceInfo_fn GetDeviceInfo;
  clCreateContext_fn CreateContext;
  clCreateContextFromType_fn CreateContextFromType;
  clRetainContext_fn RetainContext;
  clReleaseContext_fn ReleaseContext;
  clGetContextInfo_fn GetContextInfo;
  clCreateCommandQueue_fn CreateCommandQueue;
  clRetainCommandQueue_fn RetainCommandQueue;
  clReleaseCommandQueue_fn ReleaseCommandQueue;
  clGetCommandQueueInfo_fn GetCommandQueueInfo;
  clSetCommandQueueProperty_fn SetCommandQueueProperty;
  clCreateBuffer_fn CreateBuffer;
  clCreateImage2D_fn CreateImage2D;
  clCreateImage3D_fn CreateImage3D;
  clRetainMemObject_fn RetainMemObject;
  clReleaseMemObject_fn ReleaseMemObject;
  clGetSupportedImageFormats_fn GetSupportedImageFormats;
  clGetMemObjectInfo_fn GetMemObjectInfo;
  clGetImageInfo_fn GetImageInfo;
  clCreateSampler_fn CreateSampler;
  clRetainSampler_fn RetainSampler;
  clReleaseSampler_fn ReleaseSampler;
  clGetSamplerInfo_fn GetSamplerInfo;
  clCreateProgramWithSource_fn CreateProgramWithSource;
  clCreateProgramWithBinary_fn CreateProgramWithBinary;
  clRetainProgram_fn RetainProgram;
  clReleaseProgram_fn ReleaseProgram;
  clBuildProgram_fn BuildProgram;
  clUnloadCompiler_fn UnloadCompiler;
  clGetProgramInfo_fn GetProgramInfo;
  clGetProgramBuildInfo_fn GetProgramBuildInfo;
  clCreateKernel_fn CreateKernel;
  clCreateKernelsInProgram_fn CreateKernelsInProgram;
  clRetainKernel_fn RetainKernel;
  clReleaseKernel_fn ReleaseKernel;
  clSetKernelArg_fn SetKernelArg;
  clGetKernelInfo_fn GetKernelInfo;
  clGetKernelWorkGroupInfo_fn GetKernelWorkGroupInfo;
  clWaitForEvents_fn WaitForEvents;
  clGetEventInfo_fn GetEventInfo;
  clRetainEvent_fn RetainEvent;
  clReleaseEvent_fn ReleaseEvent;
  clGetEventProfilingInfo_fn GetEventProfilingInfo;
  clFlush_fn Flush;
  clFinish_fn Finish;
  clEnqueueReadBuffer_fn EnqueueReadBuffer;
  clEnqueueWriteBuffer_fn EnqueueWriteBuffer;
  clEnqueueCopyBuffer_fn EnqueueCopyBuffer;
  clEnqueueReadImage_fn EnqueueReadImage;
  clEnqueueWriteImage_fn EnqueueWriteImage;
  clEnqueueCopyImage_fn EnqueueCopyImage;
  clEnqueueCopyImageToBuffer_fn EnqueueCopyImageToBuffer;
  clEnqueueCopyBufferToImage_fn EnqueueCopyBufferToImage;
  clEnqueueMapBuffer_fn EnqueueMapBuffer;
  clEnqueueMapImage_fn EnqueueMapImage;
  clEnqueueUnmapMemObject_fn EnqueueUnmapMemObject;
  clEnqueueNDRangeKernel_fn EnqueueNDRangeKernel;
  clEnqueueTask_fn EnqueueTask;
  clEnqueueNativeKernel_fn EnqueueNativeKernel;
  clEnqueueMarker_fn EnqueueMarker;
  clEnqueueWaitForEvents_fn EnqueueWaitForEvents;
  clEnqueueBarrier_fn EnqueueBarrier;
  clGetExtensionFunctionAddress_fn GetExtensionFunctionAddress;
  clCreateFromGLBuffer_fn CreateFromGLBuffer;
  clCreateFromGLTexture2D_fn CreateFromGLTexture2D;
  clCreateFromGLTexture3D_fn CreateFromGLTexture3D;
  clCreateFromGLRenderbuffer_fn CreateFromGLRenderbuffer;
  clGetGLObjectInfo_fn GetGLObjectInfo;
  clGetGLTextureInfo_fn GetGLTextureInfo;
  clEnqueueAcquireGLObjects_fn EnqueueAcquireGLObjects;
  clEnqueueReleaseGLObjects_fn EnqueueReleaseGLObjects;
  clGetGLContextInfoKHR_fn GetGLContextInfoKHR;
  void* _reservedForD3D10KHR[6];

  /* OpenCL 1.1 */
  clSetEventCallback_fn SetEventCallback;
  clCreateSubBuffer_fn CreateSubBuffer;
  clSetMemObjectDestructorCallback_fn SetMemObjectDestructorCallback;
  clCreateUserEvent_fn CreateUserEvent;
  clSetUserEventStatus_fn SetUserEventStatus;
  clEnqueueReadBufferRect_fn EnqueueReadBufferRect;
  clEnqueueWriteBufferRect_fn EnqueueWriteBufferRect;
  clEnqueueCopyBufferRect_fn EnqueueCopyBufferRect;

  void* _reservedForDeviceFissionEXT[3];
  clCreateEventFromGLsyncKHR_fn CreateEventFromGLsyncKHR;

  /* OpenCL 1.2 */
  clCreateSubDevices_fn CreateSubDevices;
  clRetainDevice_fn RetainDevice;
  clReleaseDevice_fn ReleaseDevice;
  clCreateImage_fn CreateImage;
  clCreateProgramWithBuiltInKernels_fn CreateProgramWithBuiltInKernels;
  clCompileProgram_fn CompileProgram;
  clLinkProgram_fn LinkProgram;
  clUnloadPlatformCompiler_fn UnloadPlatformCompiler;
  clGetKernelArgInfo_fn GetKernelArgInfo;
  clEnqueueFillBuffer_fn EnqueueFillBuffer;
  clEnqueueFillImage_fn EnqueueFillImage;
  clEnqueueMigrateMemObjects_fn EnqueueMigrateMemObjects;
  clEnqueueMarkerWithWaitList_fn EnqueueMarkerWithWaitList;
  clEnqueueBarrierWithWaitList_fn EnqueueBarrierWithWaitList;
  clGetExtensionFunctionAddressForPlatform_fn GetExtensionFunctionAddressForPlatform;
  clCreateFromGLTexture_fn CreateFromGLTexture;

  /* cl_khr_d3d11_sharing, cl_khr_dx9_media_sharing */
  void* _reservedForD3DExtensions[10];

  /* cl_khr_egl_image, cl_khr_egl_event */
  void* _reservedForEGLExtensions[4];

  /* OpenCL 2.0 */
  clCreateCommandQueueWithProperties_fn CreateCommandQueueWithProperties;
  clCreatePipe_fn CreatePipe;
  clGetPipeInfo_fn GetPipeInfo;
  clSVMAlloc_fn SVMAlloc;
  clSVMFree_fn SVMFree;
  clEnqueueSVMFree_fn EnqueueSVMFree;
  clEnqueueSVMMemcpy_fn EnqueueSVMMemcpy;
  clEnqueueSVMMemFill_fn EnqueueSVMMemFill;
  clEnqueueSVMMap_fn EnqueueSVMMap;
  clEnqueueSVMUnmap_fn EnqueueSVMUnmap;
  clCreateSamplerWithProperties_fn CreateSamplerWithProperties;
  clSetKernelArgSVMPointer_fn SetKernelArgSVMPointer;
  clSetKernelExecInfo_fn SetKernelExecInfo;
  /* cl_khr_sub_groups */
  clGetKernelSubGroupInfoKHR_fn GetKernelSubGroupInfoKHR;

  /* OpenCL 2.1 */
  clCloneKernel_fn CloneKernel;
  clCreateProgramWithILKHR_fn CreateProgramWithILKHR;
  clEnqueueSVMMigrateMem_fn EnqueueSVMMigrateMem;
  clGetDeviceAndHostTimer_fn  GetDeviceAndHostTimer;
  clGetHostTimer_fn GetHostTimer;
  clGetKernelSubGroupInfoKHR_fn GetKernelSubGroupInfo;
  clSetDefaultDeviceCommandQueue_fn SetDefaultDeviceCommandQueue;

  /* OpenCL 2.2 */
  clSetProgramReleaseCallback_fn SetProgramReleaseCallback;
  clSetProgramSpecializationConstant_fn SetProgramSpecializationConstant;

} cl_icd_dispatch_table;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __OPENCL_CL_ICD_H */
