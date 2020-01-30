//
// Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
//

#ifndef __CL_LQDFLASH_AMD_H
#define __CL_LQDFLASH_AMD_H

#include "CL/cl_ext.h"

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/

extern CL_API_ENTRY cl_file_amd CL_API_CALL
clCreateSsgFileObjectAMD(cl_context context, cl_file_flags_amd flags, const wchar_t* file_name,
                         cl_int* errcode_ret) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL clGetSsgFileObjectInfoAMD(
    cl_file_amd file, cl_file_info_amd param_name, size_t param_value_size, void* param_value,
    size_t* param_value_size_ret) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL clRetainSsgFileObjectAMD(cl_file_amd file)
    CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL clReleaseSsgFileObjectAMD(cl_file_amd file)
    CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadSsgFileAMD(
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t buffer_offset,
    size_t cb, cl_file_amd file, size_t file_offset, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteSsgFileAMD(
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t buffer_offset,
    size_t cb, cl_file_amd file, size_t file_offset, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) CL_EXT_SUFFIX__VERSION_1_2;

#ifdef __cplusplus
} /*extern "C"*/
#endif /*__cplusplus*/

#endif
