/*
 * Copyright (c) 2016-2019 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * OpenCL is a trademark of Apple Inc. used under license by Khronos.
 */

#ifndef _ICD_DISPATCH_H_
#define _ICD_DISPATCH_H_

#ifndef CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#endif

// cl.h
#include <CL/cl.h>

// cl_gl.h and required files
#ifdef _WIN32
#include <windows.h>
#include <d3d9.h>
#include <d3d10_1.h>
#include <CL/cl_d3d10.h>
#include <CL/cl_d3d11.h>
#include <CL/cl_dx9_media_sharing.h>
#endif
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>
#include <CL/cl_egl.h>
#include <CL/cl_icd.h>

/*
 *
 * vendor dispatch table structure
 *
 */

struct _cl_platform_id
{
    cl_icd_dispatch *dispatch;
};

struct _cl_device_id
{
    cl_icd_dispatch *dispatch;
};

struct _cl_context
{
    cl_icd_dispatch *dispatch;
};

struct _cl_command_queue
{
    cl_icd_dispatch *dispatch;
};

struct _cl_mem
{
    cl_icd_dispatch *dispatch;
};

struct _cl_program
{
    cl_icd_dispatch *dispatch;
};

struct _cl_kernel
{
    cl_icd_dispatch *dispatch;
};

struct _cl_event
{
    cl_icd_dispatch *dispatch;
};

struct _cl_sampler
{
    cl_icd_dispatch *dispatch;
};

#endif // _ICD_DISPATCH_H_

