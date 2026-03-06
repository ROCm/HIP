/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef HIP_GL_INTEROP_H
#define HIP_GL_INTEROP_H

#include <hip/hip_common.h>

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#include "hip/amd_detail/amd_hip_gl_interop.h"
#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "hip/nvidia_detail/nvidia_hip_gl_interop.h"
#endif
#endif
