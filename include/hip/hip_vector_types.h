/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

//! hip_vector_types.h : Defines the HIP vector types.

#ifndef HIP_INCLUDE_HIP_HIP_VECTOR_TYPES_H
#define HIP_INCLUDE_HIP_HIP_VECTOR_TYPES_H

#include <hip/hip_common.h>


#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#if __cplusplus
#include <hip/amd_detail/amd_hip_vector_types.h>
#endif
#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include <vector_types.h>
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#endif
