/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_DEVICE_FUNCTIONS_H
#define HIP_INCLUDE_HIP_DEVICE_FUNCTIONS_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#endif

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/amd_detail/amd_device_functions.h>
#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include <device_functions.h>
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#endif
