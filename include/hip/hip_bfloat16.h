/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*!\file
 * \brief hip_bfloat16.h provides struct for hip_bfloat16 typedef
 */

#ifndef _HIP_BFLOAT16_H_
#define _HIP_BFLOAT16_H_

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#endif

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/amd_detail/amd_hip_bfloat16.h>
#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#warning "hip_bfloat16.h is not supported on nvidia platform"
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#endif  // _HIP_BFLOAT16_H_
