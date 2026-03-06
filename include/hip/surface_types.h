/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 *  @file  surface_types.h
 *  @brief Defines surface types for HIP runtime.
 */

#ifndef HIP_INCLUDE_HIP_SURFACE_TYPES_H
#define HIP_INCLUDE_HIP_SURFACE_TYPES_H

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"
#endif

#if !defined(__HIPCC_RTC__)
#include <hip/driver_types.h>
#endif

/**
 * An opaque value that represents a hip surface object
 */
struct __hip_surface;
typedef struct __hip_surface* hipSurfaceObject_t;

/**
 * hip surface reference
 */
struct surfaceReference {
  hipSurfaceObject_t surfaceObject;
};

/**
 * hip surface boundary modes
 */
enum hipSurfaceBoundaryMode {
  hipBoundaryModeZero = 0,
  hipBoundaryModeTrap = 1,
  hipBoundaryModeClamp = 2
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif /* !HIP_INCLUDE_HIP_SURFACE_TYPES_H */
