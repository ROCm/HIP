.. meta::
  :description: This chapter describes the device memory of the HIP ecosystem
                ROCm software.
  :keywords: AMD, ROCm, HIP, device memory

.. _device_memory:

*******************************************************************************
Device memory
*******************************************************************************

Device memory exists on the device, e.g. on GPUs in the video random access
memory (VRAM), and is accessible by the kernels operating on the device. Recent
architectures use graphics double data rate (GDDR) synchronous dynamic
random-access memory (SDRAM) such as GDDR6, or high-bandwidth memory (HBM) such
as HBM2e. Device memory can be allocated as global memory, constant, texture or
surface memory.

Global memory
================================================================================

Read-write storage visible to all threads on a given device. There are
specialized versions of global memory with different usage semantics which are
typically backed by the same hardware, but can use different caching paths.

Constant memory
================================================================================

Read-only storage visible to all threads on a given device. It is a limited
segment backed by device memory with queryable size. It needs to be set by the
host before kernel execution. Constant memory provides the best performance
benefit when all threads within a warp access the same address.

Texture memory
================================================================================

Read-only storage visible to all threads on a given device and accessible
through additional APIs. Its origins come from graphics APIs, and provides
performance benefits when accessing memory in a pattern where the
addresses are close to each other in a 2D representation of the memory.

The :ref:`texture management module <texture_management_reference>` of the HIP
runtime API reference contains the functions of texture memory.

Surface memory
================================================================================

A read-write version of texture memory, which can be useful for applications
that require direct manipulation of 1D, 2D, or 3D hipArray_t.

The :ref:`surface objects module <surface_object_reference>` of HIP runtime API
contains the functions for creating, destroying and reading surface memory.