.. meta::
  :description: This chapter describes the hardware features of the different hardware architectures.
  :keywords: AMD, ROCm, HIP, hardware, hardware features, hardware architectures

*******************************************************************************
Hardware features
*******************************************************************************

This page gives an overview of the different hardware architectures and the features they implement. Hardware features do not imply performance, that depends on the specifications found in the `Accelerator and GPU hardware specifications`_ page.

  .. list-table::
      :header-rows: 1
      :name: hardware-features-table
      
      *
        - Hardware feature
        - CDNA1
        - CDNA2
        - CDNA3
      *
        - Atomic functions on 32-bit integer values in global memory
        - ✅
        - ✅
        - ✅
      *
        - Atomic functions on 32-bit integer values in shared memory
        - ✅
        - ✅
        - ✅
      *
        - Atomic functions on 64-bit integer values in global memory
        - ✅
        - ✅
        - ✅
      *
        - Atomic functions on 64-bit integer values in shared memory
        - ✅
        - ✅
        - ✅
      *
        - Atomic addition on 32-bit floating point values in global and shared memory
        - ❌
        - ✅
        - ✅
      *
        - Atomic addition on 64-bit floating point values in global memory and shared memory
        - ❌
        - ✅
        - ✅
      *
        - Error correcting code for SRAMs IP core
        - ✅
        - ✅
        - ✅
      *
        - Memory page migration
        - ✅
        - ✅
        - ✅
      *
        - Thread group split
        - ❌
        - ✅
        - ✅
      *
        - Preload kernel arguments
        - ❌
        - ✅
        - ✅
      *
        - Packed math with 32-bit floating point values
        - ❌
        - ✅
        - ✅
      *
        - Support for float8 bfloat8
        - ❌
        - ❌
        - ✅
      *
        - Support for tfloat32
        - ❌
        - ❌
        - ✅
