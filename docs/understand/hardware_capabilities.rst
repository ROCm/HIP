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
        - RDNA1
        - CDNA1
        - RDNA2
        - CDNA2
        - RDNA3
        - CDNA3
      *
        - Atomic functions on 32-bit integer values in global memory
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Atomic functions on 32-bit integer values in shared memory
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Atomic functions on 64-bit integer values in global memory
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Atomic functions on 64-bit integer values in shared memory
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Atomic addition on 32-bit floating point values in global and shared memory
        - ❌
        - ❌
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Atomic addition on 64-bit floating point values in global memory and shared memory
        - ❌
        - ❌
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Error correcting code for SRAMs IP core
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Memory page migration
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Thread group split
        - ❌
        - ❌
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Preload kernel arguments
        - ❌
        - ❌
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Packed math with 32-bit floating point values
        - ❌
        - ❌
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Support for float8 bfloat8
        - ❌
        - ❌
        - ❌
        - ❌
        - ✅
        - ✅
      *
        - Support for tfloat32
        - ❌
        - ❌
        - ❌
        - ❌
        - ✅
        - ✅
      *
        - Maximum dimensionality of grid of thread blocks
        - 3
        - 3
        - 3
        - 3
        - 3
        - 3
      *
        - Maximum x-, y- or z-dimension of a grid of thread blocks
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
      *
        - Maximum dimensionality of thread block
        - 3
        - 3
        - 3
        - 3
        - 3
        - 3
      *
        - Maximum x-, y- or z-dimension of a block
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
      *
        - Maximum number of threads per block
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
        - :math:`2^{31} - 1`
      *
        - Wavefront size
        - 32
        - 64
        - 32
        - 64
        - 32
        - 64
      *
        - Maximum number of resident grids per device
        - ?
        - ?
        - ?
        - ?
        - ?
        - ?
      *
        - Maximum number of resident blocks per compute unit
        - ?
        - ?
        - ?
        - ?
        - ?
        - ?
      *
        - Maximum number of resident wavefronts per compute unit
        - ?
        - ?
        - ?
        - ?
        - ?
        - ?
      *
        - Maximum number of resident threads per compute unit
        - ?
        - ?
        - ?
        - ?
        - ?
        - ?
