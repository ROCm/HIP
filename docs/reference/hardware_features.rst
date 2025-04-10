.. meta::
  :description: This chapter describes the hardware features of the different hardware architectures.
  :keywords: AMD, ROCm, HIP, hardware, hardware features, hardware architectures

*******************************************************************************
Hardware features
*******************************************************************************

This page gives an overview of the different hardware architectures and the
features they implement. Hardware features do not imply performance, that
depends on the specifications found in the :doc:`rocm:reference/gpu-arch-specs`
page.

  .. list-table::
      :header-rows: 1
      :name: hardware-features-table

      *
        - Hardware feature support
        - RDNA1
        - CDNA1
        - RDNA2
        - CDNA2
        - RDNA3
        - CDNA3
      *
        - :ref:`atomic functions` on 32-bit integer values in global and shared memory
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Atomic functions on 64-bit integer values in global and shared memory
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
        - :ref:`Warp vote functions <warp_vote_functions>`
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - :ref:`Memory fence instructions <memory_fence_instructions>`
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - :ref:`Synchronization functions <synchronization_functions>`
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - :ref:`Surface functions <surface_object_reference>`
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - :ref:`float16 half precision IEEE-conformant floating-point operations<rocm:precision_support_floating_point_types>`
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - :ref:`bfloat16 16-bit floating-point operations<rocm:precision_support_floating_point_types>`
        - ❌
        - ✅
        - ❌
        - ✅
        - ✅
        - ✅
      *
        - Support for :ref:`8-bit floating-point types <rocm:precision_support_floating_point_types>`
        - ❌
        - ❌
        - ❌
        - ❌
        - ❌
        - ✅
      *
        - Support for :ref:`tensor float32 <rocm:precision_support_floating_point_types>`
        - ❌
        - ❌
        - ❌
        - ❌
        - ❌
        - ✅
      *
        - Packed math with 16-bit floating point values
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Packed math with 32-bit floating point values
        - ❌
        - ❌
        - ❌
        - ✅
        - ❌
        - ✅
      *
        - Matrix Cores
        - ❌
        - ✅
        - ❌
        - ✅
        - ❌
        - ✅
      *
        - On-Chip Error Correcting Code (ECC)
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - Maximum dimensionality of grid
        - 3
        - 3
        - 3
        - 3
        - 3
        - 3
      *
        - Maximum x-, y- and z-dimension of a grid
        - x - :math:`2^{32}-1` y - :math:`2^{16}-1` z - :math:`2^{16}-1`
        - x - :math:`2^{32}-1` y - :math:`2^{16}-1` z - :math:`2^{16}-1`
        - x - :math:`2^{32}-1` y - :math:`2^{16}-1` z - :math:`2^{16}-1`
        - x - :math:`2^{32}-1` y - :math:`2^{16}-1` z - :math:`2^{16}-1`
        - x - :math:`2^{32}-1` y - :math:`2^{16}-1` z - :math:`2^{16}-1`
        - x - :math:`2^{32}-1` y - :math:`2^{16}-1` z - :math:`2^{16}-1`
      *
        - Maximum number of threads per grid
        - :math:`2^{32} - 1`
        - :math:`2^{32} - 1`
        - :math:`2^{32} - 1`
        - :math:`2^{32} - 1`
        - :math:`2^{32} - 1`
        - :math:`2^{32} - 1`
      *
        - Maximum x-, y- and z-dimension of a block
        - :math:`1024`
        - :math:`1024`
        - :math:`1024`
        - :math:`1024`
        - :math:`1024`
        - :math:`1024`
      *
        - Maximum number of threads per block
        - :math:`1024`
        - :math:`1024`
        - :math:`1024`
        - :math:`1024`
        - :math:`1024`
        - :math:`1024`
      *
        - Wavefront size
        - 32 [1]_
        - 64
        - 32 [1]_
        - 64
        - 32 [1]_
        - 64
      *
        - Maximum number of resident blocks per compute unit
        - 40 [1]_
        - 32
        - 32 [1]_
        - 32
        - 32 [1]_
        - 32
      *
        - Maximum number of resident wavefronts per compute unit
        - 40 [1]_
        - 32
        - 32 [1]_
        - 32
        - 32 [1]_
        - 32
      *
        - Maximum number of resident threads per compute unit
        - 1280 [2]_
        - 2048
        - 1024 [2]_
        - 2048
        - 1024 [2]_
        - 2048
      *
        - Maximum number of 32-bit vector registers per thread
        - 256
        - 256 (vector) + 256 (matrix)
        - 256
        - 256 (vector) + 256 (matrix)
        - 256
        - 256 (vector) + 256 (matrix)
      *
        - Maximum number of 32-bit scalar accumulation registers per thread
        - 106
        - 104
        - 106
        - 104
        - 106
        - 104

.. [1] The RDNA architectures feature an experimental compiler option called 
   ``mwavefrontsize64``, which determines the wavefront size for kernel code
   generation. When this option is disabled, the native wavefront size of 32 is
   used, when enabled wavefront size 64 is used. This option is not supported by
   the HIP runtime.

.. [2] RDNA architectures expand the concept of the traditional compute unit
   with the so-called work group processor, which effectively includes two
   compute units, within which all threads can cooperate.
