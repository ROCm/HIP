.. meta::
  :description: Maps CUDA API syntax to HIP API syntax with an example
  :keywords: AMD, ROCm, HIP, CUDA, syntax, HIP syntax

********************************************************************************
CUDA to HIP API Function Comparison
********************************************************************************

This page introduces key syntax differences between CUDA and HIP APIs with a focused code
example and comparison table. For a complete list of mappings, visit :ref:`HIPIFY <HIPIFY:index>`.

The following CUDA code example illustrates several CUDA API syntaxes.

.. literalinclude:: ../tools/example_codes/block_reduction.cu
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp

The following table maps CUDA API functions to corresponding HIP API functions, as demonstrated in the
preceding code examples.

.. list-table::
    :header-rows: 1
    :name: syntax-mapping-table

    *
      - CUDA
      - HIP

    *
      - ``#include <cuda_runtime.h>``
      - ``#include <hip/hip_runtime.h>``

    *
      - ``cudaError_t``
      - ``hipError_t``

    *
      - ``cudaEvent_t``
      - ``hipEvent_t``

    *
      - ``cudaStream_t``
      - ``hipStream_t``

    *
      - ``cudaMalloc``
      - ``hipMalloc``

    *
      - ``cudaStreamCreateWithFlags``
      - ``hipStreamCreateWithFlags``

    *
      - ``cudaStreamNonBlocking``
      - ``hipStreamNonBlocking``

    *
      - ``cudaEventCreate``
      - ``hipEventCreate``

    *
      - ``cudaMemcpyAsync``
      - ``hipMemcpyAsync``

    *
      - ``cudaMemcpyHostToDevice``
      - ``hipMemcpyHostToDevice``

    *
      - ``cudaEventRecord``
      - ``hipEventRecord``

    *
      - ``cudaEventSynchronize``
      - ``hipEventSynchronize``

    *
      - ``cudaEventElapsedTime``
      - ``hipEventElapsedTime``

    *
      - ``cudaFree``
      - ``hipFree``

    *
      - ``cudaEventDestroy``
      - ``hipEventDestroy``

    *
      - ``cudaStreamDestroy``
      - ``hipStreamDestroy``

In summary, this comparison highlights the primary differences between CUDA and HIP APIs.
