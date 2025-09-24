.. meta::
  :description: This page lists frequently asked questions about HIP
  :keywords: AMD, ROCm, HIP, FAQ, frequently asked questions

*******************************************************************************
Frequently asked questions
*******************************************************************************

This topic provides answers to frequently asked questions from new HIP users and
users familiar with NVIDIA CUDA.

HIP Support
===========

What hardware does HIP support?
-------------------------------

HIP supports AMD and NVIDIA GPUs. See
:ref:`prerequisites of the install guide<install_prerequisites>` for detailed
information.

What operating systems does HIP support?
----------------------------------------

Linux as well as Windows are supported by ROCm. The exact versions are listed in
the system requirements for :ref:`rocm-install-on-linux:supported_distributions`
and :ref:`rocm-install-on-windows:supported-skus-win`.

.. note::
   Not all HIP runtime API functions are yet supported on Windows.
   A note is added to those functions' documentation in the
   :ref:`HIP runtime API reference<runtime_api_reference>`.

What libraries does HIP provide?
--------------------------------

HIP provides key math and AI libraries. See :doc:`rocm:reference/api-libraries`
for the full list.

What NVIDIA CUDA features does HIP support?
-------------------------------------------

The :doc:`NVIDIA CUDA runtime API supported by HIP<hipify:reference/tables/CUDA_Runtime_API_functions_supported_by_HIP>`
and :doc:`NVIDIA CUDA driver API supported by HIP<hipify:reference/tables/CUDA_Driver_API_functions_supported_by_HIP>`
pages describe which NVIDIA CUDA APIs are supported and what the equivalents are.
The :ref:`HIP runtime API reference<runtime_api_reference>` describes each API and
its limitations, if any, compared with the equivalent CUDA API.

The kernel language features are documented in the
:doc:`/how-to/hip_cpp_language_extensions` page.

Relation to other GPGPU frameworks
==================================

Is HIP a drop-in replacement for CUDA?
--------------------------------------

The `HIPIFY <https://github.com/ROCm/HIPIFY>`_ tools can automatically convert
almost all CUDA runtime code to HIP. Most device code needs no additional
conversion because HIP and CUDA have the same signatures for math and built-in
functions except for the name. HIP code provides similar performance as native
CUDA code on NVIDIA platforms, plus the benefits of being compilable for AMD
platforms.

Additional porting might be required to deal with architecture feature
queries or CUDA capabilities that HIP doesn't support.

To better understand the syntax differences, see :doc:`CUDA to HIP API Function Comparison <reference/api_syntax>`
or the :doc:`HIP porting guide <how-to/hip_porting_guide>`.

Can I install CUDA and ROCm on the same machine?
------------------------------------------------

Yes, but you require a compatible GPU to run the compiled code.

On NVIDIA platforms, can I mix HIP code with CUDA code?
-------------------------------------------------------

Yes. Most HIP types and data structures are ``typedef`` s to CUDA equivalents and
can be used interchangeably. This can be useful for iteratively porting CUDA code.

See :doc:`how-to/hip_porting_guide` for more details.

Can a HIP binary run on both AMD and NVIDIA platforms?
------------------------------------------------------

HIP is a source-portable language that can be compiled to run on AMD or NVIDIA
platforms. However, the HIP tools don't create a "fat binary" that can run on
both platforms.

Compiler related questions
==========================

hipcc detected my platform incorrectly. What should I do?
---------------------------------------------------------

The environment variable ``HIP_PLATFORM`` can be used to specify the platform
for which the code is going to be compiled with ``hipcc``. See the
:doc:`hipcc environment variables<hipcc:env>` for more information.

.. warning::

   If you specify HIP_PLATFORM=NVIDIA with hipcc, you also need to pass ``-x cu``
   to hipcc when compiling files with the ``.hip`` file extension. Otherwise,
   nvcc will not recognize the ``.hip`` file extension and will fail with
   ``nvcc fatal   : Don't know what to do with  <file>.hip``.

How to use HIP-Clang to build HIP programs?
------------------------------------------------------

:doc:`hipcc <hipcc:index>` is a compiler driver. This means it is not a compiler
but calls the appropriate compilers and sets some options.

The underlying compilers are :doc:`amdclang++ <llvm-project:index>` or
`nvcc <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`_,
depending on the platform, and can be called directly.

What is HIP-Clang?
------------------

HIP-Clang is a Clang/LLVM-based compiler used to compile HIP programs for AMD
platforms. The executable is named :doc:`amdclang++ <llvm-project:index>` on
Linux and ``clang++`` on Windows.

Can I link HIP device code with host code compiled with another compiler such as gcc, icc, or clang?
-----------------------------------------------------------------------------------------------------------

Yes. HIP generates object code that conforms to the GCC ABI, and links with libstdc++.
This means you can compile host code with the compiler of your choice and link the
generated host object code with device code.

Can HIP applications be compiled with a C compiler?
---------------------------------------------------

HIP is a C/C++ API that can be used with C compilers. However, this applies only
to the API itself. Device code and the syntax for calling kernels must be
compiled with a supported compiler like :doc:`hipcc <hipcc:index>`. The code
objects that are generated with ``hipcc`` can, however, be used with a C
compiler, as shown in the code examples below.

The following is the HIP device code, assumed to be saved in ``device.hip``:

.. code-block:: c++

  #include <hip/hip_runtime.h>

  __global__ void kernel(double* array, size_t size){
      const int x = threadIdx.x + blockIdx.x * blockDim.x;
      if(x < size){array[x] = x;}
  };

  extern "C"{
      hipError_t callKernel(int blocks, int threadsPerBlock, double* array, size_t size){
          kernel<<<blocks, threadsPerBlock, 0, hipStreamDefault>>>(array, size);
          return hipGetLastError();
      }
  }

The following is the host code, written in C, saved in ``host.c``:

.. code-block:: c

  #include <hip/hip_runtime_api.h>
  #include <stdio.h>
  #include <stdlib.h>

  #define HIP_CHECK(c) {                                \
     if (c != hipSuccess){                              \
        printf("HIP Error : %s", hipGetErrorString(c)); \
        printf(" %s %d\n", __FILE__, __LINE__);         \
        exit(c);                                        \
     }                                                  \
  }

  // Forward declaration - the implementation needs to be compiled with
  // a device compiler like hipcc or amdclang++
  hipError_t callKernel(int blocks, int threadsPerBlock, double* array, size_t size);

  int main(int argc, char** argv) {
      int blocks = 1024;
      int threadsPerBlock = 256;
      size_t arraySize = blocks * threadsPerBlock;
      double* d_array;
      double* h_array;
      h_array = (double*)malloc(arraySize * sizeof(double));

      HIP_CHECK(hipMalloc((void**)&d_array, arraySize * sizeof(double)));
      HIP_CHECK(callKernel(blocks, threadsPerBlock, d_array, arraySize));
      HIP_CHECK(hipMemcpy(h_array, d_array, arraySize * sizeof(double), hipMemcpyDeviceToHost));
      HIP_CHECK(hipFree(d_array));

      free(h_array);
      return 0;
  }

These files are then compiled and linked using

.. code-block:: shell

  hipcc -c device.hip
  gcc host.c device.o $(hipconfig --cpp_config) -L/opt/rocm/lib -lamdhip64

assuming the default installation of ROCm in ``/opt/rocm``.

How to guard code specific to the host or the GPU?
--------------------------------------------------

The compiler defines the ``__HIP_DEVICE_COMPILE__`` macro only when compiling
device code.

Refer to the :doc:`how-to/hip_porting_guide` for more information.
