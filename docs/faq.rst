*******************************************************************************
Frequently asked questions
*******************************************************************************

This topic provides answers to common and frequently asked questions from new HIP
users and users familiar with NVIDIA CUDA.

HIP Support
===========

What NVIDIA CUDA features does HIP support?
-------------------------------------------

The :doc:`NVIDIA CUDA runtime API supported by HIP<hipify:tables/CUDA_Runtime_API_functions_supported_by_HIP>`
and :doc:`NVIDIA CUDA driver API suupported by HIP<hipify:tables/CUDA_Driver_API_functions_supported_by_HIP>`
pages describe which NVIDIA CUDA APIs are supported and what the equivalents are.
The :doc:`HIP API documentation <doxygen/html/index.html>` describes each API and
its limitations, if any, compared with the equivalent CUDA API.

The kernel language features are documented in the
:doc:`/reference/cpp_language_extensions` page.

What libraries does HIP provide?
--------------------------------

HIP provides key math and AI libraries. The full list can be found at
:doc:`rocm:reference/api-libraries`

What hardware does HIP support?
-------------------------------

HIP supports AMD and NVIDIA GPUs. See the
:ref:`prerequisites of the install guide<install_prerequisites>` for detailed
information.

What operating systems does HIP support?
----------------------------------------

The supported operating systems are listed in the
:doc:`rocm:compatibility/compatibility-matrix`.

CUDA and OpenCL
===============

Is HIP a drop-in replacement for CUDA?
--------------------------------------

The `HIPIFY <https://github.com/ROCm/HIPIFY>`_ tools can automatically convert
almost all runtime code. Most device code needs no additional conversion since
HIP and CUDA have similar names for math and built-in functions. HIP code
provides the same performance as native CUDA code, plus the benefits of running
on AMD platforms.

Additional porting might be required to deal with architecture feature
queries or with CUDA capabilities that HIP doesn't support.

How does HIP compare with OpenCL?
---------------------------------

HIP offers several benefits over OpenCL:

* Device code can be written in modern C++, including templates, lambdas,
  classes and so on.
* Host and device code can be mixed in the source files.
* The HIP API is less verbose than OpenCL and is familiar to CUDA developers.
* Porting from CUDA to HIP is significantly easier than porting from CUDA to OpenCL.
* HIP uses development tools specialized for each platform: :ref:`amdclang++ <llvm-project:index>`
  for AMD GPUs or :xref:`nvcc <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`
  for NVIDIA GPUs, and profilers like :ref:`omniperf <omniperf:index>` or
  :xref:`Nsight Systems <https://developer.nvidia.com/nsight-systems>`.
* HIP provides
  * pointers and host-side pointer arithmetic.
  * device-level control over memory allocation and placement.
  * an offline compilation model.

How does porting CUDA to HIP compare to porting CUDA to OpenCL?
---------------------------------------------------------------

OpenCL differs from HIP and CUDA when considering the host runtime,
but even more so when considering the kernel code.
The HIP device code is a C++ dialect, while OpenCL is C99-based.
OpenCL does not support single-source compilation.

As a result, the OpenCL syntax differs significantly from HIP, and porting tools
must perform complex transformations, especially when it comes to templates
or other C++ features in kernels.

To better understand the syntax differences, see :doc:`here<reference/terms>` or
the :doc:`HIP porting guide <how-to/hip_porting_guide>`.

Can I install CUDA and ROCm on the same machine?
------------------------------------------------

Yes, but you still need a compatible GPU to actually run the compiled code.

HIP detected my platform incorrectly. What should I do?
-------------------------------------------------------

See the :doc:`HIP porting guide<how-to/hip_porting_guide>` under the section "Identifying HIP Runtime".

On NVIDIA platforms, can I mix HIP code with CUDA code?
-------------------------------------------------------

Yes. Most HIP types and data structures are `typedef`s to CUDA equivalents and
can be used interchangeably.

See the :doc:`how-to/hip_porting_guide` for more details.

Compiler related questions
==========================

How to use HIP-Clang to build HIP programs?
------------------------------------------------------

:ref:`hipcc <HIPCC:index>` is a compiler driver. This means it is not a compiler,
but calls the appropriate compilers and sets some options.

The underlying compilers are :ref:`amdclang++ <llvm-project:index>` or
:xref:`nvcc <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`,
depending on the platform, and can be called directly.

What is HIP-Clang?
------------------

HIP-Clang is a Clang/LLVM-based compiler used to compile HIP programs for AMD
platforms. Its executable is named :ref:`amdclang++ <llvm-project:index>` on
Linux and ``clang++`` on Windows.

Can I link HIP device code with host code compiled with another compiler such as gcc, icc, or clang?
-----------------------------------------------------------------------------------------------------------

Yes. HIP generates object code that conforms to the GCC ABI, and also links with libstdc++.
This means you can compile host code with the compiler of your choice and link the
generated host object code with device code.

Can HIP applications be compiled with a C compiler?
---------------------------------------------------

HIP is a C/C++ API that can be used with C compilers. However, this applies only
to the API itself. Device code and the syntax for calling kernels need to be
compiled with a supported compiler like :ref:`hipcc <HIPCC:index>`. The code objects that are
generated with ``hipcc`` can however be used with a C compiler, as shown in the
code examples below.

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

Miscellaneous
=============

How to create a guard for code specific to the host or the GPU?
---------------------------------------------------------------

The compiler defines the ``__HIP_DEVICE_COMPILE__`` macro only when compiling
device code.

Refer to the :doc:`how-to/hip_porting_guide` for more information.

Can a HIP binary run on both AMD and NVIDIA platforms?
------------------------------------------------------

HIP is a source-portable language that can be compiled to run on either AMD
or NVIDIA platforms. However, the HIP tools don't create a "fat binary" that can
run on either platform.
