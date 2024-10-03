*******************************************************************************
Frequently asked questions
*******************************************************************************

HIP Support
===========

What CUDA features does HIP support?
----------------------------------------

The :doc:`hipify:tables/CUDA_Runtime_API_functions_supported_by_HIP`
and :doc:`hipify:tables/CUDA_Driver_API_functions_supported_by_HIP`
pages detail what CUDA APIs are supported and what the equivalents are.
The `HIP API documentation <doxygen/html/index.html>`_ describes each API and
its limitations, if any, compared with the equivalent CUDA API.

The kernel language features are documented in the :doc:`/reference/cpp_language_extensions` page.

What libraries does HIP provide?
--------------------------------

HIP provides key math and AI libraries. The full list can be found here: :doc:`rocm:reference/api-libraries`

What hardware does HIP support?
-------------------------------

* For AMD platforms, see the :doc:`GPU support on Linux <rocm-install-on-linux:reference/system-requirements>`
  or :doc:`GPU support on Windows <rocm-install-on-windows:reference/system-requirements>`.
* For NVIDIA platforms, HIP requires unified memory and should run on any
  device with compute capability 5.0 or higher.

What operating systems does HIP support?
----------------------------------------

The supported operating systems are listed in the :doc:`rocm:compatibility/compatibility-matrix`

CUDA and OpenCL
===============

Is HIP a drop-in replacement for CUDA?
--------------------------------------

The `HIPIFY <https://github.com/ROCm/HIPIFY>`_ tools can automatically convert
almost all runtime code. Most device code needs no additional conversion since
HIP and CUDA have similar names for math and built-in functions. HIP code
provides the same performance as native CUDA code, plus the benefits of running
on AMD platforms.

Additional porting may only be required to deal with architecture feature
queries or with CUDA capabilities that HIP doesn't support.

How does HIP compare with OpenCL?
---------------------------------

HIP offers several benefits over OpenCL:

* Device code can be written in modern C++, including templates, lambdas, classes and so on.
* Host and device code can be mixed in the source files.
* The HIP API is less verbose than OpenCL and is familiar to CUDA developers.
* Porting from CUDA to HIP is significantly easier than porting from CUDA to OpenCL.
* HIP uses the best available development tools on each platform: ``amdclang++`` for AMD GPUs or ``nvcc``
  for NVIDIA GPUs, and profilers like ``omniperf`` or ``Nsight Systems``.
* HIP provides pointers and host-side pointer arithmetic.
* HIP provides device-level control over memory allocation and placement.
* HIP offers an offline compilation model.

How does porting CUDA to HIP compare to porting CUDA to OpenCL?
---------------------------------------------------------------

OpenCL is fairly different from HIP and CUDA when considering the host runtime,
but even more so when it comes to kernel code.
The device code of HIP is a C++ dialect, while OpenCL is C99-based.
OpenCL also does not support single-source compilation.

As a result, the OpenCL syntax is a lot different from HIP, and porting tools
have to perform complex transformations, especially when it comes to templates
or other C++ features in kernels.

An overview over the different syntaxes can be seen `:doc:here<reference/terms>`.

Can I install both CUDA and ROCm on the same machine?
-----------------------------------------------------

Yes. Beware, that you still need a compatible GPU to actually run the compiled code.

HIP detected my platform (``amd`` vs ``nvidia``) incorrectly. What should I do?
-------------------------------------------------------------------------------

See the `:doc:HIP porting guide<how-to/hip_porting_guide>` under the section "Identifying HIP Runtime".

On NVIDIA platforms, can I mix HIP code with CUDA code?
-------------------------------------------------------

Yes. Most HIP data structures are typedefs to CUDA equivalents and can be
intermixed.

See the :doc:`how-to/hip_porting_guide` for more details.

Compiler related questions
==========================

How to use HIP-Clang to build HIP programs?
------------------------------------------------------

``hipcc`` is just a compiler driver, meaning that it is not a compiler in itself,
but instead calls the appropriate compilers and sets some options.

The underlying compilers are ``amdclang++`` or ``nvcc``, depending on the platform,
and can be called directly.

What is HIP-Clang?
------------------

HIP-Clang is a Clang/LLVM based compiler to compile HIP programs for AMD
platforms. Its executable is called ``amdclang++`` on Linux and ``clang++`` on Windows.

Can I link HIP device code with host code compiled with another compiler such as gcc, icc, or clang?
-----------------------------------------------------------------------------------------------------------

Yes. HIP generates object code that conforms to the GCC ABI, and also links with libstdc++.
This means you can compile host code with the compiler of your choice and link the
generated host object code with device code.

Can HIP applications be compiled with a C compiler?
---------------------------------------------------

HIP is a C/C++ API that can be used with C compilers. This applies only to the
API itself, though. Device code and the syntax for calling kernels needs to be
compiled with a supported compiler like hipcc. The code objects that are
generated with hipcc can however be used with a C compiler, as shown in the
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

How to create a guard for code that is specific to the host or the GPU?
-----------------------------------------------------------------------

The compiler defines the ``__HIP_DEVICE_COMPILE__`` macro only when compiling
device code.

Refer to the :doc:`how-to/hip_porting_guide` for more information.

Can a HIP binary run on both AMD and NVIDIA platforms?
------------------------------------------------------

HIP is a source-portable language that can be compiled to run on either the AMD
or the NVIDIA platform. However, HIP tools don't create a "fat binary" that can
run on either platform.

Is the HIP runtime on Windows open source?
------------------------------------------

No, the HIP runtime on Windows depends on PAL, which is not open source.
there is no HIP repository open publicly on Windows.
