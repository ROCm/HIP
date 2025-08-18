.. meta::
  :description: This chapter describes HIP's kernel language's C++ support.
  :keywords: AMD, ROCm, HIP, C++ support

################################################################################
Kernel language C++ support
################################################################################

The HIP host API can be compiled with any conforming C++ compiler, as long as no
kernel launch is present in the code.

To compile device code and include kernel launches, a compiler with full HIP
support is needed, such as ``amdclang++``. For more information, see :doc:`ROCm
compilers <llvm-project:index>`.

In host code all modern C++ standards that are supported by the compiler can be
used. Device code compilation has some restrictions on modern C++ standards, but
in general also supports all C++ standards. The biggest restriction is the
reduced support of the C++ standard library in device code, as functions are
only compiled for the host by default. An exception to this are ``constexpr``
functions that are resolved at compile time and can be used in device code.
There are ongoing efforts to implement C++ standard library functionality with
`libhipcxx <https://github.com/ROCm/libhipcxx>`_.

********************************************************************************
Supported kernel language C++ features
********************************************************************************

This section describes HIP's kernel language C++ feature support for the
different versions of the standard.

General C++ features
===============================================================================

Exception handling
-------------------------------------------------------------------------------

An important difference between the host and device code C++ support is
exception handling. In device code, exceptions aren't available due to
the hardware architecture. The device code must use return codes to handle
errors.

Assertions
--------------------------------------------------------------------------------

The ``assert`` function is supported in device code. Assertions are used for
debugging purposes. When the input expression equals zero, the execution will be
stopped. HIP provides its own implementation for ``assert`` for usage in device
code in ``hip/hip_runtime.h``.

.. code-block:: cpp

  void assert(int input)

HIP also provides the function ``abort()`` which can be used to terminate the
application when terminal failures are detected. It is implemented using the
``__builtin_trap()`` function.

This function produces a similar effect as using CUDA's ``asm("trap")``.
In HIP, ``abort()`` terminates the entire application, while in CUDA,
``asm("trap")`` only terminates the current kernel and the application continues
to run.

printf
--------------------------------------------------------------------------------

``printf`` is supported in device code, and can be used just like in host code.

.. code-block:: cpp

  #include <hip/hip_runtime.h>

  __global__ void run_printf() { printf("Hello World\n"); }

  int main() {
    run_printf<<<dim3(1), dim3(1), 0, 0>>>();
  }

Device-Side Dynamic Global Memory Allocation
--------------------------------------------------------------------------------

Device code can use ``new`` or ``malloc`` to dynamically allocate global
memory on the device, and ``delete`` or ``free`` to deallocate global memory.

Classes
--------------------------------------------------------------------------------

Classes work on both host and device side, with some constraints on the device
side.

Member functions with the appropriate qualifiers can be called in host and
device code, and the corresponding overload is executed.

``virtual`` member functions are also supported, however calling these functions
from the host if the object was created on the device, or the other way around,
is undefined behaviour.

The ``__host__``, ``__device__``, ``__managed__``, ``__shared__`` and
``__constant__`` memory space qualifiers can not be applied to member variables.

C++11 support
===============================================================================

``constexpr``
  Full support in device code. ``constexpr`` implicitly defines ``__host__
  __device__``, so standard library functions that are marked ``constexpr`` can
  be used in device code.
  ``constexpr`` variables can be used in both host and device code.

Lambdas
  Lambdas are implicitly marked with ``__host__ __device__``. To mark them as
  only executable for the host or the device, they can be explicitly marked like
  any other function. There are restrictions on variable capture, however. Host
  and device specific variables can only be accessed on other devices or the
  host by explicitly copying them. Accessing captured the variables by
  reference, when the variable is not located on the executing device or host,
  causes undefined behaviour.

Polymorphic function wrappers
  HIP does not support the polymorphic function wrapper ``std::function``


C++14 support
===============================================================================

All `C++14 language features <https://isocpp.org/wiki/faq/cpp14-language>`_ are
supported.

C++17 support
===============================================================================

All `C++17 language features <https://en.cppreference.com/w/cpp/17>`_ are
supported.

C++20 support
===============================================================================

Most `C++20 language features <https://en.cppreference.com/w/cpp/20>`_ are
supported, but some restrictions apply. Coroutines are not available in device
code.

********************************************************************************
Compiler features
********************************************************************************

Pragma Unroll
================================================================================

The unroll pragma for unrolling loops with a compile-time constant is supported:

.. code-block:: cpp

  #pragma unroll 16 /* hint to compiler to unroll next loop by 16 */
  for (int i=0; i<16; i++) ...

.. code-block:: cpp

  #pragma unroll 1 /* tell compiler to never unroll the loop */
  for (int i=0; i<16; i++) ...

.. code-block:: cpp

  #pragma unroll /* hint to compiler to completely unroll next loop. */
  for (int i=0; i<16; i++) ...

In-Line Assembly
================================================================================

GCN ISA In-line assembly can be included in device code.

It has to be mentioned however, that in-line assembly should be used carefully.
For more information, please refer to the
:doc:`Inline ASM statements section of amdclang<llvm-project:reference/rocmcc>`.

A short example program including inline assembly can be found in
`HIP inline_assembly sample
<https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/inline_assembly>`_.

For information on what special AMD GPU hardware features are available
through assembly, please refer to the `ISA manuals of the corresponding
architecture
<https://llvm.org/docs/AMDGPUUsage.html#additional-documentation>`_.

Kernel Compilation
================================================================================

``hipcc`` now supports compiling C++/HIP kernels to binary code objects. The
file format for the binary files is usually ``.co`` which means Code Object.
The following command builds the code object using ``hipcc``.

.. code-block:: bash

  hipcc --genco --offload-arch=[TARGET GPU] [INPUT FILE] -o [OUTPUT FILE]

  [TARGET GPU] = GPU architecture
  [INPUT FILE] = Name of the file containing source code
  [OUTPUT FILE] = Name of the generated code object file

For an example on how to use these object files, refer to the `HIP module_api
sample
<https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/module_api>`_.

Architecture specific code
================================================================================

``amdclang++`` defines ``__gfx*__`` macros based on the GPU architecture to be
compiled for. These macros can be used to include GPU architecture specific
code. Refer to the sample in `HIP gpu_arch sample
<https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/gpu_arch>`_.
