.. meta::
  :description: This chapter describes the C++ support of the HIP ecosystem
  ROCm software.
  :keywords: AMD, ROCm, HIP, C++

*******************************************************************************
C++ language support
*******************************************************************************

.. _language_introduction:
Introduction
===============================================================================

The ROCm platform enables the power of combined C++ and HIP code. This code is compiled
with a ``clang`` or ``clang++`` compiler. The official compilers support the HIP
platform, but for the most up-to-date feature set, use the ``amdclang`` or ``amdclang++``
components included in the ROCm installation.

The source code is compiled according to the ``C++03``, ``C++11``, ``C++14``, ``C++17``,
and ``C++20`` standards, along with HIP-specific extensions, but is subject to
restrictions. The key restriction is the lack of standard library support. This is due to
the Single Instruction/Multiple Data (SIMD) nature of the HIP device, which makes most of
the standard library implementations not performant or useful. However, essential
operations are implemented in HIP-specific libraries such as `rocPRIM
<https://github.com/ROCm/rocprim>`_, `rocThrust <https://github.com/ROCm/rocthrust/>`_,
and `hipCUB <https://github.com/ROCm/hipcub/>`_.

.. _language_modern_c++_support:
Modern C++ support
===============================================================================

C++ is considered a modern programming language as of C++11. This section describes how
HIP supports these new C++ features.

C++11 support
-------------------------------------------------------------------------------

The C++11 standard introduced many new features. These features are supported in HIP
device code, with some notable omissions. The most significant is the lack of concurrency
support on the device. This is because the HIP device concurrency model fundamentally
differs from the C++ model used on the host side. For example, it is not necessary or
possible to start a new thread on the device.

Certain features have restrictions and clarifications. For example, any functions using
the ``constexpr`` qualifier or the new ``initializer lists``, ``std::move`` or
``std::forward`` features are implicitly considered to have the ``__host__`` and
``__device__`` execution space specifier. Also, constexpr variables that are static
members or namespace scoped can be used from both host and device, but only for read
access. Dereferencing a static ``constexpr`` outside its specified execution space causes
an error.

Lambdas are supported, but there are some extensions and restrictions on their usage. For
more information, see the `Extended lambdas`_ section below.

C++14 support
-------------------------------------------------------------------------------

The C++14 language features are supported, except for the updates to the memory model.

C++17 support
-------------------------------------------------------------------------------

All C++17 language features are supported.

C++20 support
-------------------------------------------------------------------------------

All C++20 language features are supported, but extensions and restrictions apply. C++20
introduced coroutines and modules, which fundamentally change how programs are written.
HIP doesn't support these features. However, ``consteval`` functions can be called from
host and device, even if specified for host use only.

The three-way comparison operator (spaceship operator ``<=>``) works with host and device
code.

.. _language_restrictions:
Extensions and restrictions
===============================================================================

In addition to the deviations from the standard, there are some general extensions and
restrictions to consider.

Global functions
-------------------------------------------------------------------------------

Functions that serve as an entry point for device execution are called kernels and are
specified with the ``__global__`` qualifier. To call a kernel function, use the triple
chevron operator: ``<<< >>>``. Kernel functions must have a void return type. These
functions can't:

* have a ``constexpr`` specifier
* have a parameter of type ``std::initializer_list`` or ``va_list``
* use an rvalue reference as a parameter.

Kernels can have variadic template parameters, but they can only have one parameter pack,
which must be the last item in the template parameter list.

Device space memory specifiers
-------------------------------------------------------------------------------

HIP includes device space memory specifiers to indicate whether a variable is allocated
in host or device memory and how its memory should be allocated. HIP supports the
``__device__``, ``__shared__``, ``__managed__`` and ``__constant__`` specifiers.

The ``__device__`` and ``__constant__`` specifiers define static variables, which are
allocated within global memory on the HIP devices. The only difference is that
``__constant__`` variables can't be changed after allocation. The ``__shared__``
specifier allocates the variable within shared memory, which is available for all threads
in a block.

The ``__managed__`` variable specifier is used to create global variables that are
initially undefined and unaddressed within the global symbol table. The HIP runtime
allocates managed memory and defines the symbol when it loads the device binary. A
managed variable can be accessed in both device and host code. To register managed
variables, use ``__hipRegisterManagedVariable`` in an initialization function.

It's important to know where a variable is stored because it is only available from
certain locations. Generally, variables allocated in the host memory are not accessible
from the device code, while variables allocated in the device memory are not accessible
from the host code. Dereferencing a pointer to device memory on the host results in a
segmentation fault.

Exception handling
-------------------------------------------------------------------------------

An important difference between the host and device code is exception handling. In device
code, this control flow is not available due to the hardware architecture. The device
code must use return codes to handle errors.

Kernel parameters
-------------------------------------------------------------------------------

There are some restrictions on kernel function parameters. They cannot be passed by
reference, because these functions are called from the host but run on the device. Also,
a variable number of arguments is not allowed.

Classes
-------------------------------------------------------------------------------

Classes work on both the host and device side, but there are some constraints. The static
data members need to be ``const`` qualified, and ``static`` member functions can't be
``__global__``. ``Virtual`` member functions work, but a ``virtual`` function must not be
called from the host if the parent object was created on the device, or the other way
around, because this behavior is undefined. This also means you can't pass an object with
``virtual`` functions as a parameter to a kernel.

Polymorphic function wrappers
-------------------------------------------------------------------------------

HIP doesn't support the polymorphic function wrapper ``std::function``, which was
introduced in C++11.

Extended lambdas
-------------------------------------------------------------------------------

HIP supports Lambdas, which by default work as expected.

Lambdas inherit the execution space specification from the surrounding context. For
example, in a device, the lambda can only be called from other device functions. This
also means that lambdas can't be used as a template argument for kernels unless they are
defined in a device function or a kernel.

To help develop versatile software, HIP supports an extension that makes lambdas even
more powerful. They can have ``__host__`` or ``__device__`` qualifiers. Developers can
use this feature to define lambdas in host code that can run on the device side and be
used as a template parameter for ``__global__`` functions.

Inline namespaces
-------------------------------------------------------------------------------

Inline namespaces are supported, but with a few exceptions. The following entities can't
be declared in namespace scope within an inline unnamed namespace:

* ``__managed__``, ``__device__``, ``__shared__`` and ``__constant__`` variables
* ``__global__`` function and function templates
* variables with surface or texture type
