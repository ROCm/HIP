.. meta::
  :description: This chapter describes the C++ support of the HIP ecosystem
  ROCm software.
  :keywords: AMD, ROCm, HIP, C++

*******************************************************************************
C++ language support
*******************************************************************************

The ROCm platform enables the power of combined C++ and HIP (Heterogeneous-computing
Interface for Portability) code. This code is compiled with a ``clang`` or ``clang++``
compiler. The official compilers support the HIP platform, or you can use the
``amdclang`` or ``amdclang++`` included in the ROCm installation, which are a wrapper for
the official versions.

The source code is compiled according to the ``C++03``, ``C++11``, ``C++14``, ``C++17``,
and ``C++20`` standards, along with HIP-specific extensions, but is subject to
restrictions. The key restriction is the reduced support of standard library in device
code. This is due to the fact that by default a function is considered to run on host,
except for ``constexpr`` functions, which can run on host and device as well.

.. _language_modern_cpp_support:
Modern C++ support
===============================================================================

C++ is considered a modern programming language as of C++11. This section describes how
HIP supports these new C++ features.

C++11 support
-------------------------------------------------------------------------------

The C++11 standard introduced many new features. These features are supported in HIP host
code, with some notable omissions on the device side. The rule of thumb here is that
``constexpr`` functions work on device, the rest doesn't. This means that some important
functionality like ``std::function`` is missing on the device, but unfortunately the
standard library wasn't designed with HIP in mind, which means that the support is in a
state of "works as-is".

Certain features have restrictions and clarifications. For example, any functions using
the ``constexpr`` qualifier or the new ``initializer lists``, ``std::move`` or
``std::forward`` features are implicitly considered to have the ``__host__`` and
``__device__`` execution space specifier. Also, ``constexpr`` variables that are static
members or namespace scoped can be used from both host and device, but only for read
access. Dereferencing a static ``constexpr`` outside its specified execution space causes
an error.

Lambdas are supported, but there are some extensions and restrictions on their usage. For
more information, see the `Extended lambdas`_ section below.

C++14 support
-------------------------------------------------------------------------------

The C++14 language features are supported.

C++17 support
-------------------------------------------------------------------------------

All C++17 language features are supported.

C++20 support
-------------------------------------------------------------------------------

All C++20 language features are supported, but extensions and restrictions apply. C++20
introduced coroutines and modules, which fundamentally changed how programs are written.
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
chevron operator: ``<<< >>>``. Kernel functions must have a ``void`` return type. These
functions can't:

* have a ``constexpr`` specifier
* have a parameter of type ``std::initializer_list`` or ``va_list``
* use an rvalue reference as a parameter.
* use parameters having different sizes in host and device code, e.g. long double arguments, or structs containing long double members.
* use struct-type arguments which have different layout in host and device code.

Kernels can have variadic template parameters, but only one parameter pack, which must be
the last item in the template parameter list.

Device space memory specifiers
-------------------------------------------------------------------------------

HIP includes device space memory specifiers to indicate whether a variable is allocated
in host or device memory and how its memory should be allocated. HIP supports the
``__device__``, ``__shared__``, ``__managed__``, and ``__constant__`` specifiers.

The ``__device__`` and ``__constant__`` specifiers define global variables, which are
allocated within global memory on the HIP devices. The only difference is that
``__constant__`` variables can't be changed after allocation. The ``__shared__``
specifier allocates the variable within shared memory, which is available for all threads
in a block.

The ``__managed__`` variable specifier creates global variables that are initially
undefined and unaddressed within the global symbol table. The HIP runtime allocates
managed memory and defines the symbol when it loads the device binary. A managed variable
can be accessed in both device and host code.

It's important to know where a variable is stored because it is only available from
certain locations. Generally, variables allocated in the host memory are not accessible
from the device code, while variables allocated in the device memory are not directly
accessible from the host code. Dereferencing a pointer to device memory on the host
results in a segmentation fault. Accessing device variables in host code should be done
through kernel execution or HIP functions like ``hipMemCpyToSymbol``.

Exception handling
-------------------------------------------------------------------------------

An important difference between the host and device code is exception handling. In device
code, this control flow isn't available due to the hardware architecture. The device
code must use return codes to handle errors.

Kernel parameters
-------------------------------------------------------------------------------

There are some restrictions on kernel function parameters. They cannot be passed by
reference, because these functions are called from the host but run on the device. Also,
a variable number of arguments is not allowed.

Classes
-------------------------------------------------------------------------------

Classes work on both the host and device side, but there are some constraints. The
``static`` member functions can't be ``__global__``. ``Virtual`` member functions work,
but a ``virtual`` function must not be called from the host if the parent object was
created on the device, or the other way around, because this behavior is undefined.
Another minor restriction is that ``__device__`` variables, that are global scoped must
have trivial constructors.

Polymorphic function wrappers
-------------------------------------------------------------------------------

HIP doesn't support the polymorphic function wrapper ``std::function``, which was
introduced in C++11.

Extended lambdas
-------------------------------------------------------------------------------

HIP supports Lambdas, which by default work as expected.

Lambdas have implicit host device attributes. This means that they can be executed by
both host and device code, and works the way you would expect. To make a lambda callable
only by host or device code, users can add ``__host__`` or ``__device__`` attribute. The
only restriction is that host variables can only be accessed through copy on the device.
Accessing through reference will cause undefined behavior.

Inline namespaces
-------------------------------------------------------------------------------

Inline namespaces are supported, but with a few exceptions. The following entities can't
be declared in namespace scope within an inline unnamed namespace:

* ``__managed__``, ``__device__``, ``__shared__`` and ``__constant__`` variables
* ``__global__`` function and function templates
* variables with surface or texture type
