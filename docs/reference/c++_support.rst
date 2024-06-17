.. meta::
  :description: This chapter describes the C++ support of the HIP ecosystem
  ROCm software.
  :keywords: AMD, ROCm, HIP, C++

*******************************************************************************
C++ support
*******************************************************************************

.. _language_introduction:
Introduction
===============================================================================

The main way to harness the power of the ROCm platform is through the usage C++ and HIP
code. This code is then compiled with a ``clang`` or ``clang++`` compiler. The official
versions support the HIP platform, but for the most up-to-date feature set use the
``amdclang`` or ``amdclang++`` installed with your ROCm installation.

The source code will be processed based on the ``C++03``, ``C++11``, ``C++14``, ``C++17``
or ``C++20`` standards, but supports HIP specific extensions and is subject to certain
restrictions. The largest restriction is the lack of support of the standard library.
This is mostly because the SIMD nature of the HIP device makes most of the standard
library implementations not performant or useful. The important operations are
implemented in HIP specific libraries like rocPRIM, rocThrust and hipCUB.

.. _language_modern_c++_support:
Modern C++ support
===============================================================================

The C++11 and later standards had huge additions to C++ making it a truly modern
language. In this section, we'll describe which of these features HIP supports.

C++11 support
-------------------------------------------------------------------------------

The C++11 standard introduced a myriad of new features to the language. These features
are supported in HIP device code, with some notable omissions. The biggest of which is
the lack of concurrency support on the device. This is because the HIP device concurrency
model is fundamentally different compared to the C++ model, which is used on the host
side. E.G: it doesn't make sense to start a new thread on the device.

There are some restrictions and clarifications for certain features. For example, some
new features were like functions using ``initializer lists``, ``std::move`` and
``std::forward`` or functions with ``constexpr`` qualifier are implicitly considered to
have ``__host__`` and ``__device__`` execution space specifier. Also, ``constexpr``
variables that are static member variables or namespace scoped can be used from both host
and device, but only for value access. Dereferencing a static ``constexpr`` on a different
execution space than specified will cause an error.

Lambdas work, but there are some extensions and restrictions on their usage. To learn
more about them, read the `extended lambdas`_ sections.

C++14 support
-------------------------------------------------------------------------------

The C++14 language features are supported, except for the updates to the memory model.

C++17 support
-------------------------------------------------------------------------------

All C++17 language features are supported.

C++20 support
-------------------------------------------------------------------------------

All C++20 language features are supported, but extensions and restrictions apply. C++20
introduced features that change programs foundationally with coroutines and modules, but
unfortunately these aren't supported by HIP. On the other hand, ``consteval`` functions
can be called from host and device, even if it is specified for host use only. 

The three-way comparison operator (also known as spaceship operator ``<=>``) works in
both host and device code.

.. _language_restrictions:
Extensions and Restrictions
===============================================================================

Besides the above mentioned deviations from the standard, there are more general
extensions and restrictions to consider. 

``__global__`` functions
-------------------------------------------------------------------------------

Functions that work as an entry point for device execution are called kernels and are
specified with the ``__global__`` qualifier. Calling these functions happen with the
triple chevron operator: ``<<< >>>``. Kernel functions return type must be ``void``, they
can't be ``constexpr``, can't have a parameter of type ``std::initializer_list`` or
``va_list``, can't have a parameter of rvalue reference type. Kernels can have variadic
template parameters, but only one pack, and it must be the last in the template parameter
list. 

Device Space Memory Specifiers
-------------------------------------------------------------------------------

To specify whether a variable is allocated in host or device memory, HIP has qualifiers
called device space memory specifiers. These are ``__device__``, ``__shared__``,
``__managed__`` and ``__constant__``. These are meant to specify what memory is allocated
for a variable. ``__device__`` and ``__constant__`` specifier are used for static
variables and they will be allocated on the HIP devices global memory, with the only
difference that the ``__constant__`` variables can't be changed from the device code.
``__shared__`` is used in device code, to allocate the variable on the shared memory,
which is available for all threads in a block. ``__managed__`` is also used on global
variables. A managed variable is emitted as an undefined global symbol in the device
binary and is registered by ``__hipRegisterManagedVariable`` in init functions. The HIP
runtime allocates managed memory and uses it to define the symbol when loading the device
binary. A managed variable can be accessed in both device and host code. It is important
to know where each variable is because they will be only available in certain places.
Generally, variables allocated in the host memory will not be available from device code
and variables allocated in the device memory will not be available from host code. This
can be an issue mostly with pointers in host memory, which are pointing to device memory.
Dereferencing these will cause a segmentation fault.

Exception handling
-------------------------------------------------------------------------------

An important difference between the host and device code is exception handling. In device
code it is not available, error handling needs to be done with return codes. This is
because of the hardware architecture does not allow for this kind of control flow.

Kernel parameters
-------------------------------------------------------------------------------

There are some restrictions on kernel function parameters. They cannot be passed by
reference, as these functions run on the device, but are called from the host. Also,
variable number of arguments is not allowed.

Classes
-------------------------------------------------------------------------------

Classes work on both host and device side, but there are some constraints. ``static``
data members need to be ``const`` qualified, and ``static`` member functions can't be
``__global__``. ``Virtual`` member functions work, but it's undefined behaviour to call a
virtual function from the host, when the object was created on the device, or the
other way around. This also means, that you can't pass an object with virtual functions
as a parameter to a kernel.

Polymorphic Function Wrappers
-------------------------------------------------------------------------------

Since C++11 the standard library has a polymorphic function wrapper ``std::function``.
This has an equivalent in CUDA called ``nvstd::function`` to work on CUDA enabled
devices. Unfortunately, HIP doesn't have its own version currently.

Extended Lambdas
-------------------------------------------------------------------------------

Lambdas are a powerful tool in modern C++ and are supported by the HIP ecosystem. By
default, the lambda will work as you expect, but keep in mind that they will inherit the
execution space specification from the surrounding context. For example, in a device
the lambda can only be called from other device functions. This also means that lambdas
can't be used as template argument for kernels unless they are defined in a device
function or a kernel. To help develop versatile software, HIP has an extension making
lambdas even more powerful. They can have ``__host__`` or ``__device__`` qualifiers. This
way developers can define lambdas in host code, that can run on the device side as well,
and used as template parameter for ``__global__`` functions.

Inline namespaces
-------------------------------------------------------------------------------

Inline namespaces are supported, but with a few exceptions. The following entities can't
be declared in namespace scope within an inline unnamed namespace:

* ``__managed__``, ``__device__``, ``__shared__`` and ``__constant__`` variables
* ``__global__`` function and function templates
* variables with surface or texture type
