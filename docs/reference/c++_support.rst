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
versions support the HIP platform, but for the most up to date feature set use the
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
language. In this section we'll describe which of these features HIP supports.

C++11 support
-------------------------------------------------------------------------------

The C++11 standard introduced a myriad of new features to the language. These features
are supported in HIP device code, with some notable omissions. The biggest of which is
the lack of concurrency support on the device. This is because the HIP device concurrency
model is fundamentally different compared to the C++ model, which is used on the host
side. E.G: it doesn't make sense to start a new thread on the device.

There are some restrictions and clarifications for certain features. For example lambdas
can't be used as template argument for kernels unless they are defined in a device
function or a kernel. Also some new features were like functions using ``initializer
lists``, ``std::move`` and ``std::forward`` or functions with ``constexpr`` qualifier are
implicitly considered to have ``__host__`` and ``__device__`` execution space specifier.

C++14 support
-------------------------------------------------------------------------------

The C++14 language features are supported, except for the updates to the memory model.

C++17 support
-------------------------------------------------------------------------------

All C++17 language features are supported.

C++20 support
-------------------------------------------------------------------------------

All C++20 language features are supported, but extensions and restrictions apply. For
example coroutines are not supported in device code and consteval functions can be called
from host and device, even if it is specified for host use only.


.. _language_restrictions:
Extensions and Restrictions
===============================================================================

Besides the above mentioned deviations from the standard, there are more general
extensions and restrictions to consider. 

Device Space Memory Specifiers
-------------------------------------------------------------------------------

To specify whether a variable is allocated in host or device memory HIP has qualifiers
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
Generally variables allocated in the host memory will not be available from device code
and variables allocated in the device memory will not be available from host code. This
can be an issue mostly with pointers in host memory, which are pointing to device memory.
Dereferencing these will cause a segmentation fault.

Exception handling
-------------------------------------------------------------------------------

An important difference between the host and device code is exception handling. In device
code it is not available, error handling needs to be done with return codes. This is
because of the hardware architecture does not allow for this kind o control flow.

Kernel parameters
-------------------------------------------------------------------------------

There are some restrictions on kernel function parameters. They cannot be passed by
reference, as these functions run on the device, but are called from the host. Also
variable number of arguments is not allowed.

Classes
-------------------------------------------------------------------------------

Classes work on both host and device side, but there are some constraints. ``Static``
data members need to be ``const`` qualified and ``static`` member functions can't be
``__global__``. ``Virtual`` member functions work, but it's undefined behaviour to call a
virtual function from the host, when the object was created on the device, or the
other way around. This also means, that you can't pass an object with virtual functions
as a parameter to a kernel.

Extended Lambdas
-------------------------------------------------------------------------------

Lambdas are a powerful tool in modern C++ and in HIP they have an extension making them
even more powerful. Lambdas can have ``__host__`` or ``__device__`` qualifiers. This way
developers can define lambdas in host code, that can run on the device side as well, and
used as template parameter for ``__global__`` functions.

